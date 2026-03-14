"""
ONNX-compatible real-valued STFT and ISTFT implementations.

ONNX does not support complex tensors or torch.stft/torch.istft.
These modules rewrite STFT/ISTFT using conv1d with precomputed DFT
filter matrices, following the approach from sevagh/demucs.onnx and
the Mixxx GSOC 2025 project.

All operations are real-valued and use standard ONNX-exportable ops.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_dft_filters(n_fft: int):
    """Build real and imaginary DFT filter matrices for STFT.

    Returns:
        cos_filters: (n_fft//2+1, 1, n_fft) - real part
        sin_filters: (n_fft//2+1, 1, n_fft) - imaginary part
    """
    n_freqs = n_fft // 2 + 1
    # DFT frequencies
    k = torch.arange(n_freqs, dtype=torch.float32).unsqueeze(1)  # (n_freqs, 1)
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)  # (1, n_fft)
    angles = 2.0 * math.pi * k * n / n_fft  # (n_freqs, n_fft)

    cos_filters = torch.cos(angles).unsqueeze(1)  # (n_freqs, 1, n_fft)
    sin_filters = -torch.sin(angles).unsqueeze(1)  # (n_freqs, 1, n_fft)
    return cos_filters, sin_filters


def _build_idft_filters(n_fft: int):
    """Build real and imaginary inverse DFT filter matrices for ISTFT.

    Returns:
        cos_filters: (1, n_fft//2+1, n_fft) - real part
        sin_filters: (1, n_fft//2+1, n_fft) - imaginary part
    """
    n_freqs = n_fft // 2 + 1
    k = torch.arange(n_freqs, dtype=torch.float32).unsqueeze(1)  # (n_freqs, 1)
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)  # (1, n_fft)
    angles = 2.0 * math.pi * k * n / n_fft  # (n_freqs, n_fft)

    # For inverse DFT, the first and last frequency bins contribute once,
    # others contribute twice (conjugate symmetry)
    cos_filters = torch.cos(angles)  # (n_freqs, n_fft)
    sin_filters = torch.sin(angles)  # (n_freqs, n_fft)

    # Scale: DC and Nyquist bins contribute once, others contribute twice
    scale = torch.ones(n_freqs, 1)
    scale[1:-1] = 2.0
    cos_filters = cos_filters * scale
    sin_filters = sin_filters * scale

    # Reshape for conv_transpose1d: (1, n_freqs, n_fft)
    cos_filters = cos_filters.unsqueeze(0)
    sin_filters = sin_filters.unsqueeze(0)
    return cos_filters, sin_filters


class OnnxSTFT(nn.Module):
    """ONNX-exportable STFT using conv1d with precomputed DFT filters.

    Matches torch.stft behavior with:
        center=True, pad_mode='reflect', normalized=True,
        window=hann_window, return_complex=True (as real/imag pair)
    """

    def __init__(self, n_fft: int = 4096, hop_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft, dtype=torch.float32)
        cos_filters, sin_filters = _build_dft_filters(n_fft)

        # Apply window to filters
        # cos_filters: (n_freqs, 1, n_fft), window: (n_fft,)
        cos_filters = cos_filters * window.unsqueeze(0).unsqueeze(0)
        sin_filters = sin_filters * window.unsqueeze(0).unsqueeze(0)

        # Normalization factor (matches torch.stft normalized=True)
        norm = 1.0 / math.sqrt(n_fft)
        cos_filters = cos_filters * norm
        sin_filters = sin_filters * norm

        self.register_buffer("cos_filters", cos_filters)
        self.register_buffer("sin_filters", sin_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT of input signal.

        Args:
            x: (batch, channels, frames) or (batch * channels, frames)

        Returns:
            Complex spectrogram as (batch, channels, freq_bins, time_steps, 2)
            where last dim is [real, imag].
        """
        shape = x.shape
        if x.dim() == 3:
            batch, channels, frames = shape
            x = x.reshape(batch * channels, frames)
        else:
            batch = shape[0]
            channels = 1
            frames = shape[-1]

        # Center padding (reflect)
        pad_amount = self.n_fft // 2
        x = F.pad(x, (pad_amount, pad_amount), mode="reflect")

        # Add channel dim for conv1d: (batch*ch, 1, padded_frames)
        x = x.unsqueeze(1)

        # Compute real and imaginary parts via conv1d
        real = F.conv1d(x, self.cos_filters, stride=self.hop_length)
        imag = F.conv1d(x, self.sin_filters, stride=self.hop_length)

        # Stack as last dimension: (..., 2)
        # real/imag shape: (batch*ch, n_freqs, time_steps)
        spec = torch.stack([real, imag], dim=-1)

        # Reshape back
        n_freqs = spec.shape[1]
        time_steps = spec.shape[2]
        spec = spec.reshape(batch, channels, n_freqs, time_steps, 2)

        return spec


class OnnxISTFT(nn.Module):
    """ONNX-exportable ISTFT using conv_transpose1d with overlap-add.

    Matches torch.istft behavior with:
        center=True, normalized=True, window=hann_window
    """

    def __init__(self, n_fft: int = 4096, hop_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft, dtype=torch.float32)
        cos_filters, sin_filters = _build_idft_filters(n_fft)

        # Normalization (matches torch.istft normalized=True)
        norm = 1.0 / math.sqrt(n_fft)
        cos_filters = cos_filters * norm
        sin_filters = sin_filters * norm

        # Apply window to inverse filters
        # filters: (1, n_freqs, n_fft), window: (n_fft,)
        cos_filters = cos_filters * window.unsqueeze(0).unsqueeze(0)
        sin_filters = sin_filters * window.unsqueeze(0).unsqueeze(0)

        self.register_buffer("cos_filters", cos_filters)
        self.register_buffer("sin_filters", sin_filters)

        # Overlap-add normalization depends only on the window, not on the
        # Fourier basis. Using the squared synthesis filters here introduces a
        # scale error in the reconstructed waveform.
        self.register_buffer("window_sq_kernel", (window * window).view(1, 1, n_fft))

    def forward(self, spec: torch.Tensor, length: int = 0) -> torch.Tensor:
        """Compute ISTFT to reconstruct time-domain signal.

        Args:
            spec: (batch, channels, freq_bins, time_steps, 2) where last dim is [real, imag]
            length: desired output length (0 = auto)

        Returns:
            Reconstructed signal: (batch, channels, frames)
        """
        batch, channels, n_freqs, time_steps, _ = spec.shape

        real = spec[..., 0]  # (batch, ch, freq, time)
        imag = spec[..., 1]

        # Reshape for conv_transpose1d
        real = real.reshape(batch * channels, n_freqs, time_steps)
        imag = imag.reshape(batch * channels, n_freqs, time_steps)

        # Transpose for conv_transpose1d: (batch*ch, n_freqs, time_steps)
        # cos/sin_filters: (1, n_freqs, n_fft) -> need (n_freqs, 1, n_fft) for conv_transpose1d
        cos_f = self.cos_filters.squeeze(0).unsqueeze(1)  # (n_freqs, 1, n_fft)
        sin_f = self.sin_filters.squeeze(0).unsqueeze(1)  # (n_freqs, 1, n_fft)

        # ISTFT: sum of real*cos - imag*sin across frequency bins
        # Using conv_transpose1d for overlap-add
        signal = F.conv_transpose1d(real, cos_f, stride=self.hop_length)
        signal = signal - F.conv_transpose1d(imag, sin_f, stride=self.hop_length)

        # Compute the standard overlap-add normalization envelope.
        ones = torch.ones(batch * channels, 1, time_steps, device=spec.device, dtype=spec.dtype)
        window_envelope = F.conv_transpose1d(
            ones,
            self.window_sq_kernel,
            stride=self.hop_length,
        )

        # Avoid division by zero
        window_envelope = torch.clamp(window_envelope, min=1e-8)
        signal = signal / window_envelope

        # Remove center padding
        pad_amount = self.n_fft // 2
        signal = signal[:, :, pad_amount:]

        if length > 0:
            signal = signal[:, :, :length]

        # Sum across the single output channel from conv_transpose1d
        # signal shape: (batch*ch, 1, frames)
        signal = signal.squeeze(1)

        # Reshape back
        signal = signal.reshape(batch, channels, -1)

        return signal
