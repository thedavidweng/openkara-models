"""
ONNX-compatible real-valued STFT and ISTFT implementations.

ONNX does not support complex tensors or torch.stft/torch.istft.
These modules rewrite STFT/ISTFT using conv1d with precomputed DFT
filter matrices, following the approach from sevagh/demucs.onnx and
the Mixxx GSOC 2025 project.

All operations are real-valued and use standard ONNX-exportable ops.
"""

import math
import types

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
    k = torch.arange(n_freqs, dtype=torch.float32).unsqueeze(1)
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
    angles = 2.0 * math.pi * k * n / n_fft

    cos_filters = torch.cos(angles).unsqueeze(1)
    sin_filters = -torch.sin(angles).unsqueeze(1)
    return cos_filters, sin_filters


def _build_idft_filters(n_fft: int):
    """Build real and imaginary inverse DFT filter matrices for ISTFT.

    Returns:
        cos_filters: (1, n_fft//2+1, n_fft) - real part
        sin_filters: (1, n_fft//2+1, n_fft) - imaginary part
    """
    n_freqs = n_fft // 2 + 1
    k = torch.arange(n_freqs, dtype=torch.float32).unsqueeze(1)
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
    angles = 2.0 * math.pi * k * n / n_fft

    cos_filters = torch.cos(angles)
    sin_filters = torch.sin(angles)

    scale = torch.ones(n_freqs, 1)
    scale[1:-1] = 2.0
    cos_filters = cos_filters * scale
    sin_filters = sin_filters * scale

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

        cos_filters = cos_filters * window.unsqueeze(0).unsqueeze(0)
        sin_filters = sin_filters * window.unsqueeze(0).unsqueeze(0)

        norm = 1.0 / math.sqrt(n_fft)
        cos_filters = cos_filters * norm
        sin_filters = sin_filters * norm

        self.register_buffer("cos_filters", cos_filters)
        self.register_buffer("sin_filters", sin_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT of input signal.

        Args:
            x: (batch, channels, frames)

        Returns:
            Complex spectrogram as (batch, channels, freq_bins, time_steps, 2)
            where last dim is [real, imag].
        """
        batch, channels, frames = x.shape
        x = x.reshape(batch * channels, frames)

        pad_amount = self.n_fft // 2
        x = F.pad(x, (pad_amount, pad_amount), mode="reflect")

        x = x.unsqueeze(1)

        real = F.conv1d(x, self.cos_filters, stride=self.hop_length)
        imag = F.conv1d(x, self.sin_filters, stride=self.hop_length)

        spec = torch.stack([real, imag], dim=-1)

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

        norm = 1.0 / math.sqrt(n_fft)
        cos_filters = cos_filters * norm
        sin_filters = sin_filters * norm

        cos_filters = cos_filters * window.unsqueeze(0).unsqueeze(0)
        sin_filters = sin_filters * window.unsqueeze(0).unsqueeze(0)

        self.register_buffer("cos_filters", cos_filters)
        self.register_buffer("sin_filters", sin_filters)

        self.register_buffer("window_sq_kernel", (window * window).view(1, 1, n_fft))

    def forward(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Compute ISTFT to reconstruct time-domain signal.

        Args:
            spec: (batch, channels, freq_bins, time_steps, 2) where last dim is [real, imag]
            length: desired output length

        Returns:
            Reconstructed signal: (batch, channels, frames)
        """
        batch, channels, n_freqs, time_steps, _ = spec.shape

        real = spec[..., 0]
        imag = spec[..., 1]

        real = real.reshape(batch * channels, n_freqs, time_steps)
        imag = imag.reshape(batch * channels, n_freqs, time_steps)

        cos_f = self.cos_filters.squeeze(0).unsqueeze(1)
        sin_f = self.sin_filters.squeeze(0).unsqueeze(1)

        signal = F.conv_transpose1d(real, cos_f, stride=self.hop_length)
        signal = signal - F.conv_transpose1d(imag, sin_f, stride=self.hop_length)

        ones = torch.ones(batch * channels, 1, time_steps, device=spec.device, dtype=spec.dtype)
        window_envelope = F.conv_transpose1d(
            ones,
            self.window_sq_kernel,
            stride=self.hop_length,
        )

        window_envelope = torch.clamp(window_envelope, min=1e-8)
        signal = signal / window_envelope

        pad_amount = self.n_fft // 2
        signal = signal[:, :, pad_amount:]
        signal = signal[:, :, :length]

        signal = signal.squeeze(1)

        signal = signal.reshape(batch, channels, -1)

        return signal


class RealValuedSpectrogramPatch:
    """Replace an HTDemucs model's complex spectrogram path with real-valued ops.

    HTDemucs uses complex STFT/ISTFT via torch.stft/torch.istft, which ONNX
    cannot export. This patch swaps the four spectrogram methods (_spec,
    _ispec, _magnitude, _mask) with real-valued conv1d/conv_transpose1d
    implementations built from OnnxSTFT/OnnxISTFT.

    Use as a context manager, or call apply()/restore() explicitly:

        with RealValuedSpectrogramPatch.from_model(model):
            torch.onnx.export(model, ...)

    The model must have cac=True, nfft, hop_length, segment, and samplerate
    attributes (the HTDemucs interface).
    """

    def __init__(self, model, stft_module, istft_module):
        self.model = model
        self.stft_module = stft_module
        self.istft_module = istft_module
        self._originals = {}
        self._applied = False

    @classmethod
    def from_model(cls, model):
        if not model.cac:
            raise RuntimeError(
                "RealValuedSpectrogramPatch expects an HTDemucs model with cac=True"
            )
        stft = OnnxSTFT(n_fft=model.nfft, hop_length=model.hop_length)
        istft = OnnxISTFT(n_fft=model.nfft, hop_length=model.hop_length)
        return cls(model, stft, istft)

    def apply(self):
        if self._applied:
            return
        from demucs.hdemucs import pad1d

        stft_module = self.stft_module
        istft_module = self.istft_module

        self._originals = {
            "_spec": self.model._spec,
            "_ispec": self.model._ispec,
            "_magnitude": self.model._magnitude,
            "_mask": self.model._mask,
        }

        def exportable_spec(self, x):
            hl = self.hop_length
            nfft = self.nfft
            assert hl == nfft // 4
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
            z = stft_module(x)[:, :, :-1, :, :]
            assert z.shape[-2] == le + 4, (z.shape, x.shape, le)
            z = z[:, :, :, 2: 2 + le, :]
            return z.permute(0, 1, 4, 2, 3).contiguous()

        def exportable_ispec(self, z, length=None, scale=0):
            assert scale == 0, "Scaled ISTFT export is not implemented"
            hl = self.hop_length
            z = F.pad(z, (0, 0, 0, 1))
            z = F.pad(z, (2, 2))
            pad = hl // 2 * 3
            le = hl * int(math.ceil(length / hl)) + 2 * pad

            batch, sources, channels, _, freqs, frames = z.shape
            z = z.permute(0, 1, 2, 4, 5, 3).contiguous()
            z = z.view(batch * sources, channels, freqs, frames, 2)
            x = istft_module(z, length=le)
            x = x.view(batch, sources, channels, le)
            return x[..., pad: pad + length]

        def exportable_magnitude(self, z):
            batch, channels, _, freqs, frames = z.shape
            return z.reshape(batch, channels * 2, freqs, frames)

        def exportable_mask(self, z, m):
            batch, sources, _, freqs, frames = m.shape
            return m.view(batch, sources, -1, 2, freqs, frames).contiguous()

        self.model._spec = types.MethodType(exportable_spec, self.model)
        self.model._ispec = types.MethodType(exportable_ispec, self.model)
        self.model._magnitude = types.MethodType(exportable_magnitude, self.model)
        self.model._mask = types.MethodType(exportable_mask, self.model)
        self._applied = True

    def restore(self):
        if not self._applied:
            return
        for name, method in self._originals.items():
            setattr(self.model, name, method)
        self._originals = {}
        self._applied = False

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
        return False
