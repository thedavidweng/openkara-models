"""Spectral-core trace patch for HTDemucs export (issue #23 PR 2).

Exports the Demucs neural-network core against the spectral tensor contract
(docs/spectral-core-contract.md, ``openkara.spectral-contract/v1``): the
waveform<->spectral transforms move to the application, so the exported graph
receives the spectral tensor as a public input and emits the pre-ISTFT tensor
as a public output. No STFT/ISTFT modules, dense DFT filter banks, overlap
envelopes, or waveform padding/cropping are traced into the graph.

Mechanism — HTDemucs.forward is traced unmodified (architecture and weights
stay identical to the reference), with the four spectrogram methods swapped:

``_spec``      returns the externally injected spectral tensor (the traced
               graph input) instead of computing an STFT of the waveform.
``_magnitude`` complex-as-channels view [B, C, 2, F, T] -> [B, C*2, F, T]
               (the input is already real-valued in contract layout).
``_mask``      inverse view [B, S, C*2, F, T] -> [B, S, C, 2, F, T]; the
               result is captured as the ``spectral_out`` graph output.
``_ispec``     captures its (de-normalized, pre-ISTFT) input and returns a
               scalar zero, so ``x = xt + x`` reduces the model's return
               value to the de-normalized time-branch output. The residual
               Add-with-zero is stripped from the ONNX graph afterwards
               (scripts/spectral_core_graph.py::strip_time_zero_add).

The magnitude-view mean/std normalization stays INSIDE the traced graph, per
the contract: applications never implement those statistics.

This module intentionally does not import scripts/onnx_stft.py — the dense
conv-DFT export path is scheduled for deletion once spectral-core bundles are
stable (issue #23 PR 4).
"""

import math
import types

import torch
import torch.nn as nn

# Contract constants (openkara.spectral-contract/v1). Mirrors
# scripts/spectral_reference.py; asserted against the model at patch time.
N_FFT = 4096
HOP = 1024
CONTRACT_FREQS = N_FFT // 2  # 2048 one-sided bins carried (Nyquist dropped)
SEGMENT_FRAMES = 343980
SEGMENT_SPECTRAL_FRAMES = math.ceil(SEGMENT_FRAMES / HOP)  # 336

SPECTRAL_CONTRACT_VERSION = "openkara.spectral-contract/v1"


class SpectralCoreTracePatch:
    """Swap HTDemucs spectrogram methods for spectral-core tracing.

    Use as a context manager around torch.onnx.export (or an eager forward
    for reference computation)::

        patch = SpectralCoreTracePatch.from_model(model)
        core = SpectralCoreModule(model, patch)
        with patch:
            torch.onnx.export(core, (spectral, mix), ...)

    The model must have cac=True, nfft, hop_length, segment, and samplerate
    attributes (the HTDemucs interface).
    """

    def __init__(self, model):
        self.model = model
        self._originals = {}
        self._applied = False
        self._injected = None
        self._captured = None

    @classmethod
    def from_model(cls, model):
        if not model.cac:
            raise RuntimeError(
                "SpectralCoreTracePatch expects an HTDemucs model with cac=True"
            )
        if model.nfft != N_FFT or model.hop_length != HOP:
            raise RuntimeError(
                f"model transform constants (n_fft={model.nfft}, "
                f"hop={model.hop_length}) do not match "
                f"{SPECTRAL_CONTRACT_VERSION} (n_fft={N_FFT}, hop={HOP})"
            )
        return cls(model)

    def inject(self, spectral):
        """Arm the patched ``_spec`` with the traced spectral input tensor."""
        self._injected = spectral

    def take_captured(self):
        """Return the pre-ISTFT tensor captured by the patched ``_ispec``."""
        captured = self._captured
        self._captured = None
        if captured is None:
            raise RuntimeError(
                "no spectral output captured: model forward did not call _ispec"
            )
        return captured

    def apply(self):
        if self._applied:
            return
        patch = self

        self._originals = {
            "_spec": self.model._spec,
            "_ispec": self.model._ispec,
            "_magnitude": self.model._magnitude,
            "_mask": self.model._mask,
        }

        def core_spec(self, x):
            z = patch._injected
            patch._injected = None
            if z is None:
                raise RuntimeError(
                    "_spec called without an injected spectral tensor; "
                    "run the model through SpectralCoreModule"
                )
            return z

        def core_magnitude(self, z):
            batch, channels, _, freqs, frames = z.shape
            return z.reshape(batch, channels * 2, freqs, frames)

        def core_mask(self, z, m):
            batch, sources, _, freqs, frames = m.shape
            return m.view(batch, sources, -1, 2, freqs, frames).contiguous()

        def core_ispec(self, z, length=None, scale=0):
            assert scale == 0, "Scaled ISTFT export is not implemented"
            patch._captured = z
            # Scalar zero: `x = xt + x` becomes the identity on the time
            # branch. Stripped from the exported graph afterwards.
            return z.new_zeros(())

        self.model._spec = types.MethodType(core_spec, self.model)
        self.model._ispec = types.MethodType(core_ispec, self.model)
        self.model._magnitude = types.MethodType(core_magnitude, self.model)
        self.model._mask = types.MethodType(core_mask, self.model)
        self._applied = True

    def restore(self):
        if not self._applied:
            return
        for name, method in self._originals.items():
            setattr(self.model, name, method)
        self._originals = {}
        self._injected = None
        self._captured = None
        self._applied = False

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
        return False


class SpectralCoreModule(nn.Module):
    """Traceable spectral-core boundary around a patched HTDemucs model.

    forward(spectral [B,2,2,2048,336], mix [B,2,343980])
        -> (spectral_out [B,S,2,2,2048,336], time_out [B,S,2,343980])
    """

    def __init__(self, model, patch):
        super().__init__()
        if patch.model is not model:
            raise ValueError("patch was built for a different model")
        self.model = model
        self.patch = patch

    def forward(self, spectral, mix):
        self.patch.inject(spectral)
        time_out = self.model(mix)
        spectral_out = self.patch.take_captured()
        return spectral_out, time_out


def dummy_inputs(batch=1):
    """Fixed-shape tracing inputs for the contract segment window."""
    spectral = torch.randn(
        batch, 2, 2, CONTRACT_FREQS, SEGMENT_SPECTRAL_FRAMES
    )
    mix = torch.randn(batch, 2, SEGMENT_FRAMES)
    return spectral, mix
