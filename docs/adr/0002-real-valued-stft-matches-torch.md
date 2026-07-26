# ADR-0002: Real-valued STFT/ISTFT must match torch.stft/torch.istft

> **Superseded by the spectral-core contract (issue #23); the conv-DFT export
> path was removed in PR 4.** The waveform↔spectral transforms now run in the
> application under `openkara.spectral-contract/v1` (see
> `docs/spectral-core-contract.md`); shipped cores carry no dense DFT
> filter-bank constants. This ADR is retained as the historical record of the
> conv-DFT export path that produced the (still immutable) published waveform
> release assets.

## Status

Accepted (superseded)

## Context

ONNX does not support complex-valued tensors or `torch.stft`/`torch.istft`.
The conversion pipeline replaces Demucs' complex spectrogram path with
real-valued `conv1d`/`conv_transpose1d` operations (`scripts/onnx_stft.py`,
removed with the waveform pipeline).
The exported ONNX model must produce numerically equivalent output to the
original PyTorch model (validation gate: MSE < 1e-4).

## Decision

`OnnxSTFT` and `OnnxISTFT` must match `torch.stft` / `torch.istft` with:

- `center=True` — reflect-pad by `n_fft // 2` before STFT, remove after ISTFT
- `normalized=True` — scale by `1/sqrt(n_fft)`
- `window=hann_window` — applied to both forward and inverse filters
- `return_complex=True` — represented as a real/imag pair in the last dimension

## Consequences

- **IDFT conjugate symmetry scaling**: DC and Nyquist bins contribute once;
  other bins contribute twice (conjugate symmetry). The scale factor
  (`scale[1:-1] = 2.0`) is required for correct reconstruction.
- **Overlap-add normalization uses `window²`, not squared synthesis filters**.
  Using the squared synthesis filters introduces a scale error in the
  reconstructed waveform. The normalization envelope is computed from
  `window * window` convolved with ones, independent of the Fourier basis.
- Any change to windowing, normalization, or padding must be validated against
  `torch.stft`/`torch.istft` to stay within the MSE threshold.
