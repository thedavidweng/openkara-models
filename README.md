# openkara-models

Reproducible ONNX model conversion pipeline for [OpenKara](https://github.com/thedavidweng/OpenKara).

Converts the pretrained [Demucs htdemucs](https://github.com/facebookresearch/demucs) PyTorch model to ONNX format for cross-platform audio stem separation.

## Model Details

| Property | Value |
|----------|-------|
| Source model | `htdemucs` (Hybrid Transformer Demucs) |
| Input | `[1, 2, frame_count]` — stereo audio at 44.1 kHz |
| Output | `[4, 2, frame_count]` — drums, bass, other, vocals |
| Format | ONNX (opset 17) |

## Usage

### Download pre-built model

Grab `htdemucs.onnx` and `htdemucs.onnx.sha256` from the [Releases](https://github.com/thedavidweng/openkara-models/releases) page.

### Build locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/convert_htdemucs_to_onnx.py
python scripts/validate_onnx.py
```

Output: `models/htdemucs.onnx`

## How it works

ONNX does not support complex-valued STFT/ISTFT operations used by Demucs. The conversion pipeline rewrites these as real-valued conv1d operations (DFT filter matrices), following the approach from [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) and the [Mixxx GSOC 2025 project](https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/).

## CI/CD

Pushing a tag matching `model-v*` triggers GitHub Actions to:
1. Convert the model
2. Validate ONNX output against PyTorch (MSE < 1e-4)
3. Publish the ONNX file + SHA-256 checksum as a GitHub Release

## License

MIT
