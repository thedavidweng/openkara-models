# AGENTS.md

## Cursor Cloud specific instructions

This is a Python-based ML model conversion pipeline (no web services, no databases, no Docker). It converts pretrained Demucs PyTorch models to ONNX format for the OpenKara desktop app.

### Prerequisites

- Python 3.11+ with `python3.12-venv` (or equivalent) installed at OS level.
- No GPU required; all conversion runs on CPU only.

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The CPU-only PyTorch index **must** be used before `pip install -r requirements.txt`, otherwise pip may pull the full CUDA build (~2 GB larger, unnecessary here).

### Running the pipeline

See `README.md` "Build locally" section. In short:

```bash
source .venv/bin/activate
python scripts/convert_htdemucs_to_onnx.py          # convert htdemucs (default)
python scripts/validate_onnx.py                      # validate htdemucs
python scripts/convert_htdemucs_to_onnx.py --model htdemucs_ft   # convert fine-tuned ensemble
python scripts/validate_onnx.py --model htdemucs_ft               # validate fine-tuned ensemble
```

- `htdemucs` conversion takes ~1–2 minutes on a standard VM and needs ~4 GB RAM.
- `htdemucs_ft` (4-model ensemble) takes considerably longer and needs ~7 GB+ RAM.
- First run downloads pretrained model weights from PyTorch Hub (requires internet).
- Output artifacts land in `models/` (gitignored).

### Linting / testing

There are no dedicated lint or test commands in this repo. The validation script (`scripts/validate_onnx.py`) serves as the end-to-end test — it compares ONNX output against PyTorch reference and asserts MSE < 1e-4.

### Key gotchas

- The venv **must** be activated before running scripts (they import from `demucs`, `onnx`, `onnxruntime`).
- Direct ONNX export of Demucs fails due to complex-valued STFT; the conversion script automatically falls back to a real-valued conv1d patch (`scripts/onnx_stft.py`).
- ONNX Runtime graph optimization is applied inside `scripts/convert_htdemucs_to_onnx.py`, not in CI YAML — keep it that way.
