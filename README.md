# openkara-models

Reproducible ONNX model conversion pipeline for [OpenKara](https://github.com/thedavidweng/OpenKara).

Converts the pretrained [Demucs](https://github.com/adefossez/demucs) PyTorch models to ONNX format for cross-platform audio stem separation.

## Models

### htdemucs (default)

| Property     | Value                                                                         |
| ------------ | ----------------------------------------------------------------------------- |
| Source model | `htdemucs` (Hybrid Transformer Demucs)                                        |
| Input        | `[1, 2, 343980]` — stereo audio at 44.1 kHz (fixed 7.8s segment)              |
| Output       | `[1, 4, 2, 343980]` — batch, stems (drums/bass/other/vocals), stereo, samples |
| Format       | ONNX (opset 17)                                                               |

### htdemucs_ft (fine-tuned, higher quality)

| Property     | Value                                                                         |
| ------------ | ----------------------------------------------------------------------------- |
| Source model | `htdemucs_ft` (Fine-tuned Hybrid Transformer Demucs, 4-model ensemble)        |
| Input        | `[1, 2, 343980]` — stereo audio at 44.1 kHz (fixed 7.8s segment)              |
| Output       | `[1, 4, 2, 343980]` — batch, stems (drums/bass/other/vocals), stereo, samples |
| Format       | ONNX (opset 17)                                                               |
| Note         | Ensemble of 4 fine-tuned models averaged into a single ONNX graph (~300MB+)   |

## Usage

### Download pre-built models

Grab model files from the [Releases](https://github.com/thedavidweng/openkara-models/releases) page:

- **htdemucs**: `htdemucs.onnx` + `htdemucs.onnx.sha256` (tags: `model-v*`)
- **htdemucs_ft**: `htdemucs_ft.onnx` + `htdemucs_ft.onnx.sha256` (tags: `model-ft-v*`)

### Build locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# htdemucs (default)
python scripts/convert_htdemucs_to_onnx.py
python scripts/validate_onnx.py

# htdemucs_ft (fine-tuned)
python scripts/convert_htdemucs_to_onnx.py --model htdemucs_ft
python scripts/validate_onnx.py --model htdemucs_ft
```

Output: `models/htdemucs.onnx` or `models/htdemucs_ft.onnx`

The final release artifact is the ONNX Runtime optimized graph, not the raw PyTorch export. Each final model also carries:

- `openkara.model_cache_key`: a deterministic cache-busting fingerprint for runtime compiled-model caches
- `openkara.optimized_by=onnxruntime`: marks that the shipped artifact already passed through ORT offline graph optimization

GitHub Actions is only the orchestrator here. The actual graph optimization lives in `scripts/convert_htdemucs_to_onnx.py`, so CI and any local rerun use the exact same conversion pipeline instead of duplicating optimization logic in workflow YAML.

The first shipping optimization pass is intentionally structural and deterministic. Semantic rewrites such as GELU approximation or reduction rewrites are deferred until OpenKara runtime profiling shows a specific CoreML fallback hotspot that justifies the numerical trade-off.

## Integrate with OpenKara

After a release is published, update these files in [OpenKara](https://github.com/thedavidweng/OpenKara):

**`src-tauri/src/separator/model.rs`** — model filename:

```rust
pub const EMBEDDED_MODEL_FILENAME: &str = "htdemucs.onnx";
```

**`src-tauri/src/separator/bootstrap.rs`** — download URL and checksum:

```rust
pub const MODEL_DOWNLOAD_URL: &str =
    "https://github.com/thedavidweng/openkara-models/releases/download/model-v2.0.0/htdemucs.onnx";
pub const MODEL_SHA256: &str = "<sha256 from htdemucs.onnx.sha256>";
```

**`scripts/setup.sh`** and **`.github/workflows/ci.yml`** — same URL and SHA-256 values.

The SHA-256 checksum is published alongside the model in each release (`htdemucs.onnx.sha256`).

## How it works

ONNX does not support complex-valued STFT/ISTFT operations used by Demucs. The conversion pipeline rewrites these as real-valued conv1d operations (DFT filter matrices), following the approach from [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) and the [Mixxx GSOC 2025 project](https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/).

## CI/CD

- Pushing a tag matching `model-v*` triggers conversion and release of **htdemucs**.
- Pushing a tag matching `model-ft-v*` triggers conversion and release of **htdemucs_ft**.

Each workflow:

1. Converts the model to a raw ONNX export
2. Rewrites the final artifact through ONNX Runtime offline optimization
3. Validates ONNX output against PyTorch (MSE < 1e-4) and checks optimized-artifact metadata
4. Publishes the optimized ONNX file + SHA-256 checksum as a GitHub Release

A weekly check (every Monday) monitors PyPI for new Demucs versions and opens an issue labeled `upstream-update` when a new release is detected.

For Linux-based continuation work on conversion, optimization, and packaging validation, see `docs/plans/2026-04-12-linux-agent-handoff.md`.

## License

MIT
