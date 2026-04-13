# openkara-models

Reproducible ONNX model conversion pipeline for [OpenKara](https://github.com/thedavidweng/OpenKara).

Converts the pretrained [Demucs](https://github.com/adefossez/demucs) PyTorch models to ONNX format for cross-platform audio stem separation.

## Compatibility (official ORT)

Standard release ONNX files must load with **official, pre-built ONNX Runtime** on **Linux x64, Windows x64, macOS x64, and macOS arm64** using the **CPU** execution provider (CoreML on macOS is optional). They must **not** embed operator domains that only exist in custom ORT builds—especially **`com.microsoft.nchwc`** from layout optimization at `ORT_ENABLE_ALL`.

Full policy, release gates, and a minimal Apple Silicon smoke-test snippet: **[docs/runtime-contract.md](docs/runtime-contract.md)**.

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
- `openkara.optimized_by=onnxruntime`: marks ORT offline optimization **under the [runtime contract](docs/runtime-contract.md)** (not “all optimizers including NCHWc layout rewrites”)

GitHub Actions is only the orchestrator here. The actual graph optimization lives in `scripts/convert_htdemucs_to_onnx.py`, so CI and any local rerun use the exact same conversion pipeline instead of duplicating optimization logic in workflow YAML.

The first shipping optimization pass is intentionally structural and deterministic. Semantic rewrites such as GELU approximation or reduction rewrites are deferred until OpenKara runtime profiling shows a specific CoreML fallback hotspot that justifies the numerical trade-off.

## Integrate with OpenKara

After a release is published, update the pinned **download URL**, **SHA-256**, and (when needed)
**GitHub release tag** in [OpenKara](https://github.com/thedavidweng/OpenKara). Copy exact values
from the release assets (`htdemucs.onnx.sha256`, `htdemucs_ft.onnx.sha256`).

**Current default standard pin (contract-compliant, Apple Silicon–safe):**

| Variant      | Release tag        | Asset file        |
| ------------ | ------------------ | ----------------- |
| `htdemucs`   | `model-v2.0.1`     | `htdemucs.onnx`   |
| `htdemucs_ft`| `model-ft-v2.0.1`  | `htdemucs_ft.onnx`|

**`src-tauri/src/separator/model.rs`** — embedded filename (unchanged unless you rename assets):

```rust
pub const EMBEDDED_MODEL_FILENAME: &str = "htdemucs.onnx";
```

**`src-tauri/src/separator/bootstrap.rs`** — `ModelDescriptor` entries (URL + SHA must match the
release you ship; `MODEL_DOWNLOAD_URL` / `MODEL_SHA256` are aliases of `HTDEMUCS`):

```rust
pub const HTDEMUCS: ModelDescriptor = ModelDescriptor {
    filename: "htdemucs.onnx",
    download_url: "https://github.com/thedavidweng/openkara-models/releases/download/model-v2.0.1/htdemucs.onnx",
    sha256: "8fa3dab679c59aeb049dd229f57a212c9339b3fc17ebf50541daad9e799364a1",
};

pub const HTDEMUCS_FT: ModelDescriptor = ModelDescriptor {
    filename: "htdemucs_ft.onnx",
    download_url: "https://github.com/thedavidweng/openkara-models/releases/download/model-ft-v2.0.1/htdemucs_ft.onnx",
    sha256: "0f2efbd7044182c10a6e8169b670392a3a91f904635e29329d6a3667375f5c94",
};
```

**`scripts/setup.sh`** — `MODEL_URL` / `MODEL_SHA256` for the standard `htdemucs.onnx` dev cache
(same URL and SHA as `HTDEMUCS` above).

**`.github/workflows/ci.yml`** — `MODEL_URL` / `MODEL_SHA256` env vars for the Verify job model
cache (same as `HTDEMUCS`).

**`docs/references/contracts/phase-6-model-bootstrap-contract.md`** — pinned release paths in the
bootstrap contract section.

Do **not** point default onboarding at **`model-v2.0.0`**: those artifacts could embed
`com.microsoft.nchwc` and fail on official macOS arm64 ONNX Runtime. Prefer **`v2.0.1`** or any
newer tag whose CI passed the [runtime contract](docs/runtime-contract.md) gates.

## How it works

ONNX does not support complex-valued STFT/ISTFT operations used by Demucs. The conversion pipeline rewrites these as real-valued conv1d operations (DFT filter matrices), following the approach from [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) and the [Mixxx GSOC 2025 project](https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/).

## CI/CD

- Pushing a tag matching `model-v*` triggers conversion and release of **htdemucs**.
- Pushing a tag matching `model-ft-v*` triggers conversion and release of **htdemucs_ft**.

Each workflow:

1. Converts the model to a raw ONNX export
2. Rewrites the final artifact through ONNX Runtime offline optimization (`ORT_ENABLE_EXTENDED`; see [runtime contract](docs/runtime-contract.md))
3. Validates ONNX output against PyTorch (MSE < 1e-4), checks optimized-artifact metadata, and asserts the graph contains no `com.microsoft.nchwc` nodes
4. Publishes the optimized ONNX file + SHA-256 checksum as a GitHub Release

Pull requests run a lightweight **runtime-contract** workflow (`scripts/onnx_runtime_contract.py --self-test`); full checks run on tagged release builds.

A weekly check (every Monday) monitors PyPI for new Demucs versions and opens an issue labeled `upstream-update` when a new release is detected.

For Linux-based continuation work on conversion, optimization, and packaging validation, see `docs/plans/2026-04-12-linux-agent-handoff.md`.

## License

MIT
