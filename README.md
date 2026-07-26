# openkara-models

Reproducible ONNX model conversion pipeline for [OpenKara](https://github.com/thedavidweng/OpenKara).

Converts the pretrained [Demucs](https://github.com/adefossez/demucs) PyTorch models to ONNX format for cross-platform audio stem separation.

## Compatibility (official ORT)

Standard release ONNX files must load with **official, pre-built ONNX Runtime** on **Linux x64, Windows x64, macOS x64, and macOS arm64** using the **CPU** execution provider (CoreML on macOS is optional). They must **not** embed operator domains that only exist in custom ORT builds—especially **`com.microsoft.nchwc`** from layout optimization at `ORT_ENABLE_ALL`.

Full policy, release gates, and a minimal Apple Silicon smoke-test snippet: **[docs/runtime-contract.md](docs/runtime-contract.md)**.

## Artifact catalog (authoritative)

Model and ONNX Runtime release metadata now live in the versioned artifact
catalog, not in hardcoded constants:

- `catalog/releases/<release-id>.json` — immutable release manifest.
- `catalog/channels/stable.json` — stable-channel pointer.
- `catalog/channels/candidate.json` — pre-promotion channel pointer
  (validated cross-target before the stable switch).

See **[docs/catalog-contract.md](docs/catalog-contract.md)** for the contract
and `scripts/validate_catalog.py` / `scripts/generate_catalog_release.py` for
validation and generation.

OpenKara must resolve model and runtime artifacts from the catalog, not from
hardcoded constants. The old README pin table and Rust `ModelDescriptor`
constant block have been removed to avoid duplicate release metadata. OpenKara
issue #167 switches the app to consuming `catalog/channels/stable.json` and
`catalog/releases/<release-id>.json`.

## Models

Both models ship as **spectral-core** artifacts (contract v1): the ONNX graph
takes the demucs spectral tensor plus the raw mix and returns the pre-ISTFT
spectral heads plus the time branch. The app performs the STFT/ISTFT — the
graph contains no DFT constant matrices (see [How it works](#how-it-works)).

### htdemucs (default)

| Property     | Value                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------- |
| Source model | `htdemucs` (Hybrid Transformer Demucs)                                                             |
| Inputs       | `spectral [1, 2, 2, 2048, 336]` (contract-v1 spec of `mix`) + `mix [1, 2, 343980]` stereo 44.1 kHz |
| Outputs      | `spectral_out [1, 4, 2, 2, 2048, 336]` (pre-ISTFT) + `time_out [1, 4, 2, 343980]`; stems[s] = ispec(spectral_out[:, s]) + time_out[:, s] |
| Format       | ONNX (opset 17), spectral-core contract v1                                                         |
| Size         | 199.8 MB                                                                                           |

### htdemucs_ft (fine-tuned, higher quality)

| Property     | Value                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------- |
| Source model | `htdemucs_ft` (Fine-tuned Hybrid Transformer Demucs, 4-model ensemble)                             |
| Inputs       | `spectral [1, 2, 2, 2048, 336]` (contract-v1 spec of `mix`) + `mix [1, 2, 343980]` stereo 44.1 kHz |
| Outputs      | `spectral_out [1, 4, 2, 2, 2048, 336]` (pre-ISTFT) + `time_out [1, 4, 2, 343980]`; stems[s] = ispec(spectral_out[:, s]) + time_out[:, s] |
| Format       | ONNX (opset 17), spectral-core contract v1                                                         |
| Size         | 800.2 MB                                                                                           |
| Note         | Ensemble of 4 fine-tuned models averaged into a single ONNX graph                                  |

## Usage

### Download pre-built models

Grab model files from the [Releases](https://github.com/thedavidweng/openkara-models/releases) page:

- **htdemucs**: `htdemucs.spectral.onnx` + `htdemucs.spectral.onnx.sha256` (tags: `model-spectral-v*`)
- **htdemucs_ft**: `htdemucs_ft.spectral.onnx` + `htdemucs_ft.spectral.onnx.sha256` (tags: `model-ft-spectral-v*`)

### Build locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# htdemucs (default)
python scripts/export_spectral_core.py
python scripts/validate_spectral_contract.py models/htdemucs.spectral.onnx

# htdemucs_ft (fine-tuned)
python scripts/export_spectral_core.py --model htdemucs_ft
python scripts/validate_spectral_contract.py models/htdemucs_ft.spectral.onnx
```

Output: `models/htdemucs.spectral.onnx` or `models/htdemucs_ft.spectral.onnx`

The final release artifact is the ONNX Runtime optimized graph, not the raw PyTorch export. Each final model also carries:

- `openkara.model_cache_key`: a deterministic cache-busting fingerprint for runtime compiled-model caches
- `openkara.optimized_by=onnxruntime`: marks ORT offline optimization **under the [runtime contract](docs/runtime-contract.md)** (not “all optimizers including NCHWc layout rewrites”)

GitHub Actions is only the orchestrator here. The actual graph optimization lives in `scripts/export_spectral_core.py`, so CI and any local rerun use the exact same conversion pipeline instead of duplicating optimization logic in workflow YAML.

The first shipping optimization pass is intentionally structural and deterministic. Semantic rewrites such as GELU approximation or reduction rewrites are deferred until OpenKara runtime profiling shows a specific CoreML fallback hotspot that justifies the numerical trade-off.

## Integrate with OpenKara

OpenKara resolves model and runtime artifacts from the [artifact catalog](#artifact-catalog-authoritative).
The catalog is the single source of truth for download URLs, SHA-256 digests,
sizes, and compatibility edges. Do not duplicate release metadata in OpenKara
constants — consume `catalog/channels/stable.json` and
`catalog/releases/<release-id>.json` directly (OpenKara issue #167).

## How it works

Demucs' STFT/ISTFT are complex-valued and have no portable ONNX operator, so the **spectral-core** contract (v1) keeps them out of the graph entirely. The app computes the forward STFT and the final ISTFT; the ONNX graph is the demucs core between them. It takes the demucs spectral tensor (`spectral`) plus the raw mix (`mix`) and returns the pre-ISTFT spectral heads (`spectral_out`) plus the time branch (`time_out`), and the app composes each stem as `ispec(spectral_out[:, s]) + time_out[:, s]`. Because the transforms live in the app, the graph carries **no DFT constant matrices**, and the shipped ONNX Runtime is a **23-operator reduced build** containing only the kernels this graph uses.

For `htdemucs_ft` (4-model ensemble), the exporter emits each sub-model's core to a separate graph, then merges them into a single graph that averages all four outputs. The merge step **deduplicates identical initializers** shared across the sub-models, reducing model size and improving runtime cache locality. Averaging uses `Sum + Mul(1/N)` instead of materializing a stacked intermediate tensor.

## CI/CD

- Pushing a tag matching `model-spectral-v*` triggers conversion and release of **htdemucs**.
- Pushing a tag matching `model-ft-spectral-v*` triggers conversion and release of **htdemucs_ft**.

**htdemucs** runs in a single job: export → optimize → validate → release (~5 min).

**htdemucs_ft** runs in parallel: 4 sub-model export jobs (matrix) → 1 merge + finalize job (~6 min wall time instead of ~20 min serial).

Each workflow:

1. Exports the spectral-core graph (STFT/ISTFT stay in the app; no DFT constants in the graph)
2. Rewrites the final artifact through ONNX Runtime offline optimization (`ORT_ENABLE_EXTENDED`; see [runtime contract](docs/runtime-contract.md))
3. Validates ONNX output against PyTorch (MSE < 1e-4), checks optimized-artifact metadata, and asserts the graph contains no `com.microsoft.nchwc` nodes
4. Publishes the optimized ONNX file + SHA-256 checksum as a GitHub Release (release body includes model size and validation MSE)

CI optimizations: pip wheel cache, Demucs pretrained-weights cache, concurrency cancellation for duplicate runs, pinned action SHAs, 7-day artifact retention.

Pull requests run a lightweight **runtime-contract** workflow (`scripts/onnx_runtime_contract.py --self-test`); full checks run on tagged release builds.

A weekly check (every Monday) monitors **ONNX Runtime releases** (via `scripts/detect_ort_release.py`, filtering prereleases) and **HuggingFace model-weight revisions** (via `scripts/detect_model_weight_revision.py`, using immutable commit SHAs at `adefossez/HTDemucs` and `adefossez/HTDemucs-ft`). When a change is detected, the workflow:

1. Compares the latest stable upstream tag + commit SHA against `ort/source-lock.json`
2. Opens a tracking issue for review
3. A maintainer triggers the candidate orchestration workflow (`dep-candidate.yml`) to build all runtime targets, run quality gates, and open a reviewable candidate PR with artifacts
4. A candidate PR never changes the stable pointer — stable publication remains a separate reviewed action (tag push only)

See `.github/workflows/dep-detection.yml` for detection, `.github/workflows/dep-candidate.yml` for candidate orchestration, and `.github/workflows/dep-health-monitor.yml` for automation health monitoring. Dependency update ownership is documented in `docs/dependency-update-authorities.md`.

## License

MIT
