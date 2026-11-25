# Dependency update authorities

This document defines which tool owns updates for each versioned input in
`openkara-models`. The invariant is: **one dependency has one update
authority**. No dependency is updated by two competing bots.

## Ownership matrix

| Versioned input | Authority | Config | Notes |
|-----------------|-----------|--------|-------|
| Python packages (`requirements.txt`) | Dependabot | `.github/dependabot.yml` | `pip` ecosystem, weekly. |
| GitHub Actions (`.github/workflows/*.yml`) | Dependabot | `.github/dependabot.yml` | `github-actions` ecosystem, weekly. SHA-pinned actions get SHA bumps. |
| ONNX Runtime source tag + commit | Repository-owned script | `scripts/detect_ort_release.py` | Detects new stable ORT tags (ignores prereleases), updates `ort/source-lock.json`, opens a candidate PR. Not a package-manager field — Renovate/Dependabot cannot handle source-lock JSON. |
| ONNX Runtime WebGPU plugin (`onnxruntime-ep-webgpu`) | Repository-owned script | `scripts/detect_plugin_release.py` | Polls PyPI JSON weekly, verifies wheel SHA-256 values and ORT C API/plugin compatibility against `ort/nuget-lock.json`, refreshes `ort/plugin-lock.json`, and opens a candidate PR. |
| Demucs model-weight revisions | Repository-owned script | `scripts/detect_model_weight_revision.py` | Detects HuggingFace model-weight revision changes using immutable commit SHAs (not `lastModified` timestamps), updates the model source lock, opens a candidate PR. |
| Container/toolchain images | Renovate custom manager | `renovate.json` | Custom regex managers for `ort/source-lock.json` toolchain image fields. |
| Catalog schema version | Manual | — | Schema changes require a manual PR with migration. Not automated. |

## What Dependabot owns

Dependabot handles standard package-manager ecosystems:
- `pip` (Python packages in `requirements.txt`)
- `github-actions` (actions in `.github/workflows/*.yml`)

Dependabot does NOT touch:
- `ort/source-lock.json` (ORT source tag, commit SHA, toolchain config)
- `ort/plugin-lock.json` (`onnxruntime-ep-webgpu` wheel URLs, sizes, SHA-256s, and ORT compatibility)
- Model-weight revision tracking (HuggingFace)
- Container image digests in source-lock JSON
- Catalog schema versions

## What Renovate owns

Renovate handles non-package-manager versioned fields via custom regex
managers:
- Container image references in `ort/source-lock.json` (e.g.
  `ubuntu-24.04`, `macos-14` runner images, toolchain image fields)
- Compiler version strings in `ort/source-lock.json` (e.g. `gcc-13`,
  `clang-15`)

Renovate does NOT touch:
- ORT source tag + commit SHA (handled by `scripts/detect_ort_release.py`)
- `onnxruntime-ep-webgpu` plugin wheel versions (handled by `scripts/detect_plugin_release.py`)
- Model-weight revisions (handled by `scripts/detect_model_weight_revision.py`)
- Python packages (owned by Dependabot)
- GitHub Actions (owned by Dependabot)

## What repository-owned scripts own

Two repository-owned scripts handle upstream sources that no bot can
manage (source-lock JSON, HuggingFace model weights):

- `scripts/detect_ort_release.py` — polls the `Microsoft.ML.OnnxRuntime`
  version list on nuget.org (reading the source commit from the nuspec),
  filters prereleases, compares the latest stable tag +
  commit SHA against `ort/source-lock.json`, and if different, updates the
  lock and opens a candidate PR (PR 2 of this issue).
- `scripts/detect_model_weight_revision.py` — polls the HuggingFace API
  for model-weight revision commit SHAs (not `lastModified` timestamps),
  compares against the committed lock, and if different, opens a candidate
  PR (PR 2 of this issue).

These scripts run on a schedule (PR 4 of this issue) and produce
reviewable candidate PRs. They never auto-merge or auto-promote.

## Coupled changes

Some updates require coupled changes that must pass together:
- An ORT source tag bump requires: updated `ort/source-lock.json`, rebuilt
  runtime archives (all 5 targets), updated catalog runtime entries,
  run native smoke and runtime differential checks.
- A Demucs package bump requires: re-converted ONNX models, re-run quality
  runtime checks, updated catalog model entries.
- A model-weight revision requires: re-converted ONNX models, re-run
  export fidelity and runtime checks, updated catalog model entries.

The candidate build orchestration (PR 3 of this issue) handles these
coupled changes in a single candidate PR.

## No competing bots

Dependabot and Renovate are configured to own disjoint file sets:
- Dependabot: `requirements.txt`, `.github/workflows/*.yml`
- Renovate: `ort/source-lock.json` (container + compiler fields only)

The repository-owned scripts own:
- `ort/source-lock.json` (ORT source tag + commit SHA fields only)
- `ort/plugin-lock.json` (`onnxruntime-ep-webgpu` wheel lock)
- Model-weight revision tracking

No file is owned by two authorities. If a future dependency falls into a
gap, it is assigned to exactly one authority via this document.

## Dependency health alert

The health monitor maintains a single canonical issue titled
"deps: automation health alert (stale or failed)": it edits the oldest open one
in place, creates one only if none exists, and closes it when healthy.
A newer stable ORT remains stale and actionable until the lock is upgraded;
there is no update hold or alert suppression.

ORT candidates verify `(tag, commit_sha)` against `nuget.org` via the package
nuspec (without calling the GitHub API), download `Microsoft.ML.OnnxRuntime`
from `nuget.org`, run `dotnet nuget verify --all` (requiring both Microsoft
Corporation author signature and nuget.org repository countersignature), and
refresh both `ort/nuget-lock.json` (package URL, size, SHA-256, signers, targets,
and every used member digest) and `ort/source-lock.json` (preserving existing
`submodules` and `deps.entries` when GitHub is unavailable). A mismatched
commit, failed signature verification, or missing member fails without changing
either lock.

Because `Microsoft.ML.OnnxRuntime` on nuget.org does not ship an `osx-x64`
(`x86_64-apple-darwin`) binary (Microsoft dropped `osx-x64` after 1.23.2),
`ort-build.yml` invokes `scripts/build_runtime.py --allow-carry-forward`. If the
upstream ORT git repository is reachable, it builds `x86_64-apple-darwin` from
source and fails loudly on any build or lock-verification error. Only when
upstream git clone/fetch fails does it degrade by downloading and digest-verifying
the previous `stable-v2` Intel macOS archive, running Rosetta smoke against it,
and flagging the carry-forward in `candidate-summary.md` and the workflow summary.
This preparation is not native compatibility or runtime validation: supported C
API validation, native smoke tests, and runtime differential checks must pass
before promotion.
