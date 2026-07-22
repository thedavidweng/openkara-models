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
| Demucs model-weight revisions | Repository-owned script | `scripts/detect_model_weight_revision.py` | Detects HuggingFace model-weight revision changes using immutable commit SHAs (not `lastModified` timestamps), updates the model source lock, opens a candidate PR. |
| Container/toolchain images | Renovate custom manager | `renovate.json` | Custom regex managers for `ort/source-lock.json` toolchain image fields. |
| Catalog schema version | Manual | — | Schema changes require a manual PR with migration. Not automated. |

## What Dependabot owns

Dependabot handles standard package-manager ecosystems:
- `pip` (Python packages in `requirements.txt`)
- `github-actions` (actions in `.github/workflows/*.yml`)

Dependabot does NOT touch:
- `ort/source-lock.json` (ORT source tag, commit SHA, toolchain config)
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
- Model-weight revisions (handled by `scripts/detect_model_weight_revision.py`)
- Python packages (owned by Dependabot)
- GitHub Actions (owned by Dependabot)

## What repository-owned scripts own

Two repository-owned scripts handle upstream sources that no bot can
manage (source-lock JSON, HuggingFace model weights):

- `scripts/detect_ort_release.py` — polls the ONNX Runtime GitHub
  releases API, filters prereleases, compares the latest stable tag +
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
  re-run quality + runtime quality gates, updated benchmark reports.
- A Demucs package bump requires: re-converted ONNX models, re-run quality
  gates, updated catalog model entries.
- A model-weight revision requires: re-converted ONNX models, re-run
  quality gates, updated catalog model entries.

The candidate build orchestration (PR 3 of this issue) handles these
coupled changes in a single candidate PR.

## No competing bots

Dependabot and Renovate are configured to own disjoint file sets:
- Dependabot: `requirements.txt`, `.github/workflows/*.yml`
- Renovate: `ort/source-lock.json` (container + compiler fields only)

The repository-owned scripts own:
- `ort/source-lock.json` (ORT source tag + commit SHA fields only)
- Model-weight revision tracking

No file is owned by two authorities. If a future dependency falls into a
gap, it is assigned to exactly one authority via this document.
