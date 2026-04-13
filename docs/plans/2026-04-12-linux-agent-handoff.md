# Linux Agent Handoff: Model Optimization And Packaging Validation

## Purpose

This repository publishes the ONNX model artifacts consumed by OpenKara.

The product goal is not just "export Demucs to ONNX". The real goal is:

- produce model artifacts that OpenKara can load efficiently,
- make those artifacts friendlier to ONNX Runtime CoreML session compilation on macOS,
- keep the generation pipeline deterministic so GitHub Actions and any local rerun produce the same output,
- prepare evidence for any future operator-rewrite work instead of guessing.

OpenKara has already been updated to:

- default to explicit execution providers instead of a visible `auto` setting,
- use provider-aware model session caching,
- configure CoreML sessions with static input shapes, MLProgram, and compiled-model cache directories.

That means the model artifacts produced here now directly affect real runtime performance in OpenKara.

## Current State In This Branch

This branch already changes the conversion pipeline in these ways:

- `scripts/convert_htdemucs_to_onnx.py`
  - exports a raw ONNX model,
  - runs ONNX Runtime offline graph optimization,
  - emits the optimized ONNX artifact as the final release file,
  - prints an operator inventory,
  - writes metadata keys:
    - `openkara.model_cache_key`
    - `openkara.optimized_by=onnxruntime`
- `scripts/validate_onnx.py`
  - still checks numerical parity with PyTorch,
  - now also validates the optimized-artifact metadata.
- `.github/workflows/convert.yml` and `.github/workflows/convert-ft.yml`
  - still only orchestrate conversion/validation/release,
  - do **not** duplicate optimization logic in YAML.

This separation is intentional. Graph optimization belongs in the conversion script, not in the workflow, so CI and local runs stay identical.

## What You Need To Do

You are running on Linux with the full Python/ONNX stack available. Your job is to verify and harden the optimization pipeline, then test model packaging end-to-end.

### Task 1: Verify The New Conversion Pipeline Actually Works

Run both models through the current scripts:

```bash
pip install -r requirements.txt
python scripts/convert_htdemucs_to_onnx.py
python scripts/validate_onnx.py
python scripts/convert_htdemucs_to_onnx.py --model htdemucs_ft
python scripts/validate_onnx.py --model htdemucs_ft
```

Confirm all of the following:

- conversion succeeds for both `htdemucs` and `htdemucs_ft`,
- validation succeeds for both,
- final `models/*.onnx` artifacts are the optimized outputs, not the raw intermediates,
- metadata keys are present in the final artifacts,
- operator inventory is printed during conversion.

If the new pipeline fails anywhere, fix it in this repository before moving on.

### Task 2: Inspect The Optimized Graph Output

Capture the evidence needed for later optimization work:

- record the top operators for both models before/after optimization if practical,
- compare raw export size vs optimized artifact size,
- note whether ORT optimization materially changes graph structure or just canonicalizes it,
- document any warnings emitted by ONNX Runtime while producing the optimized model.

If useful, extend the scripts to print clearer structured summaries, but keep the implementation minimal.

### Task 3: Validate Packaging Assumptions Against GitHub Actions

The release workflows are Linux-based and should remain thin wrappers over the scripts.

Check that the workflow assumptions are true:

- `models/htdemucs.onnx` and `models/htdemucs_ft.onnx` are the final optimized artifacts,
- `sha256sum` generation still works unchanged,
- artifact names do not need to change,
- release notes still accurately describe what is shipped.

If anything in workflow/docs is now inaccurate, fix it here rather than teaching CI special-case behavior.

### Task 4: Only Propose Semantic Rewrites With Evidence

Do **not** guess on `Gelu`, `Erf`, `ReduceL2`, `Unsqueeze`, or other rewrite targets.

If you think another optimization pass is warranted, first gather evidence from the generated graph:

- which operators are actually present,
- which patterns remain after ORT offline optimization,
- whether a rewrite would preserve model semantics well enough to justify the risk.

Future semantic rewrites should be proposed only if the graph evidence supports them.

## Acceptance Criteria

The handoff is successful only if all of these are true:

- both models convert successfully on Linux,
- both models validate successfully on Linux,
- final artifacts contain the optimization metadata,
- GitHub Actions workflows still match the real artifact pipeline,
- any follow-up optimization proposal is evidence-backed, not speculative.

## Files To Inspect First

- `scripts/convert_htdemucs_to_onnx.py`
- `scripts/validate_onnx.py`
- `.github/workflows/convert.yml`
- `.github/workflows/convert-ft.yml`
- `README.md`
- `docs/plans/2026-04-12-coreml-model-optimization.md`

## Constraints

- Keep graph optimization logic in `scripts/convert_htdemucs_to_onnx.py`.
- Do not duplicate conversion logic inside GitHub Actions YAML.
- Do not add speculative operator rewrites without graph evidence.
- Keep final release artifact names stable unless there is a concrete need to change them.

## What To Report Back

When you finish, report:

- exact commands run,
- whether each model converted and validated,
- any warnings/errors encountered,
- whether the optimized artifacts differ materially from raw exports,
- any recommended next optimization work, with evidence.
