# CoreML-Friendly Model Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible post-export optimization stage that makes OpenKara model artifacts easier for CoreML-backed ONNX Runtime sessions to compile, cache, and inspect.

**Architecture:** Keep the existing PyTorch-to-ONNX export flow, then add a deterministic post-processing stage in `openkara-models` that inventories operators, writes a model cache key into metadata, and emits an ORT-optimized ONNX artifact. Hold off on risky semantic rewrites such as GELU approximation until OpenKara runtime profiling proves they are needed.

**Tech Stack:** Python, `onnx`, `onnxruntime`, GitHub Actions

---

### Task 1: Add operator inventory and metadata cache-key helpers

**Files:**

- Modify: `scripts/convert_htdemucs_to_onnx.py`
- Test: `scripts/validate_onnx.py`

**Step 1: Write the failing test or validation hook**

Add a validation hook that expects the generated ONNX model to include a metadata cache key and emits a deterministic operator histogram for the final artifact.

**Step 2: Run validation to verify it fails**

Run: `python scripts/validate_onnx.py`

Expected: FAIL or missing-output behavior because the exported model currently has neither metadata nor inventory output.

**Step 3: Write minimal implementation**

Implement small helpers that:

- count ops in the final graph,
- print the top operator inventory during conversion,
- attach a stable metadata property derived from the model bytes / graph content.

**Step 4: Run validation to verify it passes**

Run: `python scripts/validate_onnx.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/convert_htdemucs_to_onnx.py scripts/validate_onnx.py
git commit -m "feat(models): add onnx inventory and cache metadata"
```

### Task 2: Add ORT offline optimization to the model pipeline

**Files:**

- Modify: `scripts/convert_htdemucs_to_onnx.py`
- Modify: `scripts/validate_onnx.py`
- Modify: `README.md`
- Test: `scripts/validate_onnx.py`

**Step 1: Write the failing validation**

Add validation that expects the final model artifact to be the ORT-optimized output, not the raw export.

**Step 2: Run validation to verify it fails**

Run: `python scripts/validate_onnx.py`

Expected: FAIL because no offline optimization output exists yet.

**Step 3: Write minimal implementation**

Use `onnxruntime.SessionOptions().optimized_model_filepath` / equivalent session-options API to produce an optimized ONNX artifact after export and before checksum generation. Keep the raw export only as a temporary intermediate file.

**Step 4: Run validation to verify it passes**

Run: `python scripts/validate_onnx.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/convert_htdemucs_to_onnx.py scripts/validate_onnx.py README.md
git commit -m "feat(models): emit ort-optimized onnx artifacts"
```

### Task 3: Teach CI to validate the optimized artifact path

**Files:**

- Modify: `.github/workflows/convert.yml`
- Modify: `.github/workflows/convert-ft.yml`
- Modify: `README.md`

**Step 1: Write the failing workflow expectation**

Update the workflow/test instructions so they assert the optimized artifact path and validation messaging, then run the local conversion workflow commands.

**Step 2: Run the local conversion commands to verify the gap**

Run: `python scripts/convert_htdemucs_to_onnx.py && python scripts/validate_onnx.py`

Expected: any mismatch between docs/workflow assumptions and actual output surfaces here before editing CI.

**Step 3: Write minimal implementation**

Adjust workflow and docs so release artifacts and validation steps match the new optimized-model pipeline.

**Step 4: Run the local conversion commands to verify they pass**

Run: `python scripts/convert_htdemucs_to_onnx.py && python scripts/validate_onnx.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add .github/workflows/convert.yml .github/workflows/convert-ft.yml README.md
git commit -m "ci(models): validate optimized model artifacts"
```

### Task 4: Reserve semantic rewrites for profile-backed follow-up work

**Files:**

- Modify: `README.md`
- Modify: `docs/plans/2026-04-12-coreml-model-optimization.md`

**Step 1: Write the failing doc expectation**

Add a doc assertion that explains why `Gelu` / `ReduceL2` rewrites are not part of the first shipping patch without CoreML profile evidence.

**Step 2: Verify the current docs miss that guidance**

Run: `rg "Gelu|ReduceL2|CoreML" README.md docs/plans/2026-04-12-coreml-model-optimization.md`

Expected: missing or incomplete rationale.

**Step 3: Write minimal implementation**

Document that first-pass optimization is structural and deterministic, and that any semantic rewrite must come after OpenKara runtime profiling demonstrates a specific unsupported/fallback-heavy pattern.

**Step 4: Verify the docs reflect the new guidance**

Run: `rg "Gelu|ReduceL2|profile|CoreML" README.md docs/plans/2026-04-12-coreml-model-optimization.md`

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-04-12-coreml-model-optimization.md
git commit -m "docs(models): gate semantic rewrites on runtime evidence"
```
