# Linux Agent Prompt

Work in `/Users/david/Development/Vibe/openkara-models`.

Read these files first:

- `docs/plans/2026-04-12-coreml-model-optimization.md`
- `docs/plans/2026-04-12-linux-agent-handoff.md`

Context:

- This repository publishes the ONNX artifacts consumed by OpenKara.
- OpenKara has already been updated to use explicit execution providers, provider-aware model session caching, and a more specific CoreML runtime setup.
- The goal here is to make the model artifacts more useful to OpenKara runtime, not just to export "some ONNX file".
- Graph optimization logic must live in `scripts/convert_htdemucs_to_onnx.py`. Do not move that logic into GitHub Actions YAML.
- Do not propose speculative operator rewrites. Any rewrite suggestion must be backed by graph evidence from the generated models.

Your tasks:

1. Install dependencies and prove the environment is ready.
2. Run conversion and validation for `htdemucs`.
3. Run conversion and validation for `htdemucs_ft`.
4. Confirm that the final published artifacts are the ORT-optimized ONNX outputs, not raw intermediates.
5. Confirm the final artifacts contain:
   - `openkara.model_cache_key`
   - `openkara.optimized_by=onnxruntime`
6. Compare raw vs optimized artifacts:
   - file size
   - node count if practical
   - top operators before/after if practical
   - any ONNX Runtime warnings during optimization
7. Verify that `.github/workflows/convert.yml` and `.github/workflows/convert-ft.yml` still accurately reflect the real artifact pipeline.
8. If the pipeline fails anywhere, fix it in this repository and re-run verification.
9. Only if the graph evidence supports it, propose the next optimization target. Otherwise explicitly say that no semantic rewrite should be proposed yet.

Required commands:

```bash
python --version
python -c "import onnx, onnxruntime; print('ok')"
pip install -r requirements.txt
python scripts/convert_htdemucs_to_onnx.py
python scripts/validate_onnx.py
python scripts/convert_htdemucs_to_onnx.py --model htdemucs_ft
python scripts/validate_onnx.py --model htdemucs_ft
```

Acceptance bar:

- Both models must convert successfully unless you hit a concrete blocker.
- Both models must validate successfully unless you hit a concrete blocker.
- Your answer must include evidence, not assumptions.
- If something fails, report the exact command, exact error, and your fix or next recommended fix.

Return your final answer in this format:

## Environment

## Commands Run

## htdemucs Results

## htdemucs_ft Results

## Raw vs Optimized Comparison

## Workflow Alignment

## Recommended Next Step

## Open Risks

Do not give a generic high-level summary. Include exact outputs, sizes, relevant file paths, and any warnings or failures.
