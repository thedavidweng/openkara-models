#!/usr/bin/env python3
"""Run the model correctness + quality suite against the corpus manifest.

For each fixture in the corpus manifest (filtered by --tier), runs the input
through:
  1. The PyTorch reference model (Demucs ensemble).
  2. The ONNX model (candidate or stable).

Then computes correctness metrics:
  - shape validation
  - NaN/Inf checks
  - MSE, MAE, max absolute error (ONNX vs PyTorch)
  - deterministic output digest (SHA-256 of the output bytes)

Emits a JSON report conforming to quality/quality-report-v1.json.

Fails (non-zero exit) if:
  - any output contains NaN or Inf;
  - any output shape does not match expected_output_shape;
  - MSE exceeds the tier threshold (pr_max or release_max).

Usage::

    python scripts/run_quality_suite.py --tier pr --onnx models/htdemucs.onnx --report report.json
    python scripts/run_quality_suite.py --tier release --onnx models/htdemucs.onnx --report report.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CORPUS_MANIFEST = ROOT / "quality" / "corpus-manifest.json"
QUALITY_REPORT_SCHEMA = ROOT / "quality" / "quality-report-v1.json"
GENERATOR_VERSION = "openkara.quality-report/v1"

# MSE thresholds: PR tier is lenient (conversion correctness), release tier
# is strict (product quality). These are the initial budgets; PR 4 freezes
# empirical baselines and may tighten them.
MSE_PR_MAX = 1e-4
MSE_RELEASE_MAX = 1e-5


def _load_corpus(tier: str | None) -> list[dict[str, Any]]:
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    fixtures = manifest["fixtures"]
    if tier:
        fixtures = [f for f in fixtures if f["tier"] == tier]
    return fixtures


def _compute_pytorch_reference(model_name: str, inp: np.ndarray) -> np.ndarray:
    """Run the PyTorch Demucs ensemble reference."""
    import torch
    from demucs_loader import load
    model = load(model_name)
    with torch.no_grad():
        t = torch.from_numpy(inp).unsqueeze(0)  # add batch dim
        out = model(t)
    result = out.squeeze(0).numpy()
    del model, t, out
    gc.collect()
    return result


def _compute_onnx_output(onnx_path: str, inp: np.ndarray) -> np.ndarray:
    """Run the ONNX model."""
    import onnxruntime as ort
    from onnx_runtime_contract import make_contract_compliant_session
    sess = make_contract_compliant_session(Path(onnx_path))
    t = inp[np.newaxis, ...]  # add batch dim
    out = sess.run(None, {sess.get_inputs()[0].name: t})
    result = out[0]
    del sess
    gc.collect()
    return result


def _output_digest(arr: np.ndarray) -> str:
    """SHA-256 of the output bytes (deterministic digest for synthetic fixtures)."""
    h = hashlib.sha256()
    h.update(arr.astype(np.float32).tobytes())
    return h.hexdigest()


def _correctness_metrics(pytorch_out: np.ndarray, onnx_out: np.ndarray,
                         expected_shape: list[int] | None) -> dict[str, Any]:
    """Compute correctness metrics comparing ONNX vs PyTorch."""
    metrics: dict[str, Any] = {
        "onnx_shape": list(onnx_out.shape),
        "pytorch_shape": list(pytorch_out.shape),
        "shape_match": list(onnx_out.shape) == list(pytorch_out.shape),
        "onnx_has_nan": bool(np.any(np.isnan(onnx_out))),
        "onnx_has_inf": bool(np.any(np.isinf(onnx_out))),
        "pytorch_has_nan": bool(np.any(np.isnan(pytorch_out))),
        "pytorch_has_inf": bool(np.any(np.isinf(pytorch_out))),
    }
    if expected_shape is not None:
        metrics["expected_shape"] = expected_shape
        metrics["expected_shape_match"] = list(onnx_out.shape) == expected_shape
    # Numerical comparison.
    if metrics["shape_match"] and not metrics["onnx_has_nan"] and not metrics["onnx_has_inf"]:
        diff = np.abs(onnx_out.astype(np.float64) - pytorch_out.astype(np.float64))
        metrics["mse"] = float((diff ** 2).mean())
        metrics["mae"] = float(diff.mean())
        metrics["max_abs_error"] = float(diff.max())
    else:
        metrics["mse"] = None
        metrics["mae"] = None
        metrics["max_abs_error"] = None
    metrics["onnx_output_digest"] = _output_digest(onnx_out)
    metrics["pytorch_output_digest"] = _output_digest(pytorch_out)
    return metrics


def run_fixture(fixture: dict[str, Any], model_name: str, onnx_path: str) -> dict[str, Any]:
    """Run one fixture through PyTorch + ONNX and return metrics."""
    import synthetic_fixtures as sf
    inp = sf.generate_fixture(fixture)
    print(f"  {fixture['fixture_id']}: {inp.shape} ...", end=" ", flush=True)
    pytorch_out = _compute_pytorch_reference(model_name, inp)
    onnx_out = _compute_onnx_output(onnx_path, inp)
    metrics = _correctness_metrics(pytorch_out, onnx_out, fixture.get("expected_output_shape"))
    metrics["fixture_id"] = fixture["fixture_id"]
    metrics["category"] = fixture["category"]
    metrics["tier"] = fixture["tier"]
    print(f"mse={metrics.get('mse')}")
    del pytorch_out, onnx_out
    gc.collect()
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the model quality suite.")
    parser.add_argument("--tier", choices=["pr", "release"], default="pr",
                        help="Fixture tier to run (default: pr).")
    parser.add_argument("--model", default="htdemucs",
                        help="PyTorch model name (default: htdemucs).")
    parser.add_argument("--onnx", required=True, type=Path,
                        help="ONNX model path.")
    parser.add_argument("--report", type=Path, default=Path("quality-report.json"),
                        help="Output report JSON path.")
    args = parser.parse_args()

    if not args.onnx.is_file():
        print(f"ERROR: ONNX model not found: {args.onnx}", file=sys.stderr)
        return 1

    fixtures = _load_corpus(args.tier)
    if not fixtures:
        print(f"ERROR: no {args.tier} fixtures in corpus manifest", file=sys.stderr)
        return 1

    print(f"Running {len(fixtures)} {args.tier}-tier fixture(s) through {args.model} + {args.onnx.name}...")
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        result = run_fixture(fixture, args.model, str(args.onnx))
        results.append(result)

    mse_threshold = MSE_PR_MAX if args.tier == "pr" else MSE_RELEASE_MAX
    report = {
        "schema_version": GENERATOR_VERSION,
        "model": args.model,
        "onnx_path": str(args.onnx),
        "tier": args.tier,
        "mse_threshold": mse_threshold,
        "results": results,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nreport: {args.report}")

    # Gate: fail on NaN/Inf, shape mismatch, or MSE exceeding threshold.
    failed = False
    for r in results:
        if r["onnx_has_nan"] or r["onnx_has_inf"]:
            print(f"FAIL: {r['fixture_id']}: NaN/Inf in output", file=sys.stderr)
            failed = True
        if r.get("expected_shape_match") is False:
            print(f"FAIL: {r['fixture_id']}: shape mismatch", file=sys.stderr)
            failed = True
        if r.get("mse") is not None and r["mse"] > mse_threshold:
            print(f"FAIL: {r['fixture_id']}: MSE {r['mse']} > {mse_threshold}", file=sys.stderr)
            failed = True
    if failed:
        return 2
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
