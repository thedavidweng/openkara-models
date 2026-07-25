#!/usr/bin/env python3
"""Verify a derived model artifact against its four-output source (issue #22).

Runs both models on deterministic synthetic fixtures with the
contract-compliant CPU session and checks, per derivation mode:

``--mode dedup`` (lossless storage transform):
    outputs must be bit-identical (max abs diff == 0).

``--mode karaoke2stem`` (deterministic projection):
    - derived output shape is exactly ``[1, 2, C, F]``;
    - derived vocals channel equals the source vocals channel bit-exactly
      (same subgraph, untouched tensor);
    - derived accompaniment equals drums+bass+other within FP32
      reassociation tolerance;
    - no NaN/Inf anywhere.

Usage::

    python scripts/verify_derived_model.py \\
        --source htdemucs.onnx --derived htdemucs.karaoke2stem.onnx \\
        --mode karaoke2stem --report verify-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import synthetic_fixtures  # noqa: E402
from onnx_runtime_contract import make_contract_compliant_session  # noqa: E402

# Small deterministic fixture set; full-window inference on the real models
# is expensive, so verification uses signals that exercise silence, wideband,
# tonal, and transient content.
DEFAULT_FIXTURES = ("white_noise", "sine_sweep", "multitone", "impulse")

ACCOMPANIMENT_ATOL = 1e-5  # FP32 reassociation between graph Sum and numpy sum


def _fixture(name: str) -> np.ndarray:
    fn = getattr(synthetic_fixtures, name)
    audio = fn()  # [channels, frames] float32, deterministic
    return audio[np.newaxis, :, :].astype(np.float32)  # [1, C, F]


def _run_all(model_path: Path, fixtures: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Run every fixture through one session, then release it (the ft
    ensemble needs ~4 GB per live session; never hold two at once)."""
    import gc
    sess = make_contract_compliant_session(model_path)
    in_name = sess.get_inputs()[0].name
    outs = {name: sess.run(None, {in_name: _fixture(name)})[0] for name in fixtures}
    del sess
    gc.collect()
    return outs


def verify_pair(source_path: Path, derived_path: Path, mode: str,
                fixtures: tuple[str, ...]) -> dict:
    src_outs = _run_all(source_path, fixtures)
    der_outs = _run_all(derived_path, fixtures)

    results = []
    ok = True
    for name in fixtures:
        src_out = src_outs[name]
        der_out = der_outs[name]

        entry: dict = {
            "fixture": name,
            "source_shape": list(src_out.shape),
            "derived_shape": list(der_out.shape),
            "derived_has_nan": bool(np.any(np.isnan(der_out))),
            "derived_has_inf": bool(np.any(np.isinf(der_out))),
        }

        if mode == "dedup":
            entry["expected_shape_ok"] = der_out.shape == src_out.shape
            diff = float(np.max(np.abs(der_out - src_out))) if entry["expected_shape_ok"] else None
            entry["max_abs_diff"] = diff
            entry["passed"] = (
                entry["expected_shape_ok"] and diff == 0.0
                and not entry["derived_has_nan"] and not entry["derived_has_inf"]
            )
        elif mode == "karaoke2stem":
            n, s, c, f = src_out.shape
            entry["expected_shape_ok"] = der_out.shape == (n, 2, c, f)
            if entry["expected_shape_ok"]:
                vocals_diff = float(np.max(np.abs(der_out[:, 0] - src_out[:, 3])))
                acc_ref = (src_out[:, 0].astype(np.float64)
                           + src_out[:, 1].astype(np.float64)
                           + src_out[:, 2].astype(np.float64))
                acc_diff = float(np.max(np.abs(der_out[:, 1].astype(np.float64) - acc_ref)))
                entry["vocals_max_abs_diff"] = vocals_diff
                entry["accompaniment_max_abs_diff"] = acc_diff
                entry["passed"] = (
                    vocals_diff == 0.0
                    and acc_diff <= ACCOMPANIMENT_ATOL
                    and not entry["derived_has_nan"] and not entry["derived_has_inf"]
                )
            else:
                entry["passed"] = False
        else:
            raise ValueError(f"unknown mode: {mode}")

        ok = ok and entry["passed"]
        results.append(entry)
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"  [{status}] {name}: {json.dumps({k: v for k, v in entry.items() if 'diff' in k})}")

    return {
        "mode": mode,
        "source": str(source_path),
        "derived": str(derived_path),
        "fixtures": results,
        "passed": ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--derived", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("dedup", "karaoke2stem"))
    parser.add_argument("--fixtures", nargs="*", default=list(DEFAULT_FIXTURES))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    for p in (args.source, args.derived):
        if not p.is_file():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1

    print(f"Verifying {args.derived.name} against {args.source.name} (mode={args.mode})")
    report = verify_pair(args.source, args.derived, args.mode, tuple(args.fixtures))

    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report: {args.report}")

    if report["passed"]:
        print("OK: derived model verified against source")
        return 0
    print("ERROR: verification FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
