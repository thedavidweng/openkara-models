#!/usr/bin/env python3
"""Generate the spectral-contract golden vectors (issue #23 PR 1).

For each deterministic fixture, produces the input waveform, the spectral
tensor (``spec``), the neural-core magnitude view, and the identity
round-trip reconstruction (``ispec`` of the un-modified spectral tensor),
all float32, written as ``.npz`` bundles. Every array's SHA-256 is pinned in
``quality/spectral-golden-v1.json`` so CI can regenerate and diff without
committing the tensors; the bundles themselves are published as release
assets (tag ``spectral-golden-v1``) for the native implementation
(OpenKara's Rust FFT) to validate against.

Usage::

    python scripts/generate_spectral_golden_vectors.py --out-dir build/spectral-golden
    python scripts/generate_spectral_golden_vectors.py --verify   # digests only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import spectral_reference as sr  # noqa: E402

MANIFEST_PATH = ROOT / "quality" / "spectral-golden-v1.json"
CONTRACT_VERSION = "openkara.spectral-contract/v1"
SR = 44100


def _fixtures() -> dict[str, np.ndarray]:
    """Deterministic [1, 2, samples] float32 fixtures. Lengths cover the
    hop-aligned and unaligned cases; content covers silence, transient,
    tonal, wideband-swept, and band-limited-noise signals."""
    t_aligned = np.arange(10240) / SR      # exactly 10 hops
    t_unaligned = np.arange(10000) / SR    # ceil -> 10 hops with right pad

    silence = np.zeros((1, 2, 10240), dtype=np.float32)

    impulse = np.zeros((1, 2, 10240), dtype=np.float32)
    impulse[:, :, 4410] = 1.0

    tone = np.sin(2 * np.pi * 440.0 * t_aligned).astype(np.float32) * 0.5
    tone = np.stack([tone, tone * 0.5])[np.newaxis]

    log_f = np.exp(np.log(20.0) + (np.log(20000.0) - np.log(20.0))
                   * t_unaligned / t_unaligned[-1])
    phase = 2 * np.pi * np.cumsum(log_f) / SR
    sweep = (np.sin(phase) * 0.3).astype(np.float32)
    sweep = np.stack([sweep, sweep])[np.newaxis]

    rng = np.random.default_rng(11)
    noise = rng.standard_normal((1, 2, 10240)) * 0.2
    nf = np.fft.rfft(noise, axis=-1)
    nf[..., int(nf.shape[-1] * 0.95):] = 0.0
    noise = np.fft.irfft(nf, n=10240, axis=-1).astype(np.float32)

    return {
        "silence-10240": silence,
        "impulse-10240": impulse,
        "tone440-10240": tone,
        "sweep-10000": sweep,
        "bandlimited-noise-10240": noise,
    }


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_vectors() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, x in _fixtures().items():
        z = sr.spec(x).astype(np.float32)
        mag = sr.magnitude(z).astype(np.float32)
        roundtrip = sr.ispec(z[:, np.newaxis].astype(np.float64), x.shape[-1])[:, 0]
        out[name] = {
            "input": x.astype(np.float32),
            "spectral": z,
            "magnitude": mag,
            "roundtrip": roundtrip.astype(np.float32),
        }
    return out


def build_manifest(vectors: dict[str, dict[str, np.ndarray]]) -> dict:
    fixtures = {}
    for name, arrays in sorted(vectors.items()):
        fixtures[name] = {
            key: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": _sha256(a)}
            for key, a in sorted(arrays.items())
        }
    return {
        "schema_version": "openkara.spectral-golden/v1",
        "contract_version": CONTRACT_VERSION,
        "generator": "scripts/generate_spectral_golden_vectors.py",
        "reference": "scripts/spectral_reference.py",
        "distribution": "release assets, tag spectral-golden-v1",
        "tolerances": {
            "fp32_implementation_max_abs": 1e-3,
            "fp64_implementation_max_abs": 1e-6,
        },
        "fixtures": fixtures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Write .npz bundles here (omit to only build digests).")
    parser.add_argument("--verify", action="store_true",
                        help="Regenerate digests and diff against the committed manifest.")
    parser.add_argument("--write-manifest", action="store_true",
                        help="Write/overwrite the committed digest manifest.")
    args = parser.parse_args()

    vectors = build_vectors()
    manifest = build_manifest(vectors)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, arrays in vectors.items():
            np.savez_compressed(args.out_dir / f"{name}.npz", **arrays)
        (args.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"OK: wrote {len(vectors)} golden bundles to {args.out_dir}")

    if args.write_manifest:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"OK: wrote {MANIFEST_PATH}")

    if args.verify:
        committed = json.loads(MANIFEST_PATH.read_text())
        if committed != manifest:
            print("ERROR: regenerated golden digests differ from the committed "
                  f"manifest {MANIFEST_PATH}", file=sys.stderr)
            return 1
        print(f"OK: golden vectors are fresh ({len(vectors)} fixtures)")

    if not (args.out_dir or args.verify or args.write_manifest):
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
