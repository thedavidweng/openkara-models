#!/usr/bin/env python3
"""Compare raw model outputs from two native ONNX Runtime archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

import archive_utils
import spectral_reference
from run_native_smoke import SPECTRAL_OUTPUT_SHAPE, _build_harness, validation_failures
from run_runtime_benchmarks import _detect_target

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 343980
TOLERANCE = 1e-4
FIXTURE_SHA256 = "9f5bdcaf036f6f2453048d7d809b5db39d5edf846ac5a792474049721549c2e1"


def windows(audio: np.ndarray):
    """Use the same half-window starts and zero padding as OpenKara."""
    if audio.shape[0] != 2 or audio.shape[1] <= FRAMES:
        raise ValueError("comparison audio must be stereo and cross a window boundary")
    for start in range(0, audio.shape[1], FRAMES // 2):
        chunk = np.zeros((1, 2, FRAMES), dtype=np.float32)
        length = min(FRAMES, audio.shape[1] - start)
        chunk[0, :, :length] = audio[:, start:start + length]
        yield chunk


def fixture_audio(path: Path) -> np.ndarray:
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise ValueError("music fixture SHA-256 mismatch")
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (2, 2, 44100):
            raise ValueError("fixture must be stereo 16-bit PCM at 44100 Hz")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return (samples.reshape(-1, 2).T / 32768).astype(np.float32)


def max_error(reference: list[np.ndarray], candidate: list[np.ndarray]) -> list[float]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("runtime output counts differ")
    errors = []
    for left, right in zip(reference, candidate):
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("runtime output shape or finiteness differs")
        errors.append(float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))))
    return errors


def qualifies(results: list[dict], target: str) -> bool:
    if not results or any(max(row["output_max_abs"]) > TOLERANCE for row in results):
        return False
    return target != "x86_64-unknown-linux-gnu" or (
        statistics.median(row["candidate_cold_load_s"] for row in results)
        <= 1.5 * statistics.median(row["stable_cold_load_s"] for row in results)
        and statistics.median(row["candidate_warm_rtf"] for row in results)
        <= 1.5 * statistics.median(row["stable_warm_rtf"] for row in results)
    )


def verify_archive(archive: Path, expected_size: int, expected_sha256: str) -> str:
    if archive.stat().st_size != expected_size:
        raise ValueError(f"runtime archive size mismatch: {archive}")
    with archive.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"runtime archive SHA-256 mismatch: {archive}")
    return digest


class Runner:
    def __init__(self, archive: Path, expected_size: int, expected_sha256: str, source: Path, target: str, work: Path):
        digest = verify_archive(archive, expected_size, expected_sha256)
        extracted = archive_utils.safe_extract(archive, work / "runtime")
        libraries = [
            p for p in extracted.rglob("*")
            if p.is_file() and (
                p.name in {"libonnxruntime.dylib", "libonnxruntime.so", "onnxruntime.dll"}
                or p.name.startswith("libonnxruntime.so.")
            )
        ]
        exact = [p for p in libraries if p.name in {
            "libonnxruntime.dylib", "libonnxruntime.so", "onnxruntime.dll"
        }]
        libraries = exact or libraries
        if len(libraries) != 1:
            raise ValueError("runtime archive must contain one loadable ORT library")
        self.library = libraries[0].resolve()
        self.harness = _build_harness(source, work / "harness").resolve()
        self.target = target
        self.digest = digest

    def run(self, model: Path, chunk: np.ndarray, work: Path) -> tuple[list[np.ndarray], dict]:
        work.mkdir()
        chunk.tofile(work / "mix.f32")
        spectral_reference.spec(chunk).astype(np.float32).tofile(work / "spectral.f32")
        result = subprocess.run([
            str(self.harness), "--lib", str(self.library), "--model", str(model.resolve()),
            "--target", self.target, "--provider", "cpu",
            "--expect-output-shape", SPECTRAL_OUTPUT_SHAPE,
            "--input-dir", str(work), "--output-dir", str(work),
            "--warmup", "1", "--iters", "3",
        ], capture_output=True, text=True, timeout=600, cwd=work)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"native harness produced invalid JSON (exit={result.returncode}); "
                f"{result.stderr[-500:]}"
            ) from exc
        report["harness_exit_code"] = result.returncode
        failures = validation_failures(report, "cpu")
        if failures:
            raise ValueError(f"native inference failed: {failures}; {result.stderr[-500:]}")
        shapes = [json.loads(shape) for shape in report["output_shape"].split(";")]
        outputs = [
            np.fromfile(work / f"output_{i}.f32", dtype=np.float32).reshape(shape)
            for i, shape in enumerate(shapes)
        ]
        return outputs, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stable", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-size", type=int, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
        parser.add_argument(f"--{name}-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", default=_detect_target())
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    clip = fixture_audio(ROOT / "tests/fixtures/piano-cc0.wav")
    with args.model.open("rb") as file:
        model_digest = hashlib.file_digest(file, "sha256").hexdigest()
    noise = np.random.default_rng(42).normal(0, 0.01, clip.shape).astype(np.float32)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stable = Runner(args.stable, args.stable_size, args.stable_sha256, args.stable_source, args.target, root / "stable")
        candidate = Runner(args.candidate, args.candidate_size, args.candidate_sha256, args.candidate_source, args.target, root / "candidate")
        for label, audio in (("noise", noise), ("piano-cc0", clip)):
            for index, chunk in enumerate(windows(audio)):
                print(f"{label} window {index}: stable", flush=True)
                stable_out, stable_report = stable.run(args.model, chunk, root / f"{label}-{index}-stable")
                print(f"{label} window {index}: candidate", flush=True)
                candidate_out, candidate_report = candidate.run(args.model, chunk, root / f"{label}-{index}-candidate")
                errors = max_error(stable_out, candidate_out)
                print(f"{label} window {index}: max errors {errors}", flush=True)
                stable_warm = statistics.median(stable_report["warm_latencies_s"])
                candidate_warm = statistics.median(candidate_report["warm_latencies_s"])
                results.append({
                    "fixture": label, "window": index, "output_max_abs": errors,
                    "stable_cold_load_s": stable_report["cold_load_s"],
                    "candidate_cold_load_s": candidate_report["cold_load_s"],
                    "stable_warm_rtf": stable_warm / (FRAMES / 44100),
                    "candidate_warm_rtf": candidate_warm / (FRAMES / 44100),
                })
    passed = qualifies(results, args.target)
    args.report.write_text(json.dumps({
        "target": args.target, "stable_sha256": stable.digest,
        "candidate_sha256": candidate.digest, "max_abs_tolerance": TOLERANCE,
        "model_sha256": model_digest, "fixture_sha256": FIXTURE_SHA256,
        "windows": results, "passed": passed,
    }, indent=2, allow_nan=False) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
