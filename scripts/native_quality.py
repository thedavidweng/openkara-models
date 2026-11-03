"""Spectral waveform adapter for the exact archive's C API, never a Python wheel."""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import archive_utils
from ort_api_version import HEADER_REL, required_headers
from run_native_smoke import _build_harness, _find_lib, _sha256_file, SPECTRAL_OUTPUT_SHAPE, validation_failures

SEGMENT_FRAMES = 343980


def _sqrt_hann(size: int) -> np.ndarray:
    """The OpenKara streaming-path chunk window: sqrt-Hann, satisfying the
    squared constant-overlap-add constraint at 50% overlap."""
    phase = 2.0 * np.pi * np.arange(size) / size
    return np.sqrt(0.5 * (1.0 - np.cos(phase)))


def _stitch_windows(run_window, inp: np.ndarray, window: int = SEGMENT_FRAMES) -> np.ndarray:
    """Run fixed-window inference over arbitrary-length input with 50%%
    overlap and squared-sqrt-Hann overlap-add — the exact stitching semantics
    of the OpenKara streaming separation path, so chunk-boundary fixtures
    measure what the product does.

    ``run_window`` maps a ``[C, window]`` array to ``[1, S, C, window]``.
    Inputs up to one window run as a single (zero-tail-padded) window; the
    raw model forward cannot take longer inputs at all (fixed ONNX shapes;
    the torch forward's train-segment handling rejects them too).
    """
    channels, n = inp.shape
    if n <= window:
        padded = np.zeros((channels, window), dtype=inp.dtype)
        padded[:, :n] = inp
        return run_window(padded)[..., :n]

    hop = window // 2
    w2 = _sqrt_hann(window) ** 2
    acc: np.ndarray | None = None
    norm = np.zeros(n, dtype=np.float64)
    for start in range(0, n, hop):
        frames = min(window, n - start)
        chunk = np.zeros((channels, window), dtype=inp.dtype)
        chunk[:, :frames] = inp[:, start:start + frames]
        out = run_window(chunk)
        if acc is None:
            acc = np.zeros(out.shape[:-1] + (n,), dtype=np.float64)
        acc[..., start:start + frames] += (
            out[..., :frames].astype(np.float64) * w2[:frames]
        )
        norm[start:start + frames] += w2[:frames]
    assert acc is not None
    return (acc / np.maximum(norm, 1e-8)).astype(np.float32)




class NativeRuntime:
    def __init__(self, archive: Path, source: Path | None, target: str, provider: str, work: Path):
        self.target, self.provider = target, provider
        self.work = work
        archive_utils.safe_extract(archive, work / "runtime")
        self.lib = _find_lib(work / "runtime").resolve()
        headers = list((work / "runtime").rglob("onnxruntime_c_api.h"))
        if len(headers) != 1:
            raise ValueError("runtime must contain exactly one C API header")
        manifests = list((work / "runtime").rglob("build-manifest.json"))
        if len(manifests) != 1:
            raise ValueError("runtime must contain exactly one build manifest")
        manifest = json.loads(manifests[0].read_text())
        if manifest["target"] != target:
            raise ValueError("native measurement target differs from runtime archive")
        packaged_source = work / "headers"
        for name in required_headers(headers[0].read_bytes(), lambda name: headers[0].with_name(name).read_bytes()):
            raw = headers[0].with_name(name).read_bytes()
            record = manifest["files"][name]
            if len(raw) != record["size"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"packaged header digest mismatch: {name}")
            if source is not None and raw != (source / HEADER_REL).with_name(name).read_bytes():
                raise ValueError("harness source must match the exact packaged C API headers")
            dest = (packaged_source / HEADER_REL).with_name(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        self.harness = _build_harness(packaged_source, work / "harness").resolve()
        size, sha = _sha256_file(archive)
        self.archive = {"name": archive.name, "sha256": sha,
                        "size": size, "installed_size": archive_utils.installed_size(archive)}

    def _window(self, model: Path, chunk: np.ndarray, warmup: int, iters: int) -> np.ndarray:
        import spectral_reference as sr
        with tempfile.TemporaryDirectory(dir=self.work) as directory:
            directory = Path(directory)
            mix = chunk[None].astype(np.float32)
            sr.spec(mix).astype(np.float32).tofile(directory / "spectral.f32")
            mix.tofile(directory / "mix.f32")
            command = [str(self.harness), "--lib", str(self.lib), "--model", str(model.resolve()),
                       "--target", self.target, "--provider", self.provider,
                       "--expect-output-shape", SPECTRAL_OUTPUT_SHAPE,
                       "--input-dir", str(directory), "--output-dir", str(directory),
                       "--warmup", str(warmup), "--iters", str(iters)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=directory)
            report = json.loads(result.stdout)
            report["harness_exit_code"] = result.returncode
            problems = validation_failures(report, self.provider)
            latencies = report.get("warm_latencies_s")
            if problems or not isinstance(latencies, list) or len(latencies) != iters:
                raise ValueError(f"native measurement failed: {problems}; {result.stderr[-500:]}")
            if any(type(v) not in (float, int) or not np.isfinite(v) or v < 0
                   for v in [*latencies, report["cold_load_s"], report["first_window_s"], report["peak_rss_kb"], report["rss_delta_kb"]]):
                raise ValueError("native measurements must be finite non-negative numbers")
            self.windows.append(report)
            shapes = [json.loads(s) for s in report["output_shape"].split(";")]
            outputs = [np.fromfile(directory / f"output_{i}.f32", dtype=np.float32).reshape(shape)
                       for i, shape in enumerate(shapes)]
            self.raw_outputs = outputs
            return (sr.ispec(outputs[0].astype(np.float64), chunk.shape[-1]) + outputs[1]).astype(np.float32)

    def separate(self, model: Path, mix: np.ndarray, warmup: int = 0, iters: int = 0) -> np.ndarray:
        self.windows = []
        return _stitch_windows(lambda chunk: self._window(model, chunk, warmup, iters), mix)

    def measure(self, model: Path, mix: np.ndarray, sample_rate: int, warmup: int, iters: int) -> tuple[np.ndarray, dict]:
        if warmup < 0 or iters < 1:
            raise ValueError("warmup >= 0 and iters >= 1 required")
        # The first pass measures actual full waveform wall time, including DSP,
        # process startup, tensor IO and per-window session creation (conservative).
        start = time.perf_counter()
        output = self.separate(model, mix)
        end_to_end_s = time.perf_counter() - start
        first_windows = self.windows
        self.separate(model, mix, warmup, iters)
        latencies = [sum(w["warm_latencies_s"][i] for w in self.windows) for i in range(iters)]
        duration = mix.shape[-1] / sample_rate
        peak = max(w["peak_rss_kb"] for w in first_windows + self.windows)
        metrics = {
            "cold_load_s": sum(w["cold_load_s"] for w in first_windows),
            "first_window_s": first_windows[0]["first_window_s"],
            "warm_median_s": statistics.median(latencies),
            "warm_p95_s": sorted(latencies)[int(0.95 * iters)], "warm_iters": iters,
            "peak_rss_kb": peak, "rss_delta_kb": max(w["rss_delta_kb"] for w in first_windows + self.windows),
            "providers": [self.provider],
            "fallback_node_count": max(w["fallback_node_count"] for w in self.windows),
            "output_shape": [list(output.shape)], "shape_errors": [],
            "frames": mix.shape[-1], "audio_duration_s": duration,
            "rtf_first": end_to_end_s / duration,
            "rtf_warm": statistics.median(latencies) / duration,
            "end_to_end_s": end_to_end_s,
        }
        return output, metrics
