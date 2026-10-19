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
from run_quality_suite import _stitch_windows


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
        for name in required_headers(headers[0].read_bytes()):
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
