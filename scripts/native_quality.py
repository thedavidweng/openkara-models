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
from runtime_headers import PinnedHeaders, packaged_provenance
from run_quality_suite import _stitch_windows


class NativeRuntime:
    def __init__(self, archive: Path, source: Path | None, target: str, provider: str, work: Path,
                 pinned_headers: PinnedHeaders | None = None):
        self.target, self.provider = target, provider
        self.work = work
        self.windows: list[dict] = []
        self.raw_outputs: list[np.ndarray] = []
        archive_utils.safe_extract(archive, work / "runtime")
        self.lib = _find_lib(work / "runtime").resolve()
        headers = list((work / "runtime").rglob("onnxruntime_c_api.h"))
        manifests = list((work / "runtime").rglob("build-manifest.json"))
        if len(manifests) != 1:
            raise ValueError("runtime must contain exactly one build manifest")
        manifest = json.loads(manifests[0].read_text())
        if manifest["target"] != target:
            raise ValueError("native measurement target differs from runtime archive")
        packaged_source = work / "headers"
        if len(headers) == 1:
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
            self.header_provenance = packaged_provenance(manifest)
        elif not headers and pinned_headers is not None:
            build = manifest.get("build", {})
            if build.get("commit_sha") != pinned_headers.commit_sha or \
                    build.get("ort_api_version") != pinned_headers.ort_api_version:
                raise ValueError("pinned headers do not match the runtime build manifest")
            pinned_closure = required_headers(
                pinned_headers.files["onnxruntime_c_api.h"], pinned_headers.files.__getitem__,
            )
            if set(pinned_closure) != set(pinned_headers.files):
                raise ValueError("pinned header set is not the complete C API closure")
            for name, raw in pinned_headers.files.items():
                if source is not None and raw != (source / HEADER_REL).with_name(name).read_bytes():
                    raise ValueError("harness source must match the pinned C API headers")
                dest = (packaged_source / HEADER_REL).with_name(name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
            self.header_provenance = pinned_headers.provenance()
        else:
            raise ValueError(
                "runtime must contain exactly one C API header, or be measured "
                "with a verified pinned header set"
            )
        self.harness = _build_harness(packaged_source, work / "harness").resolve()
        size, sha = _sha256_file(archive)
        self.archive = {"name": archive.name, "sha256": sha,
                        "size": size, "installed_size": archive_utils.installed_size(archive)}

    def _run_window(self, model: Path, inputs: dict[str, np.ndarray],
                    expected_shape_string: str, warmup: int, iters: int):
        """Run the packaged harness once and return (report, raw output tensors)."""
        with tempfile.TemporaryDirectory(dir=self.work) as directory:
            directory = Path(directory)
            for name, array in inputs.items():
                array.astype(np.float32).tofile(directory / f"{name}.f32")
            command = [str(self.harness), "--lib", str(self.lib), "--model", str(model.resolve()),
                       "--target", self.target, "--provider", self.provider,
                       "--expect-output-shape", expected_shape_string,
                       "--input-dir", str(directory), "--output-dir", str(directory),
                       "--warmup", str(warmup), "--iters", str(iters)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=directory)
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"native harness produced no JSON report: {exc}; {result.stderr[-500:]}"
                ) from exc
            report["harness_exit_code"] = result.returncode
            problems = validation_failures(report, self.provider, expected_shape_string)
            latencies = report.get("warm_latencies_s")
            if problems or not isinstance(latencies, list) or len(latencies) != iters:
                raise ValueError(f"native measurement failed: {problems}; {result.stderr[-500:]}")
            if any(type(v) not in (float, int) or not np.isfinite(v) or v < 0
                   for v in [*latencies, report["cold_load_s"], report["first_window_s"], report["peak_rss_kb"], report["rss_delta_kb"]]):
                raise ValueError("native measurements must be finite non-negative numbers")
            shapes = [json.loads(s) for s in report["output_shape"].split(";")]
            outputs = [np.fromfile(directory / f"output_{i}.f32", dtype=np.float32).reshape(shape)
                       for i, shape in enumerate(shapes)]
            if any(not np.isfinite(output).all() for output in outputs):
                raise ValueError("native outputs must be finite")
            self.windows.append(report)
            self.raw_outputs = outputs
            return report, outputs

    def _window(self, model: Path, chunk: np.ndarray, warmup: int, iters: int) -> np.ndarray:
        import spectral_reference as sr
        mix = chunk[None].astype(np.float32)
        _report, outputs = self._run_window(
            model, {"spectral": sr.spec(mix), "mix": mix},
            SPECTRAL_OUTPUT_SHAPE, warmup, iters,
        )
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

    def measure_raw(self, model: Path, inputs: dict[str, np.ndarray],
                    expected_shapes: list[list[int]], frames: int,
                    warmup: int, iters: int) -> tuple[list[np.ndarray], dict, dict]:
        """Measure one fixed window for any declared interface.

        Returns ``(raw_outputs, metrics, harness_report)``. The harness report
        already passed the strict requested-provider/fallback/shape/finiteness
        gate for this exact runtime archive.
        """
        if warmup < 0 or iters < 1:
            raise ValueError("warmup >= 0 and iters >= 1 required")
        if not expected_shapes or any(not shape or any(dim <= 0 for dim in shape)
                                      for shape in expected_shapes):
            raise ValueError("expected output shapes must be non-empty positive dimensions")
        self.windows = []
        shape_string = ";".join(
            "[" + ",".join(str(dim) for dim in shape) + "]" for shape in expected_shapes
        )
        report, outputs = self._run_window(model, inputs, shape_string, warmup, iters)
        observed = [list(output.shape) for output in outputs]
        if observed != [list(shape) for shape in expected_shapes]:
            raise ValueError(f"native output shapes {observed} != expected {expected_shapes}")
        latencies = [float(value) for value in report["warm_latencies_s"]]
        duration = frames / 44100
        metrics = {
            "cold_load_s": report["cold_load_s"],
            "first_window_s": report["first_window_s"],
            "warm_median_s": statistics.median(latencies),
            "warm_p95_s": sorted(latencies)[int(0.95 * iters)],
            "warm_iters": iters,
            "peak_rss_kb": report["peak_rss_kb"],
            "rss_delta_kb": report["rss_delta_kb"],
            "requested_provider": self.provider,
            "assigned_provider": report["provider_assignment"],
            "used_fallback": report["used_fallback"],
            "provider_node_count": report["provider_node_count"],
            "total_node_count": report["total_node_count"],
            "fallback_node_count": report["fallback_node_count"],
            "finite_output": report["finite_output"],
            "output_shapes": observed,
            "installed_size": self.archive["installed_size"],
            "archive_bytes": self.archive["size"],
            "archive_sha256": self.archive["sha256"],
            "header_provenance": self.header_provenance,
            "frames": frames,
            "audio_duration_s": duration,
            "rtf_warm": statistics.median(latencies) / duration,
        }
        return outputs, metrics, report
