# Runtime compatibility contract (OpenKara & official ORT)

This document defines what “standard” ONNX assets published from **openkara-models** must satisfy so that [OpenKara](https://github.com/thedavidweng/OpenKara) can load them with **official, pre-built ONNX Runtime** shared libraries on all supported desktop platforms—without requiring a custom ORT build or extra CMake flags.

**OpenKara default pins** (URLs + SHA-256 that the app ships) are documented in the root [README.md](../README.md) under **Integrate with OpenKara**; as of the contract fix for Apple Silicon, standard assets use at least **`model-v2.0.1`** / **`model-ft-v2.0.1`**. Do not use **`model-v2.0.0`** as the default standard download.

## Target platforms and runtime

Standard ONNX artifacts MUST be loadable with **official ONNX Runtime release packages** for at least:

- Linux x64  
- Windows x64  
- macOS x64 (Intel)  
- macOS arm64 (Apple Silicon)

OpenKara **dynamically loads** the official `onnxruntime` shared library. Releases MUST NOT assume downstream users run a **self-built ORT** with non-default kernels (for example NCHWc-specific CPU kernels enabled only in certain build configurations).

## Forbidden or conditional offline optimizations

Models distributed as the **default / standard** OpenKara download MUST NOT depend on operator domains or ops that are **not registered in official ORT on all of the platforms above**, including:

- **`com.microsoft.nchwc`** (e.g. NCHWc-fused `Conv` and related layout-specific ops)  
- Other **contrib / experimental** domains that are only registered in some ORT builds or only with specific execution providers

If the project wants **x64-only or “maximum CPU perf”** variants that use such optimizations, they MUST:

- Use a **distinct asset name**, **checksum**, and **release tag**  
- Be **clearly documented** as **not compatible** with official macOS arm64 ORT (and any other excluded platform)  
- **NOT** be the default “standard model” that OpenKara points end users to

## Allowed optimizations

Allowed optimizations include:

- Fusions and rewrites whose result remains in the **standard ONNX operator set** (`ai.onnx` / empty domain), or uses ops that **official ORT registers by default on all target platforms**  
- Metadata as already used in this repo, e.g. `openkara.model_cache_key`, `openkara.optimized_by=onnxruntime`

**Meaning of `openkara.optimized_by=onnxruntime`:** the artifact was optimized with ONNX Runtime **subject to this contract** (offline graph optimization that does **not** introduce forbidden domains such as `com.microsoft.nchwc`). It does **not** mean “every ORT optimizer at `ORT_ENABLE_ALL` was applied.”

The conversion script uses **`ORT_ENABLE_EXTENDED`** for offline optimization so layout passes that emit `com.microsoft.nchwc` are not applied when writing the shipped ONNX.

**Log severity:** contract-compliant sessions set `log_severity_level = 3` (ERROR) instead of the default 2 (WARNING) to suppress non-essential ORT warnings at the session level. This does NOT suppress the `device_discovery.cc GetPciBusId` warning that ORT 1.24+ emits on GitHub Linux runners ([microsoft/onnxruntime#27268](https://github.com/microsoft/onnxruntime/issues/27268)) — that warning comes from a statically-initialized logger in the pybind module that hardcodes WARNING and bypasses all Python API and env var control ([microsoft/onnxruntime#27092](https://github.com/microsoft/onnxruntime/issues/27092)). The warning is harmless (GPU device discovery for TRT-RTX EP, irrelevant with CPU EP). The only workaround (redirecting fd 2 during `import onnxruntime`) hides all stderr including real errors, so we accept the warning.

## Release gate (required before publishing)

Before tagging a **standard** model release, maintainers MUST:

1. **On macOS arm64**, with the **same ORT major.minor** that OpenKara declares (or the minimum supported version), run a minimal check: load the ONNX → create `InferenceSession` with **CPU EP** → run **one** dummy inference.  
2. **CPU session creation MUST succeed**; CoreML EP is optional for this gate if the machine supports it.

The repository enforces a **fast CI gate** on every push/PR:

- Protobuf scan for forbidden node domains (currently `com.microsoft.nchwc`) via `scripts/onnx_runtime_contract.py`  
- `scripts/export_spectral_core.py` re-checks the graph immediately after ORT writes the optimized file.

The **authoritative** check for releases remains loading with **official ORT** on **macOS arm64** as above.

## Versioning and migration

When fixing incompatibility (e.g. removing `com.microsoft.nchwc` from default artifacts):

- **Bump** the model release version (e.g. `v2.0.1` or `v2.1.0`)  
- Update **SHA256** and **release notes** stating that the fix addresses **Apple Silicon official ORT** failing to load graphs containing NCHWc-domain ops  

OpenKara will update download URL + SHA256; older assets (e.g. `v2.0.0`) may be marked **deprecated** or removed from default onboarding.

## Reproduce on Apple Silicon with official ORT 1.23.x

After installing matching `onnx` / `onnxruntime` wheels (same major.minor as the app, e.g. 1.23.x):

```bash
python -m venv .venv && source .venv/bin/activate
pip install "onnxruntime==1.23.*" onnx

python -c "
import numpy as np
import onnxruntime as ort

path = 'htdemucs.onnx'  # or htdemucs_ft.onnx
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess = ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
shape = inp.shape
# Replace dynamic dims with 1 for a smoke run
def fix(d):
    if isinstance(d, str):
        return 1
    return int(d)
sh = [fix(d) for d in shape]
x = np.random.randn(*sh).astype(np.float32)
sess.run(None, {inp.name: x})
print('OK: CPU InferenceSession + one run')
"
```

Optional: pass `providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']` if CoreML is installed and you want to smoke-test that path; **CPU must still work** for standard models.

## Source-built runtime distribution (issue #19)

The v2 catalog references Microsoft's official ORT 1.31.1 packages for Linux,
Apple Silicon and Windows. Intel macOS uses a full-operator CPU and CoreML
build cross-compiled from the pinned source on Apple Silicon and tested under
Rosetta. DirectML is built only by the manual comparison workflow. Windows and
Apple Silicon can also use the official WebGPU 0.5.0 plugin wheel; Intel macOS
does not provide this plugin.

The catalog identifies each package by URL, size and SHA-256 and declares its
model compatibility. The native smoke gate loads the model and runs a complete
343980-frame window. The application verifies archive and installed-file
digests before loading the library. Its ORT C API level must be compatible with
the catalog runtime's API level (31 for ORT 1.31.1).

## Related files

- `scripts/export_spectral_core.py` — spectral-core export, offline ORT optimization level, and post-write domain check  
- `scripts/onnx_runtime_contract.py` — shared protobuf domain gate and optional ORT session check  
- `scripts/validate_spectral_contract.py` — spectral-core contract gate + numerical validation against the PyTorch reference  
- `.github/workflows/runtime-contract.yml` — CI gate on pull requests and `main`
- `ort/source-lock.json` — pinned ORT source + toolchain + build config (issue #19)
- `scripts/build_runtime.py` — reproducible source build (issue #19 PR 1)
- `scripts/generate_runtime_supply_chain.py` — SBOM + provenance (issue #19 PR 3)
- `scripts/run_runtime_benchmarks.py` — native compat+perf matrix (issue #19 PR 4)
- `scripts/generate_runtime_catalog_entries.py` — catalog spec fragment (issue #19 PR 5)
- `.github/workflows/ort-build.yml` — cached Intel macOS build and Rosetta smoke
- `.github/workflows/ort-directml.yml` — manual DirectML comparison build
- `.github/workflows/official-runtime-smoke.yml` — official package smoke tests
