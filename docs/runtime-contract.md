# Runtime compatibility contract (OpenKara & official ORT)

This document defines what “standard” ONNX assets published from **openkara-models** must satisfy so that [OpenKara](https://github.com/thedavidweng/OpenKara) can load them with **official, pre-built ONNX Runtime** shared libraries on all supported desktop platforms—without requiring a custom ORT build or extra CMake flags.

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

## Release gate (required before publishing)

Before tagging a **standard** model release, maintainers MUST:

1. **On macOS arm64**, with the **same ORT major.minor** that OpenKara declares (or the minimum supported version), run a minimal check: load the ONNX → create `InferenceSession` with **CPU EP** → run **one** dummy inference.  
2. **CPU session creation MUST succeed**; CoreML EP is optional for this gate if the machine supports it.

The repository enforces a **fast CI gate** on every push/PR:

- Protobuf scan for forbidden node domains (currently `com.microsoft.nchwc`) via `scripts/onnx_runtime_contract.py`  
- `scripts/convert_htdemucs_to_onnx.py` re-checks the graph immediately after ORT writes the optimized file.

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

## Related files

- `scripts/convert_htdemucs_to_onnx.py` — offline ORT optimization level and post-write domain check  
- `scripts/onnx_runtime_contract.py` — shared protobuf domain gate and optional ORT session check  
- `scripts/validate_onnx.py` — numerical validation + domain gate + CPU session load  
- `.github/workflows/runtime-contract.yml` — CI gate on pull requests and `main`
