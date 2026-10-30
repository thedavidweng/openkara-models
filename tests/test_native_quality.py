"""Small mocked C API boundary tests, never release measurements."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import native_quality as native


def archive(tmp_path, monkeypatch, *, target='target', corrupt=False, ep=True):
    header = b'#define ORT_API_VERSION 27\n'
    if ep:
        header += b'#include "onnxruntime_ep_c_api.h"\n'
    files = {'onnxruntime_c_api.h': header, 'libonnxruntime.so': b'fake-runtime'}
    if ep:
        files['onnxruntime_ep_c_api.h'] = b'// ep header\n'
    manifest = {'target': target, 'files': {name: {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                                         for name, raw in files.items()}}
    if corrupt:
        manifest['files']['onnxruntime_c_api.h']['sha256'] = 'f' * 64
    files['build-manifest.json'] = json.dumps(manifest).encode()
    path = tmp_path / 'runtime.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name, raw in files.items():
            z.writestr(name, raw)
    def build(source, dest):
        assert (source / native.HEADER_REL).read_bytes() == header
        if ep:
            assert (source / native.HEADER_REL).with_name('onnxruntime_ep_c_api.h').read_bytes() == files['onnxruntime_ep_c_api.h']
        return tmp_path / 'harness'
    monkeypatch.setattr(native, '_build_harness', build)
    return path


def test_packaged_headers_compile_without_checkout(tmp_path, monkeypatch):
    path = archive(tmp_path, monkeypatch)
    rt = native.NativeRuntime(path, None, 'target', 'cpu', tmp_path / 'work')
    assert rt.archive['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rt.lib.read_bytes() == b'fake-runtime'


@pytest.mark.parametrize('fault', ['digest', 'target', 'source'])
def test_rejects_wrong_header_or_target(tmp_path, monkeypatch, fault):
    path = archive(tmp_path, monkeypatch, corrupt=fault == 'digest')
    source = None
    if fault == 'source':
        source = tmp_path / 'source'
        p = source / native.HEADER_REL
        p.parent.mkdir(parents=True)
        p.write_bytes(b'wrong header')
    with pytest.raises(ValueError):
        native.NativeRuntime(path, source, 'other' if fault == 'target' else 'target', 'cpu', tmp_path / 'work')


def test_measure_separates_walltime_and_native_warm_samples(tmp_path, monkeypatch):
    rt = object.__new__(native.NativeRuntime)
    calls = []
    def separate(model, mix, warmup=0, iters=0):
        calls.append((warmup, iters))
        rt.windows = [{'warm_latencies_s': [0.1, 0.3][:iters], 'cold_load_s': 0.2,
                       'first_window_s': 0.4, 'peak_rss_kb': 123, 'rss_delta_kb': 23,
                       'fallback_node_count': 0}]
        return np.zeros((1, 4, 2, 4), np.float32)
    rt.separate, rt.provider = separate, 'cpu'
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(native.time, 'perf_counter', lambda: next(clock))
    _, report = rt.measure(Path('candidate.onnx'), np.zeros((2, 4), np.float32), 4, 1, 2)
    assert calls == [(0, 0), (1, 2)]
    assert report['end_to_end_s'] == report['rtf_first'] == 2
    assert report['warm_median_s'] == report['rtf_warm'] == 0.2
    assert report['rss_delta_kb'] == 23


@pytest.mark.parametrize('exit_code,latencies', [(1, [0.1]), (0, []), (0, [float('nan')])])
def test_native_process_failure_cannot_qualify(tmp_path, monkeypatch, exit_code, latencies):
    import spectral_reference as sr
    monkeypatch.setattr(sr, 'spec', lambda x: x)
    rt = object.__new__(native.NativeRuntime)
    rt.work, rt.harness, rt.lib = tmp_path, Path('/harness'), Path('/runtime')
    rt.target, rt.provider, rt.windows = 'target', 'cpu', []
    report = {'warm_latencies_s': latencies, 'cold_load_s': 0.1, 'first_window_s': 0.2,
              'peak_rss_kb': 10, 'rss_delta_kb': 1}
    def run(command, **kwargs):
        assert command[command.index('--lib') + 1] == '/runtime'
        return SimpleNamespace(stdout=json.dumps(report), returncode=exit_code, stderr='failure')
    monkeypatch.setattr(native.subprocess, 'run', run)
    monkeypatch.setattr(native, 'validation_failures', lambda r, p, expected=None: ['exit'] if r['harness_exit_code'] else [])
    with pytest.raises(ValueError, match='native measure'):
        rt._window(Path('model'), np.zeros((2, 4), np.float32), 0, 1)


def test_provider_companion_is_not_runtime_library(tmp_path):
    (tmp_path / 'onnxruntime_providers_shared.dll').write_bytes(b'companion')
    with pytest.raises(ValueError, match='exactly one'):
        native._find_lib(tmp_path)
    (tmp_path / 'onnxruntime.dll').write_bytes(b'actual-runtime')
    assert native._find_lib(tmp_path).name == 'onnxruntime.dll'


def test_native_window_transfers_input_and_composes_both_raw_outputs(tmp_path, monkeypatch):
    import spectral_reference as sr
    monkeypatch.setattr(sr, 'spec', lambda mix: mix * 2)
    monkeypatch.setattr(sr, 'ispec', lambda spectrum, frames: spectrum)
    rt = object.__new__(native.NativeRuntime)
    rt.work, rt.harness, rt.lib = tmp_path, Path('/harness'), Path('/exact-runtime')
    rt.target, rt.provider, rt.windows = 'target', 'cpu', []
    chunk = np.arange(8, dtype=np.float32).reshape(2, 4)
    def run(command, **kwargs):
        directory = Path(command[command.index('--input-dir') + 1])
        assert command[command.index('--lib') + 1] == '/exact-runtime'
        np.testing.assert_array_equal(np.fromfile(directory / 'mix.f32', dtype=np.float32), chunk.ravel())
        np.testing.assert_array_equal(np.fromfile(directory / 'spectral.f32', dtype=np.float32), (chunk * 2).ravel())
        for i in range(2):
            np.full((1, 4, 2, 4), i + 2, np.float32).tofile(directory / f'output_{i}.f32')
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps({
            'output_shape': '[1,4,2,4];[1,4,2,4]', 'warm_latencies_s': [0.1],
            'cold_load_s': 0.2, 'first_window_s': 0.3, 'peak_rss_kb': 10, 'rss_delta_kb': 1}))
    monkeypatch.setattr(native.subprocess, 'run', run)
    monkeypatch.setattr(native, 'validation_failures', lambda report, provider, expected=None: [])
    output = rt._window(Path('candidate'), chunk, 0, 1)
    np.testing.assert_array_equal(output, np.full((1, 4, 2, 4), 5, np.float32))
    assert len(rt.raw_outputs) == 2
    assert rt.windows[0]['warm_latencies_s'] == [0.1]


def test_benchmark_routes_archive_to_native_measurement(tmp_path, monkeypatch):
    import run_runtime_benchmarks as bench
    class Measured:
        raw_outputs = [np.zeros((1, 4, 2, 4)), np.zeros((1, 4, 2, 4))]
        def measure(self, model, mix, rate, warmup, iters):
            assert (model, rate, warmup, iters) == (Path('candidate'), 44100, 1, 2)
            assert mix.shape == (2, 4)
            return None, {'audio_duration_s': 1, 'rtf_first': 1, 'rtf_warm': 1, 'end_to_end_s': 1}
    monkeypatch.setattr(bench, '_load_session', lambda *a, **k: pytest.fail('wheel used for native archive'))
    result = bench.benchmark_model(Path('archive-lib'), Path('candidate'), 4, 1, 2, native=Measured())
    assert result['interface'] == 'spectral'
    assert len(result['output_shape']) == 2


def test_runtime_initializes_window_state_for_measure_raw(tmp_path, monkeypatch):
    path = archive(tmp_path, monkeypatch)
    rt = native.NativeRuntime(path, None, 'target', 'cpu', tmp_path / 'work')
    assert rt.windows == []
    assert rt.raw_outputs == []

    rt.windows = [{'stale': True}]
    outputs = [np.zeros((1, 4, 2, 4), np.float32)]
    report = {
        'warm_latencies_s': [0.1], 'cold_load_s': 0.2, 'first_window_s': 0.3,
        'peak_rss_kb': 10, 'rss_delta_kb': 1, 'provider_assignment': 'cpu',
        'used_fallback': False, 'provider_node_count': 1, 'total_node_count': 1,
        'fallback_node_count': 1, 'finite_output': True,
    }
    monkeypatch.setattr(
        rt, '_run_window',
        lambda model, inputs, expected, warmup, iters: (report, outputs),
    )
    _outputs, metrics, _report = rt.measure_raw(
        tmp_path / 'model.onnx', {'mix': np.zeros((1, 2, 4), np.float32)},
        [[1, 4, 2, 4]], 4, 0, 1,
    )
    assert rt.windows == []
    assert metrics['warm_iters'] == 1
    assert metrics['output_shapes'] == [[1, 4, 2, 4]]
