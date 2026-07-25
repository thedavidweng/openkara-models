#!/usr/bin/env python3
"""Tests for scripts/spectral_core_patch.py (issue #23 PR 2).

torch-only (skipped in CI workflows without torch; the conversion workflow
exercises the patch against the real model). Uses a FakeModel that mimics
the HTDemucs forward tail — no demucs, no downloads, tiny tensors."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SpectralCoreTracePatchTests(unittest.TestCase):
    def _make_fake_model(self):
        """Mimics the HTDemucs forward tail with tiny tensors: the patch
        never validates tensor shapes, only the model's contract attrs."""

        class FakeHTDemucs:
            cac = True
            nfft = 4096
            hop_length = 1024

            def _spec(self, x):
                raise AssertionError("original _spec must not run")

            def _ispec(self, z, length=None, scale=0):
                raise AssertionError("original _ispec must not run")

            def _magnitude(self, z):
                raise AssertionError("original _magnitude must not run")

            def _mask(self, z, m):
                raise AssertionError("original _mask must not run")

            def __call__(self, mix):
                z = self._spec(mix)
                m = self._magnitude(z)
                net = m.unsqueeze(1).repeat(1, 4, 1, 1, 1)
                zout = self._mask(z, net)
                x = self._ispec(zout, mix.shape[-1])
                xt = mix.unsqueeze(1).repeat(1, 4, 1, 1) * 2.0
                return xt + x

        return FakeHTDemucs()

    def _run_core(self):
        from spectral_core_patch import (
            SpectralCoreModule,
            SpectralCoreTracePatch,
        )

        model = self._make_fake_model()
        patch = SpectralCoreTracePatch.from_model(model)
        core = SpectralCoreModule(model, patch)
        spectral = torch.randn(1, 2, 2, 8, 5)
        mix = torch.randn(1, 2, 40)
        with patch:
            spectral_out, time_out = core(spectral, mix)
        return spectral, mix, spectral_out, time_out

    def test_time_out_is_exactly_the_time_branch(self):
        _, mix, _, time_out = self._run_core()
        expected = mix.unsqueeze(1).repeat(1, 4, 1, 1) * 2.0
        self.assertTrue(torch.equal(time_out, expected))

    def test_spectral_out_is_the_captured_mask_view(self):
        spectral, _, spectral_out, _ = self._run_core()
        # FakeModel's "network" repeats the magnitude view across 4 sources,
        # so the captured pre-ISTFT tensor is the input spectral tensor
        # repeated on the source axis.
        expected = spectral.reshape(1, 4, 8, 5).unsqueeze(1).repeat(
            1, 4, 1, 1, 1
        ).view(1, 4, 2, 2, 8, 5)
        self.assertTrue(torch.equal(spectral_out, expected))

    def test_take_captured_without_forward_raises(self):
        from spectral_core_patch import SpectralCoreTracePatch

        patch = SpectralCoreTracePatch.from_model(self._make_fake_model())
        with self.assertRaises(RuntimeError):
            patch.take_captured()

    def test_restore_returns_original_methods(self):
        from spectral_core_patch import SpectralCoreTracePatch

        model = self._make_fake_model()
        originals = {
            name: getattr(model, name).__func__
            for name in ("_spec", "_ispec", "_magnitude", "_mask")
        }
        with SpectralCoreTracePatch.from_model(model):
            for name, original in originals.items():
                self.assertIsNot(getattr(model, name).__func__, original)
        for name, original in originals.items():
            self.assertIs(getattr(model, name).__func__, original)

    def test_rejects_non_cac_model(self):
        from spectral_core_patch import SpectralCoreTracePatch

        model = self._make_fake_model()
        model.cac = False
        with self.assertRaises(RuntimeError):
            SpectralCoreTracePatch.from_model(model)

    def test_rejects_non_contract_transform_constants(self):
        from spectral_core_patch import SpectralCoreTracePatch

        model = self._make_fake_model()
        model.nfft = 2048
        with self.assertRaises(RuntimeError):
            SpectralCoreTracePatch.from_model(model)

    def test_module_rejects_foreign_patch(self):
        from spectral_core_patch import (
            SpectralCoreModule,
            SpectralCoreTracePatch,
        )

        patch = SpectralCoreTracePatch.from_model(self._make_fake_model())
        with self.assertRaises(ValueError):
            SpectralCoreModule(self._make_fake_model(), patch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
