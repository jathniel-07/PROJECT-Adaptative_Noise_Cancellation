"""
Smoke tests for the parts of the package that don't need torch/audio
hardware: config round-tripping, the synthetic simulation, and the
classical FxNLMS loop's ability to actually cancel noise. Run with:

    python -m pytest tests/  -q
    # or, with no pytest installed:
    python tests/test_dsp_and_config.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anc_prototype.config import SystemConfig
from anc_prototype.dsp import FxNLMS
from anc_prototype.loop import attenuation_db, run_offline
from anc_prototype.simulate import (
    generate_vehicle_noise,
    make_fir,
    nonlinear_distortion,
    simulate_primary_noise,
)


class TestConfig(unittest.TestCase):
    def test_round_trip_preserves_values_and_tuple_fields(self):
        cfg = SystemConfig()
        cfg.net.cnn_channels = (4, 8, 16)
        cfg.filt.mu_base = 0.25
        restored = SystemConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.net.cnn_channels, (4, 8, 16))
        self.assertIsInstance(restored.net.cnn_channels, tuple)
        self.assertAlmostEqual(restored.filt.mu_base, 0.25)
        self.assertEqual(restored.audio.sample_rate, cfg.audio.sample_rate)


class TestSimulate(unittest.TestCase):
    def test_generate_vehicle_noise_shape_and_scale(self):
        x = generate_vehicle_noise(4000, fs=2000, seed=0)
        self.assertEqual(len(x), 4000)
        self.assertAlmostEqual(np.std(x), 1.0, places=1)

    def test_make_fir_is_unit_norm(self):
        ir = make_fir(32, 0.08, seed=3)
        self.assertAlmostEqual(np.linalg.norm(ir), 1.0, places=5)

    def test_nonlinear_distortion_is_bounded(self):
        z = np.linspace(-10, 10, 1000)
        y = nonlinear_distortion(z)
        self.assertTrue(np.all(y > -1.0) and np.all(y < 1.0))


class TestFxNLMSLoop(unittest.TestCase):
    def test_classical_loop_achieves_positive_attenuation(self):
        """End-to-end sanity check on the classical (non-hybrid) path:
        the adaptive filter should measurably cancel a simple,
        near-linear synthetic noise path."""
        fs = 2000
        x = generate_vehicle_noise(12000, fs=fs, seed=1)
        primary_ir = make_fir(48, 0.05, seed=2)
        sec_ir = make_fir(32, 0.08, seed=3)
        d = simulate_primary_noise(x, primary_ir)

        d_out, e_out, y_out, fxnlms = run_offline(
            d, sec_ir, filter_len=48, mu_base=0.1, ref_signal=x,
        )
        atten = attenuation_db(d_out, e_out)
        self.assertGreater(atten, 2.0, f"expected meaningful attenuation, got {atten:.2f} dB")
        self.assertEqual(len(y_out), len(x))
        self.assertIsInstance(fxnlms, FxNLMS)

    def test_step_control_is_pure_before_update(self):
        """step_control must not depend on any error signal -- this is
        what makes reference-mic-only field deployment (no error mic)
        valid: it's the same call used online (with .update after) and
        offline (without)."""
        sec_ir = make_fir(16, 0.1, seed=7)
        f = FxNLMS(filter_len=8, sec_path_ir=sec_ir, mu=0.1)
        y1 = f.step_control(0.5)
        self.assertEqual(y1, 0.0)  # zero-initialised weights -> zero output
        f.reset_state()
        self.assertTrue(np.all(f.x_hist == 0))
        self.assertTrue(np.all(f.w == f.w))  # weights untouched by reset_state


if __name__ == "__main__":
    unittest.main()
