"""
Synthetic defence-vehicle cabin acoustic environment.

Everything in this file is SIMULATION ONLY: it stands in for (a) a
real reference-mic recording, and (b) the real, physically-existing
noise path from source to ear that a fielded system never needs to
compute (the distortion is already present in what an error mic would
measure). Used for development, offline calibration without hardware,
and regression testing -- never called from `runtime.py` (field mode).
"""
from __future__ import annotations

import numpy as np


def generate_vehicle_noise(n_samples: int, fs: int = 2000, seed: int = 0) -> np.ndarray:
    """Synthetic reference-mic signal: engine harmonics (RPM ramps up
    then down, like accelerating and cruising) + broadband road/track
    rumble + wind buffeting. Unit-RMS normalised. Stands in for a real
    reference-mic recording."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs

    rpm = 900 + 900 * (0.5 - 0.5 * np.cos(2 * np.pi * t / (n_samples / fs)))
    firing_freq = rpm / 60.0 * 2.0
    phase = 2 * np.pi * np.cumsum(firing_freq) / fs

    engine = (1.0 * np.sin(phase)
              + 0.5 * np.sin(2 * phase)
              + 0.25 * np.sin(3 * phase))

    road = rng.normal(0, 0.4, n_samples)
    smooth = np.ones(15) / 15
    road = np.convolve(road, smooth, mode="same")

    wind = 0.15 * rng.normal(0, 1, n_samples)

    x = engine + road + wind
    x = x / (np.std(x) + 1e-8)
    return x.astype(np.float32)


def make_fir(length: int, decay: float, seed: int) -> np.ndarray:
    """A decaying random FIR impulse response, standing in for a
    physical acoustic path (primary or secondary) in simulation."""
    rng = np.random.default_rng(seed)
    ir = rng.normal(0, 1, length) * np.exp(-decay * np.arange(length))
    ir = ir / np.linalg.norm(ir)
    return ir.astype(np.float32)


def nonlinear_distortion(z: np.ndarray, alpha: float = 1.6) -> np.ndarray:
    """Mild soft-saturation nonlinearity applied in the *simulated*
    noise path, representing combustion/exhaust/turbulence distortion
    that a purely linear adaptive filter cannot fully model. In a real
    system this is never computed -- the distortion is already
    physically present in whatever the error mic records."""
    return np.tanh(alpha * z)


def simulate_primary_noise(x: np.ndarray, primary_ir: np.ndarray) -> np.ndarray:
    """The simulated noise arriving at the ear with NO cancellation --
    stands in for what a real error mic would record with the
    controller switched off. Used only to build/evaluate calibration
    data; never called from `runtime.py`."""
    n = len(x)
    P = len(primary_ir)
    d_hist = np.zeros(P)
    d = np.zeros(n, dtype=np.float32)
    xn_nl = nonlinear_distortion(x)
    for i in range(n):
        d_hist = np.concatenate(([xn_nl[i]], d_hist[:-1]))
        d[i] = np.dot(primary_ir, d_hist)
    return d
