"""
Sample-by-sample FxNLMS control loop, shared by offline calibration
(filter adaptation) and evaluation. Not used at field-deployment time:
`runtime.py` has its own, simpler loop with no error signal at all.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .dsp import FxNLMS


def run_offline(
    d: np.ndarray,
    sec_ir: np.ndarray,
    filter_len: int,
    mu_base: float,
    ref_signal: np.ndarray,
    mu_scale_signal: Optional[np.ndarray] = None,
    fxnlms: Optional[FxNLMS] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FxNLMS]:
    """
    Runs the classical loop against a known noise signal `d` (what an
    error mic would read with the controller off) -- either a
    simulated signal or a real bench recording. Used for offline
    training-time filter adaptation and for evaluation only; field
    `run` mode has no error signal and never calls this.

      d               : noise-at-ear with the controller off.
      ref_signal      : reference fed into the adaptive filter each
                         sample (raw x, or the network's x_hat).
      mu_scale_signal : per-sample step-size gate (defaults to 1.0,
                         i.e. classical fixed-step behaviour).
      fxnlms          : reuse an existing filter (e.g. to continue
                         adapting) instead of starting from zero
                         weights.

    Returns (d, e, y, fxnlms) -- the filter is returned so its
    (possibly further-adapted) weights can be frozen into a
    checkpoint.
    """
    n = len(d)
    if len(ref_signal) != n:
        raise ValueError(f"ref_signal length ({len(ref_signal)}) must match d length ({n})")
    if fxnlms is None:
        fxnlms = FxNLMS(filter_len, sec_ir, mu=mu_base)
    if mu_scale_signal is None:
        mu_scale_signal = np.ones(n, dtype=np.float32)
    elif len(mu_scale_signal) != n:
        raise ValueError("mu_scale_signal must match d in length")

    y_cancel_hist = np.zeros(len(fxnlms.sec_ir))
    e = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i in range(n):
        y_i = fxnlms.step_control(float(ref_signal[i]))
        y[i] = y_i

        y_cancel_hist = np.concatenate(([y_i], y_cancel_hist[:-1]))
        y_cancel = np.dot(fxnlms.sec_ir, y_cancel_hist)
        e_i = float(d[i]) - y_cancel
        e[i] = e_i

        fxnlms.update(e_i, mu_scale=float(mu_scale_signal[i]))

    return d, e, y, fxnlms


def attenuation_db(d: np.ndarray, e: np.ndarray, tail_fraction: float = 1 / 3) -> float:
    """Steady-state attenuation: RMS(noise) vs RMS(residual error)
    over the last `tail_fraction` of the run, in dB (higher = better)."""
    third = max(int(len(d) * tail_fraction), 1)
    rms_d = np.sqrt(np.mean(np.asarray(d[-third:], dtype=np.float64) ** 2))
    rms_e = np.sqrt(np.mean(np.asarray(e[-third:], dtype=np.float64) ** 2)) + 1e-12
    return float(20 * np.log10(rms_d / rms_e))
