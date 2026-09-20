"""
Classical adaptive filter core: Filtered-x Normalized LMS (FxNLMS).

This is the real-time-safe workhorse that actually drives the cabin
loudspeaker. The CNN-LSTM network (see network.py) only ever supplies
two *bounded* auxiliary numbers into this filter -- it can never
itself destabilise the control loop, which matters for a system a
vehicle crew relies on.
"""
from __future__ import annotations

import numpy as np


class FxNLMS:
    """
    w      : adaptive (or, once frozen, fixed) control filter driving
             the cabin loudspeaker.
    sec_ir : estimate of the secondary path S(z) (loudspeaker -> error
             mic), obtained offline via system identification (a
             calibration noise burst), re-checked periodically since
             cabin acoustics change (hatches/doors, crew, cargo).
    """

    def __init__(self, filter_len: int, sec_path_ir: np.ndarray,
                 mu: float = 0.1, eps: float = 1e-3) -> None:
        self.N = filter_len
        self.w = np.zeros(filter_len, dtype=np.float64)
        self.mu = mu
        self.eps = eps
        self.sec_ir = np.asarray(sec_path_ir, dtype=np.float64)
        self.M = len(self.sec_ir)
        self.x_hist = np.zeros(self.N)
        self.xr_hist = np.zeros(self.M)
        self.xf_hist = np.zeros(self.N)

    def step_control(self, x_n: float) -> float:
        """Anti-noise output y(n) from the CURRENT filter (before any
        update this step). x_n is the reference sample fed in --
        raw x(n) for the classical baseline, or the network's
        conditioned x_hat(n) for the hybrid controller.

        This is the ONLY method called at field-deployment time;
        `update` is offline/calibration-only once the filter is
        frozen (see runtime.py)."""
        self.x_hist = np.concatenate(([x_n], self.x_hist[:-1]))
        y_n = float(np.dot(self.w, self.x_hist))

        self.xr_hist = np.concatenate(([x_n], self.xr_hist[:-1]))
        xf_n = np.dot(self.sec_ir, self.xr_hist)
        self.xf_hist = np.concatenate(([xf_n], self.xf_hist[:-1]))
        return y_n

    def update(self, e_n: float, mu_scale: float = 1.0) -> None:
        """Normalized-LMS weight update from the just-measured error
        e(n). Requires an error-microphone reading, so in this
        prototype it is ONLY ever called during offline/bench
        calibration (see calibrate.py) -- never in `run` (field)
        mode, which has no error mic at all.

        mu_scale is the (bounded, 0.5x-1.5x) learned or fixed
        step-size gate."""
        norm = np.dot(self.xf_hist, self.xf_hist) + self.eps
        step = mu_scale * self.mu / norm
        self.w = self.w + step * e_n * self.xf_hist

    def reset_state(self) -> None:
        """Clears the internal sample-history buffers (NOT the
        learned weights `w`) -- e.g. between independent evaluation
        runs on the same filter."""
        self.x_hist[:] = 0
        self.xr_hist[:] = 0
        self.xf_hist[:] = 0
