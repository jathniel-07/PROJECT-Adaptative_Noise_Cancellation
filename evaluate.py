"""
Held-out evaluation of a calibrated checkpoint: compares the classical
fixed-step baseline, the online-adapting hybrid controller, and the
ACTUAL deployed configuration (frozen weights, reference mic only) --
then produces attenuation numbers and plots. Uses a SIMULATED test
drive by default (no hardware needed); pass a WAV pair to evaluate
against a real recorded test run instead.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audio_io import load_wav_mono
from .checkpoint import load_checkpoint
from .dsp import FxNLMS
from .loop import attenuation_db, run_offline
from .simulate import generate_vehicle_noise, make_fir, simulate_primary_noise
from .training import precompute_net_outputs

LOG = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
    test_seconds: float = 4.0,
    seed: int = 1,
    ref_wav: Optional[Path] = None,
    error_wav: Optional[Path] = None,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    net, fxnlms_frozen, cfg = load_checkpoint(checkpoint_path, device=device)

    if ref_wav is not None and error_wav is not None:
        x_test = load_wav_mono(ref_wav, cfg.audio.sample_rate)
        d_test = load_wav_mono(error_wav, cfg.audio.sample_rate)
        n = min(len(x_test), len(d_test))
        x_test, d_test = x_test[:n], d_test[:n]
        LOG.info("Evaluating against %.1fs of real recorded test audio", n / cfg.audio.sample_rate)
    else:
        test_samples = max(int(test_seconds * cfg.audio.sample_rate), cfg.net.window_size + 10)
        x_test = generate_vehicle_noise(test_samples, fs=cfg.audio.sample_rate, seed=seed)
        primary_ir = make_fir(cfg.filt.filter_len, 0.05, seed=2)
        d_test = simulate_primary_noise(x_test, primary_ir)
        LOG.info("Evaluating against %.1fs of simulated test-drive audio", test_seconds)

    sec_ir = fxnlms_frozen.sec_ir

    # (a) classical fixed-step FxNLMS baseline, raw reference, adapting online
    d_base, e_base, _, _ = run_offline(
        d_test, sec_ir, cfg.filt.filter_len, cfg.filt.mu_base, ref_signal=x_test)
    atten_base = attenuation_db(d_base, e_base)

    # precompute the frozen net's causal outputs once
    x_hat_all, mu_all = precompute_net_outputs(net, x_test, cfg.net.window_size, device=device)

    # (b) hybrid reference only, still adapting online
    d_ref_only, e_ref_only, _, _ = run_offline(
        d_test, sec_ir, cfg.filt.filter_len, cfg.filt.mu_base,
        ref_signal=x_hat_all, mu_scale_signal=np.ones_like(mu_all))
    atten_ref_only = attenuation_db(d_ref_only, e_ref_only)

    # (c) full hybrid, still adapting online (upper bound if an error mic were kept)
    d_full, e_full, _, _ = run_offline(
        d_test, sec_ir, cfg.filt.filter_len, cfg.filt.mu_base,
        ref_signal=x_hat_all, mu_scale_signal=mu_all)
    atten_full = attenuation_db(d_full, e_full)

    # (d) the ACTUAL deployed configuration: checkpoint's frozen filter
    # weights, no online adaptation at all -- reference mic in,
    # anti-noise out, exactly matching `runtime.py` / `run` mode.
    deployed = FxNLMS(cfg.filt.filter_len, sec_ir, mu=cfg.filt.mu_base, eps=cfg.filt.eps)
    deployed.w = fxnlms_frozen.w.copy()
    y_cancel_hist = np.zeros(len(sec_ir))
    e_deploy = np.zeros(len(x_test), dtype=np.float32)
    for i in range(len(x_test)):
        y_i = deployed.step_control(float(x_hat_all[i]))
        y_cancel_hist = np.concatenate(([y_i], y_cancel_hist[:-1]))
        e_deploy[i] = d_test[i] - np.dot(sec_ir, y_cancel_hist)
    atten_deploy = attenuation_db(d_test, e_deploy)

    results: Dict[str, float] = {
        "classical_baseline": atten_base,
        "hybrid_reference_only_online_adapt": atten_ref_only,
        "hybrid_full_online_adapt": atten_full,
        "deployed_frozen_reference_mic_only": atten_deploy,
    }
    for name, val in results.items():
        LOG.info("%-38s %6.2f dB", name, val)

    t = np.arange(len(x_test)) / cfg.audio.sample_rate
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    axes[0].plot(t, d_base, label="noise at ear d(n)", alpha=0.7)
    axes[0].plot(t, e_base, label="residual e(n) — classical FxNLMS", alpha=0.8)
    axes[0].set_title(f"(a) Classical fixed-step FxNLMS  |  {atten_base:.1f} dB attenuation")
    axes[0].set_xlabel("time (s)"); axes[0].legend(loc="upper right")

    axes[1].plot(t, d_test, label="noise at ear d(n)", alpha=0.7)
    axes[1].plot(t, e_deploy, label="residual e(n) — DEPLOYED (frozen, ref-mic only)", alpha=0.8)
    axes[1].set_title(f"(d) Deployed hybrid controller, reference mic only  |  {atten_deploy:.1f} dB")
    axes[1].set_xlabel("time (s)"); axes[1].legend(loc="upper right")

    axes[2].plot(t, mu_all)
    axes[2].set_title("Learned step-size gate mu_scale(n) — offline calibration use only (bounded 0.5x-1.5x)")
    axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("mu_scale")
    axes[2].set_ylim(0.4, 1.6)

    plt.tight_layout()
    plot_path = output_dir / "evaluation_results.png"
    plt.savefig(plot_path, dpi=130)
    plt.close(fig)
    LOG.info("Saved evaluation plots to %s", plot_path)

    return results
