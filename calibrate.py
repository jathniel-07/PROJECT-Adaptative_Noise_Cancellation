"""
Offline calibration pipeline.

Produces a frozen checkpoint (network + adapted filter weights) that
`runtime.py` deploys using ONLY the reference microphone. Two
calibration data sources are supported:

  source="simulate" : fully synthetic, no hardware needed -- for
                       development and regression testing.
  source="wav"       : real bench recordings. A reference-mic WAV and
                       an (uncontrolled, i.e. ANC OFF) error-mic WAV,
                       recorded simultaneously on the bench where the
                       secondary path was identified. The error mic is
                       used HERE ONLY, offline, during calibration --
                       it is never read again once the checkpoint is
                       deployed in the field.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_io import load_wav_mono
from .checkpoint import save_checkpoint
from .config import SystemConfig
from .loop import run_offline
from .network import CNNLSTMReferenceNet
from .simulate import generate_vehicle_noise, make_fir, simulate_primary_noise
from .training import build_training_dataset, precompute_net_outputs, train_reference_net

LOG = logging.getLogger(__name__)


def run_calibration(
    cfg: SystemConfig,
    output_dir: Path,
    device: str,
    source: str = "simulate",
    ref_wav: Optional[Path] = None,
    error_wav: Optional[Path] = None,
    secondary_ir_path: Optional[Path] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- secondary path S(z): should be measured on the bench -------
    if secondary_ir_path is not None:
        sec_ir = np.load(secondary_ir_path).astype(np.float32)
        LOG.info("Loaded measured secondary-path IR from %s (%d taps)", secondary_ir_path, len(sec_ir))
    else:
        LOG.warning(
            "No --secondary-ir supplied; using a SIMULATED secondary path. "
            "Replace this with a real bench measurement before fielding."
        )
        sec_ir = make_fir(cfg.filt.secondary_path_len, 0.08, seed=3)

    # ---- reference-mic calibration recording -------------------------
    if source == "wav":
        if ref_wav is None or error_wav is None:
            raise ValueError("source='wav' requires both --ref-wav and --error-wav")
        x_cal = load_wav_mono(ref_wav, cfg.audio.sample_rate)
        d_cal = load_wav_mono(error_wav, cfg.audio.sample_rate)
        n = min(len(x_cal), len(d_cal))
        x_cal, d_cal = x_cal[:n], d_cal[:n]
        LOG.info("Loaded %.1fs of real bench calibration audio (reference + error mic)",
                  n / cfg.audio.sample_rate)
    elif source == "simulate":
        LOG.info("Generating synthetic calibration (reference-mic) data...")
        x_cal = generate_vehicle_noise(cfg.train.calibration_samples, fs=cfg.audio.sample_rate,
                                        seed=cfg.train.seed)
        primary_ir = make_fir(cfg.filt.filter_len, 0.05, seed=2)
        d_cal = simulate_primary_noise(x_cal, primary_ir)
    else:
        raise ValueError(f"unknown source {source!r}; expected 'simulate' or 'wav'")

    # ---- 1. train the CNN-LSTM reference/step-size network ----------
    X, Yx, Ymu = build_training_dataset(x_cal, window_size=cfg.net.window_size)
    LOG.info("%d training windows built.", X.shape[0])

    net = CNNLSTMReferenceNet(cfg.net)
    LOG.info("Training CNN-LSTM reference/step-size network on %s...", device)
    loss_history = train_reference_net(net, X, Yx, Ymu, cfg.train, device=device)

    # ---- 2. offline-adapt the classical filter weights ---------------
    # This is the ONLY place in the whole pipeline where `FxNLMS.update`
    # runs against a real/simulated error signal. The resulting weights
    # are frozen into the checkpoint and never adapted again in the
    # field -- the deployed controller (runtime.py) only ever calls
    # `step_control`.
    x_hat_all, mu_all = precompute_net_outputs(net, x_cal, cfg.net.window_size, device=device)
    LOG.info("Offline-adapting the FxNLMS control filter from calibration data...")
    _, _, _, fxnlms = run_offline(
        d_cal, sec_ir, cfg.filt.filter_len, cfg.filt.mu_base,
        ref_signal=x_hat_all, mu_scale_signal=mu_all,
    )

    ckpt_path = output_dir / "hybrid_anc_checkpoint.pt"
    save_checkpoint(ckpt_path, net, fxnlms, cfg)
    LOG.info("Saved calibrated checkpoint to %s", ckpt_path)

    np.save(output_dir / "training_loss.npy", np.array(loss_history))
    return ckpt_path
