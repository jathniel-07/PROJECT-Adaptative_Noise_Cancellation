"""
Checkpoint bundling: a single file fully describing a calibrated
controller (network weights, frozen filter weights, secondary-path IR,
and the config that produced them), so `evaluate`/`run` never depend
on hidden global state or on being run right after `calibrate`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .config import SystemConfig
from .dsp import FxNLMS
from .network import CNNLSTMReferenceNet


def save_checkpoint(path: Path, net: CNNLSTMReferenceNet, fxnlms: FxNLMS, cfg: SystemConfig) -> None:
    payload: Dict[str, Any] = {
        "format_version": 1,
        "net_state_dict": net.state_dict(),
        "filter_w": fxnlms.w,
        "secondary_path_ir": fxnlms.sec_ir,
        "config": cfg.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str) -> Tuple[CNNLSTMReferenceNet, FxNLMS, SystemConfig]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = SystemConfig.from_dict(payload["config"])

    net = CNNLSTMReferenceNet(cfg.net)
    net.load_state_dict(payload["net_state_dict"])
    net.to(device).eval()

    fxnlms = FxNLMS(cfg.filt.filter_len, payload["secondary_path_ir"], mu=cfg.filt.mu_base, eps=cfg.filt.eps)
    fxnlms.w = np.asarray(payload["filter_w"], dtype=np.float64)

    return net, fxnlms, cfg
