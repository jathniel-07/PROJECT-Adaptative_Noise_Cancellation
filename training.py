"""
Offline supervised training of the CNN-LSTM reference/step-size
network, from a reference-mic calibration recording (simulated or
real).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import TrainingConfig
from .network import CNNLSTMReferenceNet
from .simulate import nonlinear_distortion

LOG = logging.getLogger(__name__)


def build_training_dataset(
    x: np.ndarray, window_size: int, hop: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x_hat target   : the true nonlinear reference,
                     nonlinear_distortion(x[t]). In the field this
                     would come from a calibration run where the true
                     noise-path nonlinearity is characterised (e.g.
                     dynamometer / bench testing); here (simulation)
                     it is known exactly by construction.
    mu_scale target: a classical variable-step-size heuristic (larger
                     step when local reference energy is high, i.e.
                     during transients such as RPM changes) -- the net
                     is trained to reproduce this well-understood rule,
                     generalising it using the same window features.
    """
    n = len(x)
    if n <= window_size:
        raise ValueError(f"signal length ({n}) must exceed window_size ({window_size})")
    starts = list(range(window_size - 1, n - 1, hop))
    energies = np.array([np.mean(x[t - window_size + 1:t + 1] ** 2) for t in starts])
    global_energy = float(np.mean(energies)) + 1e-8

    X = np.stack([x[t - window_size + 1:t + 1] for t in starts]).astype(np.float32)
    Yx = np.array([nonlinear_distortion(x[t]) for t in starts], dtype=np.float32)
    Ymu = np.clip(0.5 + energies / global_energy, 0.5, 1.5).astype(np.float32)
    return X, Yx, Ymu


def train_reference_net(
    net: CNNLSTMReferenceNet,
    X: np.ndarray, Yx: np.ndarray, Ymu: np.ndarray,
    train_cfg: TrainingConfig,
    device: str,
) -> List[float]:
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=train_cfg.lr)

    X_t = torch.tensor(X, device=device).unsqueeze(1)   # (N, 1, window)
    Yx_t = torch.tensor(Yx, device=device)
    Ymu_t = torch.tensor(Ymu, device=device)
    n = X_t.shape[0]

    loss_history: List[float] = []
    net.train()
    for epoch in range(train_cfg.epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, train_cfg.batch_size):
            idx = perm[i:i + train_cfg.batch_size]
            xb, yxb, ymub = X_t[idx], Yx_t[idx], Ymu_t[idx]

            x_hat, mu_scale = net(xb)
            loss = F.mse_loss(x_hat, yxb) + train_cfg.mu_loss_weight * F.mse_loss(mu_scale, ymub)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        loss_history.append(total_loss / n)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            LOG.info("epoch %3d/%d  loss=%.5f", epoch + 1, train_cfg.epochs, loss_history[-1])

    return loss_history


@torch.no_grad()
def precompute_net_outputs(
    net: CNNLSTMReferenceNet, x: np.ndarray, window_size: int, device: str, batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched causal forward pass over a whole recording: x_hat(n)
    and mu(n) only ever look at samples up to n, so this is
    functionally identical to the sample-by-sample streaming
    computation `runtime.py` does online -- just batched here for
    speed during offline calibration/evaluation."""
    net.eval()
    x_padded = np.concatenate([np.zeros(window_size - 1, dtype=np.float32), x.astype(np.float32)])
    x_t = torch.tensor(x_padded, device=device)
    windows = x_t.unfold(0, window_size, 1)     # (len(x), window_size), causal

    n = windows.shape[0]
    x_hat_all = np.zeros(n, dtype=np.float32)
    mu_all = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        wb = windows[i:i + batch_size].unsqueeze(1)  # (B, 1, window_size)
        xh, mu = net(wb)
        x_hat_all[i:i + batch_size] = xh.cpu().numpy()
        mu_all[i:i + batch_size] = mu.cpu().numpy()
    return x_hat_all, mu_all
