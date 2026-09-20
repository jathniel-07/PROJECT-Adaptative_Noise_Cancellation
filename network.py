"""
CNN + LSTM hybrid reference-conditioning network.

Trained OFFLINE on recorded/simulated reference-mic calibration data,
then frozen and run purely as a fast forward pass at deployment. Its
only input, here and everywhere it is called from elsewhere in this
package, is a short window of raw reference-microphone samples --
nothing else.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import NetworkConfig


class CNNLSTMReferenceNet(nn.Module):
    """
    CNN kernels : learnable local spectro-temporal feature extraction
                  (a learnable analysis filter bank) over a window of
                  the raw reference signal.
    LSTM        : integrates those features across the window to track
                  the slowly-varying operating regime (engine RPM,
                  road speed, terrain) -- inferred purely from the
                  acoustic reference signal, not from a separate
                  sensor feed.
    Two heads   : x_hat (nonlinearly-conditioned reference) and
                  mu_scale (bounded step-size gate). mu_scale is only
                  ever used during offline/bench filter adaptation
                  (see calibrate.py) -- the field runtime has no error
                  signal to adapt with, so it ignores this head.
    """

    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        if len(cfg.cnn_channels) != len(cfg.kernel_sizes):
            raise ValueError("cnn_channels and kernel_sizes must have the same length")

        layers = []
        in_ch = 1
        for out_ch, k in zip(cfg.cnn_channels, cfg.kernel_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )
        self.head_x = nn.Linear(cfg.lstm_hidden, 1)
        self.head_mu = nn.Linear(cfg.lstm_hidden, 1)
        self.window_size = cfg.window_size

    def forward(self, x_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_window: (batch, 1, window_size) raw reference-mic samples."""
        f = self.cnn(x_window)             # (batch, C, window_size)
        f = f.transpose(1, 2)              # (batch, window_size, C) -> LSTM sequence
        out, _ = self.lstm(f)
        last = out[:, -1, :]               # last timestep's hidden state

        x_hat = self.head_x(last).squeeze(-1)
        mu_raw = self.head_mu(last).squeeze(-1)
        mu_scale = 0.5 + torch.sigmoid(mu_raw)   # bounded [0.5, 1.5] -- can never
                                                  # push the filter outside a known-safe range
        return x_hat, mu_scale
