"""
Configuration dataclasses for the Hybrid CNN-LSTM + FxNLMS ANC prototype.

Centralising configuration here -- instead of scattering constants
through the code, as the original research script did -- is what lets
a single saved checkpoint fully describe how it was built and how it
must be run. That matters once this leaves a single developer's
machine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class AudioConfig:
    """Everything to do with the physical audio interface.

    ``reference_channel`` is the ONLY audio input the system reads
    once deployed (``run`` mode). No other vehicle sensor (tachometer,
    GPS speed, IMU, CAN-bus telemetry, ...) is ever read by this
    system -- by design, not by omission.
    """
    sample_rate: int = 2000
    block_size: int = 256
    reference_device: Optional[str] = None
    reference_channel: int = 0
    output_device: Optional[str] = None
    output_channel: int = 0


@dataclass
class FilterConfig:
    """Classical FxNLMS control-filter geometry and base step size."""
    filter_len: int = 48
    mu_base: float = 0.1
    eps: float = 1e-3
    secondary_path_len: int = 32


@dataclass
class NetworkConfig:
    """CNN-LSTM reference-conditioning network architecture."""
    window_size: int = 32
    cnn_channels: Tuple[int, ...] = (8, 16)
    kernel_sizes: Tuple[int, ...] = (7, 5)
    lstm_hidden: int = 24
    lstm_layers: int = 1


@dataclass
class TrainingConfig:
    """Offline calibration/training hyperparameters."""
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    mu_loss_weight: float = 0.3
    calibration_samples: int = 15000
    seed: int = 10


@dataclass
class SafetyConfig:
    """Runtime output protection. Every real ANC system needs an
    output amplitude limiter and a mute the operator/crew can always
    reach -- this is that switch."""
    max_output_amplitude: float = 1.0
    mute: bool = False


@dataclass
class SystemConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SystemConfig":
        net_d = dict(d.get("net", {}))
        if "cnn_channels" in net_d:
            net_d["cnn_channels"] = tuple(net_d["cnn_channels"])
        if "kernel_sizes" in net_d:
            net_d["kernel_sizes"] = tuple(net_d["kernel_sizes"])
        return SystemConfig(
            audio=AudioConfig(**d.get("audio", {})),
            filt=FilterConfig(**d.get("filt", {})),
            net=NetworkConfig(**net_d),
            train=TrainingConfig(**d.get("train", {})),
            safety=SafetyConfig(**d.get("safety", {})),
        )
