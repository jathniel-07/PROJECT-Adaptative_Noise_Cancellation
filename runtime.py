"""
Field-deployment runtime.

Reads ONLY the reference microphone and drives the cabin loudspeaker.
No error microphone, no other vehicle sensor, and no online adaptation
-- the network and filter were both frozen offline during calibration
(see calibrate.py). This is the module that answers "what does the
fielded prototype actually run".
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .audio_io import (
    LiveReferenceMicrophone,
    LoudspeakerOutput,
    ReferenceMicrophone,
    SimulatedReferenceMicrophone,
)
from .checkpoint import load_checkpoint
from .dsp import FxNLMS
from .network import CNNLSTMReferenceNet
from .simulate import generate_vehicle_noise

LOG = logging.getLogger(__name__)


class HybridANCController:
    """The frozen hybrid controller. `process_block` is the entire
    runtime data path: reference-mic samples in, anti-noise samples
    out. Nothing else feeds it."""

    def __init__(self, net: CNNLSTMReferenceNet, fxnlms: FxNLMS, window_size: int, device: str) -> None:
        self.net = net.to(device).eval()
        self.fxnlms = fxnlms
        self.window_size = window_size
        self.device = device
        self._tail = np.zeros(max(window_size - 1, 0), dtype=np.float32)

    @torch.no_grad()
    def process_block(self, ref_block: np.ndarray) -> np.ndarray:
        """ref_block: 1-D array of raw reference-mic samples.
        Returns the anti-noise output block (same length) to send to
        the loudspeaker."""
        padded = np.concatenate([self._tail, ref_block]).astype(np.float32)
        if self.window_size > 1:
            self._tail = padded[-(self.window_size - 1):]
        t = torch.from_numpy(padded).to(self.device)
        windows = t.unfold(0, self.window_size, 1).unsqueeze(1)   # (len(ref_block), 1, window)
        x_hat, _mu_scale_unused = self.net(windows)   # mu_scale is offline/calibration-only
        x_hat = x_hat.cpu().numpy()

        y_block = np.empty(len(ref_block), dtype=np.float32)
        for i in range(len(ref_block)):
            y_block[i] = self.fxnlms.step_control(float(x_hat[i]))
        return y_block


def run_field_loop(
    checkpoint_path: Path,
    device: str,
    simulate: bool = False,
    simulate_seconds: float = 10.0,
    start_muted: bool = False,
    block_size_override: Optional[int] = None,
    max_blocks: Optional[int] = None,
) -> np.ndarray:
    """Runs the frozen controller against either a live reference mic
    or (with simulate=True) a synthetic stand-in, until the source is
    exhausted or the operator interrupts (Ctrl+C). Returns the
    anti-noise waveform produced when simulate=True (useful for demos
    and tests); on real hardware the output is streamed straight to
    the loudspeaker and an empty array is returned."""
    net, fxnlms, cfg = load_checkpoint(checkpoint_path, device=device)
    if start_muted:
        cfg.safety.mute = True
    block_size = block_size_override or cfg.audio.block_size

    controller = HybridANCController(net, fxnlms, cfg.net.window_size, device=device)

    ref_source: ReferenceMicrophone
    if simulate:
        n_samples = int(simulate_seconds * cfg.audio.sample_rate)
        ref_source = SimulatedReferenceMicrophone(
            generate_vehicle_noise(n_samples, fs=cfg.audio.sample_rate, seed=int(time.time()) % 10_000)
        )
        LOG.info("Running field loop against a SIMULATED reference mic (%.1fs).", simulate_seconds)
    else:
        ref_source = LiveReferenceMicrophone(cfg.audio)
        LOG.info("Running field loop against the LIVE reference microphone. Ctrl+C to stop.")

    speaker = LoudspeakerOutput(cfg.audio, cfg.safety, simulate=simulate)

    blocks_done = 0
    try:
        while True:
            if isinstance(ref_source, SimulatedReferenceMicrophone) and ref_source.exhausted:
                break
            if max_blocks is not None and blocks_done >= max_blocks:
                break
            ref_block = ref_source.read(block_size)
            y_block = controller.process_block(ref_block)
            speaker.write(y_block)
            blocks_done += 1
    except KeyboardInterrupt:
        LOG.info("Stopping (Ctrl+C) — muting output.")
        cfg.safety.mute = True
        speaker.write(np.zeros(block_size, dtype=np.float32))
    finally:
        ref_source.close()
        out = speaker.simulated_output() if simulate else np.zeros(0, dtype=np.float32)
        speaker.close()

    return out
