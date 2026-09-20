"""
Audio input/output for the prototype.

Design rule enforced throughout this package: the ONLY input the
deployed controller reads is the reference microphone
(``ReferenceMicrophone.read``). There is no code path anywhere in this
package that reads any other vehicle sensor (tachometer, GPS speed,
IMU, CAN-bus telemetry, ...) at field-deployment time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
    _SOUNDDEVICE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host machine
    # sounddevice can fail at IMPORT time (not just "not installed") if the
    # native PortAudio library isn't present, so we catch broadly here.
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_IMPORT_ERROR = exc

from .config import AudioConfig, SafetyConfig

LOG = logging.getLogger(__name__)


class ReferenceMicrophone:
    """Abstract source of reference-microphone samples. This is the
    single audio input type the field controller ever reads from."""

    def read(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "ReferenceMicrophone":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SimulatedReferenceMicrophone(ReferenceMicrophone):
    """Replays a pre-computed array as if it were streaming from
    hardware. Used for development, offline calibration without
    hardware, and demoing the prototype loop on a laptop."""

    def __init__(self, signal: np.ndarray) -> None:
        self._signal = np.asarray(signal, dtype=np.float32)
        self._pos = 0

    def read(self, n: int) -> np.ndarray:
        end = min(self._pos + n, len(self._signal))
        block = self._signal[self._pos:end]
        if len(block) < n:
            block = np.concatenate([block, np.zeros(n - len(block), dtype=np.float32)])
        self._pos = end
        return block

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self._signal)


class LiveReferenceMicrophone(ReferenceMicrophone):
    """Streams mono blocks from ONE physical reference-microphone
    input channel via sounddevice/PortAudio.

    Requires ``pip install sounddevice`` plus a working PortAudio
    backend on the host (Linux: ``apt install libportaudio2``).
    """

    def __init__(self, cfg: AudioConfig) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice / PortAudio is not available on this machine "
                f"({_SOUNDDEVICE_IMPORT_ERROR!r}). Install it for live audio, "
                "or pass --simulate to run the prototype loop against "
                "synthetic data instead."
            )
        self._cfg = cfg
        n_channels = cfg.reference_channel + 1
        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=n_channels,
            device=cfg.reference_device,
            dtype="float32",
            blocksize=cfg.block_size,
        )
        self._stream.start()
        LOG.info(
            "Live reference microphone open: device=%r channel=%d rate=%d Hz",
            cfg.reference_device, cfg.reference_channel, cfg.sample_rate,
        )

    def read(self, n: int) -> np.ndarray:
        data, overflowed = self._stream.read(n)
        if overflowed:
            LOG.warning("Reference microphone input overflow — samples were dropped")
        return np.ascontiguousarray(data[:, self._cfg.reference_channel], dtype=np.float32)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class LoudspeakerOutput:
    """The sole output of the controller: the anti-noise waveform sent
    to the cabin loudspeaker. Every write passes through a hard
    amplitude limiter and a mute switch the operator/crew can set at
    any time via ``SafetyConfig.mute`` -- required for any real system
    people rely on."""

    def __init__(self, cfg: AudioConfig, safety: SafetyConfig, simulate: bool = False) -> None:
        self._safety = safety
        self._output_channel = cfg.output_channel
        self._simulate = simulate or sd is None
        self._history: Optional[List[np.ndarray]] = [] if self._simulate else None
        if not self._simulate:
            self._stream = sd.OutputStream(
                samplerate=cfg.sample_rate,
                channels=cfg.output_channel + 1,
                device=cfg.output_device,
                dtype="float32",
                blocksize=cfg.block_size,
            )
            self._stream.start()

    def write(self, block: np.ndarray) -> np.ndarray:
        """Limits/mutes ``block`` and sends it to the loudspeaker.
        Returns the actual (post-safety) block that was sent, e.g. for
        logging or offline plots."""
        limit = self._safety.max_output_amplitude
        safe_block = np.clip(block, -limit, limit).astype(np.float32)
        if self._safety.mute:
            safe_block = np.zeros_like(safe_block)

        if self._simulate:
            assert self._history is not None
            self._history.append(safe_block.copy())
        else:
            out = np.zeros((len(safe_block), self._stream.channels), dtype=np.float32)
            out[:, self._output_channel] = safe_block
            self._stream.write(out)
        return safe_block

    def simulated_output(self) -> np.ndarray:
        if not self._simulate:
            raise RuntimeError("simulated_output() is only valid when simulate=True")
        return np.concatenate(self._history) if self._history else np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        if not self._simulate:
            self._stream.stop()
            self._stream.close()

    def __enter__(self) -> "LoudspeakerOutput":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_wav_mono(path: Path, expected_rate: int) -> np.ndarray:
    """Loads a calibration/bench-test WAV recording as mono float32 in
    roughly [-1, 1]. Used only for OFFLINE calibration/evaluation,
    never at runtime."""
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    if rate != expected_rate:
        raise ValueError(
            f"{path}: sample rate {rate} Hz does not match the configured "
            f"AudioConfig.sample_rate ({expected_rate} Hz). Resample the "
            f"recording or update the config."
        )
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.astype(np.float32)
