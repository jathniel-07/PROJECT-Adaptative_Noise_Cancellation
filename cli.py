"""
Command-line entry point.

    anc-prototype calibrate  [--source simulate|wav] [...]
    anc-prototype evaluate   --checkpoint PATH [...]
    anc-prototype run        --checkpoint PATH [--simulate]

(or, without installing: `python -m anc_prototype ...` from the
directory containing this package.)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

from .calibrate import run_calibration
from .config import SystemConfig
from .evaluate import run_evaluation
from .runtime import run_field_loop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anc_prototype",
        description="Hybrid CNN-LSTM + FxNLMS ANC prototype (reference-mic-only field runtime).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"),
                    help="where checkpoints/plots are written (default: ./outputs)")
    sub = p.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="offline: train the network, adapt the filter, save a checkpoint")
    cal.add_argument("--source", choices=["simulate", "wav"], default="simulate",
                      help="'simulate': synthetic data, no hardware needed. "
                           "'wav': real bench recordings (needs --ref-wav/--error-wav)")
    cal.add_argument("--ref-wav", type=Path, default=None, help="bench reference-mic recording (source=wav)")
    cal.add_argument("--error-wav", type=Path, default=None,
                      help="bench error-mic recording, ANC hardware OFF (source=wav)")
    cal.add_argument("--secondary-ir", type=Path, default=None,
                      help=".npy of the measured secondary-path (speaker->error mic) impulse response")
    cal.add_argument("--epochs", type=int, default=None, help="override TrainingConfig.epochs")

    ev = sub.add_parser("evaluate", help="held-out evaluation of a checkpoint (simulated or WAV test drive)")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--ref-wav", type=Path, default=None, help="real recorded test-drive reference mic")
    ev.add_argument("--error-wav", type=Path, default=None, help="real recorded test-drive error mic")
    ev.add_argument("--test-seconds", type=float, default=4.0, help="length of the simulated test drive")

    rn = sub.add_parser("run", help="field mode: reference mic in, anti-noise out. No error mic, no adaptation.")
    rn.add_argument("--checkpoint", type=Path, required=True)
    rn.add_argument("--simulate", action="store_true",
                     help="use a synthetic reference mic instead of real audio hardware")
    rn.add_argument("--simulate-seconds", type=float, default=10.0)
    rn.add_argument("--start-muted", action="store_true",
                     help="start with output muted — recommended for first hardware bring-up")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    log = logging.getLogger("anc_prototype")

    if args.command == "calibrate":
        cfg = SystemConfig()
        if args.epochs is not None:
            cfg.train.epochs = args.epochs
        try:
            run_calibration(
                cfg, args.output_dir, DEVICE,
                source=args.source, ref_wav=args.ref_wav, error_wav=args.error_wav,
                secondary_ir_path=args.secondary_ir,
            )
        except Exception:
            log.exception("Calibration failed")
            return 1

    elif args.command == "evaluate":
        try:
            results = run_evaluation(
                args.checkpoint, args.output_dir, DEVICE,
                test_seconds=args.test_seconds, ref_wav=args.ref_wav, error_wav=args.error_wav,
            )
        except Exception:
            log.exception("Evaluation failed")
            return 1
        print("\n=== Attenuation summary (dB, higher is better) ===")
        for name, val in results.items():
            print(f"  {name:38s}: {val:6.2f} dB")

    elif args.command == "run":
        try:
            run_field_loop(
                args.checkpoint, DEVICE,
                simulate=args.simulate, simulate_seconds=args.simulate_seconds,
                start_muted=args.start_muted,
            )
        except Exception:
            log.exception("Field run failed")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
