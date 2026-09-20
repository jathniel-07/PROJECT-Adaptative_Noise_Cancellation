# Hybrid CNN-LSTM + FxNLMS ANC — Prototype

Active Noise Control (ANC) for a vehicle crew cabin: a small learned
network conditions a classical FxNLMS adaptive filter so it can handle
the mildly nonlinear noise paths a purely linear filter can't. Comfort
/ intercom-clarity subsystem — not a weapons or targeting system.

**Field deployment reads ONLY the reference microphone.** No error
microphone and no other vehicle sensor (tachometer, GPS speed, IMU,
CAN-bus telemetry, ...) is read anywhere in the running system. That's
enforced in code, not just in docs — see [How "reference-mic
only" is enforced](#how-reference-mic-only-is-enforced) below.

## What changed from the original research script

The original `hybrid_ml.py` was a single-file simulation: it trained
the network **and** let the classical filter keep adapting online
using a simulated error microphone, for both the "test drive" and
(implicitly) at deployment. That's normal for a research script, but
it means a fielded copy would need *two* microphones (reference +
error) wired into the vehicle, and would need an internet-of-things'
worth of care to keep the online adaptation loop stable outside
simulation.

This version splits that into two clearly separate phases:

1. **Calibration (offline, lab/bench, `calibrate`)** — trains the
   network and lets the classical filter adapt, using either
   synthetic data or real bench recordings (which *do* need an error
   mic, temporarily, on the bench). The result is frozen into a single
   checkpoint file.
2. **Field deployment (`run`)** — loads that checkpoint and runs a
   pure feed-forward loop: reference mic in, anti-noise out. No error
   mic, no adaptation, no other sensor. Simpler hardware, and a much
   smaller runtime attack/failure surface.

Everything else on top is "prototype hygiene": a typed, tested,
importable package instead of one long script; a config object +
checkpoint instead of hard-coded constants and a hard-coded personal
output path; logging instead of `print`; a CLI with three modes
instead of a single `main()`; and a hard output limiter + mute switch
that's actually wired in.

## Package layout

```
anc_prototype/
  config.py       SystemConfig / AudioConfig / FilterConfig / NetworkConfig / TrainingConfig / SafetyConfig
  dsp.py          FxNLMS — the classical adaptive filter core
  network.py      CNNLSTMReferenceNet — the learned conditioning network
  simulate.py     SIMULATION-ONLY synthetic vehicle noise + nonlinear path
  loop.py         shared offline control loop (calibration + evaluation only)
  training.py     dataset builder + training loop + causal batch inference
  calibrate.py    offline pipeline: train net, adapt filter, save checkpoint
  evaluate.py     held-out attenuation comparison + plots
  audio_io.py     ReferenceMicrophone (live/simulated) + LoudspeakerOutput + WAV loader
  runtime.py      HybridANCController — the actual field/deployment loop
  checkpoint.py   save/load a calibrated controller as one file
  cli.py          `calibrate` / `evaluate` / `run` subcommands
tests/
  test_dsp_and_config.py   fast tests for the non-ML core (no torch/hardware needed)
requirements.txt
pyproject.toml
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .            (installs the `anc-prototype` command)
# or, to skip the live-audio dependency entirely:
pip install numpy torch matplotlib scipy
```

`sounddevice` (for real hardware in `run` mode) needs the native
PortAudio library on the host:

```bash
# Debian/Ubuntu
sudo apt install libportaudio2
```

Everything except real hardware `run` mode works without it —
`calibrate`, `evaluate`, and `run --simulate` all use pure
numpy/torch.

Run the fast test suite any time (no torch/hardware needed):

```bash
python -m pytest tests/ -v
```

## Usage

### 1. Calibrate (offline, produces a checkpoint)

Synthetic, no hardware — good for development and CI:

```bash
python -m anc_prototype calibrate --source simulate --output-dir outputs/
```

From real bench recordings (reference mic + an *uncontrolled*, i.e.
ANC-hardware-off, error mic, recorded simultaneously where the
secondary path was identified):

```bash
python -m anc_prototype calibrate --source wav \
    --ref-wav bench/reference_mic.wav \
    --error-wav bench/error_mic.wav \
    --secondary-ir bench/secondary_path_ir.npy \
    --output-dir outputs/
```

This writes `outputs/hybrid_anc_checkpoint.pt` — a single file with
the trained network, the offline-adapted filter weights, the
secondary-path IR, and the full config used to build them.

### 2. Evaluate (sanity-check a checkpoint before fielding it)

```bash
python -m anc_prototype evaluate --checkpoint outputs/hybrid_anc_checkpoint.pt
```

Prints an attenuation summary in dB and saves
`outputs/evaluation_results.png`, including the attenuation of the
**actual deployed configuration** (frozen weights, reference mic
only) — not just the online-adapting research variants — so what you
evaluate is what you field.

### 3. Run (field mode)

Against real hardware:

```bash
python -m anc_prototype run --checkpoint outputs/hybrid_anc_checkpoint.pt --start-muted
```

`--start-muted` is recommended for first bring-up: confirm signal
levels look sane, then unmute (edit `SafetyConfig.mute`, or wire a
hardware mute switch to it — see below).

Without hardware, to see the loop run end-to-end against synthetic
audio:

```bash
python -m anc_prototype run --checkpoint outputs/hybrid_anc_checkpoint.pt --simulate
```

## Hardware wiring notes

- `AudioConfig.reference_device` / `reference_channel` select which
  physical input device/channel is the reference microphone — set
  these to match your audio interface. `output_device` /
  `output_channel` select the loudspeaker output the same way.
- The secondary path `S(z)` (loudspeaker → error mic) must be measured
  on the bench via system identification (a calibration noise burst)
  and re-checked periodically, since cabin acoustics change
  (hatches/doors, crew, cargo). Pass it to `calibrate` via
  `--secondary-ir`; without it, calibration falls back to a simulated
  secondary path and will warn loudly.
- Because the field loop never adapts online, cabin acoustic drift
  isn't tracked automatically. The intended workflow is periodic
  **depot recalibration**: temporarily reattach an error mic, rerun
  `calibrate --source wav`, redeploy the new checkpoint. How often
  depends on your acoustic validation results — treat this the same
  as any automotive-ANC product's revalidation cadence.

## Safety

- Every sample written to the loudspeaker passes through
  `SafetyConfig.max_output_amplitude` (hard clip) before it reaches
  hardware — see `LoudspeakerOutput.write` in `audio_io.py`.
- `SafetyConfig.mute` forces silence regardless of what the controller
  computes. `run` mutes automatically on Ctrl+C; wire a physical
  switch to the same flag for a crew-reachable hardware mute.
- This is a comfort/communication-clarity system and should go through
  normal automotive-acoustic validation, like any ANC product, before
  fielding — nothing here is safety- or weapons-related.

## How "reference-mic only" is enforced

- `HybridANCController.process_block` (the entire field data path,
  `runtime.py`) takes one argument: a block of reference-mic samples.
  It has no parameter for an error signal or any other input.
- `FxNLMS.update()` — the only method that needs an error reading — is
  called exactly once in the whole package: inside `calibrate.py`,
  offline. `runtime.py` never imports or calls it.
- `ReferenceMicrophone` (live or simulated) is the only audio *source*
  type `runtime.py` knows about; there's no second-channel/sensor
  input path to disable, because there isn't one to begin with.

## Known limitations / next steps

- No online field adaptation (by design — see above); if your
  acoustic environment drifts faster than your depot recalibration
  cadence, attenuation will degrade until the next recalibration.
- The nonlinear-reference training target (`Yx` in `training.py`) is
  exact in simulation because the nonlinearity is defined; a real
  bench characterisation of your specific noise path's nonlinearity
  (e.g. dynamometer testing) is needed before `calibrate --source wav`
  is fully turnkey for the network's targets (the offline *filter*
  adaptation step already supports real recordings as-is).
- `LiveReferenceMicrophone`/`LoudspeakerOutput` are thin
  `sounddevice` wrappers; swap them for your platform's audio driver
  if you're targeting embedded hardware without PortAudio support.
