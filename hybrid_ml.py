"""
Hybrid CNN-LSTM Conditioned FxNLMS — Active Noise Control for a
Defence-Vehicle Crew Cabin
=================================================================

WHAT THIS IS
------------
An Active Noise Control (ANC) system that cancels the low-frequency
engine, road/track and wind noise reaching a vehicle crew's ears --
the same class of technology used in noise-cancelling headphones and
automotive road-noise-cancellation systems, applied to a defence
vehicle cabin. The goal is crew comfort, reduced fatigue on long
missions, and clearer intercom/radio communication. It is a comfort /
communications-clarity subsystem, not a weapons or targeting system.

THE ALGORITHM ("new algorithm" requested)
------------------------------------------
Classical ANC uses the Filtered-x (Normalized) Least-Mean-Squares
algorithm -- FxLMS / FxNLMS -- to adapt a control filter in real time.
FxLMS is fast, cheap, and provably stable *for linear noise paths*,
but combustion/exhaust/turbulence effects make real vehicle noise
paths mildly NONLINEAR, which puts a hard ceiling on how much a purely
linear adaptive filter can cancel.

This script adds a small deep network in front of the classical
filter:

    reference mic  ->  [ CNN kernels ]  ->  [ LSTM ]  ->  x_hat(n), mu(n)
                                                                |
                                                                v
                                                     [ FxNLMS control filter ]
                                                                |
                                                                v
                                              cabin loudspeaker -> error mic

  * CNN kernels  : a small stack of learnable 1-D convolution kernels
                   acts like a learnable, nonlinear analysis filter
                   bank over a short window of the raw reference
                   signal -- picking out local spectro-temporal shape
                   (harmonic edges, transients) the way a hand-designed
                   filter bank would, but learned from data.
  * LSTM         : integrates those CNN features over the window to
                   track the *slowly*-varying regime (engine RPM, road
                   speed, terrain) that determines how strongly
                   nonlinear the noise path currently is.
  * Two heads    : (1) x_hat(n) -- a nonlinearly-conditioned version of
                   the reference sample that better predicts the true
                   (nonlinear) noise reaching the error mic than the
                   raw reference does; (2) mu_scale(n) -- a *bounded*
                   (0.5x-1.5x) multiplier on the adaptive filter's step
                   size, a learned generalisation of classical
                   variable-step-size LMS (bigger steps in transients,
                   smaller steps at steady state).
  * FxNLMS       : the classical, well-understood, provably-behaved
                   adaptive filter still does the actual real-time
                   cancellation. The deep network only ever produces
                   two *bounded* auxiliary numbers that feed it -- it
                   can never itself destabilise the loop, which matters
                   for a system a vehicle crew relies on.

Deep net is trained OFFLINE/in simulation (as you would from recorded
calibration runs), then frozen and run purely as a fast forward pass
at deployment; only the classical filter weights keep adapting online.

WHAT THIS SCRIPT DOES
----------------------
1. Simulates a synthetic defence-vehicle cabin noise environment
   (engine harmonics with an RPM ramp, road rumble, wind) and a mildly
   nonlinear acoustic noise path (soft saturation, representing
   combustion/exhaust/turbulence nonlinearity).
2. Implements the classical FxNLMS adaptive filter core.
3. Implements the CNN+LSTM hybrid conditioning network in PyTorch.
4. Trains it (supervised, offline) to learn the nonlinear
   reference-conditioning + step-size-gating behaviour.
5. Runs three controllers on a held-out test drive and compares
   attenuation: (a) classical fixed-step FxNLMS baseline, (b) hybrid
   with only the learned nonlinear reference, (c) full hybrid with
   both learned outputs.
6. Plots the results and prints an attenuation summary (dB).

REQUIREMENTS
-------------
    pip install numpy torch matplotlib

REAL-WORLD DEPLOYMENT NOTES
-----------------------------
* The secondary path S(z) (loudspeaker -> error mic) is assumed known
  here; in a fielded system it must be measured via offline system
  identification (a calibration noise burst) and re-checked
  periodically, since cabin acoustics change (hatches/doors, crew,
  cargo).
* Any real system needs output amplitude limiting and a hardware
  mute/bypass the crew can always reach.
* This is a comfort/communication-clarity system and should go through
  normal automotive-acoustic validation, like any ANC product, before
  fielding -- nothing here is safety- or weapons-related.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# PART 1 -- Synthetic defence-vehicle acoustic environment
# =====================================================================
def generate_vehicle_noise(n_samples, fs=2000, seed=0):
    """
    Reference-mic signal: engine harmonics (RPM ramps up then down,
    like accelerating and cruising) + broadband road/track rumble +
    wind buffeting. Normalised to unit RMS.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs

    rpm = 900 + 900 * (0.5 - 0.5 * np.cos(2 * np.pi * t / (n_samples / fs)))
    firing_freq = rpm / 60.0 * 2.0
    phase = 2 * np.pi * np.cumsum(firing_freq) / fs

    engine = (1.0 * np.sin(phase)
              + 0.5 * np.sin(2 * phase)
              + 0.25 * np.sin(3 * phase))

    road = rng.normal(0, 0.4, n_samples)
    smooth = np.ones(15) / 15
    road = np.convolve(road, smooth, mode="same")

    wind = 0.15 * rng.normal(0, 1, n_samples)

    x = engine + road + wind
    x = x / (np.std(x) + 1e-8)
    return x.astype(np.float32)


def make_fir(length, decay, seed):
    """A decaying random FIR impulse response, used to stand in for a
    physical acoustic path (primary or secondary)."""
    rng = np.random.default_rng(seed)
    ir = rng.normal(0, 1, length) * np.exp(-decay * np.arange(length))
    ir = ir / np.linalg.norm(ir)
    return ir.astype(np.float32)


def nonlinear_distortion(z, alpha=1.6):
    """Mild soft-saturation nonlinearity applied in the true noise path,
    representing combustion/exhaust/turbulence distortion that a purely
    linear adaptive filter cannot fully model."""
    return np.tanh(alpha * z)


# =====================================================================
# PART 2 -- Classical adaptive filter core: Filtered-x Normalized LMS
# =====================================================================
class FxNLMS:
    """
    The classical, real-time-safe workhorse of practical ANC systems.
    w            : adaptive control filter driving the cabin speaker
    sec_ir       : estimate of the secondary path S(z) (speaker->error
                   mic), obtained offline by system identification
    """
    def __init__(self, filter_len, sec_path_ir, mu=0.1, eps=1e-3):
        self.N = filter_len
        self.w = np.zeros(filter_len, dtype=np.float64)
        self.mu = mu
        self.eps = eps
        self.sec_ir = sec_path_ir.astype(np.float64)
        self.M = len(sec_path_ir)
        self.x_hist = np.zeros(self.N)
        self.xr_hist = np.zeros(self.M)
        self.xf_hist = np.zeros(self.N)

    def step_control(self, x_n):
        """Compute the anti-noise output y(n) using the CURRENT filter,
        before it is updated. x_n is whatever reference sample is being
        fed in (raw x(n), or the network's conditioned x_hat(n))."""
        self.x_hist = np.concatenate(([x_n], self.x_hist[:-1]))
        y_n = np.dot(self.w, self.x_hist)

        self.xr_hist = np.concatenate(([x_n], self.xr_hist[:-1]))
        xf_n = np.dot(self.sec_ir, self.xr_hist)
        self.xf_hist = np.concatenate(([xf_n], self.xf_hist[:-1]))
        return y_n

    def update(self, e_n, mu_scale=1.0):
        """Normalized LMS weight update using the just-measured error
        e(n) and the filtered-reference history. mu_scale is the
        (bounded, 0.5x-1.5x) learned or fixed step-size gate."""
        norm = np.dot(self.xf_hist, self.xf_hist) + self.eps
        step = mu_scale * self.mu / norm
        self.w = self.w + step * e_n * self.xf_hist


# =====================================================================
# PART 3 -- CNN + LSTM hybrid reference-conditioning network
# =====================================================================
class CNNLSTMReferenceNet(nn.Module):
    """
    CNN kernels : learnable local spectro-temporal feature extraction
                  (a learnable analysis filter bank) over a window of
                  the raw reference signal.
    LSTM        : integrates those features across the window to track
                  the slowly-varying operating regime.
    Two heads   : x_hat (nonlinearly-conditioned reference) and
                  mu_scale (bounded adaptive step-size gate).
    """
    def __init__(self, window_size=32, cnn_channels=(8, 16),
                 kernel_sizes=(7, 5), lstm_hidden=24, lstm_layers=1):
        super().__init__()
        assert len(cnn_channels) == len(kernel_sizes)
        layers = []
        in_ch = 1
        for out_ch, k in zip(cnn_channels, kernel_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=lstm_hidden,
                             num_layers=lstm_layers, batch_first=True)

        self.head_x = nn.Linear(lstm_hidden, 1)
        self.head_mu = nn.Linear(lstm_hidden, 1)
        self.window_size = window_size

    def forward(self, x_window):
        # x_window: (batch, 1, window_size)
        f = self.cnn(x_window)             # (batch, C, window_size)
        f = f.transpose(1, 2)              # (batch, window_size, C)  -> LSTM sequence
        out, _ = self.lstm(f)
        last = out[:, -1, :]               # last timestep's hidden state

        x_hat = self.head_x(last).squeeze(-1)
        mu_raw = self.head_mu(last).squeeze(-1)
        mu_scale = 0.5 + torch.sigmoid(mu_raw)   # bounded to [0.5, 1.5] -- can never
                                                  # push the filter outside a known-safe range
        return x_hat, mu_scale


# =====================================================================
# PART 4 -- Supervised training data (offline "calibration run")
# =====================================================================
def build_training_dataset(x, window_size=32, hop=1):
    """
    x_hat target  : the true nonlinear reference nonlinear_distortion(x[t]).
                    In the field this would come from a calibration run
                    where the true noise-path nonlinearity is
                    characterised (e.g. dynamometer / bench testing);
                    here we know it exactly because we defined it.
    mu_scale target: a classical variable-step-size heuristic (larger
                    step when local reference energy is high, i.e.
                    during transients such as RPM changes) -- the net
                    is trained to reproduce this well-understood rule,
                    generalising it using the same window features.
    """
    n = len(x)
    starts = list(range(window_size - 1, n - 1, hop))
    energies = np.array([np.mean(x[t - window_size + 1:t + 1] ** 2) for t in starts])
    global_energy = float(np.mean(energies)) + 1e-8

    X = np.stack([x[t - window_size + 1:t + 1] for t in starts]).astype(np.float32)
    Yx = np.array([nonlinear_distortion(x[t]) for t in starts], dtype=np.float32)
    Ymu = np.clip(0.5 + energies / global_energy, 0.5, 1.5).astype(np.float32)
    return X, Yx, Ymu


def train_reference_net(net, X, Yx, Ymu, epochs=60, batch_size=256, lr=1e-3,
                         device=DEVICE, mu_loss_weight=0.3):
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    X_t = torch.tensor(X, device=device).unsqueeze(1)   # (N, 1, window)
    Yx_t = torch.tensor(Yx, device=device)
    Ymu_t = torch.tensor(Ymu, device=device)
    n = X_t.shape[0]

    loss_history = []
    net.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yxb, ymub = X_t[idx], Yx_t[idx], Ymu_t[idx]

            x_hat, mu_scale = net(xb)
            loss = F.mse_loss(x_hat, yxb) + mu_loss_weight * F.mse_loss(mu_scale, ymub)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        loss_history.append(total_loss / n)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={loss_history[-1]:.5f}")

    return loss_history


# =====================================================================
# PART 5 -- Deployment: precompute net outputs causally, then run the
#           classical adaptive filter sample-by-sample (this is exactly
#           what a real-time system does, just batched here for speed;
#           x_hat(n)/mu(n) only ever look at samples up to n, so batched
#           and streaming execution are functionally identical)
# =====================================================================
def precompute_net_outputs(net, x, window_size, device=DEVICE, batch_size=4096):
    net.eval()
    x_padded = np.concatenate([np.zeros(window_size - 1, dtype=np.float32),
                                x.astype(np.float32)])
    x_t = torch.tensor(x_padded, device=device)
    windows = x_t.unfold(0, window_size, 1)     # (len(x), window_size), causal

    n = windows.shape[0]
    x_hat_all = np.zeros(n, dtype=np.float32)
    mu_all = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            wb = windows[i:i + batch_size].unsqueeze(1)  # (B, 1, window_size)
            xh, mu = net(wb)
            x_hat_all[i:i + batch_size] = xh.cpu().numpy()
            mu_all[i:i + batch_size] = mu.cpu().numpy()
    return x_hat_all, mu_all


def run_anc(x, primary_ir, sec_ir, filter_len, mu_base,
            ref_signal=None, mu_scale_signal=None):
    """
    Runs FxNLMS sample-by-sample against the true (nonlinear) noise
    environment.
      ref_signal      : reference fed into the adaptive filter each
                         sample. Defaults to raw x (classical baseline).
      mu_scale_signal  : per-sample step-size gate. Defaults to 1.0
                         (classical fixed-step baseline).
    """
    n = len(x)
    P = len(primary_ir)
    fxlms = FxNLMS(filter_len, sec_ir, mu=mu_base)

    d_hist = np.zeros(P)
    y_cancel_hist = np.zeros(len(sec_ir))
    d = np.zeros(n)
    e = np.zeros(n)
    y = np.zeros(n)

    if ref_signal is None:
        ref_signal = x
    if mu_scale_signal is None:
        mu_scale_signal = np.ones(n, dtype=np.float32)

    for i in range(n):
        # true, physically nonlinear noise arriving at the error mic
        xn_nl = nonlinear_distortion(x[i])
        d_hist = np.concatenate(([xn_nl], d_hist[:-1]))
        d_i = np.dot(primary_ir, d_hist)
        d[i] = d_i

        y_i = fxlms.step_control(ref_signal[i])
        y[i] = y_i

        y_cancel_hist = np.concatenate(([y_i], y_cancel_hist[:-1]))
        y_cancel = np.dot(sec_ir, y_cancel_hist)
        e_i = d_i - y_cancel
        e[i] = e_i

        fxlms.update(e_i, mu_scale=mu_scale_signal[i])

    return d, e, y


def attenuation_db(d, e, tail_fraction=1 / 3):
    """Steady-state attenuation: RMS(primary noise) vs RMS(residual
    error) over the last tail_fraction of the run."""
    third = int(len(d) * tail_fraction)
    rms_d = np.sqrt(np.mean(d[-third:] ** 2))
    rms_e = np.sqrt(np.mean(e[-third:] ** 2)) + 1e-12
    return 20 * np.log10(rms_d / rms_e)


# =====================================================================
# PART 6 -- Main: train, deploy, compare, plot
# =====================================================================
def main():
    fs = 2000
    filter_len = 48
    mu_base = 0.1
    window_size = 32

    primary_ir = make_fir(48, 0.05, seed=2)   # engine/road source -> ear
    sec_ir = make_fir(32, 0.08, seed=3)       # cabin speaker -> ear

    # ---- 1. "Calibration run" data for offline training -------------
    print("Generating training (calibration) data...")
    x_train = generate_vehicle_noise(15000, fs=fs, seed=10)
    X, Yx, Ymu = build_training_dataset(x_train, window_size=window_size, hop=1)
    print(f"  {X.shape[0]} training windows built.")

    # ---- 2. Train the CNN-LSTM hybrid conditioning network -----------
    print(f"Training CNN-LSTM reference/step-size network on {DEVICE}...")
    net = CNNLSTMReferenceNet(window_size=window_size)
    loss_history = train_reference_net(net, X, Yx, Ymu, epochs=60, device=DEVICE)

    # ---- 3. Held-out test drive --------------------------------------
    print("Running held-out test drive comparison...")
    x_test = generate_vehicle_noise(8000, fs=fs, seed=1)

    # (a) classical fixed-step FxNLMS baseline, raw reference
    d_base, e_base, _ = run_anc(x_test, primary_ir, sec_ir, filter_len, mu_base)
    atten_base = attenuation_db(d_base, e_base)

    # (b) + (c) hybrid: precompute the network's causal outputs once
    x_hat_all, mu_all = precompute_net_outputs(net, x_test, window_size)

    d_ref_only, e_ref_only, _ = run_anc(
        x_test, primary_ir, sec_ir, filter_len, mu_base,
        ref_signal=x_hat_all, mu_scale_signal=np.ones_like(mu_all))
    atten_ref_only = attenuation_db(d_ref_only, e_ref_only)

    d_full, e_full, _ = run_anc(
        x_test, primary_ir, sec_ir, filter_len, mu_base,
        ref_signal=x_hat_all, mu_scale_signal=mu_all)
    atten_full = attenuation_db(d_full, e_full)

    print("\n=== Steady-state attenuation (dB, higher is better) ===")
    print(f"  (a) Classical fixed-step FxNLMS (raw reference)         : {atten_base:6.2f} dB")
    print(f"  (b) + learned nonlinear reference (CNN-LSTM x_hat)      : {atten_ref_only:6.2f} dB")
    print(f"  (c) + learned step-size gate too (full hybrid)          : {atten_full:6.2f} dB")

    # ---- 4. Plots -----------------------------------------------------
    t = np.arange(len(x_test)) / fs
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))

    axes[0].plot(loss_history)
    axes[0].set_title("Offline training loss (CNN-LSTM reference/step-size network)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")

    axes[1].plot(t, d_base, label="primary noise at ear d(n)", alpha=0.7)
    axes[1].plot(t, e_base, label="residual error e(n) -- classical FxNLMS", alpha=0.8)
    axes[1].set_title(f"(a) Classical fixed-step FxNLMS  |  {atten_base:.1f} dB attenuation")
    axes[1].set_xlabel("time (s)"); axes[1].legend(loc="upper right")

    axes[2].plot(t, d_full, label="primary noise at ear d(n)", alpha=0.7)
    axes[2].plot(t, e_full, label="residual error e(n) -- full hybrid", alpha=0.8)
    axes[2].set_title(f"(c) CNN-LSTM + FxNLMS hybrid  |  {atten_full:.1f} dB attenuation")
    axes[2].set_xlabel("time (s)"); axes[2].legend(loc="upper right")

    axes[3].plot(t, mu_all)
    axes[3].set_title("Learned step-size gate mu_scale(n) (bounded 0.5x-1.5x)")
    axes[3].set_xlabel("time (s)"); axes[3].set_ylabel("mu_scale")
    axes[3].set_ylim(0.4, 1.6)

    plt.tight_layout()
    out_path = "/home/jathnielj/Documents/ANC +ML/anc_results.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved result plots to {out_path}")

    torch.save(net.state_dict(), "/home/jathnielj/Documents/ANC +ML/cnn_lstm_reference_net.pth")
    print("Saved trained network weights to cnn_lstm_reference_net.pth")


if __name__ == "__main__":
    main()