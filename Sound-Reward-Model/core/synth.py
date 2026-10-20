"""Synthetic bowing defects (make a take worse) and recording nuisances (should not change the score)."""
import numpy as np
from scipy import signal

SR = 22050
SEEN_DEFECTS = ["scratch", "wobble", "dip", "tremolo", "slow_attack"]
UNSEEN_DEFECTS = ["harsh_attack", "whistle", "beating"]
ALL_DEFECTS = SEEN_DEFECTS + UNSEEN_DEFECTS
ATTACK_DEFECTS = {"slow_attack", "harsh_attack"}
SEV = {1: 0, 2: 1, 3: 2}


def _rms(y):
    """Root-mean-square level."""
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


def _lp_noise(n, rate_hz, rng):
    """Smooth random curve with bandwidth about rate_hz."""
    k = max(2, int(SR / max(rate_hz, 0.5)))
    m = n // k + 3
    x = rng.standard_normal(m)
    t = np.linspace(0, m - 1, n + 1)[:n]
    return np.interp(t, np.arange(m), x).astype(np.float32)


def _bandnoise(n, lo, hi, rng):
    """Band-limited white noise."""
    x = rng.standard_normal(n).astype(np.float32)
    sos = signal.butter(4, [lo, min(hi, SR / 2 - 100)], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def scratch(y, sev, rng):
    """Add bursty high-frequency bow noise."""
    n = len(y)
    snr = [12.0, 6.0, 0.0][SEV[sev]] + rng.uniform(-1.5, 1.5)
    env = np.abs(_lp_noise(n, rng.uniform(15, 50), rng))
    env = env / (env.mean() + 1e-9)
    noise = _bandnoise(n, rng.uniform(1200, 2500), rng.uniform(5000, 9000), rng) * env
    noise *= _rms(y) / (_rms(noise) + 1e-9) * 10 ** (-snr / 20)
    return (y + noise).astype(np.float32)


def wobble(y, sev, rng):
    """Add unsteady pitch."""
    n = len(y)
    t = np.arange(n) / SR
    depth = [8.0, 20.0, 45.0][SEV[sev]] * rng.uniform(0.8, 1.2)
    f = rng.uniform(3, 9)
    cents = depth * (np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) + 0.5 * _lp_noise(n, 6, rng))
    ratio = 2.0 ** (cents / 1200.0)
    tau = np.cumsum(ratio) - ratio[0]
    return np.interp(tau, np.arange(n), y).astype(np.float32)


def dip(y, sev, rng):
    """Add sudden drops in volume."""
    n = len(y)
    dur = n / SR
    depth_db = [6.0, 12.0, 22.0][SEV[sev]] * rng.uniform(0.85, 1.15)
    half = min([0.06, 0.10, 0.15][SEV[sev]] * rng.uniform(0.8, 1.2), 0.22 * dur)
    k = 1 if dur < 0.6 else int(rng.integers(1, 3))
    g = np.ones(n, np.float32)
    for _ in range(k):
        c = rng.uniform(0.25, 0.85) * dur
        w = int(2 * half * SR)
        if w < 8:
            continue
        i0 = max(0, min(n - w, int(c * SR) - w // 2))
        g[i0:i0 + w] *= (1 - (1 - 10 ** (-depth_db / 20)) * signal.windows.hann(w)).astype(np.float32)
    return (y * g).astype(np.float32)


def tremolo(y, sev, rng):
    """Add periodic volume wobble."""
    n = len(y)
    t = np.arange(n) / SR
    m = [0.15, 0.35, 0.6][SEV[sev]] * rng.uniform(0.85, 1.15)
    f = rng.uniform(4, 12)
    return (y * (1 - m / 2 + m / 2 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)))).astype(np.float32)


def slow_attack(y, sev, rng):
    """Make the note start too slowly."""
    L = min(int([0.08, 0.20, 0.45][SEV[sev]] * rng.uniform(0.85, 1.15) * SR), len(y) // 2)
    g = np.ones(len(y), np.float32)
    g[:L] = np.linspace(0.02, 1.0, L) ** 1.5
    return (y * g).astype(np.float32)


def harsh_attack(y, sev, rng):
    """Add a noisy click at the start of the note."""
    n = len(y)
    L = min(int(rng.uniform(0.015, 0.04) * SR), n)
    burst = _bandnoise(L, 800, 8000, rng) * np.exp(-np.arange(L) / (0.35 * L))
    peak = np.max(np.abs(y[: max(L, int(0.05 * SR))])) + 1e-9
    burst *= peak / (np.max(np.abs(burst)) + 1e-9) * 10 ** ([-14.0, -8.0, -2.0][SEV[sev]] / 20)
    out = y.copy()
    out[:L] += burst
    return out.astype(np.float32)


def whistle(y, sev, rng):
    """Add a high whistling tone."""
    n = len(y)
    t = np.arange(n) / SR
    fw = rng.uniform(2500, 6000)
    env = np.abs(signal.hilbert(y))
    env = np.convolve(env, np.ones(200) / 200, mode="same")
    w = np.sin(2 * np.pi * fw * t) * env
    w *= _rms(y) / (_rms(w) + 1e-9) * 10 ** ([-30.0, -22.0, -14.0][SEV[sev]] / 20)
    return (y + w).astype(np.float32)


def beating(y, sev, rng):
    """Mix in a slightly detuned copy (beating)."""
    n = len(y)
    d = rng.uniform(0.004, 0.02) * (1 if sev == 1 else 1.5 if sev == 2 else 2.0)
    y2 = np.interp(np.arange(n) * (1 + d), np.arange(n), y, right=0.0)
    a = [0.35, 0.6, 0.95][SEV[sev]]
    return ((y + a * y2) / (1 + a * 0.5)).astype(np.float32)


DEFECT_FN = dict(scratch=scratch, wobble=wobble, dip=dip, tremolo=tremolo, slow_attack=slow_attack,
                 harsh_attack=harsh_attack, whistle=whistle, beating=beating)


def apply_defect(y, name, sev, rng):
    """Apply one named defect at severity 1-3."""
    return DEFECT_FN[name](y, sev, rng)


def gain(y, db):
    """Change volume by db."""
    return (y * 10 ** (db / 20)).astype(np.float32)


def noise_floor(y, snr_db, rng, pink=True):
    """Add background noise at a given SNR."""
    n = len(y)
    x = rng.standard_normal(n)
    if pink:
        X = np.fft.rfft(x)
        f = np.arange(len(X))
        f[0] = 1
        x = np.fft.irfft(X / np.sqrt(f), n)
    x = x.astype(np.float32)
    x *= _rms(y) / (_rms(x) + 1e-9) * 10 ** (-snr_db / 20)
    return (y + x).astype(np.float32)


def reverb(y, rt60, wet, rng):
    """Add synthetic room reverb."""
    L = int(rt60 * SR * 1.2) + 16
    t = np.arange(L) / SR
    ir = rng.standard_normal(L) * np.exp(-6.9 * t / rt60)
    ir[0] = 0
    wetsig = signal.fftconvolve(y, ir.astype(np.float32))[: len(y)]
    wetsig *= _rms(y) / (_rms(wetsig) + 1e-9)
    return ((1 - wet) * y + wet * wetsig).astype(np.float32)


def tilt(y, db_per_oct):
    """Tilt the spectrum (brighter or darker EQ)."""
    n = len(y)
    X = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = 1
    H = 10 ** (db_per_oct * np.log2(f / 1000.0) / 20)
    return np.fft.irfft(X * H, n).astype(np.float32)


def lowpass(y, fc):
    """Low-pass filter."""
    sos = signal.butter(4, fc, btype="low", fs=SR, output="sos")
    return signal.sosfilt(sos, y).astype(np.float32)


def highpass(y, fc):
    """High-pass filter."""
    sos = signal.butter(2, fc, btype="high", fs=SR, output="sos")
    return signal.sosfilt(sos, y).astype(np.float32)


def random_nuisance(y, rng, strength=1.0):
    """Random chain of recording effects (EQ, filters, reverb, noise, gain)."""
    if rng.random() < 0.6:
        y = tilt(y, rng.uniform(-3, 3) * strength)
    if rng.random() < 0.4:
        y = lowpass(y, rng.uniform(5000, 10000))
    if rng.random() < 0.3:
        y = highpass(y, rng.uniform(40, 120))
    if rng.random() < 0.5:
        y = reverb(y, rng.uniform(0.05, 0.35), rng.uniform(0.05, 0.3) * strength, rng)
    if rng.random() < 0.6:
        y = noise_floor(y, rng.uniform(22, 50), rng)
    return gain(y, rng.uniform(-12, 12))
