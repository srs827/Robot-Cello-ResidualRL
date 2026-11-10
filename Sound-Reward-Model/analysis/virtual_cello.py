"""Virtual cello: a physical bowed-string model (waveguide string + friction bow) controlled by bow force, speed and contact point."""
import numpy as np
from numba import njit
from scipy.signal import lfilter, firwin2

SR = 22050
MU_S, MU_D, V0 = 0.8, 0.3, 0.1
EQ = {"taps": None}
BODY = [(98, 8, 0.6), (196, 10, 1.0), (285, 12, 0.8), (440, 14, 0.6), (650, 12, 0.5), (1100, 8, 0.4), (2300, 5, 0.3)]


@njit(cache=True)
def _bow_string(n, n_bridge, n_nut, force, speed, loss, lp, seed):
    """Waveguide string bowed at one point; returns bridge signal and the number of slip phases per sample."""
    np.random.seed(seed)
    bridge = np.zeros(n_bridge)
    nut = np.zeros(n_nut)
    out = np.zeros(n)
    slip = np.zeros(n, np.int8)
    ib = 0
    inn = 0
    lp_state = 0.0
    sliding = False
    ramp = int(0.06 * SR)
    for t in range(n):
        g = min(1.0, t / ramp)
        vb = speed * g * (1.0 + 0.005 * np.random.randn())
        fb = force * (1.0 + 0.005 * np.random.randn())
        lp_state = (1 - lp) * nut[inn] + lp * lp_state
        b_in = -0.999 * bridge[ib]
        a_in = -loss * lp_state
        vh = a_in + b_in
        u = vh - vb
        dv = 0.0
        if sliding or abs(u) > fb * MU_S:
            s = 1.0 if u > 0 else -1.0
            a = abs(u) - fb * MU_D
            c = fb * (MU_S - MU_D)
            p = V0 - a
            disc = p * p + 4 * V0 * (a - c)
            root = (-p + np.sqrt(disc)) / 2 if disc >= 0 else -1.0
            if root > 0:
                dv = s * root
                sliding = True
                slip[t] = 1
            else:
                sliding = False
        v = vb + dv
        bridge[ib] = a_in + (v - vh)
        nut[inn] = b_in + (v - vh)
        out[t] = bridge[ib]
        ib = (ib + 1) % n_bridge
        inn = (inn + 1) % n_nut
    return out, slip


def body(y):
    """Cello body: a few wooden resonances plus a direct path."""
    out = 0.3 * y
    for f, q, g in BODY:
        r = np.exp(-np.pi * f / (q * SR))
        w = 2 * np.pi * f / SR
        out = out + g * (1 - r) * lfilter([1, 0, -1], [1, -2 * r * np.cos(w), r * r], y)
    return out


def set_eq(freqs, gains_db):
    """Extra body colour (FIR filter from a gain curve in dB), e.g. fitted to real robot recordings; None = off."""
    if freqs is None:
        EQ["taps"] = None
        return
    f = np.concatenate([[0], np.asarray(freqs, float), [SR / 2]])
    g = 10 ** (np.concatenate([[gains_db[0]], np.asarray(gains_db, float), [gains_db[-1]]]) / 20)
    EQ["taps"] = firwin2(1025, f / (SR / 2), g)


def helmholtz(slip, period):
    """Share of slip-to-slip intervals equal to one string period (one clean slip per period = Helmholtz motion)."""
    s = slip.astype(int)
    idx = np.flatnonzero(s)
    if len(idx) < 3:
        return 0.0
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1)
    starts = np.array([r[0] for r in runs if r[-1] - r[0] >= 2])
    if len(starts) < 3:
        return 0.0
    iv = np.diff(starts)
    return float(np.mean(np.abs(iv - period) < 0.08 * period))


def play(force, speed, beta=0.12, f0=146.83, dur=1.5, seed=0):
    """One bowed note: (audio at 22.05 kHz, Helmholtz score = share of periods with exactly one slip)."""
    n_loop = SR / f0
    nb = max(2, int(round(beta * n_loop)))
    nn = max(2, int(round(n_loop)) - nb)
    raw, slip = _bow_string(int(dur * SR), nb, nn, float(force), float(speed), 0.97, 0.25, seed)
    helm = helmholtz(slip[int(0.3 * SR):], n_loop)
    y = body(raw)
    if EQ["taps"] is not None:
        y = np.convolve(y, EQ["taps"], mode="same")
    y = y - y.mean()
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    return y.astype(np.float32), helm
