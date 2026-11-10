"""Test the reward on a virtual cello: map reward over bow force x speed vs bowed-string physics, then let an optimizer learn to bow using only the reward."""
import argparse
import json
import os
import sys

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reward"))
from reward import Reward
from virtual_cello import play, set_eq, SR

FORCES = np.logspace(-2.5, 1, 20)
SPEEDS = np.logspace(-2, 0, 12)
LIMITS = np.array([[np.log(FORCES[0]), np.log(FORCES[-1])], [np.log(SPEEDS[0]), np.log(SPEEDS[-1])], [0.06, 0.20]])
STARTS = {"too_heavy": (4.0, 0.02, 0.12), "too_light": (0.004, 0.5, 0.12), "near_bridge_heavy": (8.0, 0.1, 0.07),
          "far_light": (0.006, 0.05, 0.19), "middle": (0.05, 0.05, 0.15)}
MELODY = [146.83, 146.83, 220.0, 220.0, 220.0, 220.0, 220.0]


def _spectrum(y):
    """Average power spectrum in dB on 1/6-octave bands from 60 Hz to 10 kHz, normalised to total power."""
    f, p = welch(y, SR, nperseg=min(2048, len(y)), nfft=8192)
    p = p / (p.sum() + 1e-12)
    edges = 60 * 2 ** (np.arange(0, 41) / 6)
    return edges[:-1] * 2 ** (1 / 12), np.array([10 * np.log10(p[(f >= a) & (f < b)].mean() + 1e-12) for a, b in zip(edges[:-1], edges[1:])])


def calibrate(pieces_json, audio_dir):
    """Fit the virtual cello's body colour so clean simulated notes match the average spectrum of real robot notes."""
    P = json.load(open(pieces_json))["pieces"]
    real = []
    for k, v in P.items():
        y, _ = librosa.load(os.path.join(audio_dir, k + ".mp3"), sr=SR)
        for n in v["notes"]:
            seg = y[int(n["start"] * SR):int(n["end"] * SR)]
            if len(seg) >= int(0.3 * SR) and n["note"] in ("G2", "D3", "A3"):
                real.append((n["note"], _spectrum(seg)[1]))
    counts = {p: sum(1 for q, _ in real if q == p) for p in ("G2", "D3", "A3")}
    freqs, real_db = _spectrum(np.random.default_rng(0).standard_normal(SR))[0], np.mean([s for _, s in real], axis=0)
    set_eq(None, None)
    sim = []
    for p, c in counts.items():
        for f, v in [(1.11, 0.1), (0.534, 0.046), (2.31, 0.215), (4.806, 0.464)]:
            sim += [_spectrum(play(f, v, f0=librosa.note_to_hz(p))[0])[1]] * max(1, c // 4)
    gain = np.clip(real_db - np.mean(sim, axis=0), -30, 30)
    gain = np.convolve(np.pad(gain, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    set_eq(freqs, gain)
    return dict(freqs=freqs.round(1).tolist(), gain_db=gain.round(2).tolist(), real_notes=counts)


def grid_map(R):
    """Reward and Helmholtz motion on the force x speed grid, and the reward per bowing region."""
    H, W = np.zeros((len(FORCES), len(SPEEDS))), np.zeros((len(FORCES), len(SPEEDS)))
    for i, f in enumerate(FORCES):
        for j, v in enumerate(SPEEDS):
            y, H[i, j] = play(f, v)
            W[i, j] = R.reward(y, SR)
    reg = {"too_light": [], "helmholtz": [], "too_heavy": []}
    for j in range(len(SPEEDS)):
        rows = np.flatnonzero(H[:, j] > 0.8)
        for i in range(len(FORCES)):
            if len(rows) and H[i, j] > 0.8:
                reg["helmholtz"].append(W[i, j])
            elif len(rows) and i < rows.min():
                reg["too_light"].append(W[i, j])
            elif len(rows) and i > rows.max():
                reg["too_heavy"].append(W[i, j])
    lab = (H > 0.8).ravel()
    return dict(forces=FORCES.round(4).tolist(), speeds=SPEEDS.round(4).tolist(), helmholtz=H.round(3).tolist(),
                reward=W.round(4).tolist(), auc_helmholtz=round(float(roc_auc_score(lab, W.ravel())), 3),
                regions={k: dict(n=len(v), mean_reward=round(float(np.mean(v)), 3)) for k, v in reg.items()})


BETAS = np.logspace(np.log10(0.05), np.log10(0.25), 12)
PITCHES = {"G2": 98.0, "D3": 146.83, "A3": 220.0}


def _fit_slope(betas, forces):
    """Slope of log force vs log contact point (physics predicts -1 for the upper limit, -2 for the lower)."""
    ok = [(b, f) for b, f in zip(betas, forces) if f is not None]
    if len(ok) < 3:
        return None
    x, y = np.log([b for b, _ in ok]), np.log([f for _, f in ok])
    return round(float(np.polyfit(x, y, 1)[0]), 2)


def schelleng(R, speed=0.1):
    """Classic Schelleng diagram per string: contact point x bow force at fixed speed, physics vs reward."""
    out = {}
    for name, f0 in PITCHES.items():
        H, W = np.zeros((len(FORCES), len(BETAS))), np.zeros((len(FORCES), len(BETAS)))
        for i, f in enumerate(FORCES):
            for j, b in enumerate(BETAS):
                y, H[i, j] = play(f, speed, b, f0=f0)
                W[i, j] = R.reward(y, SR)
        fmax_phys, fmin_phys, fmax_rew = [], [], []
        for j in range(len(BETAS)):
            rows = np.flatnonzero(H[:, j] > 0.8)
            fmax_phys.append(float(FORCES[rows.max()]) if len(rows) else None)
            fmin_phys.append(float(FORCES[rows.min()]) if len(rows) else None)
            drop = [i for i in range(len(FORCES)) if W[i, j] < 0.4 and all(W[k, j] < 0.4 for k in range(i, len(FORCES)))]
            fmax_rew.append(float(FORCES[drop[0] - 1]) if drop and drop[0] > 0 else None)
        pairs = [(p, r) for p, r in zip(fmax_phys, fmax_rew) if p is not None and r is not None]
        lab = (H > 0.8).ravel()
        out[name] = dict(betas=BETAS.round(4).tolist(), helmholtz=H.round(3).tolist(), reward=W.round(4).tolist(),
                         auc_helmholtz=round(float(roc_auc_score(lab, W.ravel())), 3) if 0 < lab.sum() < len(lab) else None,
                         fmax_physics=fmax_phys, fmin_physics=fmin_phys, fmax_reward=fmax_rew,
                         slope_fmax_physics=_fit_slope(BETAS, fmax_phys), slope_fmin_physics=_fit_slope(BETAS, fmin_phys),
                         slope_fmax_reward=_fit_slope(BETAS, fmax_rew),
                         fmax_ratio_reward_vs_physics=round(float(np.median([r / p for p, r in pairs])), 2) if pairs else None)
        print("schelleng", name, {k: v for k, v in out[name].items() if k.startswith(("auc", "slope", "fmax_ratio"))}, flush=True)
    return out


def to_params(x):
    """Optimizer vector (log force, log speed, contact point) -> bowing parameters."""
    x = np.clip(x, LIMITS[:, 0], LIMITS[:, 1])
    return float(np.exp(x[0])), float(np.exp(x[1])), float(x[2])


def learn_to_bow(R, start, iters=15, pop=12, elite=4, seed=0):
    """Cross-entropy method: sample bowings, keep the ones with the highest reward, repeat (a tiny RL loop)."""
    rng = np.random.default_rng(seed)
    mean = np.array([np.log(start[0]), np.log(start[1]), start[2]])
    sd = np.array([1.0, 1.0, 0.04])
    hist = []
    for it in range(iters + 1):
        f, v, b = to_params(mean)
        y, h = play(f, v, b)
        hist.append(dict(iter=it, force=round(f, 4), speed=round(v, 4), contact=round(b, 3), reward=round(R.reward(y, SR), 4),
                         helmholtz=round(h, 3)))
        if it == iters:
            break
        xs = mean + sd * rng.standard_normal((pop, 3))
        rs = np.array([R.reward(play(*to_params(x), seed=k)[0], SR) for k, x in enumerate(xs)])
        best = xs[np.argsort(-rs)[:elite]]
        mean, sd = best.mean(axis=0), np.maximum(best.std(axis=0), [0.1, 0.1, 0.005])
    return hist


def melody(f, v, b):
    """Opening of Twinkle (D D A A A A A) bowed with one setting."""
    return np.concatenate([np.concatenate([play(f, v, b, f0=p, dur=0.6, seed=k)[0], np.zeros(int(0.05 * SR), np.float32)])
                           for k, p in enumerate(MELODY)])


def main():
    """Run the map and the learning runs, save json and before/after audio."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fast")
    ap.add_argument("--model_dir", default=None, help="folder with the model (default: checkpoints/)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "virtual_cello"))
    ap.add_argument("--audio_out", default=None, help="folder for before/after wav files (optional)")
    ap.add_argument("--calibrate", default="yes", help="yes = match the body colour to the real robot recordings first")
    ap.add_argument("--pieces", default=os.path.join(HERE, "..", "outputs", "pieces", "pieces.json"))
    ap.add_argument("--robot_audio", default=os.path.join(HERE, "..", "..", "lol", "Robot"))
    a = ap.parse_args()
    R = Reward(a.model, model_dir=a.model_dir)
    res = dict(model=a.model, calibration=calibrate(a.pieces, a.robot_audio) if a.calibrate == "yes" else None)
    res.update(map=grid_map(R), schelleng=schelleng(R), runs={})
    print("map AUC", res["map"]["auc_helmholtz"], res["map"]["regions"], flush=True)
    for name, st in STARTS.items():
        h = learn_to_bow(R, st)
        res["runs"][name] = h
        print(name, "start", h[0], "\n   end", h[-1], flush=True)
        if a.audio_out:
            os.makedirs(a.audio_out, exist_ok=True)
            for tag, row in [("before", h[0]), ("after", h[-1])]:
                sf.write(os.path.join(a.audio_out, f"{name}_{tag}_r{row['reward']:.2f}.wav"),
                         melody(row["force"], row["speed"], row["contact"]), SR)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"virtual_cello{'' if a.calibrate == 'yes' else '_uncalibrated'}.json"), "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
