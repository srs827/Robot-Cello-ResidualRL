"""Cause test: change one thing in held-out good takes (noise, volume wobble, fading end, tone-colour wobble) and measure the score drop."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "reward"))
from common import SR, MANIFEST, MODELS, OUTPUTS, load_clip
from model import clip_features
from splits import test_clips
from reward import Reward
import synth

LEVELS = [1, 2, 3, 4]


def add_noise(y, level, rng):
    """White noise at SNR 30 / 20 / 10 / 0 dB."""
    snr = [30, 20, 10, 0][level - 1]
    n = rng.standard_normal(len(y)).astype(np.float32)
    n *= np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(n ** 2)) + 1e-9) * 10 ** (-snr / 20)
    return (y + n).astype(np.float32)


def volume_wobble(y, level, rng):
    """Random (non-periodic) loudness changes of depth 15 / 30 / 50 / 75 %."""
    depth = [0.15, 0.30, 0.50, 0.75][level - 1]
    e = synth._lp_noise(len(y), 4.0, rng)
    e = e / (np.abs(e).max() + 1e-9)
    return (y * np.clip(1 + depth * e, 0.05, None)).astype(np.float32)


def fading_end(y, level, rng):
    """The last 40 % of the note fades to 70 / 50 / 30 / 10 % volume."""
    g = [0.7, 0.5, 0.3, 0.1][level - 1]
    env = np.ones(len(y), np.float32)
    k = int(0.4 * len(y))
    env[-k:] = np.linspace(1.0, g, k)
    return (y * env).astype(np.float32)


def colour_wobble(y, level, rng):
    """Tone colour swings between brighter and darker 3 times per second (EQ tilt 3 / 6 / 9 / 12 dB per octave)."""
    k = [3.0, 6.0, 9.0, 12.0][level - 1]
    m = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * np.arange(len(y)) / SR + rng.uniform(0, 6.28))
    return (m * synth.tilt(y, k) + (1 - m) * synth.tilt(y, -k)).astype(np.float32)


def quieter(y, level, rng):
    """Control: only quieter (-3 / -6 / -9 / -12 dB); should not change the score."""
    return synth.gain(y, -3.0 * level)


def static_eq(y, level, rng):
    """Control: fixed EQ tilt of +1 / +2 / +3 / +4 dB per octave; should barely change the score."""
    return synth.tilt(y, 1.0 * level)


TESTS = [("noise", add_noise, "tone_clean", 0), ("volume_wobble", volume_wobble, "volume_steady", 2),
         ("fading_end", fading_end, "ending", 4), ("colour_wobble", colour_wobble, "brightness_steady", 6),
         ("control_quieter", quieter, None, None), ("control_eq", static_eq, None, None)]


def main():
    """Run every change at 4 strengths on held-out good takes; save mean rewards, drops and aspect flags."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble12")
    ap.add_argument("--model_dir", default=MODELS)
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "explain"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    held = test_clips(m)
    ids = sorted(m[m.clip_id.isin(held) & (m.label_bin == 1) & (m.dur >= 0.5)].clip_id)
    R = Reward(a.model, a.model_dir)
    clips = [R._prepare(load_clip(c), SR)[0] for c in ids]
    base = np.array([R.reward(y) for y in clips])
    print(f"{len(ids)} held-out good takes, original mean reward {base.mean():.3f}", flush=True)
    res = dict(model=a.model, n_clips=len(ids), original_mean_reward=round(float(base.mean()), 4), tests={})
    for name, fn, aspect, fidx in TESTS:
        rows = []
        for lv in LEVELS:
            rng = np.random.default_rng(lv)
            r, flagged, fchange = [], [], []
            for y in clips:
                z = fn(y, lv, rng)
                if aspect:
                    e = R.explain(z)
                    r.append(e["reward"])
                    flagged.append(next(d["status"] != "ok" for d in e["aspects"] if d["aspect"] == aspect))
                    fchange.append(float(clip_features(z)[fidx] - clip_features(y)[fidx]))
                else:
                    r.append(R.reward(z))
            r = np.array(r)
            row = dict(level=lv, mean_reward=round(float(r.mean()), 4), mean_drop=round(float((base - r).mean()), 4),
                       share_lower=round(float((r < base - 0.01).mean()), 3))
            if aspect:
                row.update(target_aspect=aspect, share_flagged=round(float(np.mean(flagged)), 3),
                           mean_feature_change=round(float(np.mean(fchange)), 4))
            rows.append(row)
            print(name, row, flush=True)
        res["tests"][name] = dict(description=fn.__doc__, levels=rows)
    json.dump(res, open(os.path.join(a.out, "one_feature_test.json"), "w"), indent=1)
    print("saved", os.path.join(a.out, "one_feature_test.json"))


if __name__ == "__main__":
    main()
