"""Reward function for RL: audio in -> score in [0, 1], plus why the take is good/bad and what to improve."""
import json
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import SR, MODELS, resolve
from model import NetScorer, clip_windows, clip_features

REFERENCE = "reference.json"
ASPECTS = [
    dict(key="tone_clean", feature=0, higher_is_better=True, weight=0.50, name="tone cleanliness (HNR)",
         advice="cleaner tone: less scratch / noise in the sound"),
    dict(key="tone_worst_part", feature=1, higher_is_better=True, weight=0.53, name="cleanliness of the worst part",
         advice="one part of the note is scratchy: keep the tone clean through the whole note"),
    dict(key="volume_steady", feature=2, higher_is_better=False, weight=0.47, name="volume steadiness",
         advice="volume goes up and down: keep bow speed and pressure steady"),
    dict(key="ending", feature=4, higher_is_better=True, weight=0.29, name="ending holds up",
         advice="the end of the note fades or collapses: keep the bow moving to the end"),
    dict(key="brightness_steady", feature=6, higher_is_better=False, weight=0.28, name="tone colour steadiness",
         advice="tone colour changes during the note: keep the contact point steady"),
    dict(key="noise", feature=7, higher_is_better=False, weight=0.25, name="noisiness",
         advice="sound is noisy / airy: more even bow contact"),
]


class Reward:
    """Load the reward model once, then call reward(), explain() or compare() on waveforms."""

    MIN_SECONDS = 0.1
    SILENCE_DBFS = -65.0

    def __init__(self, model="fast", model_dir=None, tau=1.0, threads=4):
        model_dir = model_dir or MODELS
        ref = json.load(open(os.path.join(model_dir, REFERENCE)))
        if model not in ref["scores"]:
            raise KeyError(f"no reference scores for '{model}' in {REFERENCE}; run reward/build_reference.py --model {model}")
        self.net = NetScorer(resolve(model), model_dir, threads)
        self.ref = np.asarray(ref["scores"][model], np.float64)
        self.ref_feat = np.asarray(ref["features"], np.float64)
        self.tau = tau

    def _prepare(self, y, sr):
        """Clean any input (int/float, stereo, any sample rate) -> (mono float32 at 22.05 kHz or None, status)."""
        y = np.asarray(y)
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
        y = y.astype(np.float32)
        if y.ndim == 2:
            y = y.mean(axis=0 if y.shape[0] <= 8 else 1)
        y = y.reshape(-1)
        if y.size == 0:
            return None, "empty"
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if sr != SR:
            f = Fraction(SR, int(sr)).limit_denominator(1000)
            y = resample_poly(y, f.numerator, f.denominator).astype(np.float32)
        if len(y) < int(self.MIN_SECONDS * SR):
            return None, "too short"
        y = y - y.mean()
        if 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12) < self.SILENCE_DBFS:
            return None, "silent"
        return y, "ok"

    def _to_reward(self, raw):
        """Raw model score -> chance of beating a typical good reference take (0..1)."""
        return float(np.clip(np.mean(1.0 / (1.0 + np.exp(-(raw - self.ref) / self.tau))), 0.0, 1.0))

    def _window_scores(self, y):
        """Raw score of every 0.5 s window of a prepared clip."""
        return np.asarray(self.net.score_windows(clip_windows(y)), np.float64)

    def score(self, y, sr=SR):
        """Raw model score (only differences between takes mean anything); None for unusable audio."""
        y, status = self._prepare(y, sr)
        return None if y is None else float(self._window_scores(y).mean())

    def reward(self, y, sr=SR):
        """Reward in [0, 1] (0.5 = as good as a typical good take); 0.0 for empty/short/silent audio; never raises."""
        try:
            y, status = self._prepare(y, sr)
            return 0.0 if y is None else self._to_reward(self._window_scores(y).mean())
        except Exception:
            return 0.0

    def _aspects(self, y):
        """Each aspect's value and the share of good reference takes it beats."""
        f = clip_features(y)
        out = []
        for a in ASPECTS:
            v, ref = float(f[a["feature"]]), self.ref_feat[:, a["feature"]]
            pct = float(np.mean(ref < v) if a["higher_is_better"] else np.mean(ref > v))
            status = "ok" if pct >= 0.25 else ("weak" if pct >= 0.10 else "problem")
            out.append(dict(aspect=a["key"], name=a["name"], value=round(v, 4), better_than_good_takes=round(pct, 3),
                            status=status, advice=a["advice"] if status != "ok" else "",
                            impact=round(a["weight"] * max(0.0, 0.5 - pct), 4)))
        return out

    def explain(self, y, sr=SR):
        """Reward plus why: aspect diagnosis vs good takes, weakest part of the note, and what to improve."""
        y, status = self._prepare(y, sr)
        if y is None:
            return dict(status=status, reward=0.0)
        ws = self._window_scores(y)
        raw = float(ws.mean())
        parts = None
        if len(ws) >= 3:
            names = ["start", "middle", "end"]
            parts = {n: round(self._to_reward(g.mean()), 3) for n, g in zip(names, np.array_split(ws, 3))}
        aspects = sorted(self._aspects(y), key=lambda d: -d["impact"])
        todo = [d for d in aspects if d["status"] != "ok"]
        r = self._to_reward(raw)
        weakest = min(parts, key=parts.get) if parts else "whole note (too short to split)"
        if todo:
            summary = f"reward {r:.2f}; weakest part: {weakest}; main issue: {todo[0]['name']} -> {todo[0]['advice']}"
        else:
            summary = f"reward {r:.2f}; weakest part: {weakest}; all aspects are within the range of good takes"
        return dict(status="ok", reward=round(r, 4), raw=round(raw, 4), parts=parts, weakest_part=weakest,
                    aspects=aspects, improve=[d["advice"] for d in todo], summary=summary)

    def compare(self, y_a, y_b, sr=SR, sr_b=None):
        """Which of two takes is better, how sure the model is, and the aspects where the winner is better."""
        ea, eb = self.explain(y_a, sr), self.explain(y_b, sr_b or sr)
        if ea["status"] != "ok" or eb["status"] != "ok":
            return dict(status=f"a: {ea['status']}, b: {eb['status']}", better=None)
        p_a = float(1.0 / (1.0 + np.exp(-(ea["raw"] - eb["raw"]))))
        better = "a" if p_a >= 0.5 else "b"
        win, lose = (ea, eb) if better == "a" else (eb, ea)
        lose_map = {d["aspect"]: d for d in lose["aspects"]}
        w = {a["key"]: a["weight"] for a in ASPECTS}
        diffs = []
        for d in win["aspects"]:
            gap = d["better_than_good_takes"] - lose_map[d["aspect"]]["better_than_good_takes"]
            if gap > 0.10:
                diffs.append((w[d["aspect"]] * gap, d["name"], round(gap, 3)))
        reasons = [dict(aspect=n, gap=g) for _, n, g in sorted(diffs, reverse=True)[:3]]
        return dict(status="ok", better=better, p_a_better=round(p_a, 3), reward_a=ea["reward"], reward_b=eb["reward"],
                    why=reasons, summary=f"take {better} is better (p={max(p_a, 1 - p_a):.2f})"
                    + (f", mainly: {', '.join(r['aspect'] for r in reasons)}" if reasons else ""))


if __name__ == "__main__":
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="one file = explain it, two files = compare them")
    ap.add_argument("--model", default="fast", help="fast (1 network) or ensemble12 (12 networks)")
    a = ap.parse_args()
    R = Reward(a.model)
    clips = [sf.read(p, dtype="float32") for p in a.audio]
    if len(clips) == 1:
        print(json.dumps(R.explain(clips[0][0], clips[0][1]), indent=1))
    else:
        print(json.dumps(R.compare(clips[0][0], clips[1][0], clips[0][1], clips[1][1]), indent=1))
