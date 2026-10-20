"""Benchmark tests (robot/human test AUC, synthetic defect detection) and the reward-hacking test."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import SR, PROC, MANIFEST, load_clip
from splits import local_split, iowa_split, ROBOT_SOURCES
import synth

WIN = int(0.5 * SR)
BENCH_PKL = os.path.join(PROC, "bench.pkl")


def bench_windows(y, n_max=4):
    """Up to n_max evenly spaced 0.5 s windows of a clip."""
    if len(y) <= WIN:
        return [y]
    k = min(n_max, max(1, (len(y) - WIN) // (WIN // 2) + 1))
    return [y[s:s + WIN] for s in np.linspace(0, len(y) - WIN, k).astype(int)]


def _onset_window(y):
    """First 0.5 s of a clip (padded if shorter)."""
    w = y[:WIN]
    return np.pad(w, (0, WIN - len(w)), mode="edge") if len(w) < WIN else w


def build_bench(force=False):
    """Build (once) the fixed test sets: robot test clips, human_A3 clips and clean clips with injected defects."""
    if os.path.exists(BENCH_PKL) and not force:
        return pickle.load(open(BENCH_PKL, "rb"))
    df = pd.read_csv(MANIFEST)
    _, local_te = local_split(df)
    _, iowa_te = iowa_split(df)
    T = {}
    for name, sub in (("local_test", df[df.clip_id.isin(local_te)]), ("human_A3", df[df.source == "human_A3"])):
        T[name] = dict(ids=list(sub.clip_id), windows=[bench_windows(load_clip(c)) for c in sub.clip_id],
                       labels=sub.label_bin.values)
    gs = df[df.source == "gs_cello"]
    io = df[df.clip_id.isin(iowa_te)]
    g_good = gs[(gs.label_bin == 1) & (gs.mic == "neumann")]
    pools = list(io.sample(min(50, len(io)), random_state=1).clip_id) if len(io) else []
    pools += list(g_good.sample(min(50, len(g_good)), random_state=1).clip_id) if len(g_good) else []
    pools += list(df[df.clip_id.isin(local_te) & (df.label_bin == 1)].clip_id)
    pools += list(df[(df.source == "human_A3") & (df.label_bin == 1)].clip_id)
    base = np.stack([_onset_window(load_clip(c)).astype(np.float32) for c in pools])
    var = {}
    for di, d in enumerate(synth.ALL_DEFECTS):
        for s in (1, 2, 3):
            r = np.random.default_rng(1000 * di + s)
            var[(d, s)] = np.stack([synth.apply_defect(b, d, s, r) for b in base])
    T["synth"] = dict(ids=pools, base=base, var=var)
    pickle.dump(T, open(BENCH_PKL, "wb"))
    return T


def _auc(y, s):
    """ROC AUC, NaN if only one class."""
    y, s = np.asarray(y), np.asarray(s)
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def scorecard(scorer):
    """B3 robot-test AUC, B4 human-A3 AUC, B5 defect AUC (seen / unseen defect types)."""
    T = build_bench()
    res = {}
    for name in ("local_test", "human_A3"):
        wins, owner = [], []
        for i, ws in enumerate(T[name]["windows"]):
            wins += ws
            owner += [i] * len(ws)
        out, owner = scorer.score_windows(wins), np.array(owner)
        res[name] = _auc(T[name]["labels"], [out[owner == i].mean() for i in range(len(T[name]["ids"]))])
    S = T["synth"]
    base = scorer.score_windows(list(S["base"]))
    aucs = {}
    for d in synth.ALL_DEFECTS:
        sev3 = scorer.score_windows(list(S["var"][(d, 3)]))
        aucs[d] = _auc(np.r_[np.ones(len(base)), np.zeros(len(base))], np.r_[base, sev3])
    return {"B3_robot_test_AUC": res["local_test"], "B4_human_A3_AUC": res["human_A3"],
            "B5_seen": float(np.mean([aucs[d] for d in synth.SEEN_DEFECTS])),
            "B5_unseen": float(np.mean([aucs[d] for d in synth.UNSEEN_DEFECTS]))}


def trick_list(y, rng):
    """Cheap audio tricks that do not make the playing better (EQ, reverb, added tone, limiter, ...)."""
    n = len(y)
    t = np.arange(n) / SR
    out = {"identity": y}
    for db in (-9, 9):
        out[f"gain{db:+d}dB"] = synth.gain(y, db)
    for fc in (3000, 6000):
        out[f"lowpass{fc}"] = synth.lowpass(y, fc)
    for tl in (-6, 6):
        out[f"tilt{tl:+d}"] = synth.tilt(y, tl)
    for wet in (0.3, 0.6):
        out[f"reverb{wet}"] = synth.reverb(y, 0.3, wet, rng)
    for lvl in (0.1, 0.3):
        out[f"add_tone{lvl}"] = (y + lvl * np.max(np.abs(y)) * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    out["limiter"] = (np.tanh(4 * y / (np.max(np.abs(y)) + 1e-9)) * np.max(np.abs(y))).astype(np.float32)
    g = np.ones(n, np.float32)
    k = int(0.05 * SR)
    g[:k] = np.linspace(0, 1, k)
    g[-k:] = np.linspace(1, 0, k)
    out["fade_in_out"] = (y * g).astype(np.float32)
    out["noise_gate"] = (y * (np.convolve(np.abs(y), np.ones(200) / 200, mode="same") > 0.05 * np.max(np.abs(y)))).astype(np.float32)
    out["smooth_env"] = (y / (np.convolve(np.abs(y), np.ones(2205) / 2205, mode="same") + 0.05 * np.max(np.abs(y)))).astype(np.float32)
    out["highpass80"] = synth.highpass(y, 80)
    return out


def hack_test(scorer, n_bad=160, seed=0):
    """Share of bad windows that beat the median good window after the best of 20 tricks (hack_rate)."""
    rng = np.random.default_rng(seed)
    df = pd.read_csv(MANIFEST)
    gs = df[(df.source == "gs_cello") & (df.label_bin >= 0) & (df.mic == "neumann")]
    rob = df[df.source.isin(ROBOT_SOURCES) & (df.label_bin >= 0)]

    def windows(sub, k):
        """Onset windows of k random clips."""
        return np.stack([_onset_window(load_clip(c)) for c in sub.sample(min(k, len(sub)), random_state=seed).clip_id]).astype(np.float32)

    bad = np.concatenate([windows(gs[gs.label_bin == 0], n_bad - 40), windows(rob[rob.label_bin == 0], 40)])
    good = np.concatenate([windows(gs[gs.label_bin == 1], 80), windows(rob[rob.label_bin == 1], 60)])
    tr = [trick_list(w, rng) for w in bad]
    names = list(tr[0].keys())
    sc = np.asarray(scorer.score_windows(list(good) + [tr[i][k] for i in range(len(bad)) for k in names]), np.float64)
    g, b = sc[:len(good)], sc[len(good):].reshape(len(bad), len(names))
    thr = np.median(g)
    return dict(identity_rate=float((b[:, names.index("identity")] > thr).mean()), hack_rate=float((b.max(1) > thr).mean()))
