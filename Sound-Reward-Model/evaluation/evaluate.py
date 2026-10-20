"""Evaluate reward models: agreement with held-out human A-vs-B answers, benchmark AUCs and the hack test."""
import argparse
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, HERE)
from common import MODELS, OUTPUTS, HUMAN_PAIRS, load_clip, resolve
from model import NetScorer, clip_scores
from benchmark import scorecard, hack_test


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval of a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Exact McNemar p-value for b vs c disagreements."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def human_agreement(scorer, hp):
    """Per held-out decisive pair: did the model score the human's winner higher? -> bool array."""
    ev = hp[(hp.split == "eval") & (hp.p_a != 0.5)]
    win = np.where(ev.p_a > 0.5, ev.a, ev.b)
    lose = np.where(ev.p_a > 0.5, ev.b, ev.a)
    clips = sorted(set(win) | set(lose))
    idx = {c: i for i, c in enumerate(clips)}
    s = clip_scores(scorer, [load_clip(c) for c in clips])
    return np.array([s[idx[w]] > s[idx[l]] for w, l in zip(win, lose)])


def main():
    """Evaluate each model, print a table and save outputs/evaluation/evaluation.json and summary.csv."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ensemble12,fast,m1_pairwise", help="comma list; aliases ensemble12 / fast or a+b+c")
    ap.add_argument("--baseline", default="m1_pairwise")
    ap.add_argument("--model_dir", default=MODELS)
    ap.add_argument("--human_pairs", default=HUMAN_PAIRS)
    ap.add_argument("--skip_bench", action="store_true", help="only the human-agreement test (fast)")
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "evaluation"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hp = pd.read_csv(a.human_pairs)
    specs = a.models.split(",")
    if a.baseline not in specs:
        specs.append(a.baseline)
    res, correct = {}, {}
    for spec in specs:
        sc = NetScorer(resolve(spec), a.model_dir)
        ok = human_agreement(sc, hp)
        correct[spec] = ok
        lo, hi = wilson(int(ok.sum()), len(ok))
        r = dict(n_models=len(sc.models), human_agree=float(ok.mean()), ci_low=float(lo), ci_high=float(hi), n_pairs=int(len(ok)))
        if not a.skip_bench:
            r.update(scorecard(sc))
            r.update(hack_test(sc))
        res[spec] = r
        print(spec, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    for spec, ok in correct.items():
        if spec != a.baseline:
            b, c = int((ok & ~correct[a.baseline]).sum()), int((~ok & correct[a.baseline]).sum())
            res[spec].update(only_this_right=b, only_baseline_right=c, mcnemar_p_vs_baseline=mcnemar_exact(b, c))
    json.dump(res, open(os.path.join(a.out, "evaluation.json"), "w"), indent=1)
    pd.DataFrame(res).T.to_csv(os.path.join(a.out, "summary.csv"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
