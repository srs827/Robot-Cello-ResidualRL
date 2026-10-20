"""Which part of the note (start / middle / end windows) drives the reward score."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import MANIFEST, MODELS, OUTPUTS, load_clip, resolve
from model import NetScorer, clip_windows
from splits import ROBOT_SOURCES


def main():
    """Rank agreement between the full-clip score and a score from only the first, middle or last windows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble12")
    ap.add_argument("--model_dir", default=MODELS)
    ap.add_argument("--n", type=int, default=160, help="number of random real clips")
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "explain"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    real = m[m.source.isin(ROBOT_SOURCES + ["human_A3"])]
    real = real.sample(min(a.n, len(real)), random_state=0)
    sc = NetScorer(resolve(a.model), a.model_dir)
    parts = {"first": [], "middle": [], "last": [], "full": []}
    for cid in real.clip_id:
        ws = np.asarray(sc.score_windows(clip_windows(load_clip(cid))), np.float64)
        parts["first"].append(ws[0])
        parts["last"].append(ws[-1])
        parts["middle"].append(ws[1:-1].mean() if len(ws) > 2 else np.nan)
        parts["full"].append(ws.mean())
    full = np.array(parts["full"])
    res = {}
    for k in ("first", "middle", "last"):
        v = np.array(parts[k], float)
        ok = ~np.isnan(v)
        r, p = spearmanr(v[ok], full[ok])
        res[k] = dict(n=int(ok.sum()), spearman_with_full=round(float(r), 3), p=float(p))
        print(f"{k:7s} n={ok.sum():4d}  rank agreement with full score r = {r:+.3f}")
    json.dump(dict(model=a.model, n_clips=len(real), position_contribution=res),
              open(os.path.join(a.out, "explain_position.json"), "w"), indent=2)
    print("saved", os.path.join(a.out, "explain_position.json"))


if __name__ == "__main__":
    main()
