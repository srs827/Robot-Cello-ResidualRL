"""Which acoustic features move together with the reward score (Spearman correlation on all real recordings)."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import MANIFEST, MODELS, OUTPUTS, load_clip, resolve
from model import NetScorer, clip_scores, clip_features, PHYS_NAMES
from splits import ROBOT_SOURCES


def main():
    """Correlate the 12 features with the model score and compare good vs bad label groups."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble12")
    ap.add_argument("--model_dir", default=MODELS)
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "explain"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    real = m[m.source.isin(ROBOT_SOURCES + ["human_A3"])]
    clips = [load_clip(c) for c in real.clip_id]
    scores = clip_scores(NetScorer(resolve(a.model), a.model_dir), clips)
    feats = np.stack([clip_features(y) for y in clips])
    rows = []
    for i, name in enumerate(PHYS_NAMES):
        rho, p = spearmanr(feats[:, i], scores)
        rows.append(dict(feature=name, spearman_r=round(float(rho), 3), p=float(p)))
    rows.sort(key=lambda r: -abs(r["spearman_r"]))
    good, bad = real.label_bin.values == 1, real.label_bin.values == 0
    groups = [dict(feature=n, good_mean=round(float(feats[good, i].mean()), 3), bad_mean=round(float(feats[bad, i].mean()), 3))
              for i, n in enumerate(PHYS_NAMES)]
    for r in rows:
        print(f"{r['feature']:40s} r = {r['spearman_r']:+.3f}  p = {r['p']:.1e}")
    json.dump(dict(model=a.model, n_clips=len(clips), n_good=int(good.sum()), n_bad=int(bad.sum()),
                   feature_correlations=rows, group_means=groups),
              open(os.path.join(a.out, "explain_features.json"), "w"), indent=2)
    print("saved", os.path.join(a.out, "explain_features.json"))


if __name__ == "__main__":
    main()
