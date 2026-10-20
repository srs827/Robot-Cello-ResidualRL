"""Score the 384 good reference takes with every trained model and save reference.json (needed by reward.py)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import MODELS, ALIASES, load_clip, log
from model import NetScorer, clip_scores, clip_features, load_ckpts
from splits import training_pools

DEFAULT = ["m1_pairwise", "pw_ctl_s1", "pw_ctl_s2", "pw_ctl_s3", "pw_ctl_s4", "pw_ctl_s5", "pw_v1_s0", "pw_v1_s1",
           "pw_v2_s2", "pw_v2_s3", "pw_v3_s4", "pw_v3_s5", "distill8", "distill8b", "distill12"]


def main():
    """Save raw scores of the reference takes per model (and per ensemble alias) plus their acoustic features."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT), help="single model names (ensembles are averaged from these)")
    ap.add_argument("--model_dir", default=MODELS)
    a = ap.parse_args()
    ids = training_pools()["good"]
    clips = [load_clip(c) for c in ids]
    log(f"{len(ids)} good reference takes")
    scores = {}
    for name in a.models.split(","):
        if not load_ckpts(name, a.model_dir):
            log(f"skip {name}: no checkpoint")
            continue
        scores[name] = [round(float(v), 5) for v in clip_scores(NetScorer([name], a.model_dir), clips)]
        log(f"scored {name}")
    for alias, spec in ALIASES.items():
        names = spec.split("+")
        if all(n in scores for n in names):
            scores[alias] = [round(float(v), 5) for v in np.mean([scores[n] for n in names], axis=0)]
    feats = [[round(float(v), 5) for v in clip_features(y)] for y in clips]
    json.dump(dict(reference_clips=ids, scores=scores, features=feats), open(os.path.join(a.model_dir, "reference.json"), "w"))
    log(f"saved {os.path.join(a.model_dir, 'reference.json')} ({len(scores)} models)")


if __name__ == "__main__":
    main()
