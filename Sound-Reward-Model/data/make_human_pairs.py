"""Turn raw A-vs-B listening answers into one soft label per clip pair, split into train/eval."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import MANIFEST, HUMAN_PAIRS, DATA
from splits import test_clips


def prepare(src, out):
    """Merge answers per pair: p_a = share of votes for a (ties count half); pairs touching a test clip -> eval."""
    m = pd.read_csv(MANIFEST)
    test, known = test_clips(m), set(m.clip_id)
    df = pd.read_csv(src)
    df = df[df.a.isin(known) & df.b.isin(known) & df.winner.isin(["a", "b", "tie"])]
    rec = {}
    for r in df.itertuples():
        lo, hi = sorted((r.a, r.b))
        d = rec.setdefault((lo, hi), [0, 0, 0])
        if r.winner == "tie":
            d[2] += 1
        elif (r.winner == "a") == (r.a == lo):
            d[0] += 1
        else:
            d[1] += 1
    rows = [dict(a=lo, b=hi, n_a=na, n_b=nb, n_tie=nt, n_votes=na + nb + nt, p_a=(na + 0.5 * nt) / (na + nb + nt),
                 split="eval" if (lo in test or hi in test) else "train")
            for (lo, hi), (na, nb, nt) in rec.items()]
    hp = pd.DataFrame(rows).sort_values(["split", "a", "b"]).reset_index(drop=True)
    hp.to_csv(out, index=False)
    for s in ("train", "eval"):
        h = hp[hp.split == s]
        print(f"{s}: {len(h)} pairs ({int((h.p_a != 0.5).sum())} decisive)")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(DATA, "human_judgments.csv"))
    ap.add_argument("--out", default=HUMAN_PAIRS)
    a = ap.parse_args()
    prepare(a.src, a.out)
