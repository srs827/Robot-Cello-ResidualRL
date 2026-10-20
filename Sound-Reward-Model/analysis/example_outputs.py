"""Write example reward outputs (explain for a good and a bad take, and compare of the two)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "reward"))
from common import MODELS, OUTPUTS, load_clip
from reward import Reward

GOOD = "loc_record_audio_note_110_D3"
BAD = "loc_record_audio_note_037_C2"


def main():
    """Run explain() and compare() on two real takes and save them as json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble12")
    ap.add_argument("--model_dir", default=MODELS)
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "examples"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = Reward(a.model, a.model_dir)
    g, b = load_clip(GOOD), load_clip(BAD)
    res = {"good_take": dict(clip=GOOD, **R.explain(g)), "bad_take": dict(clip=BAD, **R.explain(b)),
           "compare_good_vs_bad": R.compare(g, b)}
    for k, v in res.items():
        print(k, "->", v["summary"])
    json.dump(res, open(os.path.join(a.out, f"examples_{a.model}.json"), "w"), indent=1)
    print("saved", os.path.join(a.out, f"examples_{a.model}.json"))


if __name__ == "__main__":
    main()
