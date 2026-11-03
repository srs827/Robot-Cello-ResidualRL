"""Does MP3 compression change the reward? Score held-out takes before and after an MP3 round trip."""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "reward"))
from common import SR, MANIFEST, OUTPUTS, load_clip
from splits import test_clips
from reward import Reward

SETTINGS = {"mp3_80kbps": 0.3, "mp3_50kbps": 0.5}


def mp3_round_trip(y, level):
    """Encode at 44.1 kHz as variable-bitrate MP3 (like the robot song files) and decode back to 22.05 kHz."""
    up = resample_poly(y, 2, 1).astype(np.float32)
    up = up / (np.abs(up).max() + 1e-9) * 0.9
    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", 2 * SR, 1, format="MP3", subtype="MPEG_LAYER_III",
                      compression_level=level, bitrate_mode="VARIABLE") as f:
        f.write(up)
    buf.seek(0)
    z, _ = sf.read(buf, dtype="float32")
    return resample_poly(z, 1, 2).astype(np.float32)


def main():
    """Mean reward of good and bad held-out takes and good-vs-bad AUC, original vs MP3."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="fast,ensemble12")
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "explain", "mp3_test.json"))
    a = ap.parse_args()
    m = pd.read_csv(MANIFEST)
    held = test_clips(m)
    t = m[m.clip_id.isin(held) & m.label_bin.isin([0, 1]) & (m.dur >= 0.5)].sort_values("clip_id")
    clips = [load_clip(c) for c in t.clip_id]
    versions = {"original": clips}
    versions.update({k: [mp3_round_trip(y, lv) for y in clips] for k, lv in SETTINGS.items()})
    lab = t.label_bin.values.astype(int)
    res = dict(n_good=int(lab.sum()), n_bad=int((1 - lab).sum()), models={})
    for name in a.models.split(","):
        R = Reward(name)
        res["models"][name] = {}
        base = None
        for v, ys in versions.items():
            r = np.array([R.reward(y, SR) for y in ys])
            base = r if base is None else base
            res["models"][name][v] = dict(mean_good=round(float(r[lab == 1].mean()), 4),
                                          mean_bad=round(float(r[lab == 0].mean()), 4),
                                          auc=round(float(roc_auc_score(lab, r)), 4),
                                          mean_change=round(float((r - base).mean()), 4),
                                          max_abs_change=round(float(np.abs(r - base).max()), 4))
            print(name, v, res["models"][name][v], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
