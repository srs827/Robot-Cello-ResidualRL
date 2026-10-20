"""Turn all raw recordings into one manifest.csv plus one silence-trimmed 22.05 kHz .npy file per clip."""
import collections
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import SR, RAW, LOCAL, AUDIO_DIR, MANIFEST, rms_dbfs

MAXLEN = 8.0


def load_trim(path, seg=None):
    """Read an audio file (optionally a time segment), resample to 22.05 kHz mono and trim silence."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if seg is not None:
        y = y[int(seg[0] * sr): int(seg[1] * sr)]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y, _ = librosa.effects.trim(y, top_db=40)
    return y[: int(MAXLEN * SR)].astype(np.float32)


def work(row):
    """Prepare one clip; returns (clip_id, duration, dBFS) or None duration if unusable."""
    cid, path, seg = row["clip_id"], row["src_path"], row.get("seg")
    out = os.path.join(AUDIO_DIR, f"{cid}.npy")
    try:
        if not os.path.exists(out):
            y = load_trim(path, seg)
            if len(y) < int(0.08 * SR) or not np.isfinite(y).all():
                return cid, None, None
            np.save(out, y)
        else:
            y = np.load(out)
        return cid, len(y) / SR, rms_dbfs(y)
    except Exception as e:
        return cid, None, str(e)


def local_rows():
    """Rows for the team's own robot and human recordings with good/bad labels."""
    srcs = [("record_audio", "record_audio/data", "record_audio/labels.json", "robot_rec"),
            ("robot_d2", "Robot/data2", "Robot/data2_labels.json", "robot"),
            ("robot_d3", "Robot/data3", "Robot/data3_labels.json", "robot"),
            ("robot_d4", "Robot/data4", "Robot/data4_labels.json", "robot"),
            ("human_A3", "Robot/human_data", "Robot/human_data_labels.json", "human")]
    rows = []
    for name, d, lj, dom in srcs:
        lj = os.path.join(LOCAL, lj)
        if not os.path.exists(lj):
            print("missing", lj)
            continue
        for fn, lab in json.load(open(lj, encoding="utf-8")).items():
            lab = str(lab).strip().lower()
            p = os.path.join(LOCAL, d, fn)
            if lab not in ("good", "bad") or not os.path.exists(p):
                continue
            stem = os.path.splitext(fn)[0]
            rows.append(dict(clip_id=f"loc_{name}_{stem}", src_path=p, source=name, domain=dom, subset=name,
                             player="robot" if dom != "human" else "human", mic="local",
                             label_bin=1 if lab == "good" else 0, defect="", group=f"loc_{name}_{stem}"))
    return rows


def gs_rows():
    """Rows for the Good-sounds cello recordings with labels and expert ratings."""
    D = os.path.join(RAW, "good_sounds")
    if not os.path.exists(os.path.join(D, "sounds.json")):
        print("Good-sounds not downloaded, skipped")
        return []
    load = lambda n: json.load(open(os.path.join(D, n)))
    sounds, takes, ratings = load("sounds.json"), load("takes.json"), load("ratings.json")
    packs = {v["id"]: v["name"] for v in load("packs.json").values()}
    rt = collections.defaultdict(dict)
    for r in ratings.values():
        rt[r["sound_id"]].setdefault(r["type"], []).append(r["mark"])
    tk = collections.defaultdict(list)
    for t in takes.values():
        tk[t["sound_id"]].append(t)
    rows = []
    for s in sounds.values():
        if s["instrument"] != "cello":
            continue
        klass = s["klass"] or ""
        lab, defect = -1, ""
        if klass.startswith("good"):
            lab = 1
        elif klass.startswith("bad"):
            lab, defect = 0, klass.split("-")[1]
        m = rt.get(s["id"], {})
        for t in tk[s["id"]]:
            p = os.path.join(D, "audio", t["filename"])
            if not os.path.exists(p):
                continue
            rows.append(dict(clip_id=f"gs_{s['id']}_{t['microphone']}", src_path=p, source="gs_cello", domain="human",
                             subset=packs[s["pack_id"]], player=s["player"], mic=t["microphone"], label_bin=lab,
                             defect=defect, klass=klass,
                             r_overall=np.mean(m["good-sound"]) if "good-sound" in m else np.nan,
                             r_attack=np.mean(m["attack"]) if "attack" in m else np.nan,
                             r_stability=np.mean(m["stability"]) if "stability" in m else np.nan,
                             r_release=np.mean(m["release"]) if "release" in m else np.nan,
                             string=s["string"], bow_velocity=s["bow_velocity"], bridge=s["bridge_position"],
                             dynamics=s["dynamics"], note=f"{s['note']}{s['octave']}", group=f"gs_{s['id']}"))
    return rows


def iowa_rows():
    """Rows for the Iowa cello notes (each long file is split into single notes)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "iowa_cello", "*.aiff"))):
        base = os.path.basename(p)
        m = re.match(r"Cello\.arco\.(pp|mf|ff)\.sul([CGDA])\.", base)
        if not m:
            continue
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        y = y.mean(1)
        iv = librosa.effects.split(librosa.resample(y, orig_sr=sr, target_sr=SR), top_db=38, frame_length=2048, hop_length=512)
        k = 0
        for a, b in iv:
            if (b - a) / SR < 0.7:
                continue
            rows.append(dict(clip_id=f"iowa_{base[:-5]}_{k:02d}", src_path=p, seg=(max(0, a / SR - 0.02), b / SR + 0.02),
                             source="iowa", domain="human", subset="iowa_arco", player="iowa", mic="neumann_anechoic",
                             label_bin=1, defect="", dynamics=m.group(1), string={"C": 4, "G": 3, "D": 2, "A": 1}[m.group(2)],
                             group=f"iowa_{base[:-5]}_{k:02d}"))
            k += 1
    return rows


def main():
    """Build the manifest and the per-clip audio cache."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    rows = local_rows() + gs_rows() + iowa_rows()
    print("rows", len(rows), collections.Counter(r["source"] for r in rows), flush=True)
    with Pool(max(1, (os.cpu_count() or 4) - 1)) as pool:
        res = pool.map(work, rows, chunksize=8)
    info = {c: (d, r) for c, d, r in res}
    keep = []
    for r in rows:
        d, rr = info[r["clip_id"]]
        if d is None:
            continue
        r["dur"], r["dbfs"] = d, rr
        r.pop("src_path")
        r.pop("seg", None)
        keep.append(r)
    df = pd.DataFrame(keep)
    df.to_csv(MANIFEST, index=False)
    print(f"saved {MANIFEST}: {len(df)} clips (dropped {len(rows) - len(df)})")
    print(df.groupby(["source", "label_bin"]).size())


if __name__ == "__main__":
    main()
