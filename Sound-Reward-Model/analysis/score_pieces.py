"""Score the robot's full-piece recordings note by note (uses the MIDI scores to find the notes)."""
import argparse
import glob
import json
import os
import sys
import time

import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reward"))
from piece_reward import PieceReward, SR


def main():
    """Run piece_reward on every recording that has a MIDI file with the same name."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=os.path.join(HERE, "..", "..", "lol", "Robot"))
    ap.add_argument("--midi_dir", default=os.path.join(HERE, "..", "..", "lol", "Midi"))
    ap.add_argument("--model", default="fast")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "pieces", "pieces.json"))
    a = ap.parse_args()
    R = PieceReward(a.model)
    res = {}
    for f in sorted(glob.glob(os.path.join(a.audio_dir, "*.mp3")) + sorted(glob.glob(os.path.join(a.audio_dir, "*.wav")))):
        name = os.path.splitext(os.path.basename(f))[0]
        mid = os.path.join(a.midi_dir, name + ".mid")
        if not os.path.exists(mid):
            continue
        y, _ = librosa.load(f, sr=SR, mono=True)
        t0 = time.time()
        r = R.piece(y, SR, midi=mid)
        r["seconds_audio"], r["seconds_compute"] = round(len(y) / SR, 1), round(time.time() - t0, 1)
        res[name] = r
        print(name, r["n_notes"], "notes, reward", r["reward"], "good share", r["share_good"], r["issue_counts"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(model=a.model, pieces=res), open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
