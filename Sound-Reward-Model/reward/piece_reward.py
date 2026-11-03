"""Score a whole piece note by note: cut the recording into notes, then reward + main issue for each note."""
import json
import os
import sys

import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import Reward, SR

HOP = 512
SKIP_ASPECTS = {"ending"}


def pitch_track(y, sr=SR):
    """Frame-wise pitch (MIDI number, -1 = no clear pitch) and f0 in Hz, every HOP samples."""
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=1000, sr=sr, hop_length=HOP)
    midi = np.where(voiced & ~np.isnan(f0), np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1).astype(int)
    return midi, f0


def _pick_cuts(db, lo, hi, k, min_gap):
    """The k deepest volume dips (bow changes) between frames lo and hi, at least min_gap frames apart."""
    w = max(2, min_gap // 2)
    depth = {i: db[max(lo, i - w):i + w + 1].max() - db[i] for i in range(lo + min_gap, hi - min_gap)
             if db[i] <= db[i - 1] and db[i] <= db[i + 1]}
    chosen = []
    for i in sorted(depth, key=lambda i: -depth[i]):
        if all(abs(i - c) >= min_gap for c in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    while len(chosen) < k:
        gaps = sorted([lo] + chosen + [hi])
        a, b = max(zip(gaps[:-1], gaps[1:]), key=lambda g: g[1] - g[0])
        chosen.append((a + b) // 2)
    return sorted(chosen)


def _trim(y, s, e, sr=SR):
    """Cut the silent start/end off one note (the model was trained on trimmed notes)."""
    seg = y[int(s * sr):int(e * sr)]
    if len(seg) < int(0.2 * sr):
        return s, e
    _, (a, b) = librosa.effects.trim(seg, top_db=30, frame_length=1024, hop_length=256)
    return (s + a / sr, s + b / sr) if b - a >= int(0.1 * sr) else (s, e)


def notes_from_midi(y, midi_path, track=None, sr=SR):
    """Align the MIDI score to the recording by pitch (DTW), split repeated notes at bow changes -> [(start_s, end_s, pitch, score_start_s)]."""
    import pretty_midi
    m = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted([n for i in m.instruments if not i.is_drum for n in i.notes], key=lambda n: (n.start, -n.pitch))
    midi_a = (track if track is not None else pitch_track(y, sr))[0]
    step = 4
    fs = sr / HOP
    aud = np.array([np.bincount(midi_a[i:i + step] + 1).argmax() - 1 for i in range(0, len(midi_a), step)])
    n_score = int(np.ceil(m.get_end_time() * fs / step)) + 1
    sco = np.full(n_score, -1)
    for n in notes:
        sco[int(n.start * fs / step):max(int(n.end * fs / step), int(n.start * fs / step) + 1)] = n.pitch
    d = np.abs(sco[:, None] - aud[None, :])
    cost = np.where(aud[None, :] < 0, 0.6, np.where(d == 0, 0.0, np.where(d == 1, 0.2, np.where(d == 12, 0.3, 1.0))))
    cost = np.where(sco[:, None] < 0, 0.5, cost)
    _, wp = librosa.sequence.dtw(C=cost)
    s2a = np.zeros(n_score)
    s2a[:] = -1
    for i, j in wp[::-1]:
        if s2a[i] < 0:
            s2a[i] = j
    s2a = np.maximum.accumulate(s2a)
    to_frame = lambda t: int(s2a[min(int(t * fs / step), n_score - 1)] * step)
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0], ref=np.max)
    db = np.convolve(db, np.ones(5) / 5, mode="same")
    runs, cur = [], [notes[0]]
    for n in notes[1:]:
        if n.pitch == cur[-1].pitch and n.start - cur[-1].end < 0.05:
            cur.append(n)
        else:
            runs.append(cur)
            cur = [n]
    runs.append(cur)
    bounds = [to_frame(r[0].start) for r in runs] + [len(db) - 1]
    gap = int(0.15 * fs)
    out = []
    for r, lo, hi in zip(runs, bounds[:-1], bounds[1:]):
        on = np.flatnonzero(np.abs(midi_a[lo:hi] - r[0].pitch) <= 1)
        if len(on):
            lo, hi = lo + max(0, on[0] - 2), lo + min(hi - lo, on[-1] + 3)
        hi = max(hi, lo + gap * len(r))
        g = max(gap, int(0.4 * (hi - lo) / len(r)))
        cuts = [lo] + (_pick_cuts(db, lo, hi, len(r) - 1, g) if len(r) > 1 else []) + [hi]
        for n, a, b in zip(r, cuts[:-1], cuts[1:]):
            s, e = _trim(y, a / fs, b / fs, sr)
            out.append((s, e, n.pitch, n.start))
    return [x for x in out if x[1] - x[0] >= 0.1]


def notes_from_pitch(y, track=None, sr=SR, min_len=0.15):
    """No score available: cut notes where the detected pitch changes or the sound stops."""
    midi = (track if track is not None else pitch_track(y, sr))[0]
    dt = HOP / sr
    out, start = [], 0
    for k in range(1, len(midi) + 1):
        if k == len(midi) or midi[k] != midi[start]:
            if midi[start] > 0 and (k - start) * dt >= min_len:
                out.append((start * dt, k * dt, int(midi[start])))
            start = k
    return out


def notes_from_log(csv_path, offset=0.0):
    """Note times from a robot run log (rows with current_event_type == Note), shifted by the recording offset."""
    import pandas as pd
    d = pd.read_csv(csv_path)
    d = d[d["current_event_type"] == "Note"].copy()
    d["pitch"] = pd.to_numeric(d["current_note_number"], errors="coerce")
    d = d.dropna(subset=["pitch"])
    new = (d.index.to_series().diff() != 1) | (d["pitch"].diff() != 0) | (d["remaining_duration_sec"].diff() > 0.05)
    out = []
    for _, g in d.groupby(new.cumsum()):
        s, e = float(g["time_elapsed_sec"].iloc[0]) + offset, float(g["time_elapsed_sec"].iloc[-1]) + offset
        if e - s >= 0.1:
            out.append((s, e, int(g["pitch"].iloc[0])))
    return out


def cents_off(f0, s, e, target, sr=SR):
    """How far the played pitch is from the target note, in cents (None if no stable pitch)."""
    f = f0[int(s * sr / HOP):int(e * sr / HOP)]
    f = f[~np.isnan(f)]
    if len(f) < 3:
        return None
    c = 1200 * np.log2(f / librosa.midi_to_hz(target))
    c = c[np.abs(c) < 600]
    return float(np.median(c)) if len(c) >= 3 else None


def timing(rows, tol=0.2):
    """Rhythm check: each note's played length (to the next note) vs the score, after removing the overall tempo."""
    pairs = [(a, b) for a, b in zip(rows[:-1], rows[1:]) if a["score_start"] is not None and b["score_start"] is not None
             and b["score_start"] > a["score_start"]]
    if len(pairs) < 3:
        return {}
    ratio = np.array([(b["start"] - a["start"]) / (b["score_start"] - a["score_start"]) for a, b in pairs])
    tempo = float(np.median(ratio))
    for (a, b), q in zip(pairs, ratio):
        a["timing_dev"] = round(float(q / tempo - 1), 3)
    dev = ratio / tempo - 1
    return dict(slower_than_score=round(tempo, 3), timing_dev_median=round(float(np.median(np.abs(dev))), 3),
                share_on_time=round(float(np.mean(np.abs(dev) <= tol)), 3))


class PieceReward(Reward):
    """Reward model for whole pieces: one reward and one main issue per note, plus the piece average."""

    def piece(self, y, sr=SR, notes=None, midi=None, check_pitch=True):
        """Score every note; notes = [(start_s, end_s, midi_pitch)] or found from the MIDI score or the pitch."""
        y, status = self._prepare(y, sr)
        if y is None:
            return dict(status=status, reward=0.0, notes=[])
        track = pitch_track(y) if (check_pitch or notes is None) else None
        if notes is None:
            notes = notes_from_midi(y, midi, track) if midi else notes_from_pitch(y, track)
        rows = []
        for k, (s, e, p, *score) in enumerate(notes):
            seg = y[int(s * SR):int(e * SR)]
            info = self.explain(seg, SR)
            row = dict(index=k, start=round(s, 3), end=round(e, 3), note=librosa.midi_to_note(p, unicode=False),
                       reward=info["reward"], issue=None, issue_name=None, score_start=round(score[0], 3) if score else None)
            if info["status"] == "ok":
                todo = [a for a in info["aspects"] if a["status"] != "ok" and a["aspect"] not in SKIP_ASPECTS]
                if todo:
                    row["issue"], row["issue_name"] = todo[0]["aspect"], todo[0]["name"]
                if check_pitch:
                    c = cents_off(track[1], s, e, p)
                    row["cents_off"] = None if c is None else round(c, 1)
            rows.append(row)
        r = np.array([x["reward"] for x in rows]) if rows else np.zeros(1)
        issues = {}
        for x in rows:
            if x["issue"]:
                issues[x["issue"]] = issues.get(x["issue"], 0) + 1
        worst = sorted(rows, key=lambda x: x["reward"])[:5]
        res = dict(status="ok", reward=round(float(r.mean()), 4), n_notes=len(rows),
                   share_good=round(float(np.mean(r >= 0.5)), 3), issue_counts=issues,
                   worst_notes=[x["index"] for x in worst])
        res.update(timing(rows))
        res["notes"] = rows
        return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="recording of the whole piece (wav / mp3)")
    ap.add_argument("--midi", default=None, help="MIDI score of the piece (best way to find the notes)")
    ap.add_argument("--log", default=None, help="robot run log csv with note times (instead of --midi)")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds between log time 0 and recording start")
    ap.add_argument("--model", default="fast")
    ap.add_argument("--out", default=None, help="save the result as json")
    a = ap.parse_args()
    y, sr = librosa.load(a.audio, sr=SR, mono=True)
    notes = notes_from_log(a.log, a.offset) if a.log else None
    res = PieceReward(a.model).piece(y, sr, notes=notes, midi=a.midi)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "notes"}, indent=1))
