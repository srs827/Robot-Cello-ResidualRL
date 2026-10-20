"""Shared paths, constants and small helpers used by every script."""
import os
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
LOCAL = os.path.join(DATA, "local")
PROC = os.path.join(DATA, "proc")
AUDIO_DIR = os.path.join(PROC, "audio")
MANIFEST = os.path.join(PROC, "manifest.csv")
HUMAN_PAIRS = os.path.join(DATA, "human_pairs.csv")
OUTPUTS = os.path.join(ROOT, "outputs")
MODELS = os.environ.get("SOUND_MODEL_DIR", os.path.join(ROOT, "checkpoints"))

SR = 22050

ENSEMBLE12 = ["m1_pairwise", "pw_ctl_s1", "pw_ctl_s2", "pw_ctl_s3", "pw_ctl_s4", "pw_ctl_s5",
              "pw_v1_s0", "pw_v1_s1", "pw_v2_s2", "pw_v2_s3", "pw_v3_s4", "pw_v3_s5"]
ENSEMBLE8 = ["m1_pairwise", "pw_ctl_s1", "pw_ctl_s2", "pw_ctl_s3", "pw_v1_s0", "pw_v1_s1", "pw_v2_s2", "pw_v2_s3"]
ALIASES = {"ensemble12": "+".join(ENSEMBLE12), "ensemble8": "+".join(ENSEMBLE8), "fast": "distill12"}


def resolve(spec):
    """Turn a model alias ('ensemble12', 'fast') or 'a+b+c' into the list of model names."""
    return ALIASES.get(spec, spec).split("+")


def load_clip(clip_id):
    """Load one prepared clip (22.05 kHz mono float32) by its id."""
    return np.load(os.path.join(AUDIO_DIR, f"{clip_id}.npy"))


def rms_dbfs(y):
    """Loudness of a waveform in dBFS."""
    return 20 * np.log10(float(np.sqrt(np.mean(np.square(y)) + 1e-12)) + 1e-9)


def log(msg):
    """Print a time-stamped message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
