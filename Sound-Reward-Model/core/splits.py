"""Fixed train/test splits and the clip pools used for training."""
import numpy as np
import pandas as pd
from common import MANIFEST

OPEN = "cello_margarita_open_strings"
ROBOT_SOURCES = ["record_audio", "robot_d2", "robot_d3", "robot_d4"]


def local_split(df, seed=0):
    """70/30 train/test split of the labelled robot clips (human_A3 is always test)."""
    rob = df[df.source.isin(ROBOT_SOURCES)].copy()
    rng = np.random.default_rng(seed)
    ids = rob.clip_id.values
    lab = rob.label_bin.values
    test = set()
    for l in (0, 1):
        sel = ids[lab == l].copy()
        rng.shuffle(sel)
        test |= set(sel[: int(0.3 * len(sel))])
    return sorted(set(ids) - test), sorted(test)


def iowa_split(df, seed=0):
    """75/25 file-level train/test split of the Iowa cello notes."""
    io = df[df.source == "iowa"]
    files = sorted({c.rsplit("_", 1)[0] for c in io.clip_id})
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    test_files = set(files[: max(1, len(files) // 4)])
    te = [c for c in io.clip_id if c.rsplit("_", 1)[0] in test_files]
    tr = [c for c in io.clip_id if c.rsplit("_", 1)[0] not in test_files]
    return tr, te


def gs_sets(df):
    """Good-sounds cello clips with good/bad labels, and the open-string clips with expert ratings."""
    g = df[df.source == "gs_cello"]
    return g[g.label_bin >= 0], g[g.subset == OPEN]


def test_clips(df):
    """Every clip that must never be trained on (robot test split + human_A3)."""
    _, te = local_split(df)
    return set(te) | set(df[df.source == "human_A3"].clip_id)


def training_pools(df=None):
    """Clip pools for training: clean good clips, robot good/bad train clips, Good-sounds sets."""
    df = pd.read_csv(MANIFEST) if df is None else df
    lt, _ = local_split(df)
    it, _ = iowa_split(df)
    lab = df.set_index("clip_id").label_bin
    good = list(it) + [c for c in lt if lab[c] == 1]
    robot_good = [c for c in lt if lab[c] == 1]
    robot_bad = [c for c in lt if lab[c] == 0]
    gs_lab, gs_rated = gs_sets(df)
    return dict(good=good, robot_good=robot_good, robot_bad=robot_bad, robot_train=lt, gs_lab=gs_lab, gs_rated=gs_rated)
