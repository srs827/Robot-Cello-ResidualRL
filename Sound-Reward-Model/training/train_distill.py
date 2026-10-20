"""Distil an ensemble of reward models into one fast model that copies the ensemble's scores."""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import SR, MODELS, MANIFEST, HUMAN_PAIRS, load_clip, log, resolve
from model import QualityNet, load_ckpts, load_net
from splits import training_pools
from train_reward import PairSampler, keep_awake, load_human, LENS
import synth

KEYS = ["x", "rate_w", "rate_l", "gg", "gb", "rg_w", "rg_l", "h_a", "h_b"]


class RealSampler(torch.utils.data.IterableDataset):
    """Batches of windows mostly from real training recordings plus ~25% synthetic-defect windows."""

    def __init__(self, real_ids, good_ids, seed, per_batch=160, synth_frac=0.25):
        self.real_ids, self.good_ids, self.seed = real_ids, good_ids, seed
        self.per_batch, self.synth_frac = per_batch, synth_frac

    def __iter__(self):
        """Yield one (per_batch, n) tensor of windows per step."""
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 7919 + wid + 4242)
        cache = {}
        clean_defects = [d for d in synth.SEEN_DEFECTS if d not in synth.ATTACK_DEFECTS]
        while True:
            n = int(float(rng.choice(LENS)) * SR)
            X, tries = [], 0
            while len(X) < self.per_batch and tries < self.per_batch * 6:
                tries += 1
                use_synth = rng.random() < self.synth_frac
                pool = self.good_ids if use_synth else self.real_ids
                c = pool[rng.integers(len(pool))]
                if c not in cache:
                    cache[c] = load_clip(c)
                y = cache[c]
                if len(y) < n:
                    continue
                w = y[int(rng.integers(0, len(y) - n + 1)):][:n].astype(np.float32)
                if use_synth:
                    w = synth.apply_defect(w, clean_defects[rng.integers(len(clean_defects))], int(rng.integers(1, 4)), rng)
                X.append(synth.random_nuisance(w, rng, 0.6))
            yield torch.from_numpy(np.stack(X))


def load_teachers(spec, model_dir):
    """Load every checkpoint of the teacher models, frozen."""
    ms = []
    for p in [c for n in resolve(spec) for c in load_ckpts(n, model_dir)]:
        m = load_net(p)
        for q in m.parameters():
            q.requires_grad_(False)
        ms.append(m)
    return ms


def build_set(a, teachers, pools):
    """Score n_batches of training windows once with the teacher ensemble -> list of (windows, scores)."""
    if a.mode == "real":
        real_ids = sorted(set(pools["good"]) | set(pools["robot_train"]) | set(pools["gs_lab"].clip_id) | set(pools["gs_rated"].clip_id))
        ds = RealSampler(real_ids, pools["good"], a.seed)
    else:
        a.groups, a.gs_pairs = 12, 12
        ds = PairSampler(a, pools, a.seed, load_human(a.human_pairs))
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers, persistent_workers=a.workers > 0,
                                     prefetch_factor=3 if a.workers > 0 else None)
    it, data, t0 = iter(dl), [], time.time()
    for i, b in enumerate(it):
        if i >= a.n_batches:
            break
        xs = [b] if a.mode == "real" else [b[k].reshape(-1, b[k].shape[-1]) if b[k].dim() == 3 else b[k] for k in KEYS if k in b]
        with torch.no_grad():
            ts = [torch.stack([m(x)["logit"] for m in teachers]).mean(0) for x in xs]
        data.append((xs, ts))
        if i % 50 == 0:
            print(f"  scoring batch {i}/{a.n_batches} | {time.time() - t0:.0f}s", flush=True)
    try:
        it._shutdown_workers()
    except Exception:
        pass
    return data


def main():
    """Train the student and save <out>/<name>_seed0.pt."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--teachers", required=True, help="e.g. ensemble12 or a+b+c")
    ap.add_argument("--init", required=True, help="model to start the student from")
    ap.add_argument("--mode", choices=["mixed", "real"], default="real", help="mixed = training-pair batches, real = mostly real recordings")
    ap.add_argument("--n_batches", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human_pairs", default=HUMAN_PAIRS)
    ap.add_argument("--models", default=MODELS, help="folder with the teacher and init checkpoints")
    ap.add_argument("--out", default=MODELS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    keep_awake()
    teachers = load_teachers(a.teachers, a.models)
    log(f"distill {a.name}: {len(teachers)} teachers, mode={a.mode}")
    data = build_set(a, teachers, training_pools(pd.read_csv(MANIFEST)))
    n_val = max(1, len(data) // 10)
    train, val = data[:-n_val], data[-n_val:]
    student = load_net(load_ckpts(a.init, a.models)[0])

    def centred(batch, grad):
        """Student and teacher scores of a batch, each centred (only relative scores matter)."""
        xs, ts = batch
        s = torch.cat([student(x)["logit"] for x in xs])
        s = s if grad else s.detach()
        t = torch.cat(ts)
        return s - s.mean(), t - t.mean()

    def val_corr():
        """Correlation between student and ensemble on the held-out batches."""
        student.eval()
        with torch.no_grad():
            cs = [centred(v, False) for v in val]
        return float(torch.corrcoef(torch.stack([torch.cat([c[0] for c in cs]), torch.cat([c[1] for c in cs])]))[0, 1])

    steps = a.epochs * len(train)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    rng, step, run, t0 = np.random.default_rng(a.seed), 0, None, time.time()
    for _ in range(a.epochs):
        for j in rng.permutation(len(train)):
            student.train()
            s, t = centred(train[j], True)
            loss = F.mse_loss(s, t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            opt.step()
            sched.step()
            run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} mse={run:.3f} held-out corr={val_corr():.3f} | {time.time() - t0:.0f}s", flush=True)
    vc = val_corr()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}_seed0.pt")
    torch.save(dict(state=student.state_dict(), cfg=dict(n_cls=21, w=24, phys=False), args=vars(a)), path)
    log(f"saved {path}; held-out correlation with the ensemble = {vc:.3f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
