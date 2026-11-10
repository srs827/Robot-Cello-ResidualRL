"""Fine-tune the fast model so clean bowing beats double-slip / crushed bowing on the virtual cello, while keeping its scores on real recordings."""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from common import SR, MODELS, MANIFEST, OUTPUTS, log
from model import load_ckpts, load_net
from splits import training_pools
from train_distill import RealSampler
from virtual_cello import play, set_eq
import synth

PITCHES = {"G2": 98.0, "D3": 146.83, "A3": 220.0}
WIN = int(0.5 * SR)


def vcello_bank(pitches, n, seed):
    """Random bowings on the virtual cello, sorted by physics into clean (Helmholtz) and bad (double slip, crushed)."""
    rng = np.random.default_rng(seed)
    clean, bad = [], []
    for k in range(n):
        f, v = np.exp(rng.uniform(np.log(0.003), np.log(10))), np.exp(rng.uniform(np.log(0.01), np.log(1)))
        y, h = play(f, v, rng.uniform(0.06, 0.2), f0=PITCHES[pitches[k % len(pitches)]], dur=1.3, seed=k)
        if h > 0.8:
            clean.append(y[int(0.3 * SR):])
        elif h < 0.2:
            bad.append(y[int(0.3 * SR):])
    return clean, bad


def window(y, rng):
    """Random 0.5 s window with random recording effects."""
    s = int(rng.integers(0, len(y) - WIN + 1))
    return synth.random_nuisance(y[s:s + WIN].astype(np.float32), rng, 0.6)


def main():
    """Train <out>/<name>_seed0.pt and copy the starting model next to it for comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fast_vc")
    ap.add_argument("--init", default="distill12")
    ap.add_argument("--train_pitches", default="G2,A3", help="D3 is kept out to test on an unseen string")
    ap.add_argument("--bank", type=int, default=4000, help="virtual cello notes to generate")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the virtual cello pairs vs keeping real scores")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_batches", type=int, default=150)
    ap.add_argument("--calibration", default=os.path.join(OUTPUTS, "virtual_cello", "virtual_cello.json"))
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "candidates"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    cal = json.load(open(a.calibration)).get("calibration")
    if cal:
        set_eq(np.array(cal["freqs"]), np.array(cal["gain_db"]))
    clean, bad = vcello_bank(a.train_pitches.split(","), a.bank, a.seed)
    log(f"virtual cello bank: {len(clean)} clean, {len(bad)} bad notes ({a.train_pitches})")
    init_path = load_ckpts(a.init, a.models)[0]
    teacher, student = load_net(init_path), load_net(init_path)
    for q in teacher.parameters():
        q.requires_grad_(False)
    pools = training_pools(pd.read_csv(MANIFEST))
    real_ids = sorted(set(pools["good"]) | set(pools["robot_train"]) | set(pools["gs_lab"].clip_id) | set(pools["gs_rated"].clip_id))
    it = iter(RealSampler(real_ids, pools["good"], a.seed, per_batch=96))
    anchor = []
    with torch.no_grad():
        for _ in range(a.n_batches):
            x = next(it)
            anchor.append((x, teacher(x)["logit"]))
    log(f"{len(anchor)} real-recording batches scored by {a.init}")
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-4)
    run, t0 = None, time.time()
    for step in range(1, a.steps + 1):
        student.train()
        x, t = anchor[int(rng.integers(len(anchor)))]
        s = student(x)["logit"]
        keep = F.mse_loss(s - s.mean(), t - t.mean())
        xc = torch.from_numpy(np.stack([window(clean[int(rng.integers(len(clean)))], rng) for _ in range(a.pairs)]))
        xb = torch.from_numpy(np.stack([window(bad[int(rng.integers(len(bad)))], rng) for _ in range(a.pairs)]))
        sc = student(torch.cat([xc, xb]))["logit"]
        pair = F.softplus(-(sc[:a.pairs] - sc[a.pairs:])).mean()
        loss = keep + a.lam * pair
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
        opt.step()
        r = (float(keep), float(pair))
        run = r if run is None else tuple(0.95 * p + 0.05 * q for p, q in zip(run, r))
        if step % 50 == 0:
            print(f"step {step}/{a.steps} keep-real mse={run[0]:.3f} clean-vs-bad loss={run[1]:.3f} | {time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(init_path, map_location="cpu", weights_only=False)
    torch.save(dict(state=student.state_dict(), cfg=ck["cfg"], args=vars(a)), os.path.join(a.out, f"{a.name}_seed0.pt"))
    shutil.copy(init_path, os.path.join(a.out, os.path.basename(init_path)))
    log(f"saved {os.path.join(a.out, a.name + '_seed0.pt')} (+ copy of {a.init} for comparison)")


if __name__ == "__main__":
    main()
