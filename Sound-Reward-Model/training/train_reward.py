"""Train one reward model from "A is better than B" pairs (synthetic defects, Good-sounds, robot labels, human answers)."""
import argparse
import ctypes
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from common import SR, MODELS, HUMAN_PAIRS, load_clip, log
from model import QualityNet
from splits import training_pools
import synth

LENS = [0.1, 0.2, 0.35, 0.5, 1.0]
RATE_GAP = 2.0


class PairSampler(torch.utils.data.IterableDataset):
    """Endless stream of training batches, each made of better/worse window pairs from every data source."""

    def __init__(self, a, pools, seed, human):
        self.a, self.seed, self.human = a, seed, human
        self.good_ids, self.rg, self.rb = pools["good"], pools["robot_good"], pools["robot_bad"]
        self.gs_lab = pools["gs_lab"].reset_index(drop=True)
        self.gs_rated = pools["gs_rated"].reset_index(drop=True)
        self.gs_good = self.gs_lab.index[self.gs_lab.label_bin == 1].values
        self.gs_bad = self.gs_lab.index[self.gs_lab.label_bin == 0].values

    def _clip(self, cid):
        """Load a clip once per worker and keep it in memory."""
        if cid not in self.cache:
            self.cache[cid] = load_clip(cid)
        return self.cache[cid]

    def _win(self, rng, cid, n, onset=False, strength=0.6):
        """Random window of n samples from a clip, with random recording effects."""
        y = self._clip(cid)
        if len(y) < n:
            return None
        s0 = 0 if onset else int(rng.integers(0, len(y) - n + 1))
        return synth.random_nuisance(y[s0:s0 + n].astype(np.float32), rng, strength)

    def _severity_group(self, rng, n):
        """Clean window, the same window with a defect at severity 1/2/3, and a second clean view."""
        y = self._clip(self.good_ids[rng.integers(len(self.good_ids))])
        if len(y) < n:
            return None
        onset = rng.random() < 0.35
        s0 = 0 if onset else int(rng.integers(0, len(y) - n + 1))
        base = y[s0:s0 + n].astype(np.float32)
        cands = [d for d in synth.SEEN_DEFECTS if onset or d not in synth.ATTACK_DEFECTS]
        d = cands[rng.integers(len(cands))]
        sa, sb = int(rng.integers(1 << 30)), int(rng.integers(1 << 30))
        X = [synth.random_nuisance(base, np.random.default_rng(sa))]
        for sev in (1, 2, 3):
            X.append(synth.random_nuisance(synth.apply_defect(base, d, sev, rng), np.random.default_rng(sa + sev)))
        X.append(synth.random_nuisance(base, np.random.default_rng(sb)))
        return np.stack(X).astype(np.float32)

    def _rated(self, rng, n):
        """Pairs of open-string windows whose expert ratings differ by at least RATE_GAP (higher rated first)."""
        W, L, tries = [], [], 0
        while len(W) < self.a.gs_pairs and tries < self.a.gs_pairs * 8:
            tries += 1
            r1 = self.gs_rated.iloc[rng.integers(len(self.gs_rated))]
            r2 = self.gs_rated.iloc[rng.integers(len(self.gs_rated))]
            if abs(r1.r_overall - r2.r_overall) < RATE_GAP:
                continue
            hi, lo = (r1, r2) if r1.r_overall > r2.r_overall else (r2, r1)
            wv = self._win(rng, hi.clip_id, n, onset=bool(rng.random() < 0.3))
            lv = self._win(rng, lo.clip_id, n, onset=bool(rng.random() < 0.3))
            if wv is None or lv is None:
                continue
            W.append(wv)
            L.append(lv)
        return W, L

    def _good_bad(self, rng, n, good_fn, bad_fn):
        """Collect gs_pairs good windows and gs_pairs bad windows."""
        G, B = [], []
        for _ in range(self.a.gs_pairs * 4):
            if len(G) >= self.a.gs_pairs and len(B) >= self.a.gs_pairs:
                break
            if len(G) < self.a.gs_pairs:
                w = good_fn(rng, n)
                if w is not None:
                    G.append(w)
            if len(B) < self.a.gs_pairs:
                w = bad_fn(rng, n)
                if w is not None:
                    B.append(w)
        m = min(len(G), len(B))
        return G[:m], B[:m]

    def _gs_good(self, rng, n):
        """Window from a Good-sounds clip labelled good."""
        r = self.gs_lab.iloc[rng.choice(self.gs_good)]
        return self._win(rng, r.clip_id, n, onset=(str(r.klass) == "good-attack"))

    def _gs_bad(self, rng, n):
        """Window from a Good-sounds clip labelled bad."""
        r = self.gs_lab.iloc[rng.choice(self.gs_bad)]
        return self._win(rng, r.clip_id, n, onset=(r.defect == "attack"))

    def _robot_good(self, rng, n):
        """Window from a robot train clip labelled good."""
        return self._win(rng, self.rg[rng.integers(len(self.rg))], n, strength=0.4)

    def _robot_bad(self, rng, n):
        """Window from a robot train clip labelled bad."""
        return self._win(rng, self.rb[rng.integers(len(self.rb))], n, strength=0.4)

    def _human(self, hrng):
        """Human-compared windows (a, b) with soft label p(a better)."""
        n = int(float(hrng.choice(LENS)) * SR)
        A, B, P, tries = [], [], [], 0
        while len(A) < self.a.gs_pairs and tries < self.a.gs_pairs * 8:
            tries += 1
            ia, ib, pa = self.human[hrng.integers(len(self.human))]
            if hrng.random() < 0.5:
                ia, ib, pa = ib, ia, 1.0 - pa
            wa, wb = self._win(hrng, ia, n, strength=0.4), self._win(hrng, ib, n, strength=0.4)
            if wa is None or wb is None:
                continue
            A.append(wa)
            B.append(wb)
            P.append(pa)
        return A, B, P

    def __iter__(self):
        """Yield one batch dict per training step."""
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid)
        hrng = np.random.default_rng(self.seed * 7919 + wid + 12345)
        self.cache = {}
        while True:
            n = int(float(rng.choice(LENS)) * SR)
            out = {}
            sev = [g for g in (self._severity_group(rng, n) for _ in range(self.a.groups)) if g is not None]
            if sev:
                out["x"] = torch.from_numpy(np.stack(sev))
            if len(self.gs_rated):
                W, L = self._rated(rng, n)
                if W:
                    out["rate_w"], out["rate_l"] = torch.from_numpy(np.stack(W)), torch.from_numpy(np.stack(L))
            if len(self.gs_good) and len(self.gs_bad):
                G, B = self._good_bad(rng, n, self._gs_good, self._gs_bad)
                if G:
                    out["gg"], out["gb"] = torch.from_numpy(np.stack(G)), torch.from_numpy(np.stack(B))
            if self.rb and self.rg:
                G, B = self._good_bad(rng, n, self._robot_good, self._robot_bad)
                if G:
                    out["rg_w"], out["rg_l"] = torch.from_numpy(np.stack(G)), torch.from_numpy(np.stack(B))
            if self.human:
                A, B, P = self._human(hrng)
                if A:
                    out["h_a"], out["h_b"] = torch.from_numpy(np.stack(A)), torch.from_numpy(np.stack(B))
                    out["h_p"] = torch.tensor(P, dtype=torch.float32)
            yield out


def bt_loss(win_logit, lose_logit, margin=0.5):
    """Pairwise (Bradley-Terry style) loss: the better window should score at least `margin` higher."""
    return F.softplus(margin - (win_logit - lose_logit)).mean()


def compute_loss(model, b, W):
    """Total training loss of one batch and its parts."""
    parts, tot = {}, torch.tensor(0.0)
    if "x" in b:
        logit = model(b["x"].view(-1, b["x"].shape[-1]))["logit"].view(-1, 5)
        ca, d1, d2, d3, cb = logit.unbind(1)
        rank = (bt_loss(ca, d1) + bt_loss(d1, d2) + bt_loss(d2, d3) + bt_loss(ca, d3, margin=1.0)).mean() / 4
        inv = ((ca - cb) ** 2).mean()
        tot = tot + W["rank"] * rank + W["inv"] * inv
        parts.update(rank=rank.item(), inv=inv.item())
    for key_w, key_l, wname, pname in (("rate_w", "rate_l", "rate_pair", "rate_pair"), ("gg", "gb", "real", "gs"),
                                       ("rg_w", "rg_l", "real", "robot")):
        if key_w in b:
            r = bt_loss(model(b[key_w])["logit"], model(b[key_l])["logit"])
            tot = tot + W[wname] * r
            parts[pname] = r.item()
    if "h_a" in b:
        d = model(b["h_a"])["logit"] - model(b["h_b"])["logit"]
        p = b["h_p"]
        r = (p * F.softplus(-d) + (1.0 - p) * F.softplus(d)).mean()
        tot = tot + W["real_human"] * r
        parts["human"] = r.item()
    return tot, parts


def keep_awake():
    """Ask Windows not to sleep while training (no effect on other systems)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def load_human(path, use_ties=True):
    """Train-split human pairs as [(clip_a, clip_b, p_a), ...]; empty if path is 'none'."""
    if not path or path.lower() == "none" or not os.path.exists(path):
        return []
    hp = pd.read_csv(path)
    hp = hp[hp.split == "train"]
    if not use_ties:
        hp = hp[hp.p_a != 0.5]
    return [(r.a, r.b, float(r.p_a)) for r in hp.itertuples()]


def main():
    """Train and save <out>/<name>_seed<seed>.pt (resumes from a checkpoint if interrupted)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--human_pairs", default=HUMAN_PAIRS, help="'none' = train without human answers")
    ap.add_argument("--out", default=MODELS)
    ap.add_argument("--groups", type=int, default=12)
    ap.add_argument("--gs_pairs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--human_weight", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=100)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    keep_awake()
    os.makedirs(a.out, exist_ok=True)
    human = load_human(a.human_pairs)
    log(f"train {a.name} seed={a.seed} steps={a.steps} human_train_pairs={len(human)}")
    ds = PairSampler(a, training_pools(), a.seed, human)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers, persistent_workers=a.workers > 0,
                                     prefetch_factor=3 if a.workers > 0 else None)
    model = QualityNet(n_cls=21, w=24, phys=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    W = dict(rank=1.0, inv=0.5, rate_pair=1.0, real=1.0, real_human=a.human_weight)
    ckpt = os.path.join(a.out, f"{a.name}_seed{a.seed}.ckpt")
    start, run = 0, {}
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start, run = s["step"], s["run"]
        log(f"resumed from step {start}")
    t0 = time.time()
    it = iter(dl)
    for step, b in enumerate(it, start=start):
        if step >= a.steps:
            break
        model.train()
        loss, parts = compute_loss(model, b, W)
        if not parts:
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k] = 0.98 * run.get(k, v) + 0.02 * v
        if step % 100 == 0:
            print(f"step {step} " + " ".join(f"{k}={v:.3f}" for k, v in run.items()) + f" | {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.ckpt_every == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, run=run), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    path = os.path.join(a.out, f"{a.name}_seed{a.seed}.pt")
    torch.save(dict(state=model.state_dict(), cfg=dict(n_cls=21, w=24, phys=False), args=vars(a)), path)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    log(f"saved {path} after {time.time() - t0:.0f}s")
    try:
        it._shutdown_workers()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
