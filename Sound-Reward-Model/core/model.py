"""Neural network that scores 0.5 s cello audio windows, plus the 12 acoustic features used to explain scores."""
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from common import SR, MODELS

N_FFT, HOP, N_MELS = 1024, 256, 64
_MEL = torch.tensor(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=30.0, fmax=10000.0), dtype=torch.float32)
_WIN = torch.hann_window(N_FFT)
_FREQS = torch.linspace(0, SR / 2, N_FFT // 2 + 1)
N_PHYS = 12


def _stft_power(x):
    """Peak-normalise each waveform and return it with its power spectrogram."""
    x = x / (x.abs().amax(dim=1, keepdim=True) + 1e-9)
    S = torch.stft(x, N_FFT, HOP, window=_WIN, return_complex=True, center=True).abs() ** 2
    return x, S


def _logmel(S):
    """Power spectrogram -> normalised log-mel image (B, 1, 64, T)."""
    M = torch.matmul(_MEL, S)
    db = 10 * torch.log10(M.clamp_min(1e-8))
    db = db - db.amax(dim=(1, 2), keepdim=True)
    db = db.clamp_min(-80.0)
    return (db / 40.0 + 1.0).unsqueeze(1)


def waveform_to_logmel(x):
    """Waveforms (B, N) -> the log-mel image the network sees."""
    return _logmel(_stft_power(x)[1])


def _autocorr_r(x, lo=22, hi=367):
    """Peak normalised autocorrelation in the cello pitch range (used for HNR)."""
    B, N = x.shape
    hi = min(hi, N - 2)
    if hi <= lo + 2:
        return torch.full((B,), 0.5)
    n_fft = 1 << (2 * N - 1).bit_length()
    X = torch.fft.rfft(x - x.mean(1, keepdim=True), n_fft)
    ac = torch.fft.irfft(X.abs() ** 2, n_fft)[:, :N]
    ac = ac / (ac[:, :1] + 1e-9)
    return ac[:, lo:hi].amax(1).clamp(1e-3, 0.999)


def phys_features(x, S):
    """12 loudness-independent acoustic features per waveform (B, 12); names in PHYS_NAMES."""
    B, N = x.shape
    r = _autocorr_r(x)
    hnr = 10 * torch.log10(r / (1 - r))
    if N >= 4500:
        k = N // 3
        rs = torch.stack([_autocorr_r(x[:, i * k:(i + 1) * k]) for i in range(3)], 1)
        hnr_min = 10 * torch.log10(rs.amin(1) / (1 - rs.amin(1)))
    else:
        hnr_min = hnr
    e = S.sum(1).clamp_min(1e-12).sqrt()
    en = e / (e.amax(1, keepdim=True) + 1e-9)
    T = en.shape[1]
    a, b = int(0.1 * T), max(int(0.9 * T), int(0.1 * T) + 1)
    core = en[:, a:b] if b - a >= 2 else en
    cv = core.std(1) / (core.mean(1) + 1e-6)
    sm = F.avg_pool1d(core.unsqueeze(1), 3, 1, 1).squeeze(1) if core.shape[1] >= 3 else core
    dip = sm.amin(1) / (sm.median(1).values + 1e-6)
    third = max(1, T // 3)
    slope = torch.log((en[:, -third:].mean(1) + 1e-6) / (en[:, :third].mean(1) + 1e-6))
    P = S.clamp_min(1e-12)
    cen = (P * _FREQS.view(1, -1, 1)).sum(1) / P.sum(1)
    cen_m, cen_s = cen.mean(1) / 3000.0, cen.std(1) / 1500.0
    band = (_FREQS > 100) & (_FREQS < 8000)
    Pb = P[:, band]
    flat = (torch.exp(torch.log(Pb).mean(1)) / Pb.mean(1)).mean(1)
    high = (P[:, _FREQS > 4000].sum(1) / P.sum(1)).mean(1)
    return torch.stack([hnr / 10, hnr_min / 10, cv, dip, slope, cen_m, cen_s, flat, high.sqrt(), en.std(1),
                        torch.log10(flat + 1e-6), (hnr - hnr_min) / 10], 1)


PHYS_NAMES = ["hnr", "hnr_min (worst third)", "energy steadiness (cv)", "dynamics dip", "energy slope (end vs start)",
              "brightness (centroid mean)", "brightness steadiness (centroid std)", "spectral flatness (noisiness)",
              "high-frequency energy (harshness)", "envelope roughness", "log flatness", "tone consistency (hnr - hnr_min)"]


def clip_features(y):
    """The 12 acoustic features of a whole clip as a numpy array."""
    y = np.asarray(y, np.float32)
    if len(y) < 2205:
        y = np.pad(y, (0, 2205 - len(y)))
    xn, S = _stft_power(torch.from_numpy(y).unsqueeze(0))
    return phys_features(xn, S).nan_to_num(0.0)[0].numpy()


class Block(nn.Module):
    """Conv2d -> BatchNorm -> ReLU (x1 or x2), then halve the frequency axis."""

    def __init__(self, cin, cout, two=False):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = nn.BatchNorm2d(cout)
        self.two = two
        if two:
            self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.b2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        """Run the block."""
        x = F.relu(self.b1(self.c1(x)))
        if self.two:
            x = F.relu(self.b2(self.c2(x)))
        return F.max_pool2d(x, (2, 1))


class TBlock(nn.Module):
    """Residual dilated 1-D convolution over time."""

    def __init__(self, c, dil):
        super().__init__()
        self.c = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.b = nn.BatchNorm1d(c)

    def forward(self, x):
        """Run the block."""
        return x + F.relu(self.b(self.c(x)))


class QualityNet(nn.Module):
    """CNN + temporal convolutions + attention pooling; one quality score ('logit') per window."""

    def __init__(self, n_cls=21, n_rate=4, w=24, d=128, drop=0.2, phys=False):
        super().__init__()
        self.blocks = nn.Sequential(Block(1, w), Block(w, 2 * w), Block(2 * w, 3 * w), Block(3 * w, 4 * w))
        self.proj = nn.Conv1d(4 * w * 4, d, 1)
        self.tcn = nn.Sequential(TBlock(d, 1), TBlock(d, 2), TBlock(d, 4))
        self.drop = nn.Dropout(drop)
        self.frame = nn.Conv1d(d, 1, 1)
        self.attn = nn.Conv1d(d, 1, 1)
        extra = 32 if phys else 0
        self.cls = nn.Linear(2 * d + extra, n_cls)
        self.rate = nn.Linear(2 * d + extra, n_rate)
        self.d = d
        self.phys = phys
        if phys:
            self.pbn = nn.BatchNorm1d(N_PHYS)
            self.pmlp = nn.Sequential(nn.Linear(N_PHYS, 32), nn.ReLU())
            self.plogit = nn.Linear(32, 1)

    def features(self, S):
        """Power spectrogram -> per-frame hidden features (B, d, T)."""
        x = self.blocks(_logmel(S))
        B, C, Fq, T = x.shape
        h = F.relu(self.proj(x.reshape(B, C * Fq, T)))
        return self.tcn(h)

    def forward(self, wav):
        """Waveforms (B, N) -> dict with 'logit' (score), 'attn', 'frame' and auxiliary heads."""
        xn, S = _stft_power(wav)
        h = self.features(S)
        fl = self.frame(h)[:, 0]
        a = torch.softmax(self.attn(h)[:, 0], dim=1)
        logit = (a * fl).sum(1)
        pooled = torch.cat([(h * a.unsqueeze(1)).sum(2), h.mean(2)], dim=1)
        if self.phys:
            pf = self.pmlp(self.pbn(phys_features(xn, S).nan_to_num(0.0)))
            logit = logit + self.plogit(pf)[:, 0]
            pooled = torch.cat([pooled, pf], dim=1)
        pooled = self.drop(pooled)
        return dict(logit=logit, frame=fl, attn=a, cls=self.cls(pooled), rate=self.rate(pooled), emb=pooled)


def clip_windows(y, win=0.5, hop=0.25, sr=SR):
    """Cut a clip into 0.5 s windows with 0.25 s hop (short clips stay one window)."""
    n, h = int(win * sr), int(hop * sr)
    if len(y) <= n:
        return [y]
    out = [y[s:s + n] for s in range(0, len(y) - n + 1, h)]
    if (len(y) - n) % h > 0.3 * n:
        out.append(y[-n:])
    return out


def load_ckpts(name, model_dir=None):
    """All checkpoint files saved under one model name."""
    return sorted(glob.glob(os.path.join(model_dir or MODELS, f"{name}_seed*.pt")))


def load_net(path):
    """Load one trained QualityNet from a checkpoint file."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = QualityNet(**ck["cfg"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


class NetScorer:
    """Scores audio windows with one model or the average of several models."""

    def __init__(self, names, model_dir=None, threads=4):
        paths = [p for n in names for p in load_ckpts(n, model_dir)]
        if not paths:
            raise FileNotFoundError(f"no checkpoints for {names} in {model_dir or MODELS}")
        self.name = "+".join(names)
        self.threads = threads
        self.models = [load_net(p) for p in paths]

    @torch.no_grad()
    def score_windows_each(self, wins, bs=256):
        """Score of every window from every model, shape (n_models, n_windows)."""
        torch.set_num_threads(self.threads)
        lens = np.array([len(w) for w in wins])
        logit = np.zeros((len(self.models), len(wins)), np.float32)
        for L in np.unique(lens):
            idx = np.where(lens == L)[0]
            for s in range(0, len(idx), bs):
                b = idx[s:s + bs]
                x = torch.from_numpy(np.stack([np.asarray(wins[j], np.float32) for j in b]))
                for k, m in enumerate(self.models):
                    logit[k, b] = m(x)["logit"].numpy()
        return logit

    def score_windows(self, wins):
        """Model-averaged score of every window."""
        return self.score_windows_each(wins).mean(0)


def clip_scores(scorer, clips):
    """Score of each whole clip = mean score of its 0.5 s windows."""
    wins, owner = [], []
    for i, y in enumerate(clips):
        ws = clip_windows(np.asarray(y, np.float32))
        wins += ws
        owner += [i] * len(ws)
    out = np.asarray(scorer.score_windows(wins), np.float64)
    owner = np.array(owner)
    return np.array([out[owner == i].mean() for i in range(len(clips))])
