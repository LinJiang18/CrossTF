"""Three learned fidelity metrics (lower is better), computed on data min-max scaled by
the real data's per-channel range:

    discriminative  |accuracy - 0.5| of a GRU classifier trained to separate real from
                    generated series, measured on a held-out 20%
    predictive      MAE of a GRU one-step-ahead predictor trained on the generated series
                    and tested on the real ones
    context_fid     Frechet distance between TS2Vec representations of the two sets

Both sets must have the same number of series. GRUs use a hidden width of 32 (the
TimeGAN reference uses n_features // 2, which is degenerate for univariate series).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error

# Native (non-cuDNN) RNN kernels, as used for the reported numbers.
torch.backends.cudnn.enabled = False

DEFAULT_HIDDEN = 32
DISC_ITERS = 2000
PRED_ITERS = 2000
BATCH = 128
TRAIN_FRAC = 0.8


class _GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1]).squeeze(-1)


class _GRUPredictor(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_features)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y)


def _split(n: int, seed: int):
    idx = np.random.RandomState(seed).permutation(n)
    k = int(n * TRAIN_FRAC)
    return idx[:k], idx[k:]


def _batches(n: int, size: int, gen: torch.Generator):
    return torch.randperm(n, generator=gen)[:size]


def discriminative_score(real: np.ndarray, gen: np.ndarray, device: str = 'cuda',
                         seed: int = 0, hidden: int = DEFAULT_HIDDEN,
                         iterations: int = DISC_ITERS) -> float:
    if real.shape != gen.shape:
        raise ValueError(f'real {real.shape} and gen {gen.shape} must match; balance first')
    torch.manual_seed(seed)
    tr, te = _split(real.shape[0], seed)
    R = torch.as_tensor(real, dtype=torch.float32, device=device)
    G = torch.as_tensor(gen, dtype=torch.float32, device=device)
    model = _GRUClassifier(real.shape[-1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    r_tr, g_tr = R[tr], G[tr]
    for _ in range(iterations):
        rb = r_tr[_batches(len(tr), BATCH, g)]
        gb = g_tr[_batches(len(tr), BATCH, g)]
        x = torch.cat([rb, gb])
        y = torch.cat([torch.ones(len(rb), device=device), torch.zeros(len(gb), device=device)])
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(R[te]), model(G[te])])
    y_true = np.r_[np.ones(len(te)), np.zeros(len(te))]
    y_hat = (logits.cpu().numpy() > 0).astype(float)
    return float(abs(accuracy_score(y_true, y_hat) - 0.5))


def predictive_score(real: np.ndarray, gen: np.ndarray, device: str = 'cuda',
                     seed: int = 0, hidden: int = DEFAULT_HIDDEN,
                     iterations: int = PRED_ITERS) -> float:
    torch.manual_seed(seed)
    G = torch.as_tensor(gen, dtype=torch.float32, device=device)
    R = torch.as_tensor(real, dtype=torch.float32, device=device)
    model = _GRUPredictor(real.shape[-1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.L1Loss()
    g = torch.Generator().manual_seed(seed)
    Xg, Yg = G[:, :-1], G[:, 1:]
    for _ in range(iterations):
        b = _batches(len(G), BATCH, g)
        opt.zero_grad()
        lossf(model(Xg[b]), Yg[b]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(R[:, :-1]).cpu().numpy()
    return float(mean_absolute_error(R[:, 1:].cpu().numpy().ravel(), pred.ravel()))


def minmax(real: np.ndarray, gen: np.ndarray):
    """Scale both sets to [0, 1] by the REAL data's per-channel range."""
    lo = real.reshape(-1, real.shape[-1]).min(axis=0)
    hi = real.reshape(-1, real.shape[-1]).max(axis=0)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return ((real - lo) / rng).astype(np.float32), ((gen - lo) / rng).astype(np.float32)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    import scipy.linalg
    mu1, s1 = a.mean(axis=0), np.cov(a, rowvar=False)
    mu2, s2 = b.mean(axis=0), np.cov(b, rowvar=False)
    covmean = scipy.linalg.sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2.0 * covmean))


def fit_encoder(real01: np.ndarray):
    """TS2Vec fitted on the (scaled) real data; one encoder scores every generated set."""
    from evaluation.ts2vec.ts2vec import TS2Vec
    torch.manual_seed(0)
    np.random.seed(0)
    m = TS2Vec(input_dims=real01.shape[-1], device=0, batch_size=8, lr=0.001,
               output_dims=320, max_train_length=3000)
    m.fit(real01, verbose=False)
    return m, m.encode(real01, encoding_window='full_series')


def score_expensive(real: np.ndarray, gen: np.ndarray, device: str = 'cuda',
                    seed: int = 0, encoder=None, real_rep=None) -> dict[str, float]:
    if real.shape[0] != gen.shape[0]:
        raise ValueError(f'real and gen must be balanced, got {real.shape[0]} vs {gen.shape[0]}')
    r, g = minmax(real, gen)
    if encoder is None:
        encoder, real_rep = fit_encoder(r)
    return {
        'discriminative': discriminative_score(r, g, device, seed),
        'predictive': predictive_score(r, g, device, seed),
        'context_fid': frechet(real_rep, encoder.encode(g, encoding_window='full_series')),
    }
