"""Three whole-series fidelity metrics (lower is better):

    wasserstein   W1 between the pooled generated and real values, / std of real values
    acd           mean |ACF_gen(l) - ACF_real(l)| over lags l = 1..50
    marginal      histogram density difference of the pooled values (50 bins)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance

MAX_LAG = 50


def acf_curve(arr: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Mean sample ACF at lags 1..L over all series and channels -> [L]."""
    n, t, c = arr.shape
    L = min(max_lag, max(0, t - 2))
    if L < 1:
        return np.zeros(0)
    x = arr.transpose(0, 2, 1).reshape(-1, t).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    den = np.einsum('ij,ij->i', x, x)
    ok = den > 1e-12
    x, den = x[ok], den[ok]
    return np.array([np.mean(np.einsum('ij,ij->i', x[:, :-lag], x[:, lag:]) / den)
                     for lag in range(1, L + 1)])


def _histogram(x, n_bins):
    a, b = x.min().item(), x.max().item()
    delta = (b - a) / n_bins
    bins = torch.arange(a, b + 1e-8, step=delta)
    count = torch.histc(x, n_bins).float()
    count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss:
    """Marginal-distribution metric: mean absolute difference between the real
    histogram density and the generated density evaluated on the real bins."""

    def __init__(self, x_real: torch.Tensor, n_bins: int = 50):
        self.densities, self.locs, self.deltas = [], [], []
        for i in range(x_real.shape[2]):
            d, b = _histogram(x_real[..., i].reshape(-1, 1), n_bins)
            self.densities.append(d)
            self.deltas.append(b[1:2] - b[:1])
            self.locs.append(0.5 * (b[1:] + b[:-1]))

    def __call__(self, x_fake: torch.Tensor) -> torch.Tensor:
        loss = []
        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            half = self.deltas[i] / 2. - dist
            counter = ((half * (half >= 0.).float()) > 0.).float()
            density = counter.mean(0) / self.deltas[i]
            loss.append(torch.mean(torch.abs(density - self.densities[i]), 0))
        return torch.stack(loss).mean()


class TrunkRef:
    """Real side of the three whole-series metrics, built once per dataset."""

    def __init__(self, real: np.ndarray):
        self.flat = real.reshape(-1).astype(np.float64)
        self.flat_sd = float(self.flat.std())
        self.acf = acf_curve(real)
        self.histo = HistoLoss(torch.as_tensor(real, dtype=torch.float32), n_bins=50)

    def score(self, gen: np.ndarray) -> dict[str, float]:
        gen_acf = acf_curve(gen)
        L = min(len(gen_acf), len(self.acf))
        return {'wasserstein': float(wasserstein_distance(
                    gen.reshape(-1).astype(np.float64), self.flat) / self.flat_sd),
                'acd': float(np.mean(np.abs(gen_acf[:L] - self.acf[:L]))) if L else np.nan,
                'marginal': float(self.histo(torch.as_tensor(gen, dtype=torch.float32)).item())}
