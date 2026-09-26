"""The four extreme-fidelity quantities.

Each series is scored on two sides:

    level score  z_t = |x_t - median(x)|          (N, T)
    jump score   w_t = |x_{t+1} - x_t|             (N, T-1)

Thresholds are the (1 - TAIL) quantiles of the pooled REAL scores and are held fixed for
every generated set, so the number of exceedances is free to differ from the real data:

    u_lev = quantile(all real z, 1 - TAIL),   u_jmp = quantile(all real w, 1 - TAIL)

Per series, two properties on each side:

    lev_rate      (soft) fraction of time steps with z > u_lev    how often it is extreme
    lev_excess    mean of z - u_lev over those steps (0 if none)  how far past u it goes
    jmp_rate      same on w and u_jmp
    jmp_excess

The reported metric for each quantity is the Wasserstein-1 distance between the N
generated and the N real per-series values, divided by the standard deviation of the
real values (`distance`). Lower is better. The same per-series quantities (through
`_side`) also enter the coefficient solve in `crosstf/pot.py`.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8
TAIL = 0.05
# Width of the logistic in the soft exceedance count, as a fraction of the threshold.
TAU_REL = 0.01
KEYS = ('lev_rate', 'lev_excess', 'jmp_rate', 'jmp_excess')


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def scores(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, T, 1) -> level scores (N, T) and jump scores (N, T-1), in the data's units."""
    x = arr[:, :, 0].astype(np.float64)
    return np.abs(x - np.median(x, axis=1, keepdims=True)), np.abs(np.diff(x, axis=1))


def thresholds(real: np.ndarray, tail: float = TAIL) -> tuple[float, float]:
    """(u_lev, u_jmp) from the pooled real scores."""
    z, w = scores(real)
    return float(np.quantile(z, 1.0 - tail)), float(np.quantile(w, 1.0 - tail))


def _side(score: np.ndarray, u: float) -> dict[str, np.ndarray]:
    """Per-series rate and excess of one score matrix at threshold u.

    The rate is a soft count (logistic in (score - u)), so that it varies smoothly with
    the data; the excess is 0 for a series with no exceedance, so no series is dropped.
    """
    mask = score > u
    n_exc = mask.sum(axis=1)
    out = {'rate': _sigmoid((score - u) / (TAU_REL * abs(u) + EPS)).mean(axis=1)}
    out['excess'] = np.where(mask, score - u, 0.0).sum(axis=1) / np.maximum(n_exc, 1)
    return out


def per_series(arr: np.ndarray, u_lev: float, u_jmp: float) -> dict[str, np.ndarray]:
    """One value per series for each of the four quantities."""
    z, w = scores(arr)
    out = {f'lev_{k}': v for k, v in _side(z, u_lev).items()}
    out.update({f'jmp_{k}': v for k, v in _side(w, u_jmp).items()})
    return {k: out[k] for k in KEYS}


def distance(gen_vals: np.ndarray, real_vals: np.ndarray, fn) -> float:
    """`fn`(generated values, real values) divided by the std of the real values."""
    g, r = gen_vals[np.isfinite(gen_vals)], real_vals[np.isfinite(real_vals)]
    sd = r.std()
    return float(fn(g, r) / sd) if g.size and r.size and sd > 1e-12 else np.nan
