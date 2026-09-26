"""Objective and constraint of the coefficient solve.

Objective: a hierarchical peaks-over-threshold (POT) divergence between the generated and
the real set, summed over the level side and the jump side:

    |xi_g - xi_r|  +  W1(NegBin_g, NegBin_r) / mean(n_r)  +  W1(Gamma_g, Gamma_r) / mean(m_r)

  xi    the GPD shape of the tail, fitted by maximum likelihood to the exceedances pooled
        over all series (a single series has too few exceedances to fit a shape);
  n_i   per series, the (soft) number of exceedances; the population is summarised by a
        negative binomial fitted by moments;
  m_i   per series, the mean excess over the threshold; summarised by a Gamma fitted by
        moments.

The thresholds are the 95% quantiles of the pooled real scores (see
`evaluation/extreme_quantities.py`) and are held fixed for every generated set. The
fitted-distribution summaries keep the objective smooth in the coefficients and distinct
from the reported metrics, which compare the raw per-series values with W1.

Constraint: the energy distance between the generated and real sets of whole series.
"""

from __future__ import annotations

import numpy as np

from evaluation import extreme_quantities as EQ

MIN_EXC = 200            # minimum pooled exceedances needed to fit a GPD
EPS = 1e-12


def _pwm_gpd(y: np.ndarray) -> tuple[float, float]:
    """Probability-weighted-moments fit of GPD(xi, sigma) (Hosking & Wallis, 1987).
    Used only as a fallback when the maximum-likelihood fit fails."""
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = len(y)
    a0 = y.mean()
    w = (n - np.arange(1, n + 1)) / (n - 1)
    a1 = float((w * y).mean())
    if a0 - 2 * a1 <= EPS:
        return 0.0, max(a0, EPS)
    xi = 2.0 - a0 / (a0 - 2 * a1)
    sigma = 2 * a0 * a1 / (a0 - 2 * a1)
    return float(np.clip(xi, -0.5, 0.95)), float(max(sigma, EPS))


def _mle_gpd(y):
    """Maximum-likelihood GPD fit with the location fixed at 0."""
    from scipy.stats import genpareto
    y = np.asarray(y, dtype=np.float64)
    try:
        xi, _, sigma = genpareto.fit(y, floc=0)
        if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError
        return float(np.clip(xi, -2.0, 0.95)), float(max(sigma, EPS))
    except Exception:
        return _pwm_gpd(y)


def _fit_side(scores, u):
    """(exceedance rate, xi, sigma) of one score matrix at threshold u, or None."""
    exc = scores[scores > u] - u
    if len(exc) < MIN_EXC:
        return None
    xi, sig = _mle_gpd(exc)
    return float((scores > u).mean()), xi, sig


def _mom_negbin(n):
    """Negative binomial (r, p) by moments; the Poisson limit when not over-dispersed."""
    m, v = float(np.mean(n)), float(np.var(n))
    if not np.isfinite(m) or not np.isfinite(v) or m <= EPS:
        return None
    if v <= m * (1.0 + 1e-6):
        return ('poisson', m)
    return ('nbinom', m * m / (v - m), m / v)


def _mom_gamma(x):
    """Gamma (shape, scale) by moments."""
    m, v = float(np.mean(x)), float(np.var(x))
    if not np.isfinite(m) or not np.isfinite(v) or m <= EPS or v <= EPS:
        return None
    return m * m / v, v / m


def _count_cdf(fit, k):
    from scipy.stats import nbinom, poisson
    return (poisson.cdf(k, fit[1]) if fit[0] == 'poisson'
            else nbinom.cdf(k, fit[1], fit[2]))


def _count_w1(fr, fg, scale):
    """W1 between two count distributions, sum_k |F_g(k) - F_r(k)|, divided by `scale`."""
    from scipy.stats import nbinom, poisson
    hi = []
    for f in (fr, fg):
        hi.append(poisson.ppf(0.9999, f[1]) if f[0] == 'poisson'
                  else nbinom.ppf(0.9999, f[1], f[2]))
    kmax = int(min(max(np.nanmax(hi), 1.0) * 1.5 + 10, 1e5))
    k = np.arange(0, kmax + 1)
    return float(np.sum(np.abs(_count_cdf(fg, k) - _count_cdf(fr, k)))) / max(scale, EPS)


_GAMMA_GRID = np.linspace(1e-3, 1.0 - 1e-3, 1024)


def _gamma_w1(fr, fg, scale):
    """W1 between two Gammas on a quantile grid, divided by `scale`."""
    from scipy.stats import gamma as _gamma
    qr = _gamma.ppf(_GAMMA_GRID, fr[0], scale=fr[1])
    qg = _gamma.ppf(_GAMMA_GRID, fg[0], scale=fg[1])
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(trapz(np.abs(qg - qr), _GAMMA_GRID)) / max(scale, EPS)


class HierRef:
    """The real side of the hierarchical POT divergence, built once per dataset."""

    def __init__(self, real: np.ndarray, tail: float = EQ.TAIL):
        self.u_lev, self.u_jmp = EQ.thresholds(real, tail)
        z, w = EQ.scores(real)
        self.fit = {}
        for name, s, u in (('lev', z, self.u_lev), ('jmp', w, self.u_jmp)):
            got = _fit_side(s, u)
            if got is None:
                raise ValueError(f'{name}: too few exceedances to fit')
            nb, ga, sn, sm = self._populations(real, name)
            if nb is None or ga is None:
                raise ValueError(f'{name}: cannot fit the real populations')
            self.fit[name] = dict(xi=got[1], nb=nb, ga=ga, scale_n=sn, scale_m=sm)

    def _populations(self, arr, name):
        """(NegBin fit, Gamma fit, mean count, mean excess) for one side."""
        z, w = EQ.scores(arr)
        s, u = (z, self.u_lev) if name == 'lev' else (w, self.u_jmp)
        per = EQ._side(s, u)
        n = np.asarray(per['rate'], float) * s.shape[1]      # soft count
        m = np.asarray(per['excess'], float)
        n, m = n[np.isfinite(n)], m[np.isfinite(m)]
        return _mom_negbin(n), _mom_gamma(m), float(np.mean(n)), float(np.mean(m))

    def divergence(self, gen: np.ndarray) -> float:
        """Sum over the two sides of |dxi| + W1(counts) + W1(mean excess). A side whose
        generated populations cannot be fitted is charged 10."""
        tot = 0.0
        for name in ('lev', 'jmp'):
            f = self.fit[name]
            nb, ga, _, _ = self._populations(gen, name)
            if nb is None or ga is None:
                tot += 10.0
                continue
            got = _fit_side(EQ.scores(gen)[0 if name == 'lev' else 1],
                            self.u_lev if name == 'lev' else self.u_jmp)
            dxi = 10.0 if got is None else abs(got[1] - f['xi'])
            tot += dxi + _count_w1(f['nb'], nb, f['scale_n']) \
                       + _gamma_w1(f['ga'], ga, f['scale_m'])
        return float(tot)


def energy_distance(x: np.ndarray, y: np.ndarray, sub: int = 384, draws: int = 3,
                    seed: int = 0) -> float:
    """Energy distance 2E||X-Y|| - E||X-X'|| - E||Y-Y'|| between two sets of series.

    Both sets are divided by the std of `y`. Each draw uses a random subsample of `sub`
    series per side and the draws are averaged; the self terms are U-statistics (the
    diagonal is excluded).
    """
    rng = np.random.default_rng(seed)
    x = x[:, :, 0] if x.ndim == 3 else x
    y = y[:, :, 0] if y.ndim == 3 else y
    sd = float(y.std()) or 1.0
    x, y = x.astype(np.float64) / sd, y.astype(np.float64) / sd

    def mean_pdist(u, v, same=False):
        tot, cnt = 0.0, 0
        for i in range(0, len(u), 128):
            d = np.linalg.norm(u[i:i + 128, None, :] - v[None, :, :], axis=2)
            tot += d.sum(); cnt += d.size
        if same:
            cnt -= len(u)
        return tot / max(cnt, 1)

    vals = []
    for _ in range(draws):
        a = x[rng.choice(len(x), min(sub, len(x)), replace=False)]
        b = y[rng.choice(len(y), min(sub, len(y)), replace=False)]
        vals.append(2 * mean_pdist(a, b) - mean_pdist(a, a, True) - mean_pdist(b, b, True))
    return float(np.mean(vals))
