"""Reproduce Table 1: extreme-sensitive cross-view asymmetry across seven datasets.

    python rebuttal/reproduce_Table1.py                          # Table 1 (GPU)
    python rebuttal/reproduce_Table1.py --datasets ecg traffic   # a subset

Every number is recomputed from the released time and frequency branches
(rebuttal/pretrained/{Time,Frequency}_<dataset><seed>, training seeds 4 and 5) on one
GPU, about 15 minutes for all seven datasets. Per-dataset results are cached in
rebuttal/outputs/Table1/<dataset>.json; a rerun computes only the alphas the cache lacks
(--force recomputes everything).

What is measured
----------------
One global fusion coefficient alpha is applied at every step of the fusion window
(sigma <= 0.35, steps 12..17 of the 18-step EDM schedule); before the window the time
branch runs alone. alpha = 0 is the pure frequency branch, alpha = 1 the pure time branch
(Time-Only). Every alpha starts from the same noise and prefix (sampling seed 101), so
two alphas differ only by alpha. Table 1 summarises each quantity q by its relative
max-min range over alpha in {0, 0.25, 0.5, 0.75, 0.9, 1}:

    range(q) = (max_a q(a) - min_a q(a)) / |q(a = 1)|

    Loss      one-step denoising loss of the fused denoiser,
              L(a) = E||r_F + a (r_T - r_F)||^2 = A + 2aB + a^2 C,
              where r_T, r_F are the reconstruction errors D(x0 + sigma*eps) - x0 of
              the time and frequency branches on the real (normalised) data, one shared
              noise draw. Range per (training seed, window step), then averaged.
    Extreme   the four extreme quantities (lev_rate, lev_excess, jmp_rate, jmp_excess;
              evaluation/extreme_quantities.py), each the W1 distance to the real data
              divided by the std of the real values. Range averaged over 4 metrics x 2
              training seeds.
    Overall   Wasserstein distance to the real data (evaluation/trunk.py), same
              construction, averaged over the 2 training seeds.

Ext./Loss and Ext./Overall are per-dataset ratios of the unrounded ranges. The Median
row takes the column medians of the three ranges, rounded to the printed precision
(0.1%), and divides them, as in the paper.

--raw (needs evaluation/prior_metrics, which is not part of this release) additionally
prints, per dataset, the raw value of every metric at five fusion
coefficients, each ONE alpha shared by all metrics: Frequency-Only (alpha = 0);
Time-Only (alpha = 1); the best and the worst alpha of the grid 0, step, ..., 1.5
(--step, default 0.25) by Extreme (gmean), the geometric mean of our four extreme
quantities (SELECT); and alpha*, the single alpha our solver selects (alpha_single in
results/CrossTF/<dataset>/alpha.json). Every value carries its change relative to
Time-Only. Metrics:
the geometric mean and each of our four extreme quantities, the three overall metrics
(Wasserstein, ACD, marginal), and the nine extreme metrics of E4GEN
(evaluation/prior_metrics). Each value is the mean over the two training seeds; lower is
better for every metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                # the CrossTF repository
CACHE = ROOT / 'rebuttal' / 'outputs' / 'Table1'
PRETRAINED = ROOT / 'rebuttal' / 'pretrained'

DATASETS = {'traffic': 'Traffic', 'transportation': 'Transportation',
            'temperature': 'Temperature', 'electricity': 'Electricity',
            'precipitation': 'Precipitation', 'ecg': 'ECG', 'building': 'Building'}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]                 # Table 1: alpha in [0, 1]
ASTAR = 'alpha_star'                                      # key of the set at our alpha*
SELECT = 'extreme_gmean'          # --raw: metric that picks the best / worst grid alpha
PRIOR = ('E4GEN',)                # --raw: prior-work metric families computed and shown
EXTREME = ['lev_rate', 'lev_excess', 'jmp_rate', 'jmp_excess']
OVERALL = ['wasserstein', 'acd', 'marginal']
NAMES = {'extreme_gmean': 'Extreme (gmean)', 'lev_rate': 'Mag-Rate',
         'lev_excess': 'Mag-Excess', 'jmp_rate': 'Chg-Rate', 'jmp_excess': 'Chg-Excess',
         'wasserstein': 'Wass.', 'acd': 'ACD', 'marginal': 'Marg.'}
WINDOW = 0.35                                             # fusion window: sigma <= 0.35
TRAIN_SEEDS = [4, 5]
SAMPLE_SEED = 101
MAX_SERIES = 512                                          # real series used for the loss
BATCH = 128
CACHE_VERSION = 3


# ---------------------------------------------------------------------------
# Recomputation from the released branches (GPU)
# ---------------------------------------------------------------------------

def select_grid(step: float) -> list[float]:
    """The --raw grid: 0, step, 2 step, ..., 1.5."""
    n = int(round(1.5 / step))
    if n < 1 or abs(n * step - 1.5) > 1e-9:
        raise SystemExit(f'--step {step} must divide 1.5')
    return [round(k * step, 6) for k in range(n + 1)]


def key_of(a: float) -> str:
    """Cache key of one alpha ('0.25', '1.0', ...)."""
    return str(float(round(a, 6)))


def recompute(ds: str, device: str, prior: bool, alphas: list[float], old: dict | None,
              path: Path) -> dict:
    """Loss coefficients and metric values along alpha for one dataset. Only what `old`
    (the cached result) lacks is computed; the cache is rewritten as work completes."""
    a_star = float(json.load(open(ROOT / 'results' / 'CrossTF' / ds / 'alpha.json'))
                   ['alpha_single'])
    d = old or {'version': CACHE_VERSION, 'dataset': ds, 'seeds': TRAIN_SEEDS,
                'sample_seed': SAMPLE_SEED, 'alpha_star': a_star, 'curves': {},
                'per_seed': None}
    curves = d['curves']
    metrics = ['extreme_gmean']
    if prior:
        from evaluation import prior_metrics as PM
        prior_keys = [k for f in PRIOR for k in PM.FAMILIES[f]]
        metrics += prior_keys

    def done(key, ms):
        return all(key in curves.get(m, {}).get(str(sd), {}) for m in ms for sd in TRAIN_SEEDS)

    wanted = [(key_of(a), a) for a in alphas] + [(ASTAR, a_star)]
    todo = [(k, a) for k, a in wanted if not done(k, metrics)]
    if d['per_seed'] and not todo:
        print(f'  [{ds}] using cached {path.relative_to(ROOT)}', flush=True)
        return d

    def save():
        d['alphas'] = sorted({float(k) for k in curves.get('extreme_gmean', {}).get('4', {})
                              if k != ASTAR})
        json.dump(d, open(path, 'w'), indent=1)

    import torch

    from crosstf.branches import load_pair
    from crosstf.evaluator import Evaluator
    from crosstf.fusion import SIGMAS
    from evaluation import extreme_quantities as EQ
    from evaluation.trunk import TrunkRef
    from scipy.stats import wasserstein_distance
    from utils.utils_data import load_npy_dataset

    torch.backends.cudnn.enabled = True
    cfg = str(ROOT / 'configs' / f'{ds}.yaml')
    t0 = time.time()
    print(f'  [{ds}] {len(todo)} alpha sets to compute', flush=True)

    # 1. Loss coefficients A, B, C per (training seed, window step).
    if not d['per_seed']:
        d['per_seed'] = loss_coefficients(ds, cfg, device, load_pair, load_npy_dataset,
                                          SIGMAS)
        save()
        print(f'  [{ds}] loss coefficients done ({time.time() - t0:.0f}s)', flush=True)
    if not todo:
        return d

    # 2. Our metrics of the generated sets (regenerated for every alpha still missing
    # something; generation is deterministic, so a set is the same on every run).
    from crosstf.branches import branch_args
    data_path = ROOT / Path(branch_args(cfg, ds, TRAIN_SEEDS[0], 'time', device).data_path)
    real = np.load(data_path).astype(np.float32)
    u_lev, u_jmp = EQ.thresholds(real)
    real_ext = EQ.per_series(real, u_lev, u_jmp)
    tref = TrunkRef(real)

    ev = Evaluator(cfg, ds, TRAIN_SEEDS, [SAMPLE_SEED], BATCH, device, WINDOW, str(PRETRAINED))
    ev.build_prefix()
    generated = {}

    def put(row, sd, key):
        for k, v in row.items():
            curves.setdefault(k, {}).setdefault(str(sd), {})[key] = float(v)

    for key, al in todo:
        for sd in TRAIN_SEEDS:
            g = ev.generate(sd, {i: al for i in ev.window}, SAMPLE_SEED)
            ge = EQ.per_series(g, u_lev, u_jmp)
            row = {k: EQ.distance(ge[k], real_ext[k], wasserstein_distance) for k in EXTREME}
            row['extreme_gmean'] = float(np.exp(np.mean(np.log([row[k] for k in EXTREME]))))
            row.update(tref.score(g))
            put(row, sd, key)
            if prior:                   # kept only for the prior-work metrics below
                generated[(key, sd)] = g
        print(f'  [{ds}] alpha {key} generated and scored ({time.time() - t0:.0f}s)', flush=True)
    del ev
    torch.cuda.empty_cache()
    save()

    # 3. Prior-work metrics, after all sets exist: they train small networks, which must
    # not touch the global state the sampler runs under.
    if prior:
        real64 = np.load(data_path).astype(np.float64)
        for (key, sd), g in generated.items():
            if done(key, prior_keys):
                continue
            put(PM.score(real64, g, ds, device, families=PRIOR), sd, key)
            save()
            print(f'  [{ds}] prior metrics alpha {key} seed {sd} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return d


def loss_coefficients(ds, cfg, device, load_pair, load_npy_dataset, SIGMAS) -> dict:
    """A, B, C of the fused one-step loss per (training seed, window step)."""
    import torch
    per_seed = {}
    for sd in TRAIN_SEEDS:
        mt, mf, a = load_pair(cfg, ds, sd, device, str(PRETRAINED))
        norm, _, _, _, _ = load_npy_dataset(a.data_path, seed=sd)
        x0 = torch.as_tensor(norm, dtype=torch.float32)
        if x0.shape[0] > MAX_SERIES:
            x0 = x0[np.random.default_rng(0).choice(x0.shape[0], MAX_SERIES, replace=False)]
        x0 = x0.to(device)
        noise = torch.randn(x0.shape, device=device,
                            generator=torch.Generator(device=device).manual_seed(sd))
        rows = {}
        for st, sig in enumerate(SIGMAS):
            if sig > WINDOW:
                continue
            rT = _errors(mt, x0, sig, noise).reshape(x0.shape[0], -1)
            rF = _errors(mf, x0, sig, noise).reshape(x0.shape[0], -1)
            u = rT - rF
            A = float((rF * rF).sum(1).mean())
            B = float((rF * u).sum(1).mean())
            C = float((u * u).sum(1).mean())
            rows[str(st)] = {'sigma': float(sig), 'A': A, 'B': B, 'C': C, 'L1': A + 2 * B + C}
        per_seed[str(sd)] = rows
        del mt, mf
        torch.cuda.empty_cache()
    return per_seed


def _errors(model, x0, sigma, noise):
    """Reconstruction error D(x0 + sigma * eps) - x0 of one branch (EMA weights)."""
    import torch
    outs = []
    with torch.no_grad(), model.ema_scope():
        model.eval()
        s = torch.as_tensor(float(sigma), device=x0.device, dtype=x0.dtype)
        for i in range(0, x0.shape[0], BATCH):
            xb, nb = x0[i:i + BATCH], noise[i:i + BATCH]
            outs.append((model.net(xb + float(sigma) * nb, s).to(torch.float64) - xb).cpu())
    return torch.cat(outs)


# ---------------------------------------------------------------------------
# Table 1 from the curves
# ---------------------------------------------------------------------------

def _range(curve: dict) -> float:
    """Relative max-min range of one curve over ALPHAS, w.r.t. alpha = 1."""
    v = np.array([curve[str(a)] for a in ALPHAS], dtype=float)
    return float((v.max() - v.min()) / abs(curve['1.0']))


def metric_range(curves: dict, keys: list[str], seeds) -> float:
    """Mean range over the given metrics x training seeds."""
    return float(np.mean([_range(curves[k][str(s)]) for k in keys for s in seeds]))


def loss_range(per_seed: dict) -> float:
    """Mean range of L(a) = A + 2aB + a^2 C over (seed, step) cells inside the window."""
    out = []
    for steps in per_seed.values():
        for r in steps.values():
            if r['sigma'] > WINDOW:
                continue
            L = np.array([r['A'] + 2 * a * r['B'] + a * a * r['C'] for a in ALPHAS])
            out.append((L.max() - L.min()) / r['L1'])
    return float(np.mean(out))


def print_table(results: dict) -> None:
    rows = []
    for ds, d in results.items():
        loss = loss_range(d['per_seed'])
        ext = metric_range(d['curves'], EXTREME, d['seeds'])
        ovr = metric_range(d['curves'], ['wasserstein'], d['seeds'])
        rows.append((DATASETS[ds], loss, ovr, ext, ext / loss, ext / ovr))

    # Median row: medians rounded to the printed 0.1%, then divided (paper convention)
    med = [round(100 * float(np.median([r[i] for r in rows])), 1) for i in (1, 2, 3)]
    med_row = ('Median', med[0] / 100, med[1] / 100, med[2] / 100,
               med[2] / med[0], med[2] / med[1])

    head = '%-15s %8s %9s %9s | %10s %13s' % (
        'Dataset', 'Loss', 'Overall', 'Extreme', 'Ext./Loss', 'Ext./Overall')
    rule = '-' * len(head)
    fmt = '%-15s %7.1f%% %8.1f%% %8.1f%% | %9.2fx %12.2fx'
    print('\nTable 1: Extreme-sensitive cross-view asymmetry (alpha in [0, 1])')
    print('Relative max-min range: Loss, Overall, Extreme; Ratio: Ext./Loss, Ext./Overall')
    print(rule); print(head); print(rule)
    for r in rows:
        print(fmt % (r[0], 100 * r[1], 100 * r[2], 100 * r[3], r[4], r[5]))
    print(rule)
    print(fmt % (med_row[0], 100 * med_row[1], 100 * med_row[2], 100 * med_row[3],
                 med_row[4], med_row[5]))
    print(rule)


# ---------------------------------------------------------------------------
# Raw values along alpha (--raw)
# ---------------------------------------------------------------------------

def _seed_mean(curves: dict, k: str, key: str) -> float:
    """Mean over training seeds, skipping a seed where the metric is undefined (NaN)."""
    v = np.array([curves[k][s][key] for s in curves[k]], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')


def _cell(v: float, ref: float) -> str:
    """Raw value and its change vs Time-Only."""
    if not np.isfinite(v):
        return 'n/a'
    s = '%.4g' % v
    if np.isfinite(ref) and ref != 0:
        s += ' [%+.1f%%]' % (100 * (v - ref) / abs(ref))
    return s


def raw_tables(results: dict, grid: list[float]) -> str:
    """Per dataset, every metric at five fusion coefficients, each ONE alpha for all
    metrics: Frequency-Only, Time-Only, the best and the worst grid alpha by SELECT, and
    our alpha*."""
    from evaluation.prior_metrics import FAMILIES, NAMES as PRIOR_NAMES
    names = {**NAMES, **PRIOR_NAMES}
    groups = [('Ours', ['extreme_gmean'] + EXTREME), ('Ours (overall)', OVERALL)]
    groups += [(fam, FAMILIES[fam]) for fam in PRIOR]
    sel = names[SELECT]
    lines = ['# Raw metric values at five fusion coefficients', '',
             'One alpha shared by every step of the fusion window (sigma <= 0.35). Each '
             'column is ONE alpha for every metric: alpha = 0 (Frequency-Only), alpha = 1 '
             '(Time-Only), the grid alpha with the lowest / highest %s over alpha in '
             '{%s} (grid best / worst), and alpha*, the single alpha our solver selects. '
             'Mean of training seeds 4 and 5, sampling seed 101. Lower is better for every '
             'metric; brackets: change relative to Time-Only.'
             % (sel, ', '.join('%g' % a for a in grid)), '']
    for ds, d in results.items():
        c, a_star = d['curves'], d['alpha_star']
        g = {a: _seed_mean(c, SELECT, key_of(a)) for a in grid}
        best, worst = min(g, key=g.get), max(g, key=g.get)
        cols = [key_of(0.0), key_of(1.0), key_of(best), key_of(worst), ASTAR]
        lines += [f'## {DATASETS[ds]}', '',
                  '| Family | Metric | Frequency-Only (a=0) | Time-Only (a=1) '
                  f'| Grid best (a={best:g}) | Grid worst (a={worst:g}) '
                  f'| Ours (a*={a_star:.2f}) |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for fam, keys in groups:
            for k in keys:
                ref = _seed_mean(c, k, key_of(1.0))
                cells = [_cell(_seed_mean(c, k, col), float('nan') if col == key_of(1.0)
                               else ref) for col in cols]
                lines.append(f'| {fam} | {names[k]} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description='Reproduce Table 1 of the paper.')
    ap.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument('--raw', action='store_true',
                    help='also print the raw value of every metric along alpha, including '
                         'the extreme metrics of prior work')
    ap.add_argument('--step', type=float, default=0.25,
                    help='--raw: spacing of the alpha grid over [0, 1.5] (default 0.25); '
                         'alphas already in the cache are reused')
    ap.add_argument('--force', action='store_true', help='ignore cached results')
    ap.add_argument('--device', default='cuda')
    ns = ap.parse_args()
    if ns.raw:
        try:
            import evaluation.prior_metrics  # noqa: F401
        except ImportError:
            raise SystemExit('--raw needs evaluation/prior_metrics, which is not part of '
                             'this release; run without --raw for Table 1.')
    datasets = [d for d in DATASETS if d in ns.datasets]   # keep the paper's row order
    grid = select_grid(ns.step)
    alphas = sorted(set(ALPHAS) | set(grid)) if ns.raw else ALPHAS

    os.chdir(ROOT)                      # configs point at ./data/...
    sys.path.insert(0, str(ROOT))
    CACHE.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in datasets:
        p = CACHE / f'{ds}.json'
        old = None
        if p.exists() and not ns.force:
            old = json.load(open(p))
            if old.get('version') != CACHE_VERSION:
                old = None
        results[ds] = recompute(ds, ns.device, ns.raw, alphas, old, p)
    print_table(results)
    if ns.raw:
        text = raw_tables(results, grid)
        (CACHE / 'raw.md').write_text(text + '\n')
        print('\n' + text)
        print(f'(also written to {(CACHE / "raw.md").relative_to(ROOT)})')


if __name__ == '__main__':
    main()
