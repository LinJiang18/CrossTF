"""The ten reported metrics for one dataset (all lower is better).

    lev_rate lev_excess jmp_rate jmp_excess    extreme fidelity (extreme_quantities.py)
    wasserstein acd marginal                   whole-series fidelity (trunk.py)
    discriminative predictive context_fid      learned fidelity (fidelity.py)

The real side is the whole dataset. A generated set must have the same length and number
of channels; if it has more series than the real data (SDForger), a random subset of the
real size is drawn (seed 0); a set with fewer series is skipped.

    # every generated set under results/ (released) and outputs/ (regenerated)
    python evaluation/evaluate.py --dataset ecg --all
    # specific files
    python evaluation/evaluate.py --dataset ecg --gen results/CrossTF/ecg/CrossTF_ecg4_generatedata.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import extreme_quantities as EQ          # noqa: E402
from evaluation.trunk import TrunkRef                    # noqa: E402

EXTREME = list(EQ.KEYS)
TRUNK = ['wasserstein', 'acd', 'marginal']
EXPENSIVE = ['discriminative', 'predictive', 'context_fid']
ALL = EXTREME + TRUNK + EXPENSIVE


def real_path(ds: str) -> Path:
    import yaml
    cfg = yaml.safe_load(open(ROOT / 'configs' / f'{ds}.yaml'))
    return (ROOT / cfg['data_path']).resolve()


def discover(ds: str) -> dict[str, Path]:
    """{label: file} for every generated set of `ds` under results/ and outputs/."""
    from evaluation.make_table import discover as _discover
    found: dict[str, Path] = {}
    for root in (ROOT / 'results', ROOT / 'outputs'):
        if root.is_dir():
            for method, files in _discover(root, ds).items():
                for f in files:
                    found[str(f.relative_to(ROOT))[:-len('_generatedata.npy')]] = f
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--gen', nargs='*', default=[], help='generated .npy files to score')
    ap.add_argument('--all', action='store_true', help='score every generated set found')
    ap.add_argument('--cheap-only', action='store_true',
                    help='skip the three learned metrics (no GPU needed)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None, help='write the scores as JSON here')
    ns = ap.parse_args()
    ds = ns.dataset

    cfgs = discover(ds) if ns.all else {}
    for g in ns.gen:
        cfgs[Path(g).stem.replace('_generatedata', '')] = Path(g)
    if not cfgs:
        raise SystemExit('nothing to score: pass --gen FILE ... or --all')

    rp = real_path(ds)
    real = np.load(rp).astype(np.float32)
    u_lev, u_jmp = EQ.thresholds(real)
    real_ext = EQ.per_series(real, u_lev, u_jmp)
    trunk_ref = TrunkRef(real)
    print(f'===== {ds}  real {real.shape} from {rp.name}   u_lev={u_lev:.4g}  u_jmp={u_jmp:.4g}')

    rows: dict[str, dict[str, float]] = {}
    enc = rep = None
    for label, gp in cfgs.items():
        gen = np.load(gp).astype(np.float32)
        if gen.shape[1:] != real.shape[1:]:
            print(f'  {label}: SKIPPED, shape {gen.shape} vs real {real.shape}')
            continue
        if gen.shape[0] != real.shape[0]:
            if gen.shape[0] < real.shape[0]:
                print(f'  {label}: SKIPPED, {gen.shape[0]} < {real.shape[0]} series')
                continue
            keep = np.random.RandomState(0).choice(gen.shape[0], real.shape[0], replace=False)
            gen = gen[np.sort(keep)]
        t0 = time.time()
        g_ext = EQ.per_series(gen, u_lev, u_jmp)
        row = {q: EQ.distance(g_ext[q], real_ext[q], wasserstein_distance) for q in EXTREME}
        row.update(trunk_ref.score(gen))
        if not ns.cheap_only:
            from evaluation import fidelity as F
            if enc is None:
                r01, _ = F.minmax(real, gen)
                enc, rep = F.fit_encoder(r01)
            row.update(F.score_expensive(real, gen, device=ns.device, encoder=enc, real_rep=rep))
        rows[label] = row
        print(f'  {label}  ({time.time() - t0:.1f}s)', flush=True)

    cols = [c for c in ALL if any(c in r for r in rows.values())]
    w = max(len(l) for l in rows) + 2 if rows else 10
    print('\n' + 'config'.ljust(w) + ''.join(f'{c[:12]:>13s}' for c in cols))
    for label, r in rows.items():
        print(label.ljust(w) + ''.join(
            f'{r[c]:>13.4f}' if c in r and np.isfinite(r[c]) else f'{"-":>13s}' for c in cols))
    if ns.out:
        Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({'dataset': ds, 'u_lev': u_lev, 'u_jmp': u_jmp,
                   'n': int(real.shape[0]), 'rows': rows}, open(ns.out, 'w'), indent=2)
        print(f'\nwrote {ns.out}')


if __name__ == '__main__':
    main()
