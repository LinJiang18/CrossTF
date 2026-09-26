"""Per-dataset result tables: every method, seven metrics, averaged over training seeds.

    python evaluation/make_table.py                      # released samples in results/
    python evaluation/make_table.py --root outputs       # your own regenerated samples
    python evaluation/make_table.py --datasets ecg traffic --out tables/

Columns (all lower is better):

    Mag-Rate  Chg-Rate  Mag-Excess  Chg-Excess    extreme fidelity (extreme_quantities.py)
    Wass.     Marg.     ACD                       overall fidelity (trunk.py)

Mag-* are the level-side quantities (lev_rate, lev_excess), Chg-* the jump-side ones
(jmp_rate, jmp_excess). Each method's score is the mean over its training seeds; in each
column the best method is bold and the second best underlined (after rounding to four
decimals). A generated set with more series than the real data (SDForger) is compared on
a random subset of the real size (seed 0).

Generated sets are found under --root as
    <Method>/[seed<N>/]<dataset>/<Method>_<dataset>_generatedata.npy     (baselines)
    CrossTF/<dataset>/CrossTF_<dataset><seed>_generatedata.npy           (released CrossTF)
    CrossTF_<dataset><seed>/CrossTF_<dataset><seed>_generatedata.npy     (main.py output)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import extreme_quantities as EQ          # noqa: E402
from evaluation.trunk import TrunkRef                    # noqa: E402

DATASETS = ['building', 'ecg', 'electricity', 'precipitation', 'temperature', 'traffic',
            'transportation']
DISPLAY = {'building': 'Building', 'ecg': 'ECG', 'electricity': 'Electricity',
           'precipitation': 'Precipitation', 'temperature': 'Temperature',
           'traffic': 'Traffic', 'transportation': 'Transportation'}
COLUMNS = [('lev_rate', 'Mag-Rate'), ('jmp_rate', 'Chg-Rate'), ('lev_excess', 'Mag-Excess'),
           ('jmp_excess', 'Chg-Excess'), ('wasserstein', 'Wass.'), ('marginal', 'Marg.'),
           ('acd', 'ACD')]
GROUPS = [('GAN', ['TimeGAN']), ('VAE', ['KoVAE']), ('LLM', ['SDForger']),
          ('Diffusion', ['DiffTime', 'Diffusion-TS', 'FourierDiffusion', 'ImagenTime']),
          ('TF-Based', ['RF-Diffusion']), ('Extreme-Aware', ['FIDE', 'HeavyDiff', 'E4GEN']),
          ('Ours', ['CrossTF'])]
LABEL = {'FourierDiffusion': 'FourierDiff'}


def real_path(ds: str) -> Path:
    import yaml
    cfg = yaml.safe_load(open(ROOT / 'configs' / f'{ds}.yaml'))
    return (ROOT / cfg['data_path']).resolve()


def discover(root: Path, ds: str) -> dict[str, list[Path]]:
    """{method: [generated file per training seed]} for one dataset."""
    found: dict[str, set[Path]] = {}
    for f in root.rglob('*_generatedata.npy'):
        stem = f.name[:-len('_generatedata.npy')]
        if f.parent.name == ds and stem.endswith(f'_{ds}'):
            method = stem[:-len(f'_{ds}')]
        else:
            m = re.fullmatch(rf'(CrossTF)_{ds}\d+', stem)
            if not m:
                continue
            method = m.group(1)
        found.setdefault(method, set()).add(f.resolve())
    return {k: sorted(v) for k, v in found.items()}


def score_dataset(ds: str, root: Path) -> dict[str, dict]:
    real = np.load(real_path(ds)).astype(np.float32)
    u_lev, u_jmp = EQ.thresholds(real)
    real_ext = EQ.per_series(real, u_lev, u_jmp)
    ref = TrunkRef(real)
    out = {}
    for method, files in sorted(discover(root, ds).items()):
        per_seed = []
        for f in files:
            gen = np.load(f).astype(np.float32)
            if gen.shape[1:] != real.shape[1:] or gen.shape[0] < real.shape[0]:
                print(f'  skipping {f.relative_to(root)}: shape {gen.shape} vs {real.shape}')
                continue
            if gen.shape[0] > real.shape[0]:
                keep = np.random.RandomState(0).choice(gen.shape[0], real.shape[0], replace=False)
                gen = gen[np.sort(keep)]
            g_ext = EQ.per_series(gen, u_lev, u_jmp)
            row = {q: EQ.distance(g_ext[q], real_ext[q], wasserstein_distance) for q in EQ.KEYS}
            row.update(ref.score(gen))
            per_seed.append(row)
        if per_seed:
            out[method] = {'n_seeds': len(per_seed),
                           'mean': {k: float(np.mean([r[k] for r in per_seed])) for k, _ in COLUMNS},
                           'per_seed': per_seed}
    return out


def _ranks(scores: dict[str, dict]) -> dict[str, dict[str, int]]:
    """Rank (0 = best, 1 = second) of each method in each column, after rounding."""
    ranks: dict[str, dict[str, int]] = {m: {} for m in scores}
    for k, _ in COLUMNS:
        vals = sorted({round(s['mean'][k], 4) for s in scores.values()})
        for m, s in scores.items():
            ranks[m][k] = vals.index(round(s['mean'][k], 4))
    return ranks


def markdown(ds: str, scores: dict[str, dict]) -> str:
    rk = _ranks(scores)
    lines = [f'### {DISPLAY[ds]}', '',
             '| Type | Method | ' + ' | '.join(c for _, c in COLUMNS) + ' |',
             '|---|---|' + '---:|' * len(COLUMNS)]
    for group, methods in GROUPS:
        for m in methods:
            if m not in scores:
                continue
            cells = []
            for k, _ in COLUMNS:
                v = f'{scores[m]["mean"][k]:.4f}'
                cells.append(f'**{v}**' if rk[m][k] == 0 else f'<u>{v}</u>' if rk[m][k] == 1 else v)
            lines.append(f'| {group} | {LABEL.get(m, m)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def latex(ds: str, scores: dict[str, dict]) -> str:
    rk = _ranks(scores)
    rows = []
    for group, methods in GROUPS:
        present = [m for m in methods if m in scores]
        for i, m in enumerate(present):
            cells = []
            for k, _ in COLUMNS:
                v = f'{scores[m]["mean"][k]:.4f}'
                cells.append(rf'\textbf{{{v}}}' if rk[m][k] == 0
                             else rf'\underline{{{v}}}' if rk[m][k] == 1 else v)
            head = (rf'\multirow{{{len(present)}}}{{*}}{{{group}}}' if i == 0 else '')
            rows.append(f'  {head} & {LABEL.get(m, m)} & ' + ' & '.join(cells) + r' \\')
        if present:
            rows.append(r'  \midrule')
    rows[-1] = r'  \bottomrule'
    return '\n'.join([
        r'\begin{tabular}{ll ccccccc}', r'  \toprule',
        r'  & & \multicolumn{4}{c}{Extreme Fidelity} & \multicolumn{3}{c}{Overall Fidelity} \\',
        r'  \cmidrule(lr){3-6} \cmidrule(lr){7-9}',
        '  Type & Method & ' + ' & '.join(c for _, c in COLUMNS) + r' \\', r'  \midrule',
        *rows, r'\end{tabular}'])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=str(ROOT / 'results'),
                    help='directory holding the generated sets (default: results/)')
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--out', default=None,
                    help='also write <dataset>.md, <dataset>.tex and scores.json here')
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    out = Path(ns.out) if ns.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    everything = {}
    for ds in ns.datasets:
        scores = score_dataset(ds, root)
        if not scores:
            print(f'no generated sets for {ds} under {root}')
            continue
        everything[ds] = scores
        md = markdown(ds, scores)
        print(md + '\n')
        if out:
            (out / f'{ds}.md').write_text(md + '\n')
            (out / f'{ds}.tex').write_text(latex(ds, scores) + '\n')
    if out:
        json.dump(everything, open(out / 'scores.json', 'w'), indent=1)
        print(f'wrote {out}/')


if __name__ == '__main__':
    main()
