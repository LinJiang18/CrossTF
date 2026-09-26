"""Copy one run's generated set from main.py's OUTPUT/ to the shared outputs/ layout.

main.py writes `OUTPUT/<run_name>/generate_data_unnorm_<run_name>.npy` (original units).
This script checks its shape against the real dataset and saves it, as float32, to
`common.generated_path(method, dataset, seed)`. No rescaling is done: the `_unnorm` file
is already in the original units.

    python collect.py --method E4GEN --dataset precipitation --run_name precipitation_E4GEN
    python collect.py --method E4GEN --dataset precipitation --run_name precipitation_E4GEN_s2 --seed 2
    python collect.py --method HeavyDiff --dataset precipitation --run_name precipitation_heavydiff
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--method', required=True, choices=['E4GEN', 'HeavyDiff'])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--run_name', required=True, help='the --name given to main.py')
    p.add_argument('--seed', type=int, default=None,
                   help='output subfolder seed<N>; omit for the flat layout (seed 1 / single run)')
    ns = p.parse_args()

    src = HERE / 'OUTPUT' / ns.run_name / f'generate_data_unnorm_{ns.run_name}.npy'
    if not src.is_file():
        sys.exit(f'missing {src.relative_to(HERE)}')
    gen = np.load(src).astype(np.float32)
    real = np.load(common.data_path(ns.dataset))
    if real.ndim == 2:
        real = real[..., None]
    if gen.shape != real.shape:
        sys.exit(f'shape mismatch: generated {gen.shape} vs real {real.shape}')

    out = common.generated_path(ns.method, ns.dataset, ns.seed)
    np.save(out, gen)
    print(f'{ns.method} {ns.dataset} seed={ns.seed}: {gen.shape} -> {out.relative_to(common.REPO_ROOT)}')


if __name__ == '__main__':
    main()
