"""Convert the seven .npy datasets into the parquet layout the fms-dgt time_series tasks read.

For each dataset this writes `<DGT_DATA_DIR>/public/time_series/<npy stem>.parquet`, which is
the file referenced by `fms-dgt/tasks/public/time_series/<Stem>_multisample/task.yaml`.

Layout (the multisample convention of the fms-dgt time_series databuilder):

  * one column per trajectory, `TOY-0` .. `TOY-{N-1}`, each column holding that
    trajectory's T values -- so the frame is T rows x N columns, the transpose of the npy;
  * plus `TOY-{N}`, a copy of `TOY-0`. fms-dgt requires strictly more matching columns
    than `train_samples`, so without it `train_samples: N` is rejected;
  * raw values, no scaling. SDForger standard-scales internally and inverts before it
    returns, so the generated series come back in the original units.

`DGT_DATA_DIR` defaults to `baselines/SDForger/work/data`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # baselines/ -> common.py
import common  # noqa: E402

DEFAULT_WORK = HERE / 'work'


def task_stem(name: str) -> str:
    """Task directory stem, e.g. electricity -> Electricity (leading token of the npy name)."""
    return Path(common.DATASETS[name]).stem.split('_seq_')[0]


def write_parquet(npy: Path, out: Path, prefix: str = 'TOY-') -> tuple[int, int]:
    x = np.load(npy)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or x.shape[2] != 1:
        raise SystemExit(f'expected (N, T, 1), got {x.shape} from {npy.name}')
    n, t, _ = x.shape
    df = pd.DataFrame(x.reshape(n, t).T, columns=[f'{prefix}{i}' for i in range(n)])
    df[f'{prefix}{n}'] = df[f'{prefix}0'].copy()      # the required spare column
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return n, t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--datasets', nargs='*', default=None,
                   help=f'default: all of {sorted(common.DATASETS)}')
    p.add_argument('--data-dir', default=None,
                   help='fms-dgt data root (default: $DGT_DATA_DIR, else baselines/SDForger/work/data)')
    p.add_argument('--force', action='store_true', help='rewrite existing parquet files')
    ns = p.parse_args()

    data_dir = Path(ns.data_dir or os.environ.get('DGT_DATA_DIR') or DEFAULT_WORK / 'data')
    names = ns.datasets or sorted(common.DATASETS)
    print(f'{"dataset":<16}{"N":>6}{"T":>5}  parquet')
    for name in names:
        npy = common.data_path(name)
        parquet = data_dir / 'public' / 'time_series' / f'{npy.stem}.parquet'
        if parquet.exists() and not ns.force:
            df = pd.read_parquet(parquet)
            t, n = df.shape[0], df.shape[1] - 1
            state = 'exists'
        else:
            n, t = write_parquet(npy, parquet)
            state = 'written'
        print(f'{name:<16}{n:>6}{t:>5}  {parquet.name} ({state}; task {task_stem(name)}_multisample)')


if __name__ == '__main__':
    main()
