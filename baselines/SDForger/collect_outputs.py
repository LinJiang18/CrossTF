"""fms-dgt jsonl output -> the (N', T, 1) .npy layout shared by all baselines.

The SDForger databuilder emits one `TimeSeriesOutputData` per sample, saved as
`<DGT_OUTPUT_DIR>/time_series/<Task>_multisample/final_data.jsonl`, one line per sample:

    {"task_name": ..., "generated_time_series": {"TOY-": [v1, ..., vT]}}

It calls `inverse_standard_scale_generated` before returning, so the values are already
in the original units; no rescaling is done here, only reshaping and saving.

Writes `outputs/SDForger/<dataset>/SDForger_<dataset>_generatedata.npy` (float32,
(N', T, 1)) and a small `summary.json`. N' is whatever SDForger returned: after
de-duplication and its IQR filter it is typically 100-103% of N. Nothing is padded or
truncated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # baselines/ -> common.py
import common  # noqa: E402

METHOD = 'SDForger'
CHANNEL = 'TOY-'


def generated_jsonl(task_out_dir: Path) -> Path:
    """The one file holding the generated samples.

    fms-dgt writes the same samples to both `data.jsonl` and `final_data.jsonl`
    (byte-identical), so a `*.jsonl` glob would read every series twice. Read exactly one.
    """
    for name in ('final_data.jsonl', 'data.jsonl'):
        f = task_out_dir / name
        if f.is_file():
            return f
    raise SystemExit(f'no final_data.jsonl or data.jsonl under {task_out_dir}')


def read_jsonl(task_out_dir: Path) -> np.ndarray:
    rows = []
    with open(generated_jsonl(task_out_dir)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = rec.get('generated_time_series')
            if not ts:
                continue
            # one channel in the multisample setup; take it whatever it is named
            series = ts.get(CHANNEL) or next(iter(ts.values()))
            rows.append(np.asarray(series, dtype=np.float32))
    if not rows:
        raise SystemExit(f'no generated series under {task_out_dir}')
    lens = {len(r) for r in rows}
    if len(lens) != 1:
        raise SystemExit(f'ragged series lengths {sorted(lens)} under {task_out_dir}')
    return np.stack(rows)[..., None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--task-out-dir', required=True,
                   help='<DGT_OUTPUT_DIR>/time_series/<Task>_multisample')
    p.add_argument('--seed', type=int, default=None,
                   help='if given, write under outputs/SDForger/seed<N>/ (default: no seed subdir)')
    p.add_argument('--extra', default='{}', help='JSON merged into summary.json')
    ns = p.parse_args()

    gen = read_jsonl(Path(ns.task_out_dir))
    real = np.load(common.data_path(ns.dataset))
    if real.ndim == 2:
        real = real[..., None]

    if gen.shape[1] != real.shape[1]:
        raise SystemExit(f'T mismatch: generated {gen.shape[1]} vs real {real.shape[1]}')

    out_file = common.generated_path(METHOD, ns.dataset, ns.seed)
    np.save(out_file, gen.astype(np.float32))

    summary = {'dataset': ns.dataset, 'model': METHOD,
               'data_file': common.DATASETS[ns.dataset],
               'shape_real': list(real.shape), 'shape_generated': list(gen.shape),
               'n_real': int(real.shape[0]), 'n_generated': int(gen.shape[0]),
               'scale': 'original (SDForger inverts its StandardScaler before returning)'}
    summary.update(json.loads(ns.extra))
    (out_file.parent / 'summary.json').write_text(json.dumps(summary, indent=2))
    diff = gen.shape[0] - real.shape[0]
    print(f'{ns.dataset}: generated {gen.shape} vs real {real.shape} ({diff:+d})')


if __name__ == '__main__':
    main()
