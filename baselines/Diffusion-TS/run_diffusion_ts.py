#!/usr/bin/env python3
"""Diffusion-TS on one of the seven benchmark datasets.

Diffusion-TS: Interpretable Diffusion for General Time Series Generation
(Yuan & Qiao, ICLR 2024), upstream at github.com/Y-debug-sys/Diffusion-TS.

Nothing here reimplements the method. The model, the diffusion, the training loop and the
sampler are the vendored upstream files next to this script:

  * `Diffusion_TS` is instantiated through upstream's own `instantiate_from_config`;
  * `Trainer.train()` and `Trainer.sample()` are called as they ship (the per-step losses
    are collected by handing `Trainer` a logger that writes jsonl, through the hook
    upstream already calls, not a modified loop);
  * the dataloader comes from upstream's `build_dataloader`.

What this file supplies is the parts upstream leaves to the user: which datasets, a
capacity for a data shape upstream ships no config for (see config.yaml), and the inverse
normalisation upstream leaves commented out in its `main.py`:

    samples = trainer.sample(...)
    if dataset.auto_norm:
        samples = unnormalize_to_zero_to_one(samples)
        # samples = dataset.scaler.inverse_transform(...)   <- commented out upstream

We apply the full inverse (`dataset.unnormalize`), i.e. that commented line plus the
[-1, 1] -> [0, 1] step, so the generated set is in the data's own units.

Seeds
-----
The paper reports three runs per dataset. `--seed-index {1,2,3}` selects one of them; the
numeric seed passed to `seed_everything` is

    seed-index 1 -> 12345   (upstream main.py's default, also `run.seed` in config.yaml)
    seed-index 2 -> 2
    seed-index 3 -> 3

and the generated set is written to `outputs/Diffusion-TS/seed<index>/<dataset>/`.

Usage
-----
    python run_diffusion_ts.py --dataset electricity --seed-index 1
    python run_diffusion_ts.py --dataset ecg --steps 20 --n-generate 16   # smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # baselines/, for common.py
import common                                      # noqa: E402
sys.path.insert(0, str(HERE))                      # vendored upstream packages

from engine.solver import Trainer                                   # noqa: E402
from Data.build_dataloader import build_dataloader                  # noqa: E402
from Utils.io_utils import instantiate_from_config, seed_everything  # noqa: E402

MODEL_NAME = 'Diffusion-TS'

# seed-index -> numeric seed actually used for the reported results.
PAPER_SEEDS = {1: 12345, 2: 2, 3: 3}


class JsonlLogger:
    """Upstream's Logger interface, narrowed to what Trainer touches.

    Trainer calls `log_info(...)` and `add_scalar(tag=, scalar_value=, global_step=)`.
    Writing jsonl from here keeps the training loop itself upstream's.
    """

    def __init__(self, path: Path):
        self.f = open(path, 'w', buffering=1)

    def log_info(self, info, check_primary=True):
        logging.info('%s', info)

    def add_scalar(self, **kw):
        self.f.write(json.dumps({'step': kw.get('global_step'),
                                 'train_loss': float(kw.get('scalar_value'))}) + '\n')

    def add_scalars(self, **kw):
        pass

    def close(self):
        self.f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--config', default=str(HERE / 'config.yaml'))
    p.add_argument('--seed-index', type=int, default=1, choices=sorted(PAPER_SEEDS),
                   help='which of the three reported runs (see module docstring)')
    p.add_argument('--seed', type=int, default=None,
                   help='override the numeric seed (default: the one for --seed-index)')
    p.add_argument('--work-dir', default=os.environ.get('DTS_WORK', str(HERE / 'work')),
                   help='checkpoints and training logs (default: baselines/Diffusion-TS/work)')
    p.add_argument('--steps', type=int, default=None,
                   help='overrides solver.max_epochs (a step count); smoke tests only')
    p.add_argument('--n-generate', type=int, default=None, help='smoke tests only')
    p.add_argument('--out', default=None,
                   help='explicit output .npy path (default: common.generated_path)')
    ns = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    npy_path = common.data_path(ns.dataset)
    if not npy_path.is_file():
        raise SystemExit(f'no data at {npy_path}')

    raw = np.load(npy_path, allow_pickle=False)
    if raw.ndim != 3:
        raise SystemExit(f'expected (N, T, C), got {raw.shape}')
    n_data, seq_len, n_feat = (int(x) for x in raw.shape)

    config = yaml.safe_load(Path(ns.config).read_text())
    run_cfg = config.pop('run')
    seed = ns.seed if ns.seed is not None else PAPER_SEEDS[ns.seed_index]
    if ns.steps is not None:
        config['solver']['max_epochs'] = int(ns.steps)

    work_dir = Path(ns.work_dir) / f'seed{ns.seed_index}' / ns.dataset
    work_dir.mkdir(parents=True, exist_ok=True)

    # per-dataset holes in the shared config
    config['model']['params']['seq_length'] = seq_len
    config['model']['params']['feature_size'] = n_feat
    config['dataloader']['train_dataset']['params'].update(
        {'name': ns.dataset, 'data_path': str(npy_path), 'window': seq_len})
    config['solver']['results_folder'] = str(work_dir / 'checkpoints')

    # Snapshot before instantiate_from_config writes the optimizer into
    # config['solver']['scheduler']['params'].
    cfg_snapshot = copy.deepcopy(config)

    seed_everything(seed)
    args = SimpleNamespace(name=ns.dataset, save_dir=str(work_dir), gpu=0, tensorboard=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dl_info = build_dataloader(config, args)
    dataset = dl_info['dataset']
    logging.info('%s: (N,T,C)=%s | train rows %d', ns.dataset, raw.shape, len(dataset))

    model = instantiate_from_config(config['model']).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    logging.info('%s parameters: %d | device %s | seed %d', MODEL_NAME, n_params, device, seed)

    jlogger = JsonlLogger(work_dir / 'train_metrics.jsonl')
    trainer = Trainer(config=config, args=args, model=model,
                      dataloader=dl_info, logger=jlogger)
    trainer.log_frequency = int(run_cfg.get('log_frequency', 100))

    t0 = time.time()
    trainer.train()                       # upstream's loop, unmodified
    train_min = (time.time() - t0) / 60
    logging.info('training done in %.1f min', train_min)

    # ---- generate exactly N samples (never duplicated), in chunks of sample_batch_size.
    n_generate = ns.n_generate if ns.n_generate is not None else run_cfg.get('n_generate')
    n_generate = int(n_generate) if n_generate else n_data
    chunk = min(int(run_cfg['sample_batch_size']), n_generate)
    t1 = time.time()
    samples = trainer.sample(num=n_generate, size_every=chunk,
                             shape=[seq_len, n_feat])
    sample_min = (time.time() - t1) / 60
    logging.info('generated %s in %.1f min', samples.shape, sample_min)
    jlogger.close()

    # ---- back to the data's own units (see the module docstring).
    gen_orig = dataset.unnormalize(samples).astype(np.float32)
    out = Path(ns.out) if ns.out else common.generated_path(MODEL_NAME, ns.dataset,
                                                            ns.seed_index)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, gen_orig)

    summary = {
        'model': MODEL_NAME,
        'dataset': ns.dataset,
        'seed_index': ns.seed_index,
        'seed': seed,
        'shape': list(gen_orig.shape),
        'n_generated': int(gen_orig.shape[0]),
        'n_real': n_data,
        'n_params': int(n_params),
        'train_minutes': round(train_min, 2),
        'sample_minutes': round(sample_min, 2),
        'hyperparameters': {'sample_chunk': chunk,
                            **cfg_snapshot['model']['params'],
                            'max_epochs_is_gradient_steps':
                                cfg_snapshot['solver']['max_epochs'],
                            'base_lr': cfg_snapshot['solver']['base_lr'],
                            'batch_size': cfg_snapshot['dataloader']['batch_size'],
                            'gradient_accumulate_every':
                                cfg_snapshot['solver']['gradient_accumulate_every']},
        'environment': {'python': sys.version.split()[0], 'torch': torch.__version__},
    }
    with open(out.parent / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info('wrote %s', out.name)


if __name__ == '__main__':
    main()
