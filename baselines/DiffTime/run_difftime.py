"""DiffTime on one benchmark dataset: train, then sample N series.

DiffTime -- Coletta, Gopalakrishnan, Borrajo, Vyetrenko, "On the Constrained Time-Series
Generation Problem", NeurIPS 2023 (arXiv:2307.01717) -- has no public code release. The code
in this directory is upstream CSDI (ermongroup/CSDI @ 7f24a43), which DiffTime is built on,
plus an unconditional wrapper (`CSDI_Unconditional` in main_model.py, `dataset_npy.py`).
This script imports the denoiser, the loss and the sampler from there and does not
reimplement or modify any of them. Hyperparameters come from `config.yaml` (Table 5 of the
DiffTime paper). See UPSTREAM.md for the deviations from the paper.

The generated set is written, in the ORIGINAL data scale, to
`outputs/DiffTime/seed<N>/<dataset>/DiffTime_<dataset>_generatedata.npy`
(see baselines/common.py), next to a `summary.json`. The trained weights and the loss log go
to `--run-dir` (default `checkpoints/DiffTime/seed<N>/<dataset>/`).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))                # baselines/, for common.py

import common                                                              # noqa: E402
from main_model import CSDI_Unconditional                                  # noqa: E402
from dataset_npy import NpySeriesDataset, collate_unconditional            # noqa: E402

METHOD = 'DiffTime'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_csdi_config(hp: dict) -> dict:
    """Shape `config.yaml` into the nested dict CSDI_base / diff_CSDI expect."""
    return {
        'train': {
            'batch_size': int(hp['batch_size']),
            'lr': float(hp['lr']),
        },
        'diffusion': {
            'layers': int(hp['layers']),
            'channels': int(hp['channels']),
            'nheads': int(hp['nheads']),
            'diffusion_embedding_dim': int(hp['diffusion_embedding_dim']),
            'beta_start': float(hp['beta_start']),
            'beta_end': float(hp['beta_end']),
            'num_steps': int(hp['num_steps']),
            'schedule': str(hp['schedule']),
            'is_linear': bool(hp['is_linear']),
        },
        'model': {
            'is_unconditional': True,
            'timeemb': int(hp['timeemb']),
            'featureemb': int(hp['featureemb']),
            'target_strategy': str(hp['target_strategy']),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--config', default=str(HERE / 'config.yaml'))
    p.add_argument('--seed', type=int, default=None, help='overrides run.seed of the config')
    p.add_argument('--run-dir', default=None,
                   help='where model.pth and train_metrics.jsonl go '
                        '(default: <repo>/checkpoints/DiffTime/seed<N>/<dataset>)')
    p.add_argument('--max-steps', type=int, default=None,
                   help='smoke test only: stop after this many gradient steps')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ns = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    npy_path = common.data_path(ns.dataset)
    if not npy_path.is_file():
        raise SystemExit(f'{ns.dataset}: no data at {npy_path}')

    with open(ns.config) as f:
        cfg = yaml.safe_load(f)
    hp, runcfg = cfg['difftime'], cfg['run']
    seed = ns.seed if ns.seed is not None else int(runcfg['seed'])
    if ns.max_steps is not None:
        hp['max_steps'] = int(ns.max_steps)

    gen_path = common.generated_path(METHOD, ns.dataset, seed)
    out_dir = gen_path.parent
    run_dir = (Path(ns.run_dir) if ns.run_dir else
               common.REPO_ROOT / 'checkpoints' / METHOD / f'seed{seed}' / ns.dataset)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = torch.device(ns.device)

    # ---- data ------------------------------------------------------------------------
    raw = np.load(npy_path, allow_pickle=False)
    if raw.ndim != 3 or raw.shape[-1] != 1:
        raise SystemExit(f'{ns.dataset}: CSDI_Unconditional is univariate, expected (N, T, 1), '
                         f'got {raw.shape}')
    n_real, seq_len, _ = raw.shape
    raw = raw.astype(np.float32, copy=False)
    logging.info('%s: (N,T,C)=%s', ns.dataset, raw.shape)

    # z-score over the whole array (univariate, so this is CSDI's per-feature standardisation);
    # inverted before writing.
    use_zscore = bool(runcfg.get('zscore', True))
    if use_zscore:
        mean, std = float(raw.mean()), float(raw.std())
        if std < 1e-12:
            raise SystemExit(f'{ns.dataset}: std {std} too small to z-score')
        train_array = ((raw - mean) / std).astype(np.float32)
    else:
        mean, std = 0.0, 1.0
        train_array = raw
    logging.info('z-score: mean=%.6g std=%.6g (enabled=%s)', mean, std, use_zscore)

    # Table 5's "epochs 10000", read literally: 10000 passes over this dataset, i.e.
    # epochs * ceil(N / batch_size) gradient steps (630k for ecg at N=1000).
    steps_per_epoch = -(-n_real // int(hp['batch_size']))          # ceil
    if 'max_steps' in hp and hp['max_steps'] is not None:
        max_steps = int(hp['max_steps'])                            # smoke-test override
        epochs_planned = max_steps / steps_per_epoch
    else:
        epochs_planned = int(hp['epochs'])
        max_steps = epochs_planned * steps_per_epoch
    logging.info('%d epochs x %d steps/epoch = %d gradient steps',
                 epochs_planned if isinstance(epochs_planned, int) else -1,
                 steps_per_epoch, max_steps)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        NpySeriesDataset(train_array),
        batch_size=int(hp['batch_size']), shuffle=True,
        num_workers=int(runcfg.get('num_workers', 0)),
        collate_fn=collate_unconditional, generator=g,
        pin_memory=(device.type == 'cuda'),
    )

    # ---- model -----------------------------------------------------------------------
    csdi_config = build_csdi_config(hp)
    model = CSDI_Unconditional(csdi_config, str(device)).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    logging.info('DiffTime/CSDI parameters: %d', n_params)

    # Adam with Table 5's lr and weight decay. No lr scheduler (DiffTime's Algorithm 3 has none).
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp['lr']),
                                 weight_decay=float(hp['weight_decay']))

    # ---- train: Algorithm 3, one batch and one gradient step per iteration -------------
    log_every = min(1000, max(1, max_steps // 100))
    t0 = time.time()
    model.train()
    step, window, losses_all = 0, [], []
    it = iter(train_loader)
    with open(run_dir / 'train_metrics.jsonl', 'w', buffering=1) as mf:
        while step < max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, is_train=1)
            loss.backward()
            optimizer.step()
            step += 1
            v = float(loss.item())
            window.append(v)
            losses_all.append(v)
            if step % log_every == 0 or step == max_steps:
                rec = {'step': step,
                       'loss': v,
                       'loss_mean_window': float(np.mean(window)),
                       'samples_seen': step * int(hp['batch_size']),
                       'epoch_equiv': round(step * int(hp['batch_size']) / n_real, 3)}
                mf.write(json.dumps(rec) + '\n')
                logging.info('step %d/%d  loss %.6f  mean(%d) %.6f  epoch_equiv %.2f',
                             step, max_steps, v, len(window), rec['loss_mean_window'],
                             rec['epoch_equiv'])
                window = []
    train_min = (time.time() - t0) / 60
    logging.info('training done in %.1f min (%d steps)', train_min, step)

    torch.save(model.state_dict(), run_dir / 'model.pth')

    # ---- generate: exactly N samples, no duplication ----------------------------------
    t1 = time.time()
    seed_everything(seed)
    model.eval()
    generated = model.sample_unconditional(
        num_samples=n_real, seq_len=seq_len,
        batch_size=int(runcfg.get('sample_batch_size', 64)))
    sample_min = (time.time() - t1) / 60
    if generated.shape != (n_real, seq_len, 1):
        raise SystemExit(f'{ns.dataset}: sampler returned {generated.shape}, '
                         f'expected {(n_real, seq_len, 1)}')
    logging.info('generated %s in %.1f min', generated.shape, sample_min)

    # ---- write, in the ORIGINAL scale --------------------------------------------------
    gen_orig = (generated * std + mean).astype(np.float32)
    np.save(gen_path, gen_orig)

    summary = {
        'model': METHOD,
        'dataset': ns.dataset,
        'seed': seed,
        'data_file': npy_path.name,
        'shape': list(gen_orig.shape),
        'n_real': int(n_real),
        'n_generated': int(gen_orig.shape[0]),
        'scale': 'original (z-score inverted before writing)',
        'normalisation': {'kind': 'zscore_global' if use_zscore else 'none',
                          'mean': mean, 'std': std},
        'n_params': int(n_params),
        'train_minutes': round(train_min, 2),
        'sample_minutes': round(sample_min, 2),
        'steps_trained': int(step),
        'steps_per_epoch': int(steps_per_epoch),
        'epochs_trained': round(step / steps_per_epoch, 2),
        'final_loss': float(losses_all[-1]),
        'final_loss_mean_last_100': float(np.mean(losses_all[-100:])),
        'hyperparameters': dict(hp),
        'hyperparameter_source': 'Table 5, arXiv:2307.01717 (DiffTime paper appendix)',
        'code_source': 'ermongroup/CSDI @ 7f24a43 + CSDI_Unconditional (main_model.py)',
        'env': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info('wrote %s', gen_path)


if __name__ == '__main__':
    main()
