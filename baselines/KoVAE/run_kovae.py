"""KoVAE on one of the seven benchmark datasets.

Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs
(Naiman et al., ICLR 2024), upstream at github.com/azencot-group/KoVAE.

The training loop, the optimiser, the data pipeline and the sampling procedure below are
taken from upstream's `run_regular.py`. The model is imported from the vendored `models/`
directory next to this script; nothing here reimplements KoVAE.

Differences from upstream's `run_regular.py` (I/O only, not method):

  1. Hyperparameters come from `config.yaml`, which carries upstream's argparse defaults.
  2. Data is read from `common.data_path(dataset)` (an (N, T, C) .npy, per-channel
     min-max to [0, 1]); the generated set is written in original units to
     `common.generated_path('KoVAE', dataset)`.
  3. The post-hoc discriminative score upstream computes at the end of training is not
     run (generated sets are scored by the shared evaluation code).
  4. `tf.keras.utils.set_random_seed` is not called: tensorflow is not otherwise used,
     and torch/numpy are seeded as upstream does. neptune logging is dropped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # baselines/, for common.py
import common                                      # noqa: E402
sys.path.insert(0, str(HERE))                      # vendored upstream packages

from models.kovae import KoVAE                                          # noqa: E402
from utils.utils import agg_losses, log_losses                          # noqa: E402
from utils.utils_data import extreme_npy_loading, inverse_MinMax_NTC    # noqa: E402

MODEL_NAME = 'KoVAE'


def set_seed_device(seed):
    """Upstream `utils.set_seed_device`, minus the tensorflow seeding."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('cuda is available')
    else:
        device = torch.device("cpu")
    return device


def build_args(cfg, dataset, npy_path, seed):
    a = argparse.Namespace(**dict(cfg['kovae']))
    a.seed = int(seed)
    a.dataset = dataset
    a.npy_path = str(npy_path)
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--config', default=str(HERE / 'config.yaml'))
    p.add_argument('--seed', type=int, default=None, help='overrides run.seed (10)')
    p.add_argument('--epochs', type=int, default=None,
                   help='overrides kovae.epochs; smoke tests only')
    p.add_argument('--num-workers', type=int, default=None,
                   help='overrides run.num_workers')
    p.add_argument('--work-dir', default=os.environ.get('KOVAE_WORK', str(HERE / 'work')),
                   help='training logs (default: baselines/KoVAE/work)')
    p.add_argument('--out', default=None,
                   help='explicit output .npy path (default: common.generated_path)')
    ns = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cfg = OmegaConf.to_object(OmegaConf.load(ns.config))
    npy_path = common.data_path(ns.dataset)
    if not npy_path.is_file():
        raise SystemExit(f'no data at {npy_path}')
    seed = ns.seed if ns.seed is not None else int(cfg['run']['seed'])
    if ns.epochs is not None:
        cfg['kovae']['epochs'] = int(ns.epochs)
    if ns.num_workers is not None:
        cfg['run']['num_workers'] = int(ns.num_workers)

    work_dir = Path(ns.work_dir) / ns.dataset
    work_dir.mkdir(parents=True, exist_ok=True)

    args = build_args(cfg, ns.dataset, npy_path, seed)
    args.device = set_seed_device(seed)

    # ---- data: per-channel min-max to [0, 1] over the whole array.
    arr_norm, scaler, t_file, c_file = extreme_npy_loading(str(npy_path))
    args.seq_len, args.inp_dim = t_file, c_file
    logging.info('%s: (N,T,C)=%s', ns.dataset, arr_norm.shape)

    train_set = Data.TensorDataset(torch.from_numpy(arr_norm))

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                   shuffle=True, num_workers=int(cfg['run']['num_workers']),
                                   worker_init_fn=seed_worker, generator=g)

    model = KoVAE(args).to(device=args.device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    n_params = sum(q.numel() for q in model.parameters())
    logging.info('KoVAE parameters: %d', n_params)
    logging.info('%s', vars(args))

    # ---- train: upstream's loop.
    t0 = time.time()
    with open(work_dir / 'train_metrics.jsonl', 'w', buffering=1) as mf:
        for epoch in range(0, args.epochs):
            model.train()
            losses_agg_tr = []
            for i, data in enumerate(train_loader, 1):
                X = data[0].to(args.device)
                optimizer.zero_grad()
                x_rec, Z_enc, Z_enc_prior = model(X)
                losses = model.loss(X, x_rec, Z_enc, Z_enc_prior)
                losses[0].backward()
                optimizer.step()
                losses_agg_tr = agg_losses(losses_agg_tr, losses)
            log_losses(epoch, losses_agg_tr, model.names)
            mf.write(json.dumps({'epoch': epoch,
                                 **{n: float(np.mean(v)) for n, v in
                                    zip(model.names, losses_agg_tr)}}) + '\n')
    train_min = (time.time() - t0) / 60
    logging.info('training done in %.1f min', train_min)

    # ---- generate: upstream re-seeds before sampling and draws one batch per training
    # batch, so the generated set has exactly N samples.
    args.device = set_seed_device(seed)
    model.eval()
    with torch.no_grad():
        generated = [model.sample_data(d[0].shape[0]).detach().cpu().numpy()
                     for d in train_loader]
    generated = np.vstack(generated)
    logging.info('generated %s', generated.shape)

    gen_orig = inverse_MinMax_NTC(generated, scaler).astype(np.float32)
    out = Path(ns.out) if ns.out else common.generated_path(MODEL_NAME, ns.dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, gen_orig)

    summary = {'model': MODEL_NAME, 'dataset': ns.dataset, 'seed': seed,
               'shape': list(gen_orig.shape),
               'n_generated': int(gen_orig.shape[0]),
               'n_real': int(arr_norm.shape[0]),
               'n_params': int(n_params),
               'train_minutes': round(train_min, 1),
               'kovae': {k: v for k, v in cfg['kovae'].items()}}

    with open(out.parent / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info('wrote %s', out.name)


if __name__ == '__main__':
    main()
