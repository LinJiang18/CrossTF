"""Train TimeGAN on one dataset and write N generated series.

The model, its losses and its sampler come from `tsgm.models.timeGAN.TimeGAN` (pip
package tsgm 0.1.0) and are not reimplemented here. This script only loads the data,
min-max scales it, sets the training budget, and writes the generated set.

Run `patch_tsgm_torch.py` once per environment first (lets tsgm use the GPU under the
Keras torch backend). Then:

    python run_timegan.py --dataset precipitation            # settings from config.yaml

Output: `outputs/TimeGAN/<dataset>/TimeGAN_<dataset>_generatedata.npy` and `summary.json`;
tsgm's per-epoch losses go to `$TIMEGAN_WORK/<dataset>/train_metrics.jsonl` (default
`baselines/TimeGAN/work/`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

# The reported runs used the torch backend (with patch_tsgm_torch.py applied).
os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import yaml
import keras
import tsgm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common                                             # noqa: E402

METHOD = 'TimeGAN'
WORK = Path(os.environ.get('TIMEGAN_WORK', HERE / 'work'))

DEVIATIONS = {
    'implementation_tsgm_not_paper_code': (
        "the model is tsgm's TimeGAN port, not the paper's own jsyoon0823/TimeGAN "
        "(TensorFlow 1.x, which does not run on current GPUs / CUDA 12)."
    ),
    'training_length_10000_iterations_not_50000': (
        "upstream runs 50000 iterations per phase; tsgm's fit() has no default (epochs is "
        "required). The budget is 10000 iterations per phase (20% of upstream) for cost."
    ),
    'budget_in_iterations_not_epochs': (
        "the budget is set in iterations, upstream's unit, and converted per dataset to "
        "epochs = ceil(iterations / ceil(N / batch_size)), so every dataset gets the same "
        "number of updates regardless of N."
    ),
    'batch_size_128_not_tsgm_256': (
        "batch_size follows upstream's 128; tsgm's own default is 256. hidden_dim 24, "
        "n_layers 3, module gru and gamma 1.0 are upstream defaults that tsgm agrees with."
    ),
    'torch_backend_patched': (
        "Keras torch backend; patch_tsgm_torch.py lets tsgm 0.1.0 use CUDA and fixes two "
        "crashes on that path, without changing the model or its training."
    ),
    'seed_added': "upstream sets no seed; python/numpy/backend are seeded with 10.",
    'no_train_test_split': "trains on the whole array and samples N series.",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)   # seeds python, numpy and the active backend


def minmax_scale(x):
    """Per-channel min-max to [0, 1] over the whole array."""
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-8), x_min, x_max


def minmax_inverse(x_scaled, x_min, x_max):
    return x_scaled * (x_max - x_min + 1e-8) + x_min


def to_numpy(out):
    if hasattr(out, 'detach'):          # torch tensor
        return out.detach().cpu().numpy()
    if hasattr(out, 'numpy'):           # tf.Tensor
        return out.numpy()
    return np.asarray(out)


def write_train_metrics(model, path):
    """Write tsgm's per-epoch losses (training_losses_history) as jsonl.

    Guarded: this runs after the samples are on disk, and a change in tsgm's internal
    tracker must not cost a finished run. tsgm swallows exceptions inside its training
    steps and returns a zero loss instead, so losses pinned at 0.0 here mean a broken run.
    """
    try:
        hist = model.training_losses_history
        series = {str(k): [float(v) for v in np.asarray(hist[k]).ravel()] for k in hist}
        n = max((len(v) for v in series.values()), default=0)
        with open(path, 'w') as f:
            for i in range(n):
                row = {'step': i}
                row.update({k: v[i] for k, v in series.items() if i < len(v)})
                f.write(json.dumps(row) + '\n')
        return sorted(series)
    except Exception as exc:                                  # noqa: BLE001
        logging.warning('could not write train_metrics.jsonl: %s', exc)
        return []


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--config', default=str(HERE / 'config.yaml'))
    p.add_argument('--seed', type=int, default=None, help='overrides run.seed')
    p.add_argument('--iterations', type=int, default=None,
                   help='overrides run.iterations_per_phase')
    p.add_argument('--seed-subdir', action='store_true',
                   help='write under outputs/TimeGAN/seed<N>/ (default: no seed subdir)')
    ns = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    with open(ns.config) as f:
        cfg = yaml.safe_load(f)

    hp = cfg['timegan']
    seed = ns.seed if ns.seed is not None else int(cfg['run']['seed'])
    iterations = (ns.iterations if ns.iterations is not None
                  else int(cfg['run']['iterations_per_phase']))
    work_dir = WORK / ns.dataset
    work_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    x = np.load(common.data_path(ns.dataset)).astype('float32')
    if x.ndim == 2:
        x = x[..., None]
    n, t, c = x.shape
    batch_size = int(hp['batch_size'])

    # upstream counts iterations; tsgm counts epochs. Convert, and report both.
    steps_per_epoch = math.ceil(n / batch_size)
    epochs = math.ceil(iterations / steps_per_epoch)
    actual_iterations = epochs * steps_per_epoch
    logging.info('%s: (N,T,C)=%s, keras backend %s', ns.dataset, x.shape,
                 keras.backend.backend())
    logging.info('budget %d iter/phase -> %d epochs x %d steps = %d iterations/phase',
                 iterations, epochs, steps_per_epoch, actual_iterations)

    x_scaled, x_min, x_max = minmax_scale(x)

    model = tsgm.models.timeGAN.TimeGAN(
        seq_len=t,
        n_features=c,
        hidden_dim=int(hp['hidden_dim']),
        n_layers=int(hp['n_layers']),
        batch_size=batch_size,
        module=str(hp['module']),
        gamma=float(hp['gamma']),
    )
    lr = float(hp['lr'])
    model.compile(
        d_optimizer=keras.optimizers.Adam(lr),
        g_optimizer=keras.optimizers.Adam(lr),
        emb_optimizer=keras.optimizers.Adam(lr),
        supgan_optimizer=keras.optimizers.Adam(lr),
        ae_optimizer=keras.optimizers.Adam(lr),
        emb_loss=keras.losses.MeanSquaredError(),
        clf_loss=keras.losses.BinaryCrossentropy(),
    )

    t0 = time.time()
    model.fit(x_scaled, epochs=epochs)
    train_min = (time.time() - t0) / 60
    logging.info('training done in %.1f min', train_min)

    # tsgm's own sampler, asked for exactly N.
    gen_scaled = to_numpy(model.generate(n))
    gen = minmax_inverse(gen_scaled, x_min, x_max).astype(np.float32)
    logging.info('generated %s', gen.shape)
    if gen.shape[0] != n:
        logging.warning('asked for %d samples, tsgm returned %d', n, gen.shape[0])

    out_file = common.generated_path(METHOD, ns.dataset, seed if ns.seed_subdir else None)
    np.save(out_file, gen)

    loss_keys = write_train_metrics(model, work_dir / 'train_metrics.jsonl')

    n_params = 0
    for sub in ('generator', 'discriminator', 'embedder', 'recovery', 'supervisor'):
        m = getattr(model, sub, None)
        if m is not None and hasattr(m, 'count_params'):
            n_params += int(m.count_params())

    summary = {
        'dataset': ns.dataset,
        'seed': seed,
        'data_file': common.DATASETS[ns.dataset],
        'shape': list(gen.shape),
        'n_generated': int(gen.shape[0]),
        'n_real': int(n),
        'n_params': n_params,
        'train_minutes': round(train_min, 1),
        'keras_backend': keras.backend.backend(),
        'timegan': dict(hp),
        'training_budget': {
            'iterations_per_phase_requested': iterations,
            'iterations_per_phase_actual': actual_iterations,
            'epochs_per_phase': epochs,
            'steps_per_epoch': steps_per_epoch,
            'upstream_iterations_per_phase': 50000,
        },
        'loss_keys': loss_keys,
        'deviations': DEVIATIONS,
    }
    with open(out_file.parent / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info('wrote %s', out_file)


if __name__ == '__main__':
    main()
