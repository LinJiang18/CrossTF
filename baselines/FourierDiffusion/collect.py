"""Copy one FourierDiffusion run into the shared output layout.

Reads  runs/<run_name>/<run_name>_generatedata.npy   (written by cmd/sample.py,
       already destandardised and inverse-DFT'd, i.e. in original units)
writes <repo>/outputs/FourierDiffusion/<dataset>/FourierDiffusion_<dataset>_generatedata.npy
       and a summary.json next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # baselines/, for common.py
import common  # noqa: E402

METHOD = 'FourierDiffusion'


def n_trainable(run_dir: Path) -> int | None:
    ckpts = list(run_dir.glob('*.ckpt'))
    if len(ckpts) != 1:
        return None
    from fdiff.models.score_models import ScoreModule
    model = ScoreModule.load_from_checkpoint(
        checkpoint_path=ckpts[0], map_location='cpu', weights_only=False)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--run-name', required=True, help='project_name used for train/sample')
    p.add_argument('--runs-dir', default=str(HERE / 'runs'))
    p.add_argument('--max-epochs', type=int, required=True)
    ns = p.parse_args()

    run_dir = Path(ns.runs_dir) / ns.run_name
    gen = np.load(run_dir / f'{ns.run_name}_generatedata.npy').astype(np.float32)
    real = np.load(common.data_path(ns.dataset))
    if gen.shape != real.shape:
        raise SystemExit(f'{ns.dataset}: generated {gen.shape} vs real {real.shape}')
    if not np.isfinite(gen).all():
        raise SystemExit(f'{ns.dataset}: generated data contains NaN/Inf')

    out = common.generated_path(METHOD, ns.dataset)
    np.save(out, gen)
    summary = {
        'method': METHOD, 'dataset': ns.dataset, 'seed': 42,
        'data_file': common.DATASETS[ns.dataset],
        'shape': list(gen.shape),
        'n_params': n_trainable(run_dir),
        'hyperparams': {'fourier_transform': True, 'fourier_noise_scaling': True,
                        'standardize': True, 'noise_scheduler': 'vpsde',
                        'd_model': 72, 'num_layers': 10, 'n_head': 12,
                        'lr_max': 1e-3, 'likelihood_weighting': False,
                        'max_epochs': ns.max_epochs, 'batch_size': 64,
                        'train_ratio': 0.9, 'num_diffusion_steps': 1000,
                        'sample_batch_size': 200},
    }
    with open(out.parent / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'{ns.dataset}: {gen.shape} -> {out}')


if __name__ == '__main__':
    main()
