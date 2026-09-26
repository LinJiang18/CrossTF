"""FIDE on one dataset, with the GEV regulariser repaired so that it trains.

FIDE: Frequency-Inflated Conditional Diffusion Model for Extreme-Aware Time Series
Generation (Galib, Tan, Luo; NeurIPS 2024). Upstream: https://github.com/galib19/FIDE

This runner replaces upstream's `main_temperature_example.py`. The model, the noise
schedule, `add_noise` and the sampler are imported unchanged from `Code/`; only the
loss is re-implemented here, so every change to the method is visible in this file.

Upstream's `train_utilities.get_loss` computes its extreme-value regulariser as

    reg_loss = np.mean(gev_model.logpdf(bm_pred.detach().cpu().numpy()))
    reg_loss = -0.05 * torch.tensor(reg_loss, dtype=torch.float32).to(device)

The numpy/scipy round-trip severs the autograd graph, so the term is a constant and
contributes no gradient (gradients with the regulariser on and off are bitwise
identical); the released code therefore trains a plain conditional DDPM. Three repairs,
all inside the regulariser:

  1. the GEV log-density is evaluated in torch, so gradient flows to the parameters;
  2. the x0 estimate is the DDPM one, `(x_t - sqrt(1-a) * eps) / sqrt(a)`, instead of
     upstream's `x_noisy - pred_noise`;
  3. `lambda_1 = linear_decay(i)` is applied as a per-sample weight (upstream computes
     it and discards it).

`--no-regularizer` gives the as-released model (plain conditional DDPM); it was not used
for the reported results.
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

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.fft import irfft, rfft, rfftfreq
from scipy.stats import genextreme
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
FIDE_CODE = HERE / 'Code'
sys.path.insert(0, str(HERE.parent))  # baselines/, for common.py
import common  # noqa: E402

METHOD = 'FIDE'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gev_logpdf_torch(x, c, loc, scale, eps=1e-6, floor=-50.0):
    """scipy.stats.genextreme.logpdf, in torch, differentiable in x.

    c, loc and scale are the scipy fit on the real block maxima and stay fixed; only x
    carries gradient. scipy's shape convention is c = -xi.

    The density is 0 outside `1 - c*y > 0`. Clamping that argument alone is not enough
    when c < 0 (Frechet case, heavy tail): 1/c is then negative, so `t**(1/c)` explodes
    at the clamp instead of vanishing, and the regulariser can dominate the DDPM loss
    by many orders of magnitude. Flooring the log-density keeps an out-of-support
    sample at a large but bounded penalty.
    """
    y = (x - loc) / scale
    if abs(c) < 1e-12:                                   # Gumbel limit
        lp = -math.log(scale) - y - torch.exp(-y)
    else:
        t = torch.clamp(1.0 - c * y, min=eps)            # support boundary
        lp = -math.log(scale) + (1.0 / c - 1.0) * torch.log(t) - t ** (1.0 / c)
    return torch.clamp(lp, min=floor)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    p.add_argument('--config', default=str(HERE / 'config.yaml'))
    p.add_argument('--out-dir', default=None,
                   help='write <out-dir>/<dataset>/ instead of the shared outputs/FIDE/<dataset>/')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--no-regularizer', action='store_true',
                   help='as-released FIDE: upstream\'s regulariser contributes no '
                        'gradient, so this trains the same plain conditional DDPM.')
    ns = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cfg = OmegaConf.to_object(OmegaConf.load(ns.config))
    hp = cfg['fide']
    npy_path = common.data_path(ns.dataset)

    seed = ns.seed if ns.seed is not None else int(cfg['run']['seed'])
    n_epochs = ns.epochs if ns.epochs is not None else int(hp['n_epochs'])
    use_reg = not ns.no_regularizer
    if ns.out_dir is None:
        gen_path = common.generated_path(METHOD, ns.dataset)
        out_dir = gen_path.parent
    else:
        out_dir = Path(ns.out_dir) / ns.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_path = out_dir / f'{METHOD}_{ns.dataset}_generatedata.npy'

    set_seed(seed)

    x_native = np.load(npy_path).astype(np.float32)
    if x_native.ndim == 2:
        x_native = x_native[..., None]
    num_samples, seq_len, dim = x_native.shape
    if dim != 1:
        raise SystemExit(f'FIDE is univariate only; got C={dim}')

    # model.py reads seq_len from this env var at import time, so it must be set first.
    os.environ['FIDE_SEQ_LEN'] = str(seq_len)
    if str(FIDE_CODE) not in sys.path:
        sys.path.insert(0, str(FIDE_CODE))
    for name in ('model', 'train_utilities', 'sampling', 'temperature_inverse'):
        sys.modules.pop(name, None)

    from model import TransformerModel                     # noqa: E402
    import train_utilities as train_ut                     # noqa: E402
    from sampling import ddpm_sample                       # noqa: E402
    from temperature_inverse import invert_fft_enhancement  # noqa: E402

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('%s: (N,T,C)=%s on %s', ns.dataset, x_native.shape, device)

    # ---- upstream's preprocessing: global StandardScaler, then a high-frequency boost.
    scaler = StandardScaler().fit(x_native.reshape(-1, 1))
    x_scaled = scaler.transform(x_native.reshape(-1, 1)).reshape(num_samples, seq_len, 1)
    real_scaled = torch.from_numpy(x_scaled.astype(np.float32)).to(device)

    t_grid = (torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
              .view(1, seq_len, 1).expand(num_samples, seq_len, 1))

    # GEV on the block maxima of the scaled (pre-boost) data, as upstream does.
    bm_value = real_scaled.max(dim=1).values.reshape(-1, 1, 1)
    bm_flat = bm_value.detach().cpu().numpy().reshape(-1)
    gev_c, gev_loc, gev_scale = genextreme.fit(bm_flat)
    gev_frozen = genextreme(gev_c, loc=gev_loc, scale=gev_scale)
    bm_samples_gev = gev_frozen.rvs(size=num_samples, random_state=seed)
    logging.info('GEV fit: c=%.4f loc=%.4f scale=%.4f', gev_c, gev_loc, gev_scale)

    # high-frequency boost (upstream's, same indices as the inverse)
    rd = real_scaled.detach().cpu().numpy()
    rd_fft = rfft(rd, axis=1)
    rd_freq = rfftfreq(seq_len)
    top_n = max(1, int((rd_fft.shape[1] * float(hp['percentage_of_freq_enhanced'])) / 100))
    top_indices = np.argsort(rd_freq)[-top_n:]
    boosted = rd_fft.copy()
    boosted[:, top_indices, 0] *= float(hp['freq_enhance_c'])
    rd_enh = irfft(boosted.reshape(-1, rd_freq.shape[0]), n=seq_len, axis=1)
    real_enh = torch.from_numpy(rd_enh.reshape(num_samples, seq_len, 1).astype(np.float32)).to(device)

    batch_size = ns.batch_size if ns.batch_size is not None else int(hp['batch_size'])
    eff_bs = min(batch_size, num_samples)
    loader = DataLoader(TensorDataset(real_enh, t_grid, bm_value),
                        batch_size=eff_bs, shuffle=True, drop_last=False)
    steps_per_epoch = len(loader)

    model = TransformerModel(dim=1, hidden_dim=int(hp['hidden_dim']),
                             max_i=train_ut.diffusion_steps).to(device)
    train_ut.model = model
    train_ut.is_regularizer = False        # upstream's detached path stays off; ours is below
    optim = torch.optim.Adam(model.parameters())
    n_params = sum(q.numel() for q in model.parameters())
    logging.info('params %d | batch %d | steps/epoch %d | epochs %d | regulariser %s',
                 n_params, eff_bs, steps_per_epoch, n_epochs, use_reg)

    reg_weight = float(hp['reg_weight'])
    t0 = time.time()
    metrics_path = out_dir / 'train_metrics.jsonl'
    with open(metrics_path, 'w', buffering=1) as mf:
        for epoch in range(n_epochs):
            ep = {'total': [], 'ddpm': [], 'reg': []}
            for x_b, t_b, bm_b in loader:
                optim.zero_grad()

                i = torch.randint(0, train_ut.diffusion_steps, size=(x_b.shape[0],))
                i = i.view(-1, 1, 1).expand_as(x_b[..., :1]).to(x_b)
                x_noisy, noise = train_ut.add_noise(x_b, t_b, i)
                pred_noise = model(x_noisy, t_b, i, bm_b)
                ddpm_loss = torch.sqrt(torch.mean((pred_noise - noise) ** 2))

                if use_reg:
                    alpha = train_ut.alphas[i.long()].to(x_b)
                    # REPAIR 2: the DDPM x0 estimate, not x_noisy - pred_noise.
                    pred_x0 = (x_noisy - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
                    bm_pred = pred_x0.max(dim=1).values                     # [B, 1], on the graph
                    # REPAIR 1: torch GEV log-density, so this actually backpropagates.
                    logp = gev_logpdf_torch(bm_pred, gev_c, gev_loc, gev_scale)
                    # REPAIR 3: lambda_1 is used rather than discarded.
                    w = train_ut.linear_decay(i[:, 0, :].reshape(-1, 1)).to(x_b)
                    reg_loss = -reg_weight * (w * logp).mean()
                    loss = ddpm_loss + reg_loss
                else:
                    reg_loss = torch.zeros((), device=x_b.device)
                    loss = ddpm_loss

                loss.backward()
                optim.step()
                ep['total'].append(float(loss.detach()))
                ep['ddpm'].append(float(ddpm_loss.detach()))
                ep['reg'].append(float(reg_loss.detach()))
            mf.write(json.dumps({'epoch': epoch,
                                 **{k: float(np.mean(v)) for k, v in ep.items()}}) + '\n')
            if epoch < 5 or epoch % 50 == 0:
                logging.info('epoch %d total=%.6f ddpm=%.6f reg=%.6f', epoch,
                             np.mean(ep['total']), np.mean(ep['ddpm']), np.mean(ep['reg']))
    train_min = (time.time() - t0) / 60
    logging.info('training done in %.1f min', train_min)

    # ---- sample N, conditioned on GEV draws, with upstream's sampler unchanged.
    model.eval()
    bm_cond = torch.tensor(bm_samples_gev, dtype=torch.float32, device=device).reshape(-1, 1, 1)
    samples = ddpm_sample(t_grid, bm_cond)
    seq_np = samples.detach().cpu().numpy().astype(np.float32)

    mean = float(scaler.mean_[0])
    scale_ = float(scaler.scale_[0]) if scaler.scale_[0] != 0 else 1.0
    seq_2d = invert_fft_enhancement(seq_np.reshape(num_samples, seq_len), seq_len,
                                    float(hp['freq_enhance_c']),
                                    int(hp['percentage_of_freq_enhanced']))
    gen = (seq_2d * scale_ + mean).astype(np.float32)[..., np.newaxis]

    np.save(gen_path, gen)
    logging.info('generated %s -> %s', gen.shape, gen_path)

    summary = {
        'method': METHOD, 'dataset': ns.dataset, 'seed': seed,
        'data_file': npy_path.name,
        'shape': list(gen.shape),
        'n_params': int(n_params), 'train_minutes': round(train_min, 1),
        'fide': dict(hp),
        'regularizer_active': bool(use_reg),
        'training': {'epochs': n_epochs, 'batch_size': eff_bs,
                     'steps_per_epoch': steps_per_epoch,
                     'gradient_steps': n_epochs * steps_per_epoch},
        'gev_fit': {'c': float(gev_c), 'loc': float(gev_loc), 'scale': float(gev_scale)},
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
