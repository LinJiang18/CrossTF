"""Train RF-Diffusion on one dataset, sample N series from it, and write the generated set.

Why this file rather than upstream `train.py`: `train.py::train`, the single-GPU path,
maps task ids to the wrong models -- task_id 0 builds `tfdiff_eeg`, 1 builds
`tfdiff_mimo`, everything else builds `tfdiff_WiFi`, while `train_distributed` has the
correct 0/1/2/3 -> WiFi/FMCW/MIMO/EEG mapping. We run on one GPU, so this runner builds
`tfdiff_WiFi` and `SignalDiffusion` directly and trains them with upstream's
`tfdiffLearner` and data collator.

Sampling uses `SignalDiffusion.robust_sampling`, the full reverse chain from pure noise.
Upstream's own `inference.py` calls `native_sampling`, which degrades the real data and
takes a single restore step -- a reconstruction, not generation.

Run `prepare_data.py` first. Then:

    python run_rfdiff.py --dataset precipitation --epochs 200 --batch-size 32 --seed 1

Checkpoints and tensorboard logs go to `$RFDIFF_WORK/{model,log}/<dataset>/` (default
`baselines/RF-Diffusion/work/`); training resumes from the latest checkpoint there.
The generated set goes to `outputs/RF-Diffusion/<dataset>/`.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))              # tfdiff/, complex/
sys.path.insert(0, str(HERE.parent))       # common.py
import common                                             # noqa: E402

from tfdiff.params import AttrDict, params_wifi          # noqa: E402
from tfdiff.diffusion import SignalDiffusion             # noqa: E402
from tfdiff.wifi_model import tfdiff_WiFi                # noqa: E402
from tfdiff.learner import tfdiffLearner                 # noqa: E402
from tfdiff.dataset import from_path                     # noqa: E402

METHOD = 'RF-Diffusion'
WORK = Path(os.environ.get('RFDIFF_WORK', HERE / 'work'))
BLUR = 0.1 ** 2      # README deviation 3: params_wifi's 1e-10 flattens the blur kernel to 1


def build_params(ds: str, t: int, max_iter, batch_size: int) -> AttrDict:
    p = AttrDict(dict(params_wifi))
    p.override(dict(
        task_id=0,
        sample_rate=t, input_dim=1, extra_dim=[1], cond_dim=6,
        data_dir=[str(WORK / 'dataset' / ds / 'raw')],
        model_dir=str(WORK / 'model' / ds),
        log_dir=str(WORK / 'log' / ds),
        batch_size=batch_size, max_iter=max_iter,
        blur_schedule=(BLUR * np.ones(params_wifi.max_step)).tolist(),
    ))
    return p


@torch.no_grad()
def generate(model, diffusion, p, n_want: int, device) -> np.ndarray:
    """robust_sampling from pure noise, in batches, with the constant condition."""
    model.eval()
    out, bs = [], p.batch_size
    while sum(len(o) for o in out) < n_want:
        cond = torch.zeros(bs, p.cond_dim, 2, device=device)
        x0 = diffusion.robust_sampling(model, cond, device)      # [B, T, 1, 2]
        out.append(x0[..., 0].squeeze(-1).float().cpu().numpy())  # real part -> [B, T]
        print(f'  sampled {sum(len(o) for o in out)}/{n_want}', flush=True)
    return np.concatenate(out)[:n_want]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset', required=True, choices=sorted(common.DATASETS))
    ap.add_argument('--epochs', type=int, default=200,
                    help='training length; max_iter = epochs * iterations per epoch '
                         'unless --max-iter is given')
    ap.add_argument('--max-iter', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-samples', type=int, default=None,
                    help='number of series to generate (default: N of the real set; '
                         'smaller values are for testing only)')
    ap.add_argument('--seed-subdir', action='store_true',
                    help='write under outputs/RF-Diffusion/seed<N>/ (default: no seed subdir)')
    ns = ap.parse_args()
    device = torch.device(ns.device)
    torch.manual_seed(ns.seed); np.random.seed(ns.seed)

    real = np.load(common.data_path(ns.dataset)).astype(np.float32)     # (N, T, 1)
    n, t = real.shape[0], real.shape[1]
    n_gen = ns.n_samples if ns.n_samples is not None else n
    p = build_params(ns.dataset, t, None, ns.batch_size)
    print(f'{ns.dataset}: N={n} T={t}  blur={BLUR}', flush=True)

    model = tfdiff_WiFi(p).to(device)
    # tfdiffLearner reads `model.device`, which only DistributedDataParallel provides;
    # setting the attribute here keeps tfdiff/ untouched.
    model.device = device
    print(f'  params: {sum(q.numel() for q in model.parameters()):,}', flush=True)
    dataset = from_path(p)
    max_iter = ns.max_iter if ns.max_iter is not None else ns.epochs * len(dataset)
    p.override(dict(max_iter=max_iter))
    n_ep = max_iter / len(dataset)
    # lr follows the scheduler installed below (deviation 6): halved every epochs//4
    # epochs, not every epoch as upstream does.
    n_halvings = n_ep // max(1, ns.epochs // 4)
    print(f'  {len(dataset)} iters/epoch, max_iter={max_iter} ({n_ep:.0f} epochs), '
          f'lr {p.learning_rate:.1e} -> {p.learning_rate * 0.5**n_halvings:.2e} '
          f'({n_halvings:.0f} halvings)', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=p.learning_rate)
    learner = tfdiffLearner(p.log_dir, p.model_dir, model, dataset, opt, p)
    learner.is_master = True
    # README deviation 6. Upstream sets StepLR(opt, 1, gamma=0.5) and steps it once per
    # epoch, so lr = 1e-3 * 0.5^epoch and training is numerically dead by epoch ~15.
    # Halve four times across the whole run instead, ending at lr/16.
    learner.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, ns.epochs // 4), gamma=0.5)
    learner.restore_from_checkpoint()
    t0 = time.time()
    learner.train(max_iter=max_iter)
    train_min = (time.time() - t0) / 60
    print(f'  trained {train_min:.1f} min', flush=True)

    diffusion = SignalDiffusion(p)
    gen = generate(learner.model, diffusion, p, n_gen, device)    # (n_gen, T), normalised

    # README deviation 5. The collator normalises every series by its own mean and std, so
    # the model never sees an amplitude and what it returns has none. Draw one (mean, std)
    # pair per generated series from the real per-series pairs, so the marginal
    # distribution of amplitudes is preserved rather than collapsed onto one global scale.
    st = np.load(WORK / 'dataset' / ns.dataset / 'stats.npz')
    rng = np.random.default_rng(ns.seed)
    pick = rng.integers(0, len(st['mean']), size=len(gen))
    g = gen.astype(np.float64)
    g = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + 1e-12)
    gen_scaled = (g * st['std'][pick][:, None] + st['mean'][pick][:, None]).astype(np.float32)

    seed_dir = ns.seed if ns.seed_subdir else None
    out_file = common.generated_path(METHOD, ns.dataset, seed_dir)
    np.save(out_file, gen_scaled[:, :, None])
    json.dump({'dataset': ns.dataset, 'model': METHOD, 'seed': ns.seed,
               'data_file': common.DATASETS[ns.dataset],
               'shape_real': list(real.shape),
               'shape_generated': list(gen_scaled[:, :, None].shape),
               'n_params': int(sum(q.numel() for q in model.parameters())),
               'train_minutes_this_process': round(train_min, 1), 'max_iter': max_iter,
               'epochs': round(max_iter / len(dataset), 1),
               'blur_schedule': BLUR, 'sampler': 'robust_sampling',
               'scale': 'per-series (mean, std) drawn from the real set (README deviation 5)'},
              open(out_file.parent / 'summary.json', 'w'), indent=2)
    print(f'  wrote {out_file}', flush=True)


if __name__ == '__main__':
    main()
