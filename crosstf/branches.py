"""Loading the two trained branches of one (dataset, training seed)."""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

from omegaconf import OmegaConf

from models.model import CrossTFBranch
from utils.utils import restore_checkpoint
from utils.utils_args import resolve_run_identity
from utils.utils_config import apply_data_from_npy
from utils.utils_data import gen_dataloader

# Keys the command line normally supplies and a bare yaml does not carry.
_OPTIONAL_DEFAULTS = {
    'num_workers': 0, 'log_dir': './logs',
    'beta1': 1e-5, 'betaT': 1e-2, 'deterministic': False,
}


def branch_args(config_file: str, name: str, seed: int, view: str,
                device: str = 'cuda') -> SimpleNamespace:
    """The args namespace one branch was trained with."""
    cfg = dict(OmegaConf.to_object(OmegaConf.load(config_file)))
    for key, default in _OPTIONAL_DEFAULTS.items():
        cfg.setdefault(key, default)
    args = SimpleNamespace(**cfg)
    args.config_file = config_file
    args.name = name
    args.seed = int(seed)
    args.view = view
    resolve_run_identity(args)
    apply_data_from_npy(args)
    args.device = device
    return args


def resolve_ckpt(args: SimpleNamespace, project: str, mode: str) -> str:
    """The checkpoint file under logs/<project>/ written by `main.py --train`."""
    log_root = getattr(args, 'log_dir_root', None) or './logs'
    matches = sorted(glob.glob(os.path.join(log_root, project, f'pred={mode}-*')))
    if not matches:
        raise FileNotFoundError(
            f'No checkpoint under {log_root}/{project}/. Train it first:\n'
            f'  python main.py --config_file {args.config_file} '
            f'--name {args.name} --seed {args.seed} --train --{mode}'
        )
    if len(matches) == 1:
        return matches[0]
    want = (
        f'pred={mode}-bs={args.batch_size}--lr={args.learning_rate:.4f}-'
        f'ch_mult={args.ch_mult}-attn_res={args.attn_resolution}-unet_ch={args.unet_channels}'
    )
    exact = [m for m in matches if os.path.basename(m).startswith(want)]
    if len(exact) != 1:
        raise RuntimeError(f'Cannot pick a checkpoint for {project}: {len(exact)} of '
                           f'{len(matches)} match {want!r}')
    return exact[0]


def load_model(args: SimpleNamespace, ckpt: str, device: str) -> CrossTFBranch:
    train_loader, _ = gen_dataloader(args)
    model = CrossTFBranch(args=args, device=device).to(device)
    if model.stft_embedder is not None:
        model.init_stft_embedder(train_loader)
    state = dict(model=model, epoch=0)
    ema = model.model_ema if model.use_ema else None
    restore_checkpoint(ckpt, state, device=device, ema_model=ema)
    model.eval()
    if ema is not None:
        ema.copy_to(model.net)
    return model


def load_pair(config_file: str, name: str, seed: int, device: str,
              log_root: str | None = None):
    """(time model, frequency model, the time branch's args) for one training seed."""
    pair, first = {}, None
    for view in ('time', 'frequency'):
        a = branch_args(config_file, name, seed, view, device)
        if log_root:
            a.log_dir_root = log_root
        pair[view] = load_model(a, resolve_ckpt(a, a.project_name, view), device)
        if view == 'time':
            first = a
    return pair['time'], pair['frequency'], first
