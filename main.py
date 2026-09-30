"""CrossTF entry point.

    # train one branch, then generate from it
    python main.py --config_file configs/ecg.yaml --seed 4 --train  --time
    python main.py --config_file configs/ecg.yaml --seed 4 --sample --time
    python main.py --config_file configs/ecg.yaml --seed 4 --train  --frequency

    # solve the fusion coefficients (needs both branches of every seed in
    # `crosstf.seeds`), then generate with them
    python main.py --config_file configs/ecg.yaml --train  --crosstf
    python main.py --config_file configs/ecg.yaml --seed 4 --sample --crosstf

Checkpoints go to logs/<run>/, generated sets to outputs/<run>/<run>_generatedata.npy,
where <run> is Time_<name><seed>, Frequency_<name><seed> or CrossTF_<name><seed>.
"""
import json
import logging
import math
import os

import numpy as np
import torch
import torch.multiprocessing

from models.model import CrossTFBranch
from models.sampler import DiffusionProcess
from utils.logger import PrintLogger
from utils.utils import (create_model_name_and_dir, log_config_and_tags, print_model_params,
                         restore_state, save_checkpoint)
from utils.utils_args import parse_args
from utils.utils_config import apply_data_from_npy, get_output_paths
from utils.utils_data import denormalize_npy, gen_dataloader

torch.multiprocessing.set_sharing_strategy('file_system')

# How often a per-step record is appended to train_metrics.jsonl.
METRICS_LOG_EVERY = 50

# Sampling seed used by `--sample --crosstf` for each training seed. Disjoint from the
# sampling seeds the coefficients are solved on (101, 202).
SAMPLE_SEED_FOR = {4: 303, 5: 404, 6: 505}


def generate_samples(model, args, num_samples, batch_size):
    """Sample single-branch trajectories; returns normalised series [N, T, C]."""
    generated = []
    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            process = DiffusionProcess(args, model.net, (args.seq_len, args.ts_channels))
            remaining = num_samples
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                x_ts = process.sampling(sampling_number=current_batch)
                generated.append(x_ts.detach().cpu().numpy())
                del x_ts
                remaining -= current_batch
            del process
    return np.vstack(generated)


def release_gpu_memory(*objects):
    for obj in objects:
        if obj is not None:
            del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build(args, logger):
    """Data, model and the STFT normalisation statistics (computed from the training
    data, so --sample builds the loader too)."""
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, _ = gen_dataloader(args)
    model = CrossTFBranch(args=args, device=args.device).to(args.device)
    if getattr(model, 'stft_embedder', None) is not None:
        model.init_stft_embedder(train_loader)
    print_model_params(logger, model)
    logging.info(f'prediction_mode={args.prediction_mode}')
    return model, train_loader


def sample(args, model):
    """Load the trained weights and write the generated set of one branch."""
    state = dict(model=model, epoch=0)
    ema_model = model.model_ema if args.ema else None
    if not os.path.exists(args.log_dir):
        raise FileNotFoundError(f'--sample needs a trained checkpoint, none at {args.log_dir}. '
                                f'Run the same command with --train first.')
    restore_state(args, state, ema_model=ema_model)
    args.sample_seed = int(args.sample_seed) if args.sample_seed is not None \
        else int(torch.seed()) & 0xFFFFFFFF
    torch.random.manual_seed(args.sample_seed)
    gen_batch_size = getattr(args, 'gen_batch_size', None) or args.batch_size
    logging.info(f'Generating {args.num_samples} samples (sample_seed={args.sample_seed})')
    generated_norm = generate_samples(model, args, args.num_samples, gen_batch_size)
    generated = denormalize_npy(generated_norm, args.data_min, args.data_max).astype(np.float32)
    path = get_output_paths(args)
    np.save(path, generated)
    logging.info(f'Saved {generated.shape} to {path}')


def train(args, model, train_loader, logger):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    state = dict(model=model, epoch=0)
    init_epoch = 0
    if args.resume:
        ema_model = model.model_ema if args.ema else None
        init_epoch = restore_state(args, state, ema_model=ema_model)

    output_dir = getattr(args, 'output_dir', None) or './outputs'
    metrics_log_path = os.path.join(output_dir, args.project_name, 'train_metrics.jsonl')
    os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
    metrics_log_file = open(metrics_log_path, 'a' if args.resume else 'w', buffering=1)
    global_step = init_epoch * len(train_loader)

    logging.info(f'Training for {args.epochs} epochs.')
    for epoch in range(init_epoch, args.epochs):
        model.train()
        model.epoch = epoch
        logger.log_name_params('train/epoch', epoch)
        epoch_loss_sum, epoch_loss_n, last_to_log = 0.0, 0, None
        for data in train_loader:
            x_ts = data[0].to(args.device)         # clean 1D series [B, T, C]
            optimizer.zero_grad()
            loss, to_log = model.loss_fn(x_ts)
            last_to_log = to_log
            loss_val = float(loss.detach().item())
            if not math.isfinite(loss_val):
                logging.warning(f'Non-finite loss at epoch {epoch}: {loss_val}')
                del x_ts, loss
                continue
            if global_step % METRICS_LOG_EVERY == 0:
                record = {'step': global_step, 'epoch': epoch}
                record.update(to_log)
                metrics_log_file.write(json.dumps(record) + '\n')
            global_step += 1
            epoch_loss_sum += loss_val
            epoch_loss_n += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            model.on_train_batch_end()
            del x_ts, loss
        if last_to_log is not None:
            for key, value in last_to_log.items():
                logger.log(f'train/{key}', value, epoch)
        logging.info(f'Epoch {epoch} mean loss: {epoch_loss_sum / max(epoch_loss_n, 1):.6e}')
        state['epoch'] = epoch + 1
        ema_model = model.model_ema if args.ema else None
        save_checkpoint(args.log_dir, state, epoch + 1, ema_model)
    metrics_log_file.close()
    logging.info(f'Training complete. Checkpoint: {args.log_dir}')
    release_gpu_memory(optimizer)


def crosstf_train(args):
    """Solve the per-step fusion coefficients and write logs/CrossTF_<name>/alpha.json."""
    from crosstf import save_alpha
    from crosstf.solver import solve
    cfg = dict(getattr(args, 'crosstf', None) or {})
    if getattr(args, 'solve_seeds', None):
        cfg['seeds'] = list(args.solve_seeds)
    result = solve(args.config_file, args.name,
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   log_root=args.log_dir, **cfg)
    path = save_alpha(result, args.name, args.log_dir)
    logging.info(f'alpha = {result["alpha"]} written to {path}')
    return result


def crosstf_sample(args):
    """Generate from the two trained branches with the solved coefficients applied
    inside the cross-view window."""
    from crosstf import load_alpha
    from crosstf.evaluator import Evaluator
    rec = load_alpha(args.name, args.log_dir, getattr(args, 'alpha_file', None))
    alpha = rec['alpha']
    alpha = [float(v) for v in alpha] if isinstance(alpha, list) else float(alpha)
    seed = int(args.seed)
    sample_seed = int(args.sample_seed) if args.sample_seed is not None \
        else SAMPLE_SEED_FOR.get(seed, 303)
    logging.info(f'alpha = {alpha}, window sigma <= {rec["window_sigma"]}, '
                 f'training seed {seed}, sample_seed {sample_seed}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = Evaluator(args.config_file, args.name, [seed], [sample_seed],
                   getattr(args, 'gen_batch_size', None) or args.batch_size,
                   device, rec['window_sigma'], args.log_dir)
    ev.build_prefix()
    if isinstance(alpha, list):
        if len(alpha) != len(ev.window):
            raise SystemExit(f'{len(alpha)} coefficients for a window of {len(ev.window)} steps')
        alphas = dict(zip(ev.window, alpha))
    else:
        alphas = {i: alpha for i in ev.window}
    generated = ev.generate(seed, alphas, sample_seed)
    args.project_name = f'CrossTF_{args.name}{seed}'
    args.dataset = args.project_name
    path = get_output_paths(args)
    np.save(path, generated)
    logging.info(f'Saved {generated.shape} to {path}')


def main(args):
    apply_data_from_npy(args)

    if args.view == 'crosstf':
        if args.action == 'train':
            crosstf_train(args)
        else:
            crosstf_sample(args)
        return

    name = create_model_name_and_dir(args)
    logging.info(args)
    with PrintLogger() as logger:
        log_config_and_tags(args, logger, name)
        model, train_loader = build(args, logger)
        if args.action == 'sample':
            release_gpu_memory(train_loader)
            sample(args, model)
        else:
            train(args, model, train_loader, logger)
        release_gpu_memory(model)


if __name__ == '__main__':
    args = parse_args()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
