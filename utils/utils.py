import logging
import os

import torch


def save_checkpoint(ckpt_path, state, epoch, ema_model=None):
    saved_state = {
        'epoch': epoch,
        'model': state['model'].state_dict(),
    }
    if ema_model is not None:
        saved_state['ema_model'] = ema_model.state_dict()
    torch.save(saved_state, ckpt_path)


def restore_checkpoint(ckpt_path, state, device='cuda:0', ema_model=None):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logging.warning(f'No checkpoint found at {ckpt_path}. Returned the same state as input')
        return state
    loaded_state = torch.load(ckpt_path, map_location=device)
    state['epoch'] = loaded_state['epoch']
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    if 'ema_model' in loaded_state and ema_model is not None:
        ema_model.load_state_dict(loaded_state['ema_model'])
    logging.info('Successfully loaded previous state')
    return state


def log_config_and_tags(args, logger, name):
    logger.log_name_params('config/hyperparameters', vars(args))
    logger.log_name_params('config/name', name)


def create_model_name_and_dir(args):
    """Checkpoint file name, which encodes the branch's hyperparameters.

    Sets `args.log_dir` to `<log_dir>/<run identity>/<name>`.
    """
    mode = getattr(args, 'prediction_mode', None) or 'time'
    name = (f'pred={mode}-'
            f'bs={args.batch_size}-'
            f'-lr={args.learning_rate:.4f}-'
            f'ch_mult={args.ch_mult}-'
            f'attn_res={args.attn_resolution}-'
            f'unet_ch={args.unet_channels}'
            )
    if getattr(args, 'num_blocks', None) is not None:
        name += f'-nb={args.num_blocks}'
    if mode == 'time':
        assert (args.delay is not None and args.embedding is not None)
        name += f'-delay={args.delay}-{args.embedding}'
    if mode == 'frequency':
        assert (args.n_fft is not None and args.hop_length is not None)
        name += f'-stft={args.n_fft}-{args.hop_length}'
    args.log_dir = '%s/%s/%s' % (args.log_dir, args.dataset, name)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    return name


def restore_state(args, state, ema_model=None):
    logging.info('restoring checkpoint from: {}'.format(args.log_dir))
    restore_checkpoint(args.log_dir, state, device=getattr(args, 'device', 'cuda:0'),
                       ema_model=ema_model)
    return state['epoch']


def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info('number of model parameters: {}'.format(params_num))
    logger.log_name_params('config/params_num', params_num)
