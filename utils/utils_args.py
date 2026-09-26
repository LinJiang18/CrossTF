import argparse
import os

from omegaconf import OmegaConf


def parse_args():
    """Command-line arguments; any config key left unset on the command line is taken
    from the yaml given by --config_file."""
    parser = argparse.ArgumentParser(
        description='CrossTF: train/sample the time and frequency branches, '
                    'solve the fusion coefficients, and sample with them.')
    parser.add_argument('--seed', type=int, default=0, help='training seed')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of dataloader workers')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='where checkpoints are written')

    # --- diffusion process ---
    parser.add_argument('--beta1', type=float, default=1e-5)
    parser.add_argument('--betaT', type=float, default=1e-2)
    parser.add_argument('--deterministic', action='store_true', default=False)

    parser.add_argument('--config_file', '--config', dest='config_file', type=str,
                        default='./configs/ecg.yaml', help='dataset config (see configs/)')

    # --- what to do, and with which view ---
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', dest='action', action='store_const', const='train',
                        help='fit a branch (with --crosstf: solve the fusion coefficients)')
    action.add_argument('--sample', dest='action', action='store_const', const='sample',
                        help='generate from a trained branch (with --crosstf: from both '
                             'branches with the solved coefficients)')
    view = parser.add_mutually_exclusive_group(required=True)
    view.add_argument('--time', dest='view', action='store_const', const='time',
                      help='time branch (delay embedding)')
    view.add_argument('--frequency', dest='view', action='store_const', const='frequency',
                      help='frequency branch (STFT)')
    view.add_argument('--crosstf', dest='view', action='store_const', const='crosstf',
                      help='cross-view fusion of the two trained branches')

    # --- training ---
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    # --- data ---
    parser.add_argument('--data_path', type=str, default=None,
                        help='preprocessed (N, T, C) .npy dataset')
    parser.add_argument('--name', type=str, default=None,
                        help='dataset name, e.g. `electricity`; defaults to the config file '
                             'name. Runs are stored as Time_<name><seed>, '
                             'Frequency_<name><seed> and CrossTF_<name>.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='where generated sets are written')
    parser.add_argument('--alpha_file', type=str, default=None,
                        help='fusion coefficients for --sample --crosstf (default: '
                             'logs/CrossTF_<name>/alpha.json)')
    parser.add_argument('--sample_seed', type=int, default=None,
                        help='sampling seed for --sample --crosstf (default: fixed per '
                             'training seed, see main.py)')

    # --- image transformations ---
    parser.add_argument('--n_fft', type=int)
    parser.add_argument('--hop_length', type=int)
    parser.add_argument('--delay', type=int)
    parser.add_argument('--embedding', type=int)

    # --- model ---
    parser.add_argument('--img_resolution', type=int)
    parser.add_argument('--unet_channels', type=int)
    parser.add_argument('--num_blocks', type=int, default=None)
    parser.add_argument('--ch_mult', type=int, nargs='+')
    parser.add_argument('--attn_resolution', type=int, nargs='+')
    parser.add_argument('--diffusion_steps', type=int)
    parser.add_argument('--ema', type=bool)
    parser.add_argument('--ema_warmup', type=int)
    parsed_args = parser.parse_args()

    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config_file))
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    resolve_run_identity(parsed_args)
    return parsed_args


VIEW_LABEL = {'time': 'Time', 'frequency': 'Frequency', 'crosstf': 'CrossTF'}


def resolve_run_identity(args):
    """--name + --seed + view -> args.project_name, the directory name of the run under
    logs/ and outputs/. The fusion coefficients are solved from several training seeds,
    so the CrossTF run of a dataset carries no seed."""
    stem = getattr(args, 'name', None)
    if not stem:
        cfg_file = getattr(args, 'config_file', None)
        if cfg_file:
            stem = os.path.splitext(os.path.basename(cfg_file))[0]
    if not stem:
        raise ValueError('--name is required (and no --config_file to infer it from)')
    stem = str(stem)
    if stem[-1].isdigit():
        raise ValueError(f'--name must not end in a digit, got {stem!r}: the run identity '
                         f'is <View>_<name><seed>.')
    args.name = stem
    args.prediction_mode = args.view
    if args.view == 'crosstf':
        args.project_name = f'CrossTF_{stem}'
    else:
        args.project_name = f'{VIEW_LABEL[args.view]}_{stem}{int(args.seed)}'
    return args.project_name
