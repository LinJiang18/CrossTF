import os

import numpy as np
import torch

from models.img_transformations import DelayEmbedder


def extract_dataset_name(data_path):
    """Dataset name from a file such as ECG_seq_250_sample_1000.npy."""
    basename = os.path.splitext(os.path.basename(data_path))[0]
    if '_seq_' in basename:
        return basename.split('_seq_')[0]
    return basename


def delay_embed_roundtrip_ok(seq_len, input_channels, delay, embedding):
    """Check that ts -> image -> ts is lossless (shape match, no zero gaps, MAE ~ 0)."""
    embedder = DelayEmbedder('cpu', seq_len, delay, embedding)
    sample = torch.randn(2, seq_len, input_channels)
    try:
        image = embedder.ts_to_img(sample, pad=True)
        reconstructed = embedder.img_to_ts(image)
        if reconstructed.shape != sample.shape:
            return False
        zero_frac = (reconstructed.numpy() == 0).mean()
        mae = (reconstructed - sample).abs().mean().item()
        return zero_frac == 0 and mae < 1e-5
    except (RuntimeError, IndexError):
        return False


def infer_attn_resolution(img_resolution):
    resolutions = []
    current = img_resolution
    while current >= 4:
        resolutions.append(int(current))
        current //= 2
    return resolutions if resolutions else [4]


_REQUIRED = ('delay', 'embedding', 'img_resolution', 'n_fft', 'hop_length',
             'ch_mult', 'attn_resolution', 'batch_size', 'gen_batch_size')


def apply_data_from_npy(args, data_path=None):
    """Read (N, T, C) from the dataset file and derive the geometry of both 2D views.

    The diffusion itself always runs on the 1D series; `prediction_mode` selects which 2D
    representation (delay embedding for `time`, STFT for `frequency`) the denoiser uses.
    `crosstf` loads both trained branches, so both views' fields are filled in.
    """
    data_path = data_path or getattr(args, 'data_path', None)
    if not data_path:
        raise ValueError('data_path must be set in the config or via --data_path')
    missing = [k for k in _REQUIRED if getattr(args, k, None) is None]
    if missing:
        raise ValueError(f'config is missing {missing}')

    raw = np.load(data_path)
    if raw.ndim != 3:
        raise ValueError(f'Expected 3D array (N, T, C), got shape {raw.shape} from {data_path}')

    num_samples, seq_len, ts_channels = raw.shape
    args.data_path = data_path
    args.source_dataset = extract_dataset_name(data_path)
    args.project_name = getattr(args, 'project_name', None) or args.source_dataset
    args.dataset = args.project_name
    args.seq_len = seq_len
    args.ts_channels = ts_channels
    args.num_samples = num_samples
    args.input_size = ts_channels

    mode = getattr(args, 'prediction_mode', None)
    if mode is None:
        raise ValueError('prediction_mode is unset: pass --time, --frequency or --crosstf')
    args.prediction_mode = str(mode).lower()
    if args.prediction_mode not in ('time', 'frequency', 'crosstf'):
        raise ValueError(f'prediction_mode must be time|frequency|crosstf, got {args.prediction_mode}')

    need_delay = args.prediction_mode in ('time', 'crosstf')
    need_stft = args.prediction_mode in ('frequency', 'crosstf')

    if need_stft:
        # torchaudio's centre-padded STFT: n_fft // 2 + 1 bins by this many frames.
        stft_h = args.n_fft // 2 + 1
        stft_pad = args.n_fft // 2
        stft_w = (seq_len + 2 * stft_pad - args.n_fft) // args.hop_length + 1
        args.stft_raw_hw = (stft_h, stft_w)
        args.freq_img_resolution = max(stft_h, stft_w)
        args.freq_input_channels = 2 * ts_channels
        if getattr(args, 'freq_attn_resolution', None) is None:
            args.freq_attn_resolution = infer_attn_resolution(args.freq_img_resolution)

    if need_delay:
        args.time_img_resolution = int(args.img_resolution)
        if not delay_embed_roundtrip_ok(seq_len, ts_channels, args.delay, args.embedding):
            raise ValueError(
                f'Delay-embedding roundtrip failed for {args.project_name}: '
                f'seq_len={seq_len}, delay={args.delay}, embedding={args.embedding}'
            )

    if need_stft and not need_delay:
        # Frequency branch alone: the STFT image is the only 2D canvas.
        args.img_resolution = args.freq_img_resolution
        if max(args.attn_resolution) > args.img_resolution:
            args.attn_resolution = infer_attn_resolution(args.img_resolution)

    # The diffusion state is the 1D series, so its channels are the series' channels.
    args.input_channels = ts_channels
    args.use_stft = bool(need_stft)
    return args


def get_output_paths(args):
    output_root = getattr(args, 'output_dir', None) or './outputs'
    project_name = getattr(args, 'project_name', None) or args.dataset
    dataset_dir = os.path.join(output_root, project_name)
    os.makedirs(dataset_dir, exist_ok=True)
    return os.path.join(dataset_dir, f'{project_name}_generatedata.npy')
