import numpy as np
import torch
import torch.utils.data as Data


def MinMaxScaler(data, return_scalers=False):
    """Min-max normalise along the first axis."""
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """Min-max scale `data` with the given min and max."""
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def load_npy_dataset(data_path, seed=0):
    """Load an (N, T, C) array and min-max normalise it per channel.

    Returns (normalised data, raw data, normalised data shuffled with `seed`, min, max).
    Only the shuffled copy is used for training; the raw array keeps its file order.
    """
    raw = np.load(data_path).astype(np.float32)
    if raw.ndim != 3:
        raise ValueError(f'Expected 3D array (N, T, C), got shape {raw.shape} from {data_path}')

    flat = raw.reshape(-1, raw.shape[-1])
    min_vals = np.min(flat, axis=0)
    max_vals = np.max(flat, axis=0)
    norm_flat = (flat - min_vals) / (max_vals - min_vals + 1e-7)
    norm_data = norm_flat.reshape(raw.shape)

    idx = np.random.RandomState(seed).permutation(len(norm_data))
    shuffled_norm = norm_data[idx]
    return norm_data, raw, shuffled_norm, min_vals, max_vals


def denormalize_npy(data, min_vals, max_vals):
    return data * (max_vals - min_vals + 1e-7) + min_vals


def gen_dataloader(args):
    """Training loader for an (N, T, C) .npy dataset. Also caches the raw data and the
    normalisation constants on `args`."""
    norm_data, raw_data, train_data, min_vals, max_vals = load_npy_dataset(args.data_path, args.seed)
    args.full_data = norm_data
    args.raw_data = raw_data
    args.data_min = min_vals
    args.data_max = max_vals

    tensor = torch.Tensor(train_data)
    dataset = Data.TensorDataset(tensor)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    return loader, loader
