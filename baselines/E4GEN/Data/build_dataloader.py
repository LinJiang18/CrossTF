import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import joblib
from typing import Any, Optional, Tuple


class Normalizer:
    def __init__(self, scaler, auto_norm=True):
        self.scaler = scaler
        self.auto_norm = auto_norm

    def normalize(self, x):
        # x: (M, C)
        x = self.scaler.transform(x)
        if self.auto_norm:
            x = x * 2.0 - 1.0
        return x

    def unnormalize(self, x):
        # x: (M, C)
        if self.auto_norm:
            x = (x + 1.0) / 2.0
        return self.scaler.inverse_transform(x)


def normalize_threshold(normalizer, threshold, feature_size=1):
    """
    Convert threshold from original (raw) space to normalized space using the same
    normalizer as the data (MinMaxScaler + *2-1 to [-1, 1]).

    Args:
        normalizer: Normalizer instance with .normalize(x) where x is (M, C).
        threshold: scalar or array of shape (C,) in original space.
        feature_size: number of channels C (used when threshold is scalar).

    Returns:
        float if feature_size==1 and threshold is scalar, else np.ndarray of shape (C,).
    """
    if np.isscalar(threshold) or isinstance(threshold, (int, float)):
        threshold = np.array([float(threshold)] * feature_size, dtype=np.float64)
    else:
        threshold = np.asarray(threshold, dtype=np.float64).flatten()
        if threshold.size != feature_size:
            raise ValueError(
                "threshold size {} must match feature_size {}.".format(threshold.size, feature_size)
            )
    x = threshold.reshape(1, -1)  # (1, C)
    out = normalizer.normalize(x)  # (1, C)
    if out.size == 1:
        return float(out.flat[0])
    return out.flatten()


class SplitWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        """
        data: np.ndarray of shape (M, C) — long series, sliced into windows of length seq_length.
        seq_length: int
        """
        self.data = torch.from_numpy(data).float()
        self.seq_length = seq_length

        # number of full chunks
        self.num_chunks = self.data.shape[0] // seq_length

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        return self.data[start:end]


class PrechunkedDataset(torch.utils.data.Dataset):
    """Normalized samples already shaped (N, L, C); one batch item per row."""

    def __init__(self, data_nlc: np.ndarray):
        x = np.asarray(data_nlc)
        if x.ndim != 3:
            raise ValueError("PrechunkedDataset expects (N, L, C), got shape={}".format(x.shape))
        self.data = torch.from_numpy(x.astype(np.float32, copy=False)).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

def split_raw_for_train_val(
    data: np.ndarray, seq_length: int, val_holdout_fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D (M, C): split on seq_length-aligned row boundaries (contiguous windows).
    3D (N, L, C): split on sample axis N; requires L == seq_length (no reflow).

    The last ``val_holdout_fraction`` of windows/samples are validation.

    Returns:
        data_train, data_val — val empty array if no split applies.
    """
    data = np.asarray(data)
    if data.ndim == 3:
        N, L, C = data.shape
        if L != seq_length:
            raise ValueError(
                "3D data has L={} but config seq_length={}; they must match.".format(L, seq_length)
            )
        if N < 2:
            return data, np.empty((0, L, C), dtype=data.dtype)
        frac = float(val_holdout_fraction)
        if frac <= 0.0:
            return data, np.empty((0, L, C), dtype=data.dtype)
        if abs(frac - 0.5) < 1e-9:
            n_train = N // 2
        else:
            n_val_target = int(round(N * frac))
            n_val_target = max(1, min(n_val_target, N - 1))
            n_train = N - n_val_target
        n_val = N - n_train
        if n_train < 1 or n_val < 1:
            return data, np.empty((0, L, C), dtype=data.dtype)
        return data[:n_train], data[n_train:]

    if data.ndim != 2:
        raise ValueError("split_raw_for_train_val expects 2D (M, C) or 3D (N, L, C), got shape={}".format(data.shape))

    M, C = data.shape
    n_chunks = M // seq_length
    if n_chunks < 2:
        return data[: n_chunks * seq_length], np.empty((0, C), dtype=data.dtype)
    usable_len = n_chunks * seq_length
    data = data[:usable_len]
    frac = float(val_holdout_fraction)
    if frac <= 0.0:
        return data, np.empty((0, C), dtype=data.dtype)
    # Exact half/half when frac==0.5: first floor(n/2) windows train, rest validation.
    if abs(frac - 0.5) < 1e-9:
        n_train_chunks = n_chunks // 2
    else:
        n_val_target = int(round(n_chunks * frac))
        n_val_target = max(1, min(n_val_target, n_chunks - 1))
        n_train_chunks = n_chunks - n_val_target
    n_val = n_chunks - n_train_chunks
    if n_train_chunks < 1 or n_val < 1:
        return data, np.empty((0, C), dtype=data.dtype)
    split_row = n_train_chunks * seq_length
    return data[:split_row], data[split_row:]


def split_to_chunks(data, seq_length):
    """
    If data is 2D: (M, C) -> split into (N, seq_length, C) by trimming.
    If data is 3D: (N, L, C) -> require L == seq_length, then return as-is.
    """
    x = np.asarray(data)

    if x.ndim == 2:
        M, C = x.shape
        num_chunks = M // seq_length
        trimmed = x[: num_chunks * seq_length]
        chunks = trimmed.reshape(num_chunks, seq_length, C)
        return chunks

    if x.ndim == 3:
        N, L, C = x.shape
        if L != seq_length:
            raise ValueError(
                "data is (N, L, C) but L != seq_length: L={}, seq_length={}".format(L, seq_length)
            )
        return x

    raise ValueError(f"data must be 2D (M, C) or 3D (N, L, C), got shape={x.shape}")

def load_npy_or_npz_X(path):
    """
    Load data from .npy or .npz.
    - .npz: return arr["X"]
    Post:
      - if 1D: (M,) -> (M,1)
      - if 2D: (M,C) keep (contiguous stream; windows are formed by chunking)
      - if 3D: (N,L,C) keep — each of N rows is one sample of length L
    """
    obj = np.load(path, allow_pickle=False)

    if isinstance(obj, np.lib.npyio.NpzFile):
        if "X" not in obj.files:
            raise KeyError(f"NPZ file has no 'X'. available keys: {obj.files}")
        x = obj["X"]
        obj.close()
    else:
        x = obj

    x = np.asarray(x)

    if x.ndim == 1:
        x = x[:, None]

    return x


def extreme_same_file_train_val_holdout(config: dict) -> bool:
    """
    True when stage-2/3 should train only on the train split of ``train.params.data_path``:
    ``val_holdout_fraction`` is set and there is no explicit
    ``extreme_model.prediction.validation_data_path``.

    Diffusion training (``--train``) always uses the full aligned dataset; this flag only
    affects whether the DataLoader built here is restricted to the train split.
    """
    train_params = config.get("train", {}).get("params", {})
    data_path = train_params.get("data_path", "") or ""
    val_holdout_raw = train_params.get("val_holdout_fraction")
    vf: Optional[float] = None
    if val_holdout_raw is not None:
        v = float(val_holdout_raw)
        if 0.0 < v < 1.0:
            vf = v
    ext_cfg = config.get("extreme_model", {}).get("prediction", {})
    validation_data_path = ext_cfg.get("validation_data_path")
    if validation_data_path == "":
        validation_data_path = None
    return vf is not None and validation_data_path is None and bool(data_path)


def build_dataloader(
    config: dict,
    args: Any,
    restrict_dataset_to_train_split: bool = False,
):

    # -------- config --------
    train_params = config["train"]["params"]
    npy_path = train_params["data_path"]
    seq_length = train_params["seq_length"]
    batch_size = train_params["batch_size"]
    val_holdout_raw = train_params.get("val_holdout_fraction")
    val_holdout_fraction: Optional[float] = None
    if val_holdout_raw is not None:
        val_holdout_fraction = float(val_holdout_raw)
        if not (0.0 < val_holdout_fraction < 1.0):
            val_holdout_fraction = None

    # -------- load raw data --------
    # data = np.load(npy_path)
    # if data.ndim == 1:
    #     data = data[:, None]
    # # (M, C)
    data = load_npy_or_npz_X(npy_path)

    if data.ndim == 3:
        N, L, C = data.shape
        if L != seq_length:
            raise ValueError(
                "3D data_path has L={} but train.params.seq_length={}; align config or reshape data.".format(
                    L, seq_length
                )
            )
    elif data.ndim == 2:
        M, _C = data.shape
        n_keep = (M // seq_length) * seq_length
        if n_keep < seq_length:
            raise ValueError(
                "2D data_path has fewer than one full window: M={}, seq_length={}, path={}.".format(
                    M, seq_length, npy_path
                )
            )
        data = data[:n_keep]

    # Full aligned stream / samples: scaler + origin_*.npy always match this (same size as chunked dataset).
    scaler = MinMaxScaler()
    if data.ndim == 2:
        scaler.fit(data)
    else:
        scaler.fit(data.reshape(-1, data.shape[-1]))

    normalizer = Normalizer(scaler, auto_norm=True)
    origin_unnorm = split_to_chunks(data, seq_length)  # (N, L, C)
    if data.ndim == 2:
        data_norm_full = normalizer.normalize(data)
        origin_norm = split_to_chunks(data_norm_full, seq_length)
    else:
        flat = data.reshape(-1, data.shape[-1])
        data_norm_full = normalizer.normalize(flat).reshape(data.shape)
        origin_norm = data_norm_full

    # -------- save both (full aligned dataset; N matches len(dataset) when not restricted) --------
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f"origin_data_unnorm_{args.name}.npy"), origin_unnorm)
    np.save(os.path.join(args.output_dir, f"origin_data_norm_{args.name}.npy"), origin_norm)
    joblib.dump({"scaler": scaler, "auto_norm": True}, os.path.join(args.output_dir, "normalizer.joblib"))

    # -------- DataLoader: full data for diffusion / sampling; optional train split for extreme stages --------
    use_train_split = (
        bool(restrict_dataset_to_train_split)
        and val_holdout_fraction is not None
    )
    if use_train_split:
        data_train, _data_val = split_raw_for_train_val(data, seq_length, val_holdout_fraction)
        if data_train.size == 0:
            raise ValueError(
                "val_holdout_fraction={} leaves no training data for data_path={} (seq_length={}).".format(
                    val_holdout_fraction, npy_path, seq_length
                )
            )
        if data.ndim == 2:
            data_norm = normalizer.normalize(data_train)
        else:
            flat_tr = data_train.reshape(-1, data_train.shape[-1])
            data_norm = normalizer.normalize(flat_tr).reshape(data_train.shape)
    else:
        data_norm = data_norm_full

    if data_norm.ndim == 3:
        dataset = PrechunkedDataset(data_norm)
    else:
        dataset = SplitWindowDataset(data_norm, seq_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return {"dataloader": dataloader, "normalizer": normalizer, "scaler": scaler}


def load_normalizer_from_disk(output_dir):
    """Load normalizer from normalizer.joblib; returns None if file not found."""
    path = os.path.join(output_dir, "normalizer.joblib")
    if not os.path.isfile(path):
        return None
    d = joblib.load(path)
    return Normalizer(d["scaler"], d.get("auto_norm", True))


def unnormalize_sample(samples: np.ndarray, normalizer) -> np.ndarray:
    """
    samples: np.ndarray, shape (B, T, 1)
    normalizer: instance of Normalizer(scaler, auto_norm=...)
    return: np.ndarray, shape (B, T, 1) in original scale
    """
    samples = np.asarray(samples)
    assert samples.ndim == 3 , f"expected dim = 3"

    B, T, C = samples.shape
    x2d = samples.reshape(B * T, C)
    x2d = normalizer.unnormalize(x2d)
    return x2d.reshape(B, T, C)

