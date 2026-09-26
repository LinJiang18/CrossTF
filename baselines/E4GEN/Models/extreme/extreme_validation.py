"""
Validation metrics and evaluation flow for the extreme semantic predictor.

Metrics:
  1. num_events_mae: mean absolute error in the number of extreme events per sample.
  2. interval_length_mae: mean absolute error in interval length (in points), over valid events.
  3. interval_position_mae: mean absolute error in interval position (start/end), over valid events.
  4. pmax_mae: mean absolute error in pmax, over valid events.

Data: optional separate validation .npy, or the same file as training with
``train.params.val_holdout_fraction`` (last fraction of complete windows held out).
Diffusion training uses the full aligned dataset; this holdout matches the train split
used for stage-2/3 when no separate ``validation_data_path`` is set.

Visualization: two panels per sample—top = x0_hat (diffusion-TS) + predicted extremes (red);
bottom = x0 (ground truth) + actual extremes (blue). Title shows timestep t for x0_hat.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from Data.build_dataloader import (
    load_npy_or_npz_X,
    load_normalizer_from_disk,
    normalize_threshold,
    split_raw_for_train_val,
    split_to_chunks,
)
from Models.extreme.extreme_semantic import ExtremeLabels, ExtremeSemantic
from Models.extreme.extreme_prediction import (
    ExtremeSemanticPredictor,
    run_frozen_xpred_step,
    build_extreme_labels_from_x0,
)
from Utils.io_utils import checkpoint_run_dir


# ---------------------------------------------------------------------------
# 1) Pure metric computation (pred vs labels, batched)
# ---------------------------------------------------------------------------


def compute_num_events_mae(
    labels: ExtremeLabels,
    pred_m: torch.Tensor,
) -> float:
    """
    Mean absolute error in the number of extreme events per sample.

    Args:
        labels: ground truth ExtremeLabels; labels.m is (B, K_max).
        pred_m: predicted validity mask (B, K_max) bool.

    Returns:
        Scalar: mean over batch of |n_pred[b] - n_gt[b]|.
    """
    # n_gt[b], n_pred[b]: number of valid events per sample
    n_gt = labels.m.sum(dim=1).float()   # (B,)
    n_pred = pred_m.sum(dim=1).float()  # (B,)
    return (n_pred - n_gt).abs().mean().item()


def compute_interval_length_mae(
    labels: ExtremeLabels,
    pred_ts: torch.Tensor,
    pred_te: torch.Tensor,
) -> float:
    """
    Mean absolute error in interval length (in points), averaged over valid events only.

    len = te - ts + 1. Only (b, e) with labels.m[b, e] == 1 are included.

    Args:
        labels: ground truth (ts, te, m).
        pred_ts, pred_te: (B, K_max) predicted start/end indices.

    Returns:
        Scalar: mean of |len_pred - len_gt| over valid (b, e).
    """
    len_gt = (labels.te - labels.ts + 1).float()
    len_pred = (pred_te - pred_ts + 1).float().clamp(min=0)
    mask = labels.m.float()
    if mask.sum() < 1:
        return 0.0
    return ((len_pred - len_gt).abs() * mask).sum().item() / mask.sum().item()


def compute_interval_position_mae(
    labels: ExtremeLabels,
    pred_ts: torch.Tensor,
    pred_te: torch.Tensor,
) -> float:
    """
    Mean absolute error in interval position (start and end), averaged over valid events.

    pos_err[b,e] = (|ts_hat - ts_gt| + |te_hat - te_gt|) / 2. Only (b, e) with labels.m[b, e] == 1.

    Args:
        labels: ground truth (ts, te, m).
        pred_ts, pred_te: (B, K_max) predicted start/end indices.

    Returns:
        Scalar: mean position error in points.
    """
    pos_err = (
        (pred_ts.float() - labels.ts.float()).abs() +
        (pred_te.float() - labels.te.float()).abs()
    ) / 2.0
    mask = labels.m.float()
    if mask.sum() < 1:
        return 0.0
    return (pos_err * mask).sum().item() / mask.sum().item()


def compute_pmax_mae(
    labels: ExtremeLabels,
    pred_pmax: torch.Tensor,
) -> float:
    """
    Mean absolute error in pmax, averaged over valid events.

    Args:
        labels: ground truth (pmax, m).
        pred_pmax: (B, K_max) predicted pmax.

    Returns:
        Scalar: mean of |pmax_hat - pmax_gt| over valid (b, e).
    """
    mask = labels.m.float()
    if mask.sum() < 1:
        return 0.0
    diff = (pred_pmax - labels.pmax).abs() * mask
    return diff.sum().item() / mask.sum().item()


def compute_all_metrics(
    labels: ExtremeLabels,
    pred_ts: torch.Tensor,
    pred_te: torch.Tensor,
    pred_m: torch.Tensor,
    pred_pmax: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute the four validation metrics given batched labels and predictions.

    Args:
        labels: ground truth ExtremeLabels (ts, te, pmax, m), shapes (B, K_max).
        pred_ts, pred_te: (B, K_max) predicted start/end.
        pred_m: (B, K_max) bool predicted validity.
        pred_pmax: (B, K_max) predicted pmax.

    Returns:
        Dict with keys: num_events_mae, interval_length_mae, interval_position_mae, pmax_mae.
    """
    return {
        "num_events_mae": compute_num_events_mae(labels, pred_m),
        "interval_length_mae": compute_interval_length_mae(labels, pred_ts, pred_te),
        "interval_position_mae": compute_interval_position_mae(labels, pred_ts, pred_te),
        "pmax_mae": compute_pmax_mae(labels, pred_pmax),
    }


# ---------------------------------------------------------------------------
# 2) Validation data loading and normalization
# ---------------------------------------------------------------------------


class NormalizerSimple:
    """MinMax scale to [0,1] then linear map to [-1, 1] (match training)."""

    def __init__(self, data_2d: np.ndarray):
        try:
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            raise ImportError("sklearn is required for validation; install scikit-learn.")
        self.scaler = MinMaxScaler()
        self.scaler.fit(data_2d)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return (x * 2.0 - 1.0).astype(np.float32)

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        npy = x.detach().cpu().numpy()
        out = self.normalize(npy)
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def load_validation_npy(
    path: str,
    seq_length: int,
    normalizer: Optional[Any] = None,
) -> Tuple[torch.Tensor, Any]:
    """
    Load validation .npy and return normalized x0 tensor (N, L, C) and the normalizer used.

    Args:
        path: path to .npy (e.g. Data/Extreme_Data/toyexample_timeseries/topexample_seq_200_sample_1000_validation.npy).
        seq_length: L to chunk sequences.
        normalizer: if provided, use it to normalize; else fit MinMaxScaler on loaded data and map to [-1,1].

    Returns:
        x0: (N, L, C) float tensor, normalized in [-1, 1].
        normalizer: the normalizer instance (for reuse).
    """
    data = load_npy_or_npz_X(path)
    chunks = split_to_chunks(data, seq_length)
    flat = chunks.reshape(-1, chunks.shape[-1])
    if normalizer is None:
        normalizer = NormalizerSimple(flat)
    x_np = normalizer.normalize(flat).reshape(chunks.shape)
    x0 = torch.from_numpy(x_np).float()
    return x0, normalizer


def load_validation_npy_from_train_file_split(
    train_data_path: str,
    seq_length: int,
    val_holdout_fraction: float,
    normalizer: Optional[Any] = None,
) -> Tuple[torch.Tensor, Any]:
    """
    Same file as training: take the validation tail (same ``split_raw_for_train_val`` as stage-2 train split).
    """
    data = load_npy_or_npz_X(train_data_path)
    _data_train, data_val = split_raw_for_train_val(data, seq_length, val_holdout_fraction)
    if data_val.size == 0:
        c = int(data.shape[-1])
        return torch.empty(0, seq_length, c).float(), normalizer
    chunks = split_to_chunks(data_val, seq_length)
    flat = chunks.reshape(-1, chunks.shape[-1])
    if normalizer is None:
        normalizer = NormalizerSimple(flat)
    x_np = normalizer.normalize(flat).reshape(chunks.shape)
    x0 = torch.from_numpy(x_np).float()
    return x0, normalizer


# ---------------------------------------------------------------------------
# 3) Run predictor on a batch and get predictions
# ---------------------------------------------------------------------------


def run_predictor_batch(
    diffusion_model: torch.nn.Module,
    predictor: ExtremeSemanticPredictor,
    x0: torch.Tensor,
    semantic_extractor: ExtremeSemantic,
    threshold: Union[float, torch.Tensor],
    t_min: int,
    t_max: int,
    device: Union[torch.device, str],
) -> tuple[ExtremeLabels, Dict[str, torch.Tensor]]:
    """
    For a batch of real x0 (B, L, C): build labels from x0, get x0_hat via frozen diffusion,
    run predictor to get per-point extreme logits and backbone features (no decoding).

    Returns:
        labels: ExtremeLabels from x0.
        pred: dict with extreme_logits (B,L), F (B,L,d), t (B,), x0_hat (B,L,C).
    """
    x0 = x0.to(device)
    diffusion_model.eval()
    predictor.eval()
    with torch.no_grad():
        labels = build_extreme_labels_from_x0(x0, threshold, semantic_extractor)
        x0_hat, _, t = run_frozen_xpred_step(
            diffusion_model, x0, t_min=t_min, t_max=t_max
        )
        out = predictor(x0_hat, t, tau=0.5)  # tau only used for optional decoded fields; we use raw logits
    return labels, {
        "extreme_logits": out["extreme_logits"],
        "F": out["F"],
        "t": t,
        "x0_hat": x0_hat,
    }


def decode_batch_with_tau(
    predictor: ExtremeSemanticPredictor,
    extreme_logits: torch.Tensor,
    F: torch.Tensor,
    t: torch.Tensor,
    tau: float,
    x0_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode extreme logits with threshold tau and compute pmax_hat. Returns (ts_hat, te_hat, m_hat, pmax_hat)."""
    ts_hat, te_hat, m_hat = predictor.decode_intervals_from_logits(
        extreme_logits, tau, predictor.K_max
    )
    pmax_hat = predictor._predict_pmax(F, x0_hat, ts_hat, te_hat, t)
    return ts_hat, te_hat, m_hat, pmax_hat


def tune_tau_on_validation(
    validation_data_path: Optional[str],
    seq_length: int,
    diffusion_model: torch.nn.Module,
    predictor: ExtremeSemanticPredictor,
    K_max: int,
    threshold: Union[float, torch.Tensor],
    t_min: int,
    t_max: int,
    device: Union[torch.device, str],
    batch_size: int = 64,
    normalizer: Optional[Any] = None,
    threshold_candidates: Optional[List[float]] = None,
    logger: Optional[Any] = None,
    train_data_path_for_split: Optional[str] = None,
    val_holdout_fraction: Optional[float] = None,
    extreme_tail: str = "upper",
) -> float:
    """
    Load validation data, run predictor in batches, tune tau by minimizing interval_position_mae.
    Returns tau_best (use as decoding threshold for stage 2 and inference).
    """
    if threshold_candidates is None:
        threshold_candidates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    if (
        train_data_path_for_split
        and val_holdout_fraction is not None
        and val_holdout_fraction > 0.0
    ):
        x0_val, _ = load_validation_npy_from_train_file_split(
            train_data_path_for_split,
            seq_length,
            val_holdout_fraction,
            normalizer=normalizer,
        )
    else:
        if not validation_data_path:
            return 0.5
        x0_val, _ = load_validation_npy(validation_data_path, seq_length, normalizer=normalizer)
    x0_val = x0_val.to(device)
    semantic_extractor = ExtremeSemantic(
        K_max=K_max, seq_length=seq_length, extreme_tail=extreme_tail
    )
    batch_results: List[tuple[ExtremeLabels, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    n = x0_val.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = x0_val[start:end]
        labels, pred = run_predictor_batch(
            diffusion_model,
            predictor,
            batch,
            semantic_extractor,
            threshold,
            t_min,
            t_max,
            device,
        )
        batch_results.append((
            labels,
            pred["extreme_logits"],
            pred["F"],
            pred["t"],
            pred["x0_hat"],
        ))
    tau_best = 0.5
    if len(batch_results) == 0:
        return tau_best
    # We tune tau using a combination of interval position error and event-count error.
    # score(tau) = pos_mae(tau) + lambda_num * num_events_mae(tau)
    # lambda_num balances the two terms (num_events_mae is typically smaller in scale).
    lambda_num = 5.0
    best_score = float("inf")
    best_pos_mae = float("inf")
    best_num_mae = float("inf")
    for tau in threshold_candidates:
        all_ts, all_te, all_lts, all_lte, all_lpmax, all_lm, all_m_pred = [], [], [], [], [], [], []
        for (labels_b, logits_b, F_b, t_b, x0_hat_b) in batch_results:
            ts_b, te_b, m_b, _ = decode_batch_with_tau(
                predictor, logits_b, F_b, t_b, tau, x0_hat_b
            )
            all_ts.append(ts_b)
            all_te.append(te_b)
            all_m_pred.append(m_b)
            all_lts.append(labels_b.ts)
            all_lte.append(labels_b.te)
            all_lpmax.append(labels_b.pmax)
            all_lm.append(labels_b.m)
        labels_tau = ExtremeLabels(
            ts=torch.cat(all_lts, dim=0),
            te=torch.cat(all_lte, dim=0),
            pmax=torch.cat(all_lpmax, dim=0),
            m=torch.cat(all_lm, dim=0),
        )
        pred_m_tau = torch.cat(all_m_pred, dim=0)
        pred_ts_tau = torch.cat(all_ts, dim=0)
        pred_te_tau = torch.cat(all_te, dim=0)
        pos_mae = compute_interval_position_mae(
            labels_tau,
            pred_ts_tau,
            pred_te_tau,
        )
        num_mae = compute_num_events_mae(labels_tau, pred_m_tau)
        score = pos_mae + lambda_num * num_mae
        if score < best_score:
            best_score = score
            best_pos_mae = pos_mae
            best_num_mae = num_mae
            tau_best = tau
    if logger is not None:
        logger.log_info(
            "Tuned decoding tau on validation: tau_best={:.4f} "
            "(interval_position_mae={:.4f}, num_events_mae={:.4f}, score={:.4f})".format(
                tau_best, best_pos_mae, best_num_mae, best_score
            )
        )
    return tau_best


# ---------------------------------------------------------------------------
# 3b) Visualization: x0, x0_hat, actual extremes (blue), predicted extremes (red)
# ---------------------------------------------------------------------------


def _to_1d(y: torch.Tensor) -> np.ndarray:
    """(L,) or (L,C) -> (L,) by taking first channel or squeezing."""
    a = y.detach().cpu().numpy()
    if a.ndim == 2:
        a = a[:, 0]
    return np.asarray(a, dtype=np.float64)


def _unnormalize_with_normalizer(
    normalizer: Any,
    x: Union[torch.Tensor, np.ndarray],
    feature_size: int = 1,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Unnormalize x from [-1, 1] to original space using normalizer.unnormalize.
    x: shape (..., C) with C = feature_size (e.g. (B, L, C) or (B, K_max) for pmax with C=1).
    """
    if not hasattr(normalizer, "unnormalize"):
        return x
    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        device = x.device
        dtype = x.dtype
        x = x.detach().cpu().numpy()
    shape = x.shape
    flat = x.reshape(-1, feature_size)
    out = normalizer.unnormalize(flat)
    out = out.reshape(shape)
    if is_tensor:
        out = torch.from_numpy(out).to(device=device, dtype=dtype)
    return out


def plot_pmax_distribution(
    pmax_gt: torch.Tensor,
    pmax_pred: torch.Tensor,
    m_valid: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot marginal distributions of predicted and actual pmax (no pairing).
    One figure: two overlaid histograms (actual vs predicted) over valid event slots.
    When save_path is set, also writes bar densities to a .txt file (one value per bar per line).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = m_valid.detach().cpu().numpy()
    pg = pmax_gt.detach().cpu().numpy()
    pp = pmax_pred.detach().cpu().numpy()
    pg_flat = pg[m]
    pp_flat = pp[m]
    if len(pg_flat) == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    nbins = min(30, max(1, len(pg_flat) // 5))
    n_gt, bin_edges, _ = ax.hist(
        pg_flat,
        bins=nbins,
        alpha=0.6,
        label="Actual",
        color="blue",
        density=True,
    )
    n_pred, _, _ = ax.hist(
        pp_flat,
        bins=bin_edges,
        alpha=0.6,
        label="Predicted",
        color="red",
        density=True,
    )
    ax.set_xlabel("pmax")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of pmax (valid events)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def plot_num_events_distribution(
    m_gt: torch.Tensor,
    m_pred: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of per-sample event counts: ground truth vs predicted.
    x-axis = number of events (0,1,2,...) and y-axis = number of samples.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Number of events per sample
    n_gt = m_gt.sum(dim=1).detach().cpu().numpy().astype(int)
    n_pred = m_pred.sum(dim=1).detach().cpu().numpy().astype(int)
    if len(n_gt) == 0:
        return
    k_max = int(max(n_gt.max(), n_pred.max()))
    bins = np.arange(0, k_max + 2)
    hist_gt, _ = np.histogram(n_gt, bins=bins)
    hist_pred, _ = np.histogram(n_pred, bins=bins)
    x = np.arange(0, k_max + 1)
    width = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(x - width / 2, hist_gt, width, label="Ground truth", color="blue", alpha=0.8)
    ax.bar(x + width / 2, hist_pred, width, label="Predicted", color="red", alpha=0.8)
    ax.set_xlabel("Number of events per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of event count")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def plot_extreme_comparison(
    x0: Union[torch.Tensor, np.ndarray],
    x0_hat: Union[torch.Tensor, np.ndarray],
    labels: ExtremeLabels,
    pred_ts: torch.Tensor,
    pred_te: torch.Tensor,
    pred_m: torch.Tensor,
    pred_pmax: torch.Tensor,
    sample_idx: int = 0,
    t_value: Optional[int] = None,
    save_path: Optional[str] = None,
    extreme_tail: str = "upper",
) -> Tuple[Any, Any]:
    """
    Plot one sample: single figure with x0 (ground truth) as base line.
    Blue: predicted endpoints, range, and pmax arrow.
    Red: ground truth extreme positions (mask-like, 1=extreme).

    Args:
        x0: (L,) or (L, C) one sample (ground truth).
        x0_hat: (L,) or (L, C) one sample (diffusion-TS prediction); unused in plot.
        labels: ExtremeLabels with B=1; ts, te, pmax, m (1, K_max).
        pred_ts, pred_te, pred_m, pred_pmax: (1, K_max) for one sample (predicted).
        sample_idx: index label for title.
        t_value: diffusion timestep; shown in title.
        save_path: if set, save figure to this path.
        extreme_tail: \"upper\" -> pred pmax marker x = argmax(x0) on [ts,te]; \"lower\" -> argmin.

    Returns:
        (ax, ax): single axes returned twice for backward compatibility.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization; install matplotlib.")

    if isinstance(x0, torch.Tensor):
        x0_1d = _to_1d(x0)
    else:
        x0_1d = np.asarray(x0)
        if x0_1d.ndim == 2:
            x0_1d = x0_1d[:, 0]

    L = len(x0_1d)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Base: x0 (ground truth)
    ax.plot(np.arange(L), x0_1d, color="black", linewidth=1.0, label="x0 (ground truth)")

    # Blue: predicted endpoints, range, pmax arrow
    pts = pred_ts.detach().cpu().numpy()
    pte = pred_te.detach().cpu().numpy()
    pmax_p = pred_pmax.detach().cpu().numpy()
    pm = pred_m.detach().cpu().numpy()
    if pred_ts.dim() == 2:
        pts, pte = pts[0], pte[0]
        pmax_p, pm = pmax_p[0], pm[0]
    for e in range(len(pm)):
        if not pm[e]:
            continue
        s, end = int(pts[e]), int(pte[e])
        s, end = max(0, s), min(L - 1, end)
        if end < s:
            continue
        ax.axvspan(s, end + 1, alpha=0.15, color="blue")
        ax.scatter([s], [x0_1d[s]], color="blue", s=30, zorder=3, label="pred endpoints" if e == 0 else None)
        ax.scatter([end], [x0_1d[end]], color="blue", s=30, zorder=3)
        seg_gt = x0_1d[s : end + 1]
        if seg_gt.size > 0:
            if str(extreme_tail).lower() == "lower":
                peak_time_idx = s + int(np.argmin(seg_gt))
            else:
                peak_time_idx = s + int(np.argmax(seg_gt))
            ax.scatter(
                [peak_time_idx],
                [pmax_p[e]],
                color="blue",
                s=40,
                marker="^",
                zorder=3,
                label="pred pmax" if e == 0 else None,
            )

    # Red: ground truth extreme positions (mask: 1=extreme)
    ts_gt = labels.ts.detach().cpu().numpy()
    te_gt = labels.te.detach().cpu().numpy()
    m_gt = labels.m.detach().cpu().numpy()
    if labels.ts.dim() == 2:
        ts_gt, te_gt = ts_gt[0], te_gt[0]
        m_gt = m_gt[0]
    gt_mask_indices = []
    for e in range(len(m_gt)):
        if not m_gt[e]:
            continue
        s, end = int(ts_gt[e]), int(te_gt[e])
        s, end = max(0, s), min(L - 1, end)
        for i in range(s, end + 1):
            gt_mask_indices.append(i)
    if gt_mask_indices:
        gt_mask_indices = np.unique(gt_mask_indices)
        ax.scatter(
            gt_mask_indices,
            x0_1d[gt_mask_indices],
            color="red",
            s=12,
            zorder=4,
            label="gt extreme (mask)",
        )

    t_str = str(t_value) if t_value is not None else "N/A"
    ax.set_title(f"Sample {sample_idx}: x0 (ground truth), t={t_str}")
    ax.set_xlim(0, L - 1)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    return ax, ax


def plot_extreme_comparison_batch(
    x0_batch: torch.Tensor,
    x0_hat_batch: torch.Tensor,
    labels: ExtremeLabels,
    pred_ts: torch.Tensor,
    pred_te: torch.Tensor,
    pred_m: torch.Tensor,
    pred_pmax: torch.Tensor,
    indices: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
    max_plots: int = 10,
    t_batch: Optional[torch.Tensor] = None,
    extreme_tail: str = "upper",
) -> List[Tuple[Any, Any]]:
    """
    Plot comparison for several samples: top = x0_hat + predicted (red), bottom = x0 + actual (blue).

    Args:
        x0_batch: (B, L, C).
        x0_hat_batch: (B, L, C).
        labels: ExtremeLabels (B, K_max).
        pred_ts, pred_te, pred_m, pred_pmax: (B, K_max).
        indices: which batch indices to plot; default [0, ..., min(max_plots-1, B-1)].
        save_dir: if set, save each figure as sample_0.png, sample_1.png, ...
        max_plots: max number of figures when indices is None.
        t_batch: (B,) timesteps used for x0_hat; if set, shown in title.

    Returns:
        List of (ax_top, ax_bottom) per plotted sample.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization; install matplotlib.")

    B = x0_batch.shape[0]
    if indices is None:
        indices = list(range(min(max_plots, B)))
    axes_list = []
    for i in indices:
        if i >= B:
            continue
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"sample_{i}.png")
        t_value = int(t_batch[i].item()) if t_batch is not None and i < t_batch.shape[0] else None
        ax_top, ax_bottom = plot_extreme_comparison(
            x0_batch[i],
            x0_hat_batch[i],
            ExtremeLabels(
                ts=labels.ts[i : i + 1],
                te=labels.te[i : i + 1],
                pmax=labels.pmax[i : i + 1],
                m=labels.m[i : i + 1],
            ),
            pred_ts[i : i + 1],
            pred_te[i : i + 1],
            pred_m[i : i + 1],
            pred_pmax[i : i + 1],
            sample_idx=i,
            t_value=t_value,
            save_path=save_path,
            extreme_tail=extreme_tail,
        )
        axes_list.append((ax_top, ax_bottom))
    return axes_list


# ---------------------------------------------------------------------------
# 4) Full validation flow: load data, run model, aggregate metrics
# ---------------------------------------------------------------------------


def run_extreme_validation(
    seq_length: int,
    diffusion_model: torch.nn.Module,
    predictor: ExtremeSemanticPredictor,
    K_max: int,
    threshold: Union[float, torch.Tensor],
    t_min: int,
    t_max: int,
    device: Union[torch.device, str],
    validation_data_path: Optional[str] = None,
    batch_size: int = 64,
    normalizer: Optional[Any] = None,
    vis_save_dir: Optional[str] = None,
    num_vis: int = 10,
    validation_tune_threshold: bool = True,
    threshold_candidates: Optional[List[float]] = None,
    train_data_path_for_split: Optional[str] = None,
    val_holdout_fraction: Optional[float] = None,
    extreme_tail: str = "upper",
) -> Dict[str, float]:
    """
    End-to-end validation: load validation .npy or a held-out tail of the training file,
    run diffusion + predictor in batches,
    optionally tune decoding threshold tau by minimizing interval_position_mae, then aggregate metrics.

    Args:
        validation_data_path: path to validation .npy when not using same-file split.
        train_data_path_for_split / val_holdout_fraction: if set, load validation tail from this file
        (must match ``split_raw_for_train_val`` / ``train.params.val_holdout_fraction``).
        seq_length: sequence length L.
        diffusion_model: frozen diffusion model (on device).
        predictor: trained ExtremeSemanticPredictor (on device).
        K_max: max events per sample (must match predictor and extractor).
        threshold: threshold for extreme extraction (GT labels).
        t_min, t_max: timestep window for x0_hat.
        device: device.
        batch_size: batch size for validation loops.
        normalizer: optional; if None, fit on validation data.
        vis_save_dir: if set, save comparison plots for the first num_vis samples here.
        num_vis: number of sample plots when vis_save_dir is set.
        validation_tune_threshold: if True, select tau that minimizes interval_position_mae on validation set.
        threshold_candidates: list of tau values to try when validation_tune_threshold is True; default [0.2, ..., 0.8].

    Returns:
        Dict with num_events_mae, interval_length_mae, interval_position_mae, pmax_mae, and optionally tau_best.
    """
    if threshold_candidates is None:
        threshold_candidates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    if (
        train_data_path_for_split
        and val_holdout_fraction is not None
        and val_holdout_fraction > 0.0
    ):
        x0_val, normalizer = load_validation_npy_from_train_file_split(
            train_data_path_for_split,
            seq_length,
            val_holdout_fraction,
            normalizer=normalizer,
        )
    else:
        if not validation_data_path:
            return {}
        x0_val, normalizer = load_validation_npy(
            validation_data_path, seq_length, normalizer=normalizer
        )
    if x0_val.shape[0] == 0:
        return {}
    x0_val = x0_val.to(device)
    semantic_extractor = ExtremeSemantic(
        K_max=K_max, seq_length=seq_length, extreme_tail=extreme_tail
    )
    feature_size = x0_val.shape[-1]
    use_unnorm = hasattr(normalizer, "unnormalize")

    # Pass 1: run all batches, collect (labels, extreme_logits, F, t, x0_hat) per batch
    batch_results: List[tuple[ExtremeLabels, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    n = x0_val.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = x0_val[start:end]
        labels, pred = run_predictor_batch(
            diffusion_model,
            predictor,
            batch,
            semantic_extractor,
            threshold,
            t_min,
            t_max,
            device,
        )
        batch_results.append((
            labels,
            pred["extreme_logits"],
            pred["F"],
            pred["t"],
            pred["x0_hat"],
        ))

    # Tune tau by minimizing interval_position_mae (boundary-only objective)
    tau_best = 0.5
    if validation_tune_threshold and len(batch_results) > 0:
        best_mae = float("inf")
        for tau in threshold_candidates:
            all_ts = []
            all_te = []
            all_m = []
            all_pmax = []
            all_lts = []
            all_lte = []
            all_lpmax = []
            all_lm = []
            for (labels_b, logits_b, F_b, t_b, x0_hat_b) in batch_results:
                ts_b, te_b, m_b, pmax_b = decode_batch_with_tau(predictor, logits_b, F_b, t_b, tau, x0_hat_b)
                all_ts.append(ts_b)
                all_te.append(te_b)
                all_m.append(m_b)
                all_pmax.append(pmax_b)
                all_lts.append(labels_b.ts)
                all_lte.append(labels_b.te)
                all_lpmax.append(labels_b.pmax)
                all_lm.append(labels_b.m)
            labels_tau = ExtremeLabels(
                ts=torch.cat(all_lts, dim=0),
                te=torch.cat(all_lte, dim=0),
                pmax=torch.cat(all_lpmax, dim=0),
                m=torch.cat(all_lm, dim=0),
            )
            pos_mae = compute_interval_position_mae(
                labels_tau,
                torch.cat(all_ts, dim=0),
                torch.cat(all_te, dim=0),
            )
            if pos_mae < best_mae:
                best_mae = pos_mae
                tau_best = tau

    # Decode all batches with tau_best and concatenate
    all_ts_hat = []
    all_te_hat = []
    all_m_hat = []
    all_pmax_hat = []
    all_labels_ts = []
    all_labels_te = []
    all_labels_pmax = []
    all_labels_m = []
    first_batch_for_vis = None
    for idx, (labels_b, logits_b, F_b, t_b, x0_hat_b) in enumerate(batch_results):
        ts_b, te_b, m_b, pmax_b = decode_batch_with_tau(predictor, logits_b, F_b, t_b, tau_best, x0_hat_b)
        all_ts_hat.append(ts_b)
        all_te_hat.append(te_b)
        all_m_hat.append(m_b)
        all_pmax_hat.append(pmax_b)
        all_labels_ts.append(labels_b.ts)
        all_labels_te.append(labels_b.te)
        all_labels_pmax.append(labels_b.pmax)
        all_labels_m.append(labels_b.m)
        if idx == 0:
            first_batch_for_vis = (labels_b, x0_hat_b, ts_b, te_b, m_b, pmax_b, t_b)

    labels_cat = ExtremeLabels(
        ts=torch.cat(all_labels_ts, dim=0),
        te=torch.cat(all_labels_te, dim=0),
        pmax=torch.cat(all_labels_pmax, dim=0),
        m=torch.cat(all_labels_m, dim=0),
    )
    pred_ts = torch.cat(all_ts_hat, dim=0)
    pred_te = torch.cat(all_te_hat, dim=0)
    pred_m = torch.cat(all_m_hat, dim=0)
    pred_pmax = torch.cat(all_pmax_hat, dim=0)

    # Visualization: first batch with tau_best (skipped unless vis_save_dir is set)
    if first_batch_for_vis is not None and vis_save_dir is not None:
        labels_vis, x0_hat_vis, ts_vis, te_vis, m_vis, pmax_vis, t_vis = first_batch_for_vis
        batch_vis = x0_val[: min(batch_size, n)]
        if use_unnorm:
            batch_plot = _unnormalize_with_normalizer(normalizer, batch_vis, feature_size)
            x0_hat_plot = _unnormalize_with_normalizer(normalizer, x0_hat_vis, feature_size)
            labels_plot = ExtremeLabels(
                ts=labels_vis.ts,
                te=labels_vis.te,
                pmax=_unnormalize_with_normalizer(normalizer, labels_vis.pmax, 1),
                m=labels_vis.m,
            )
            pmax_hat_plot = _unnormalize_with_normalizer(normalizer, pmax_vis, 1)
        else:
            batch_plot = batch_vis
            x0_hat_plot = x0_hat_vis
            labels_plot = labels_vis
            pmax_hat_plot = pmax_vis
        plot_extreme_comparison_batch(
            batch_plot,
            x0_hat_plot,
            labels_plot,
            ts_vis,
            te_vis,
            m_vis,
            pmax_hat_plot,
            indices=list(range(min(num_vis, batch_plot.shape[0]))),
            save_dir=vis_save_dir,
            t_batch=t_vis,
            extreme_tail=str(extreme_tail),
        )

    # Unnormalize to original space for metrics when normalizer supports it
    if use_unnorm:
        labels_cat = ExtremeLabels(
            ts=labels_cat.ts,
            te=labels_cat.te,
            pmax=_unnormalize_with_normalizer(normalizer, labels_cat.pmax, 1),
            m=labels_cat.m,
        )
        pred_pmax = _unnormalize_with_normalizer(normalizer, pred_pmax, 1)

    # Summarize pmax distributions (valid events only) in original space when available.
    pmax_dist_stats: Dict[str, float] = {}
    with torch.no_grad():
        mask_valid = labels_cat.m.bool()
        if mask_valid.any():
            pg = labels_cat.pmax[mask_valid].detach().cpu().numpy().astype(float)
            pp = pred_pmax[mask_valid].detach().cpu().numpy().astype(float)
            # Basic moments
            pmax_dist_stats["pmax_gt_mean"] = float(pg.mean())
            pmax_dist_stats["pmax_pred_mean"] = float(pp.mean())
            pmax_dist_stats["pmax_gt_std"] = float(pg.std())
            pmax_dist_stats["pmax_pred_std"] = float(pp.std())
            # Range
            pmax_dist_stats["pmax_gt_min"] = float(pg.min())
            pmax_dist_stats["pmax_gt_max"] = float(pg.max())
            pmax_dist_stats["pmax_pred_min"] = float(pp.min())
            pmax_dist_stats["pmax_pred_max"] = float(pp.max())
            # Selected quantiles
            for q in (10.0, 25.0, 50.0, 75.0, 90.0):
                key_gt = f"pmax_gt_p{int(q)}"
                key_pred = f"pmax_pred_p{int(q)}"
                pmax_dist_stats[key_gt] = float(np.percentile(pg, q))
                pmax_dist_stats[key_pred] = float(np.percentile(pp, q))

    if vis_save_dir is not None:
        pmax_dist_path = os.path.join(vis_save_dir, "pmax_distribution.png")
        plot_pmax_distribution(
            labels_cat.pmax,
            pred_pmax,
            labels_cat.m,
            save_path=pmax_dist_path,
        )
        num_events_path = os.path.join(vis_save_dir, "num_events_distribution.png")
        plot_num_events_distribution(
            labels_cat.m,
            pred_m,
            save_path=num_events_path,
        )

    metrics = compute_all_metrics(labels_cat, pred_ts, pred_te, pred_m, pred_pmax)
    metrics["tau_best"] = tau_best
    # Attach pmax distribution summary statistics for further analysis.
    metrics.update(pmax_dist_stats)
    return metrics


# ---------------------------------------------------------------------------
# 5) Entry point for main.py: run validation after train_extreme
# ---------------------------------------------------------------------------


def _load_diffusion_and_predictor_for_validation(
    config: dict,
    args: Any,
    diffusion_model: torch.nn.Module,
    device: Union[torch.device, str],
    logger: Optional[Any] = None,
) -> ExtremeSemanticPredictor:
    """
    Load diffusion checkpoint into diffusion_model and build predictor from config,
    then load checkpoint_predictor.pt. Used when running validation without training.
    """
    base_folder = config["solver"]["results_folder"]
    seq_len = config["model"]["params"]["seq_length"]
    results_folder = checkpoint_run_dir(base_folder, args.name, seq_len)
    ckpt_path = results_folder / "checkpoint_backbone.pt"
    predictor_ckpt_path = results_folder / "checkpoint_predictor.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {ckpt_path}. "
            "Train diffusion first with --train, then run validation."
        )
    if not predictor_ckpt_path.exists():
        raise FileNotFoundError(
            f"Predictor checkpoint not found: {predictor_ckpt_path}. "
            "Train extreme predictor first (run_stage2_train_extreme) or ensure checkpoint_predictor.pt exists."
        )

    ckpt = torch.load(str(ckpt_path), map_location=device)
    diffusion_model.load_state_dict(ckpt["model"], strict=True)
    diffusion_model.eval()
    for p in diffusion_model.parameters():
        p.requires_grad_(False)

    extreme_model = config.get("extreme_model", {})
    ext_cfg = extreme_model.get("prediction", {})
    extreme_tail = extreme_model.get("extreme_tail", "upper")
    seq_length = int(config["model"]["params"]["seq_length"])
    feature_size = int(config["model"]["params"]["feature_size"])
    num_timesteps = int(config["model"]["params"].get("timesteps", 1000))
    threshold_raw = float(ext_cfg.get("threshold", 0.5))
    normalizer = load_normalizer_from_disk(getattr(args, "output_dir", "."))
    if normalizer is not None:
        thr_use = float(np.asarray(normalize_threshold(normalizer, threshold_raw, feature_size)).ravel()[0])
    else:
        thr_use = threshold_raw
    predictor = ExtremeSemanticPredictor(
        seq_length=seq_length,
        feature_size=feature_size,
        K_max=int(ext_cfg.get("K_max", 5)),
        num_timesteps=num_timesteps,
        d_model=int(ext_cfg.get("d_model", 128)),
        n_heads=int(ext_cfg.get("n_heads", 4)),
        n_layers=int(ext_cfg.get("n_layers", 2)),
        dropout=float(ext_cfg.get("dropout", 0.1)),
        context_left=int(ext_cfg.get("context_left", 0)),
        context_right=int(ext_cfg.get("context_right", 0)),
        threshold=thr_use,
        extreme_tail=str(extreme_tail),
    )
    pred_ckpt = torch.load(str(predictor_ckpt_path), map_location=device)
    predictor.load_state_dict(pred_ckpt["model"], strict=True)
    predictor = predictor.to(device)
    predictor.eval()

    if logger is not None:
        logger.log_info("Loaded diffusion from {} and predictor from {}.".format(ckpt_path, predictor_ckpt_path))
    return predictor


def run_extreme_validation_entry(
    config: dict,
    args: Any,
    diffusion_model: torch.nn.Module,
    predictor: Optional[ExtremeSemanticPredictor] = None,
    device: Optional[Union[torch.device, str]] = None,
    logger: Optional[Any] = None,
    vis_save_dir: Optional[str] = None,
    num_vis: int = 10,
) -> Dict[str, float]:
    """
    Single entry for extreme predictor validation. Reads paths and hyperparams from config/args,
    runs run_extreme_validation, logs metrics, optionally saves comparison plots.

    When predictor is None, loads diffusion and predictor from checkpoints (validation-only mode:
    no training, only run metrics and optional plots).

    Args:
        config: full YAML config (model.params, extreme_model.prediction, train.params).
        args: CLI args (output_dir, name, extreme_milestone, etc.).
        diffusion_model: the diffusion model (will load checkpoint into it when predictor is None).
        predictor: if None, load from checkpoint_predictor.pt (and load diffusion checkpoint).
        device: device.
        logger: optional Logger for logging metrics.
        vis_save_dir: if set, save comparison plots under this directory. Default None: no figures.
        num_vis: number of sample plots when vis_save_dir is set.

    Returns:
        Dict with num_events_mae, interval_length_mae, interval_position_mae, pmax_mae.
    """
    if device is None:
        device = next(diffusion_model.parameters()).device
    if predictor is None:
        predictor = _load_diffusion_and_predictor_for_validation(
            config, args, diffusion_model, device, logger
        )

    seq_length = int(config["model"]["params"]["seq_length"])
    train_params = config.get("train", {}).get("params", {})
    ext_cfg = config.get("extreme_model", {}).get("prediction", {})

    data_path = train_params.get("data_path", "") or ""
    val_holdout_raw = train_params.get("val_holdout_fraction")
    val_holdout_fraction = None
    if val_holdout_raw is not None:
        vf = float(val_holdout_raw)
        if 0.0 < vf < 1.0:
            val_holdout_fraction = vf

    validation_data_path = ext_cfg.get("validation_data_path")
    if validation_data_path == "":
        validation_data_path = None

    # 1) Explicit validation .npy wins. 2) Else same-file holdout (matches stage-2 train/val split).
    # 3) Else legacy: {train_basename}_validation.npy next to training file.
    use_same_file_split = (
        validation_data_path is None
        and val_holdout_fraction is not None
        and bool(data_path)
    )

    if use_same_file_split:
        if not os.path.isfile(data_path):
            if logger is not None:
                logger.log_info(
                    "Extreme validation skipped: train data_path not found: {}".format(data_path)
                )
            return {}
        validation_data_path = None
    elif validation_data_path:
        if not os.path.isfile(validation_data_path):
            if logger is not None:
                logger.log_info(
                    "Extreme validation skipped: validation file not found: {}".format(validation_data_path)
                )
            return {}
    else:
        if not data_path:
            raise ValueError(
                "Extreme validation needs train.params.data_path, or extreme_model.prediction.validation_data_path."
            )
        base_dir = os.path.dirname(data_path)
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        validation_data_path = os.path.join(base_dir, base_name + "_validation.npy")
        if not os.path.isfile(validation_data_path):
            if logger is not None:
                logger.log_info(
                    "Extreme validation skipped: validation file not found: {}".format(validation_data_path)
                )
            return {}

    K_max = int(ext_cfg.get("K_max", 5))
    threshold_raw = float(ext_cfg.get("threshold", 0.5))
    t_min = int(ext_cfg.get("t_min", 0))
    t_max = int(ext_cfg.get("t_max", 200))
    batch_size = int(ext_cfg.get("batch_size", train_params.get("batch_size", 64)))
    feature_size = int(config["model"]["params"].get("feature_size", 1))

    # Threshold is in original space: load training normalizer and normalize threshold (and validation data)
    normalizer = load_normalizer_from_disk(getattr(args, "output_dir", "."))
    if normalizer is not None:
        threshold = normalize_threshold(normalizer, threshold_raw, feature_size)
        val_normalizer = normalizer
    else:
        threshold = threshold_raw
        val_normalizer = None

    validation_tune_threshold = bool(ext_cfg.get("validation_tune_threshold", True))
    # Threshold tuning candidates: prefer explicit list; else build from interval if provided.
    threshold_interval = ext_cfg.get("threshold_interval")
    threshold_candidates = ext_cfg.get("threshold_candidates")
    if threshold_candidates is not None:
        threshold_candidates = [float(x) for x in threshold_candidates]
    elif threshold_interval is not None:
        step = float(threshold_interval)
        if step > 0.0:
            # Build [0, 1] inclusive with given step, e.g. 0.05 -> 0.0, 0.05, ..., 1.0
            n_steps = int(round(1.0 / step))
            threshold_candidates = [i * step for i in range(n_steps + 1)]
        else:
            threshold_candidates = None

    extreme_tail = config.get("extreme_model", {}).get("extreme_tail", "upper")
    metrics = run_extreme_validation(
        seq_length=seq_length,
        diffusion_model=diffusion_model,
        predictor=predictor,
        K_max=K_max,
        threshold=threshold,
        t_min=t_min,
        t_max=t_max,
        device=device,
        validation_data_path=validation_data_path,
        batch_size=batch_size,
        normalizer=val_normalizer,
        vis_save_dir=vis_save_dir,
        num_vis=num_vis,
        validation_tune_threshold=validation_tune_threshold,
        threshold_candidates=threshold_candidates,
        train_data_path_for_split=data_path if use_same_file_split else None,
        val_holdout_fraction=val_holdout_fraction if use_same_file_split else None,
        extreme_tail=str(extreme_tail),
    )

    if logger is not None and metrics:
        logger.log_info("Extreme validation metrics: num_events_mae={:.4f}, interval_length_mae={:.4f}, "
                        "interval_position_mae={:.4f}, pmax_mae={:.4f}".format(
                            metrics.get("num_events_mae", 0),
                            metrics.get("interval_length_mae", 0),
                            metrics.get("interval_position_mae", 0),
                            metrics.get("pmax_mae", 0),
                        ))
        if validation_tune_threshold:
            logger.log_info("Decoding threshold tau_best (by interval_position_mae): {:.4f}".format(metrics.get("tau_best", 0.5)))
        if vis_save_dir:
            logger.log_info("Extreme comparison plots saved to: {}".format(vis_save_dir))

    # Save tau_best for use during ControlNet sampling (same folder as checkpoint_predictor.pt)
    base_folder = config.get("solver", {}).get("results_folder", "./Checkpoint")
    seq_len = config.get("model", {}).get("params", {}).get("seq_length")
    if metrics and seq_len is not None and getattr(args, "name", None):
        results_folder = checkpoint_run_dir(base_folder, args.name, seq_len)
        results_folder.mkdir(parents=True, exist_ok=True)
        tau_best = float(metrics.get("tau_best", 0.5))
        hyperparams_path = results_folder / "validation_hyperparams.json"
        with open(hyperparams_path, "w") as f:
            json.dump({"tau_best": tau_best}, f, indent=2)
        if logger is not None:
            logger.log_info("Saved tau_best={:.4f} to {} for sampling.".format(tau_best, hyperparams_path))

    return metrics


# ---------------------------------------------------------------------------
# Usage example (e.g. after training, with diffusion and predictor loaded):
#
#   from Models.extreme.extreme_validation import run_extreme_validation
#
#   metrics = run_extreme_validation(
#       seq_length=200,
#       diffusion_model=diffusion_model,
#       predictor=predictor,
#       K_max=5,
#       threshold=0.5,
#       t_min=0,
#       t_max=200,
#       device=device,
#       validation_data_path=".../file_validation.npy",
#       batch_size=64,
#   )
#   # Or same file as training with train.params.val_holdout_fraction:
#   # train_data_path_for_split=".../file.npy", val_holdout_fraction=0.5
#   print(metrics)  # num_events_mae, interval_length_mae, interval_position_mae, pmax_mae
