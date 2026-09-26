"""
Training framework for the extreme semantic predictor r.

This module wires together:
  - A frozen diffusion x-pred model (E4GEN or compatible),
  - ExtremeSemantic for label extraction,
  - ExtremeStage2Dataset for (x0_hat, t, ExtremeLabels) batches,
  - ExtremeSemanticPredictor as the model to be trained.

Loss definition and detailed training logic will be added step by step.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from Data.build_dataloader import (
    build_dataloader,
    extreme_same_file_train_val_holdout,
    normalize_threshold,
)
from Utils.io_utils import checkpoint_run_dir
from Models.extreme.extreme_semantic import (
    ExtremeLabels,
    ExtremeSemantic,
    normalize_extreme_tail,
)
from Models.extreme.extreme_prediction import (
    ExtremeStage2Dataset,
    ExtremeSemanticPredictor,
)
from Models.extreme.extreme_validation import tune_tau_on_validation


@dataclass
class ExtremePredictorConfig:
    """
    Lightweight config holder for stage-2 extreme predictor training.

    This can be constructed from a larger YAML config in main.py.
    """

    K_max: int
    threshold: Union[float, torch.Tensor]
    t_min: int
    t_max: int
    batch_size: int
    num_epochs: int
    lr: float = 1e-4
    context_left: int = 0
    context_right: int = 0
    warmup_epochs: int = 0
    lambda_boundary: float = 1.0
    lambda_pmax: float = 1.0
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    boundary_epochs: int = 0
    pmax_epochs: int = 0
    pmax_warmup_steps: int = 0  # stage 2: first N steps use GT intervals for pmax loss
    decoding_tau: float = 0.5
    validation_data_path: Optional[str] = None
    # When set with train_data_path, tune tau from held-out tail of training file (see train.params.val_holdout_fraction).
    train_data_path: Optional[str] = None
    val_holdout_fraction: Optional[float] = None
    threshold_candidates: Optional[list] = None
    # Left tail: near-threshold band ε = k*std(pmax_gt); penalize over-prediction there.
    pmax_loss_near_threshold_k: float = 0.4
    pmax_loss_near_over_weight: float = 1.5
    pmax_loss_near_under_weight: float = 0.8
    # Right tail: magnitude weight and underestimation penalty for non-near events.
    pmax_loss_magnitude_alpha: float = 0.35
    pmax_loss_tail_under_weight: float = 1.2
    pmax_loss_tail_over_weight: float = 1.0
    extreme_tail: str = "upper"


def _match_intervals_and_pmax_loss(
    ts_hat: torch.Tensor,
    te_hat: torch.Tensor,
    m_hat: torch.Tensor,
    pmax_hat: torch.Tensor,
    ts_gt: torch.Tensor,
    te_gt: torch.Tensor,
    m_gt: torch.Tensor,
    pmax_gt: torch.Tensor,
    threshold_value: float,
    pmax_gt_std: float = 0.0,
    near_threshold_k: float = 0.4,
    near_over_weight: float = 1.5,
    near_under_weight: float = 0.8,
    magnitude_alpha: float = 0.0,
    tail_under_weight: float = 1.0,
    tail_over_weight: float = 1.0,
    extreme_tail: str = "upper",
) -> torch.Tensor:
    """
    Match pred intervals to GT; return weighted mean |pmax_hat - pmax_gt|.
    Upper: near-threshold band uses (pmax_gt - thr) in [0, epsilon].
    Lower: band uses (thr - pmax_gt) in [0, epsilon].
    """
    B, K = ts_hat.shape
    device = pmax_hat.device
    signed_errors = []
    pmax_vals = []
    for b in range(B):
        pred_idx = [e for e in range(K) if m_hat[b, e].item()]
        gt_idx = [e for e in range(K) if m_gt[b, e].item()]
        if not pred_idx or not gt_idx:
            continue
        pred_centers = ((ts_hat[b, pred_idx].float() + te_hat[b, pred_idx].float()) / 2.0).detach().cpu()
        gt_centers = ((ts_gt[b, gt_idx].float() + te_gt[b, gt_idx].float()) / 2.0).detach().cpu()
        cost = (pred_centers.unsqueeze(1) - gt_centers.unsqueeze(0)).abs().numpy()
        used_pred, used_gt = set(), set()
        pairs = []
        for _ in range(min(len(pred_idx), len(gt_idx))):
            best_val, best = float("inf"), None
            for i in range(len(pred_idx)):
                if i in used_pred:
                    continue
                for j in range(len(gt_idx)):
                    if j not in used_gt and cost[i, j] < best_val:
                        best_val, best = cost[i, j], (i, j)
            if best is None:
                break
            i, j = best
            used_pred.add(i)
            used_gt.add(j)
            pairs.append((pred_idx[i], gt_idx[j]))
        for pi, gj in pairs:
            err = pmax_hat[b, pi] - pmax_gt[b, gj]
            signed_errors.append(err)
            pmax_vals.append(pmax_gt[b, gj])

    if not signed_errors:
        return (pmax_hat.sum() * 0.0).to(device)

    errors = torch.stack(signed_errors)
    abs_err = errors.abs()
    pmax_vals_t = torch.stack(pmax_vals)
    thr = torch.as_tensor(threshold_value, device=device, dtype=pmax_hat.dtype)
    tail = normalize_extreme_tail(extreme_tail)
    if tail == "upper":
        delta = pmax_vals_t - thr
    else:
        delta = thr - pmax_vals_t

    epsilon = near_threshold_k * max(pmax_gt_std, 1e-8)
    near = (delta >= 0) & (delta <= epsilon)

    w_near_over = torch.full_like(errors, near_over_weight, device=device, dtype=pmax_hat.dtype)
    w_near_under = torch.full_like(errors, near_under_weight, device=device, dtype=pmax_hat.dtype)
    w_tail_under = torch.full_like(errors, tail_under_weight, device=device, dtype=pmax_hat.dtype)
    w_tail_over = torch.full_like(errors, tail_over_weight, device=device, dtype=pmax_hat.dtype)
    w_dir = torch.where(
        near,
        torch.where(errors > 0, w_near_over, w_near_under),
        torch.where(errors < 0, w_tail_under, w_tail_over),
    )

    thr_safe = thr.clamp(min=1e-8)
    w_mag = 1.0 + magnitude_alpha * (delta.clamp(min=0.0) / thr_safe)

    w = w_dir * w_mag
    return (w * abs_err).sum() / w.sum().clamp(min=1e-8)


def _collate_extreme_batch(samples: list) -> dict:
    """Collate batch where each sample has 'extreme' as ExtremeLabels; stack into batched ExtremeLabels."""
    keys = samples[0].keys()
    collated = {}
    for k in keys:
        if k == "extreme":
            labels_list = [s["extreme"] for s in samples]
            collated["extreme"] = ExtremeLabels(
                ts=torch.cat([lb.ts for lb in labels_list], dim=0),
                te=torch.cat([lb.te for lb in labels_list], dim=0),
                pmax=torch.cat([lb.pmax for lb in labels_list], dim=0),
                m=torch.cat([lb.m for lb in labels_list], dim=0),
            )
        else:
            collated[k] = torch.cat([s[k] for s in samples], dim=0)
    return collated


class ExtremePredictorSolver:
    """
    Orchestrates stage-2 training of ExtremeSemanticPredictor.

    Usage pattern (roughly):
        solver = ExtremePredictorSolver(
            cfg=extreme_cfg,
            diffusion_model=diffusion_model_frozen,
            base_dataset=some_dataset_with_x0,
            device=device,
        )
        solver.train()
    """

    def __init__(
        self,
        cfg: ExtremePredictorConfig,
        diffusion_model: nn.Module,
        base_dataset: Dataset,
        device: Union[torch.device, str] = "cuda",
        normalizer: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.diffusion_model = diffusion_model.to(device)
        self.base_dataset = base_dataset
        self.device = torch.device(device)
        self.normalizer = normalizer

        # Ensure diffusion model is frozen (no gradients) and in eval mode.
        self.diffusion_model.eval()
        for p in self.diffusion_model.parameters():
            p.requires_grad_(False)

        # Infer basic shapes/params from diffusion model
        seq_length = getattr(self.diffusion_model, "seq_length")
        feature_size = getattr(self.diffusion_model, "feature_size")
        num_timesteps = getattr(self.diffusion_model, "num_timesteps")

        # Threshold in the same (normalized) space as x0_hat / pmax for scheme C/E-4.
        if self.normalizer is not None:
            thr_norm = normalize_threshold(self.normalizer, cfg.threshold, feature_size)
        else:
            thr_norm = float(cfg.threshold)
        # For 1D feature_size this is a scalar; ensure we store a float.
        if isinstance(thr_norm, (list, tuple)):
            thr_norm = float(thr_norm[0])
        self.threshold_norm = float(thr_norm)

        # Label extractor
        self.semantic_extractor = ExtremeSemantic(
            K_max=cfg.K_max,
            seq_length=seq_length,
            extreme_tail=cfg.extreme_tail,
        )

        # Dataset and dataloader for stage-2 (normalizer: threshold in config is raw space, normalize before use)
        self.dataset = ExtremeStage2Dataset(
            base_dataset=base_dataset,
            diffusion_model=self.diffusion_model,
            semantic_extractor=self.semantic_extractor,
            threshold=cfg.threshold,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            normalizer=normalizer,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate_extreme_batch,
        )

        # std(pmax_gt) in normalized space for left-tail band ε = k*std
        pmax_list = []
        n_sample = min(len(self.dataset), 2000)
        for idx in range(n_sample):
            item = self.dataset[idx]
            lb = item["extreme"]
            if lb.m.any():
                pmax_list.append(lb.pmax[lb.m].detach().cpu())
        if pmax_list:
            all_pmax = torch.cat(pmax_list, dim=0).float()
            self.pmax_gt_std = float(all_pmax.std().item())
            if self.pmax_gt_std < 1e-8:
                self.pmax_gt_std = 0.0
        else:
            self.pmax_gt_std = 0.0

        # Extreme semantic predictor model
        self.model = ExtremeSemanticPredictor(
            seq_length=seq_length,
            feature_size=feature_size,
            K_max=cfg.K_max,
            num_timesteps=num_timesteps,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            context_left=cfg.context_left,
            context_right=cfg.context_right,
            threshold=self.threshold_norm,
            extreme_tail=cfg.extreme_tail,
        ).to(self.device)

        # Optimizer (scheduler can be added later if needed)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
        )

    def compute_loss(
        self,
        batch: Dict[str, Any],
        pred: Dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        """
        stage "boundary": only L_extreme (stage 1). stage "pmax": only L_pmax on matched pairs (stage 2).
        """
        labels = batch["extreme"]
        device = pred["extreme_logits"].device
        ts_gt = labels.ts.to(device)
        te_gt = labels.te.to(device)
        pmax_gt = labels.pmax.to(device)
        m_gt = labels.m.to(device)
        B, L = pred["extreme_logits"].shape
        K = pred["pmax_hat"].shape[1]

        gt_mask = torch.zeros(B, L, device=device, dtype=torch.float32)
        for b in range(B):
            for e in range(K):
                if m_gt[b, e]:
                    i_start = int(ts_gt[b, e].clamp(0, L - 1).item())
                    i_end = int(te_gt[b, e].clamp(0, L - 1).item())
                    gt_mask[b, i_start : i_end + 1] = 1.0
        num_pos = gt_mask.sum().clamp(min=1.0)
        num_neg = float(B * L) - num_pos
        pos_weight_val = float(num_neg / num_pos)
        pos_weight = torch.tensor([pos_weight_val], device=device, dtype=pred["extreme_logits"].dtype)
        L_extreme = F.binary_cross_entropy_with_logits(
            pred["extreme_logits"],
            gt_mask,
            pos_weight=pos_weight,
            reduction="mean",
        )

        if stage == "boundary":
            return self.cfg.lambda_boundary * L_extreme

        L_pmax = _match_intervals_and_pmax_loss(
            pred["ts_hat"],
            pred["te_hat"],
            pred["m_hat"],
            pred["pmax_hat"],
            ts_gt,
            te_gt,
            m_gt,
            pmax_gt,
            threshold_value=self.threshold_norm,
            pmax_gt_std=getattr(self, "pmax_gt_std", 0.0),
            near_threshold_k=getattr(self.cfg, "pmax_loss_near_threshold_k", 0.4),
            near_over_weight=getattr(self.cfg, "pmax_loss_near_over_weight", 1.5),
            near_under_weight=getattr(self.cfg, "pmax_loss_near_under_weight", 0.8),
            magnitude_alpha=getattr(self.cfg, "pmax_loss_magnitude_alpha", 0.0),
            tail_under_weight=getattr(self.cfg, "pmax_loss_tail_under_weight", 1.0),
            tail_over_weight=getattr(self.cfg, "pmax_loss_tail_over_weight", 1.0),
            extreme_tail=getattr(self.cfg, "extreme_tail", "upper"),
        )
        return self.cfg.lambda_pmax * L_pmax

    def _freeze_boundary_keep_pmax(self) -> None:
        """Freeze backbone and extreme head; only pmax_head trained in stage 2."""
        for name, p in self.model.named_parameters():
            if "pmax_head" not in name:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

    def _unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    def train(self, logger: Optional[Any] = None) -> None:
        """
        Stage 1: train boundary only (L_extreme). Then tune tau on validation set.
        Stage 2: freeze boundary, train pmax on matched pairs using tau_best.
        """
        if logger is not None:
            self.logger = logger
        log = getattr(self, "logger", None)
        self.model.train()
        self._unfreeze_all()
        boundary_epochs = getattr(self.cfg, "boundary_epochs", 0) or 0
        pmax_epochs = getattr(self.cfg, "pmax_epochs", 0) or 0
        if boundary_epochs <= 0 and pmax_epochs <= 0:
            boundary_epochs = self.cfg.num_epochs
            pmax_epochs = 0
        tau = getattr(self.cfg, "decoding_tau", 0.5)
        total_steps = boundary_epochs * len(self.dataloader) + pmax_epochs * len(self.dataloader)
        if log is not None:
            tic = time.time()
            log.log_info(
                "Extreme predictor: stage1 (boundary) {} ep, stage2 (pmax) {} ep".format(
                    boundary_epochs, pmax_epochs
                )
            )
        pbar = tqdm(total=total_steps, desc="loss: ---")

        # ---------- Stage 1: boundary only ----------
        for epoch in range(boundary_epochs):
            for batch in self.dataloader:
                x0_hat = batch["x0_hat"].to(self.device)
                t = batch["t"].to(self.device)
                labels = batch["extreme"]
                pred = self.model(x0_hat, t, tau=tau)
                loss = self.compute_loss({"x0_hat": x0_hat, "t": t, "extreme": labels}, pred, stage="boundary")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f"[s1 boundary] loss: {loss.item():.6f}")
                pbar.update(1)

        # ---------- Tune tau on validation (approach A): use tau_best for stage 2 ----------
        use_val_split = (
            getattr(self.cfg, "val_holdout_fraction", None) is not None
            and float(getattr(self.cfg, "val_holdout_fraction", 0) or 0) > 0.0
            and getattr(self.cfg, "train_data_path", None)
        )
        use_val_file = bool(getattr(self.cfg, "validation_data_path", None))
        if pmax_epochs > 0 and (use_val_file or use_val_split):
            seq_length = getattr(self.diffusion_model, "seq_length")
            tau = tune_tau_on_validation(
                validation_data_path=self.cfg.validation_data_path,
                seq_length=seq_length,
                diffusion_model=self.diffusion_model,
                predictor=self.model,
                K_max=self.cfg.K_max,
                threshold=self.cfg.threshold,
                t_min=self.cfg.t_min,
                t_max=self.cfg.t_max,
                device=self.device,
                batch_size=self.cfg.batch_size,
                normalizer=getattr(self, "normalizer", None),
                threshold_candidates=getattr(self.cfg, "threshold_candidates", None),
                logger=log,
                train_data_path_for_split=self.cfg.train_data_path if use_val_split else None,
                val_holdout_fraction=self.cfg.val_holdout_fraction if use_val_split else None,
                extreme_tail=getattr(self.cfg, "extreme_tail", "upper"),
            )
            if log is not None:
                log.log_info("Using validation-tuned tau_best={:.4f} for stage 2".format(tau))

        # ---------- Stage 2: freeze boundary, train pmax; warmup with GT intervals then predicted ----------
        pmax_warmup_steps = getattr(self.cfg, "pmax_warmup_steps", 0) or 0
        if pmax_epochs > 0:
            self._freeze_boundary_keep_pmax()
            self.optimizer = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.cfg.lr,
            )
            step_s2 = 0
            for epoch in range(pmax_epochs):
                for batch in self.dataloader:
                    x0_hat = batch["x0_hat"].to(self.device)
                    t = batch["t"].to(self.device)
                    labels = batch["extreme"]
                    pred = self.model(x0_hat, t, tau=tau)
                    ts_gt = labels.ts.to(self.device)
                    te_gt = labels.te.to(self.device)
                    pmax_gt = labels.pmax.to(self.device)
                    m_gt = labels.m.to(self.device)
                    if step_s2 < pmax_warmup_steps:
                        pmax_hat_gt = self.model._predict_pmax(
                            pred["F"], x0_hat, ts_gt, te_gt, t
                        )
                        if m_gt.any():
                            loss = (pmax_hat_gt[m_gt] - pmax_gt[m_gt]).abs().mean()
                        else:
                            loss = pred["pmax_hat"].sum() * 0.0
                        loss = self.cfg.lambda_pmax * loss
                        pbar.set_description(f"[s2 pmax warmup] loss: {loss.item():.6f}")
                    else:
                        loss = self.compute_loss(
                            {"x0_hat": x0_hat, "t": t, "extreme": labels}, pred, stage="pmax"
                        )
                        pbar.set_description(f"[s2 pmax] loss: {loss.item():.6f}")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    step_s2 += 1

        pbar.close()
        if log is not None:
            log.log_info("Extreme predictor training done, time: {:.2f}s".format(time.time() - tic))


def train_extreme_predictor(
    cfg: ExtremePredictorConfig,
    diffusion_model: nn.Module,
    base_dataset: Dataset,
    device: Union[torch.device, str] = "cuda",
    normalizer: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> ExtremeSemanticPredictor:
    """
    Convenience functional API to train the extreme semantic predictor.
    Returns the trained predictor model.
    normalizer: if provided, config threshold is in raw space and will be normalized before use.
    logger: optional; passed to train() for tau tuning log and stage logs.
    """
    solver = ExtremePredictorSolver(
        cfg=cfg,
        diffusion_model=diffusion_model,
        base_dataset=base_dataset,
        device=device,
        normalizer=normalizer,
    )
    solver.train(logger=logger)
    return solver.model


def run_stage2_train_extreme(
    config: dict,
    args,
    diffusion_model: nn.Module,
    device: torch.device,
    logger=None,
) -> ExtremeSemanticPredictor:
    """
    One-shot entry for stage-2 extreme predictor training.
    Loads diffusion checkpoint, builds config/dataset, trains r.
    Call this from main when args.train_extreme (after diffusion is trained).
    """
    # 1) Build dataloader, get base_dataset (train split only when same-file holdout; diffusion uses full data)
    dataloader_info = build_dataloader(
        config,
        args,
        restrict_dataset_to_train_split=extreme_same_file_train_val_holdout(config),
    )
    base_dataset = dataloader_info["dataloader"].dataset

    # 2) Load diffusion checkpoint into diffusion_model
    base_folder = config["solver"]["results_folder"]
    seq_len = config["model"]["params"]["seq_length"]
    results_folder = checkpoint_run_dir(base_folder, args.name, seq_len)
    ckpt_path = results_folder / "checkpoint_backbone.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {ckpt_path}. "
            "Train diffusion first with --train, then run --train_extreme."
        )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    diffusion_model.load_state_dict(ckpt["model"], strict=True)
    diffusion_model.eval()
    for p in diffusion_model.parameters():
        p.requires_grad_(False)

    # 3) Build ExtremePredictorConfig from config["extreme_model"]["prediction"]
    extreme_model = config.get("extreme_model", {})
    ext_cfg = extreme_model.get("prediction", {})
    extreme_tail = extreme_model.get("extreme_tail", "upper")
    train_params = config.get("train", {}).get("params", {})
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

    use_same_file_split = (
        validation_data_path is None
        and val_holdout_fraction is not None
        and bool(data_path)
    )
    if use_same_file_split:
        train_data_path_for_cfg = data_path
        val_frac_for_cfg = val_holdout_fraction
        validation_data_path_for_cfg = None
    else:
        train_data_path_for_cfg = None
        val_frac_for_cfg = None
        if not validation_data_path and data_path:
            base_dir = os.path.dirname(data_path)
            base_name = os.path.splitext(os.path.basename(data_path))[0]
            validation_data_path = os.path.join(base_dir, base_name + "_validation.npy")
        validation_data_path_for_cfg = validation_data_path
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
    ep_cfg = ExtremePredictorConfig(
        K_max=int(ext_cfg.get("K_max", 5)),
        threshold=float(ext_cfg.get("threshold", 0.5)),
        t_min=int(ext_cfg.get("t_min", 0)),
        t_max=int(ext_cfg.get("t_max", 200)),
        batch_size=int(ext_cfg.get("batch_size", config["train"]["params"]["batch_size"])),
        num_epochs=int(ext_cfg.get("num_epochs", 100)),
        lr=float(ext_cfg.get("lr", 1e-4)),
        context_left=int(ext_cfg.get("context_left", 0)),
        context_right=int(ext_cfg.get("context_right", 0)),
        warmup_epochs=int(ext_cfg.get("warmup_epochs", 0)),
        lambda_boundary=float(ext_cfg.get("lambda_boundary", 1.0)),
        lambda_pmax=float(ext_cfg.get("lambda_pmax", 1.0)),
        d_model=int(ext_cfg.get("d_model", 128)),
        n_heads=int(ext_cfg.get("n_heads", 4)),
        n_layers=int(ext_cfg.get("n_layers", 2)),
        dropout=float(ext_cfg.get("dropout", 0.1)),
        boundary_epochs=int(ext_cfg.get("boundary_epochs", 0)),
        pmax_epochs=int(ext_cfg.get("pmax_epochs", 0)),
        pmax_warmup_steps=int(ext_cfg.get("pmax_warmup_steps", 0)),
        decoding_tau=float(ext_cfg.get("decoding_tau", 0.5)),
        validation_data_path=validation_data_path_for_cfg,
        train_data_path=train_data_path_for_cfg,
        val_holdout_fraction=val_frac_for_cfg,
        threshold_candidates=threshold_candidates,
        pmax_loss_near_threshold_k=float(ext_cfg.get("pmax_loss_near_threshold_k", 0.4)),
        pmax_loss_near_over_weight=float(ext_cfg.get("pmax_loss_near_over_weight", 1.5)),
        pmax_loss_near_under_weight=float(ext_cfg.get("pmax_loss_near_under_weight", 0.8)),
        pmax_loss_magnitude_alpha=float(ext_cfg.get("pmax_loss_magnitude_alpha", 0.35)),
        pmax_loss_tail_under_weight=float(ext_cfg.get("pmax_loss_tail_under_weight", 1.2)),
        pmax_loss_tail_over_weight=float(ext_cfg.get("pmax_loss_tail_over_weight", 1.0)),
        extreme_tail=str(extreme_tail),
    )

    # 4) Train and return (pass normalizer so threshold in config is treated as raw space)
    if logger is not None:
        logger.log_info(
            "Stage-2: training extreme predictor (diffusion ckpt: checkpoint_backbone.pt)"
        )
    normalizer = dataloader_info.get("normalizer")
    model = train_extreme_predictor(
        cfg=ep_cfg,
        diffusion_model=diffusion_model,
        base_dataset=base_dataset,
        device=device,
        normalizer=normalizer,
        logger=logger,
    )

    # 5) Save extreme predictor in the same folder as diffusion checkpoints (single file)
    predictor_ckpt_path = results_folder / "checkpoint_predictor.pt"
    torch.save({"model": model.state_dict()}, str(predictor_ckpt_path))
    if logger is not None:
        logger.log_info("Saved extreme predictor to {}".format(predictor_ckpt_path))

    return model
