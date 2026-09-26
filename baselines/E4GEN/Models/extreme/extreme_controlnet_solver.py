"""
Training framework for ExtremeControlNet (stage A).

Unified stage: L_recon (same as Diffusion-TS) + L_ext (masked pointwise L1 on GT extreme regions)
               + L_pmax (weighted L1 on pmax over GT intervals).
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
from torch.nn.utils import clip_grad_norm_
from Data.build_dataloader import (
    build_dataloader,
    extreme_same_file_train_val_holdout,
    normalize_threshold,
)
from Utils.io_utils import checkpoint_run_dir
from Models.extreme.extreme_semantic import (
    ExtremeLabels,
    ExtremeSemantic,
    build_extreme_control_fields,
    normalize_extreme_tail,
)
from Models.interpretable_diffusion.model_utils import extract


@dataclass
class ExtremeControlNetConfig:
    """Config for ControlNet stage-A training."""

    K_max: int
    threshold: Union[float, torch.Tensor]
    t_min: int
    t_max: int
    batch_size: int
    boundary_epochs: int
    pmax_epochs: int
    lr: float = 1e-4
    max_grad_norm: float = 1.0
    lambda_boundary: float = 1.0
    lambda_pmax: float = 1.0
    boundary_epsilon_stage2: float = 0.1  # keep weak L_boundary in stage A2
    # Soft exceed: upper p_ext = sigmoid(k * max_c(x-thr)); lower uses max_c(thr-x)
    soft_exceed_k: float = 10.0
    extreme_tail: str = "upper"
    # pmax loss params (same as extreme_predictor_solver)
    pmax_loss_near_threshold_k: float = 0.4
    pmax_loss_near_over_weight: float = 1.5
    pmax_loss_near_under_weight: float = 0.8
    pmax_loss_magnitude_alpha: float = 0.35
    pmax_loss_tail_under_weight: float = 1.2
    pmax_loss_tail_over_weight: float = 1.0


def _segment_extremum(
    x0_hat: torch.Tensor,
    ts: torch.Tensor,
    te: torch.Tensor,
    m: torch.Tensor,
    extreme_tail: str = "upper",
) -> torch.Tensor:
    """
    For each (b, e) with m[b,e]=True, max (upper) or min (lower) of x0_hat over [ts, te].
    Returns (B, K) with zeros where m is False.
    """
    B, L, C = x0_hat.shape
    K = ts.shape[1]
    device = x0_hat.device
    use_min = normalize_extreme_tail(extreme_tail) == "lower"
    out = torch.zeros(B, K, device=device, dtype=x0_hat.dtype)
    for b in range(B):
        for e in range(K):
            if not m[b, e].item():
                continue
            s = int(ts[b, e].clamp(0, L - 1).item())
            e_end = int(te[b, e].clamp(0, L - 1).item())
            s, e_end = max(0, s), min(L - 1, e_end)
            if e_end < s:
                continue
            seg = x0_hat[b, s : e_end + 1, :]
            out[b, e] = seg.min() if use_min else seg.max()
    return out


def _pmax_loss_gt_intervals(
    x0_hat: torch.Tensor,
    labels_gt: ExtremeLabels,
    threshold_value: float,
    pmax_gt_std: float = 0.0,
    near_threshold_k: float = 0.4,
    near_over_weight: float = 1.5,
    near_under_weight: float = 0.8,
    magnitude_alpha: float = 0.35,
    tail_under_weight: float = 1.2,
    tail_over_weight: float = 1.0,
    extreme_tail: str = "upper",
) -> torch.Tensor:
    """
    Weighted L1 on predicted segment extremum vs pmax_gt (GT field: max or min per tail).
    """
    pmax_pred = _segment_extremum(
        x0_hat, labels_gt.ts, labels_gt.te, labels_gt.m, extreme_tail=extreme_tail
    )
    ts_gt = labels_gt.ts
    te_gt = labels_gt.te
    m_gt = labels_gt.m
    pmax_gt = labels_gt.pmax

    B, K = pmax_pred.shape
    device = pmax_pred.device
    signed_errors = []
    pmax_vals = []
    for b in range(B):
        for e in range(K):
            if not m_gt[b, e].item():
                continue
            signed_errors.append(pmax_pred[b, e] - pmax_gt[b, e])
            pmax_vals.append(pmax_gt[b, e])

    if not signed_errors:
        return (pmax_pred.sum() * 0.0).to(device)

    errors = torch.stack(signed_errors)
    abs_err = errors.abs()
    pmax_vals_t = torch.stack(pmax_vals)
    thr = torch.as_tensor(threshold_value, device=device, dtype=pmax_pred.dtype)
    tail = normalize_extreme_tail(extreme_tail)
    if tail == "upper":
        delta = pmax_vals_t - thr
    else:
        delta = thr - pmax_vals_t

    epsilon = near_threshold_k * max(pmax_gt_std, 1e-8)
    near = (delta >= 0) & (delta <= epsilon)

    w_near_over = torch.full_like(errors, near_over_weight, device=device, dtype=pmax_pred.dtype)
    w_near_under = torch.full_like(errors, near_under_weight, device=device, dtype=pmax_pred.dtype)
    w_tail_under = torch.full_like(errors, tail_under_weight, device=device, dtype=pmax_pred.dtype)
    w_tail_over = torch.full_like(errors, tail_over_weight, device=device, dtype=pmax_pred.dtype)
    w_dir = torch.where(
        near,
        torch.where(errors > 0, w_near_over, w_near_under),
        torch.where(errors < 0, w_tail_under, w_tail_over),
    )
    thr_safe = thr.clamp(min=1e-8)
    w_mag = 1.0 + magnitude_alpha * (delta.clamp(min=0.0) / thr_safe)
    w = w_dir * w_mag
    return (w * abs_err).sum() / w.sum().clamp(min=1e-8)


class ExtremeControlNetSolver:
    """
    Orchestrates stage-A ControlNet training.
    Unified training: reconstruction + extreme-region supervision + pmax supervision.
    """

    def __init__(
        self,
        cfg: ExtremeControlNetConfig,
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

        # Freeze backbone; only control_net trains
        for name, p in self.diffusion_model.named_parameters():
            if "control_net" not in name:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

        seq_length = getattr(self.diffusion_model, "seq_length")
        feature_size = getattr(self.diffusion_model, "feature_size")

        if self.normalizer is not None:
            thr_norm = normalize_threshold(self.normalizer, cfg.threshold, feature_size)
        else:
            thr_norm = float(cfg.threshold)
        if isinstance(thr_norm, (list, tuple)):
            thr_norm = float(thr_norm[0])
        self.threshold_norm = float(thr_norm)

        self.semantic_extractor = ExtremeSemantic(
            K_max=cfg.K_max,
            seq_length=seq_length,
            extreme_tail=cfg.extreme_tail,
        )

        # pmax_gt_std for left-tail band
        pmax_list = []
        n_sample = min(len(base_dataset), 2000)
        for idx in range(n_sample):
            x0 = base_dataset[idx]
            if not isinstance(x0, torch.Tensor):
                x0 = torch.as_tensor(x0).float()
            if x0.dim() == 2:
                x0 = x0.unsqueeze(0)  # (1, L, C)
            # base_dataset from build_dataloader returns normalized data
            lb = self.semantic_extractor.extract_from_threshold(x0, self.threshold_norm)
            if lb.m.any():
                pmax_list.append(lb.pmax[lb.m].detach().cpu())
        if pmax_list:
            all_pmax = torch.cat(pmax_list, dim=0).float()
            self.pmax_gt_std = float(all_pmax.std().item())
            if self.pmax_gt_std < 1e-8:
                self.pmax_gt_std = 0.0
        else:
            self.pmax_gt_std = 0.0

        # DataLoader: simple batch of (B, L, C)
        self.dataloader = DataLoader(
            base_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.control_net_params = [p for p in self.diffusion_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.control_net_params, lr=cfg.lr)

    def _get_batch_x0(self, batch: torch.Tensor) -> torch.Tensor:
        """Ensure batch is (B, L, C) on device."""
        x0 = batch.to(self.device).float()
        if x0.dim() == 2:
            x0 = x0.unsqueeze(-1)
        return x0

    def _l_recon(
        self,
        x0_hat: torch.Tensor,
        x_start: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Same as Diffusion-TS unconditional: L1 + Fourier + loss_weight."""
        train_loss = F.l1_loss(x0_hat, x_start, reduction="none")
        if getattr(self.diffusion_model, "use_ff", False):
            ff_weight = getattr(self.diffusion_model, "ff_weight", 1.0)
            fft1 = torch.fft.fft(x0_hat.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(x_start.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = (
                F.l1_loss(torch.real(fft1), torch.real(fft2), reduction="none")
                + F.l1_loss(torch.imag(fft1), torch.imag(fft2), reduction="none")
            )
            train_loss = train_loss + ff_weight * fourier_loss
        train_loss = train_loss.reshape(train_loss.shape[0], -1).mean(dim=1)
        train_loss = train_loss * extract(
            self.diffusion_model.loss_weight, t, train_loss.shape
        )
        return train_loss.mean()

    def _l_boundary(
        self,
        x0_hat: torch.Tensor,
        labels_gt: ExtremeLabels,
    ) -> torch.Tensor:
        """BCE on soft p_ext vs y_gt."""
        B, L, C = x0_hat.shape
        K = labels_gt.m.shape[1]
        device = x0_hat.device

        # GT mask (B, L)
        gt_mask = torch.zeros(B, L, device=device, dtype=torch.float32)
        for b in range(B):
            for e in range(K):
                if labels_gt.m[b, e]:
                    i_start = int(labels_gt.ts[b, e].clamp(0, L - 1).item())
                    i_end = int(labels_gt.te[b, e].clamp(0, L - 1).item())
                    gt_mask[b, i_start : i_end + 1] = 1.0

        thr = torch.as_tensor(
            self.threshold_norm, device=device, dtype=x0_hat.dtype
        )
        if normalize_extreme_tail(self.cfg.extreme_tail) == "upper":
            s = (x0_hat - thr).max(dim=-1)[0]  # (B, L)
        else:
            s = (thr - x0_hat).max(dim=-1)[0]
        k = self.cfg.soft_exceed_k
        logits = k * s

        num_pos = gt_mask.sum().clamp(min=1.0)
        num_neg = float(B * L) - num_pos
        pos_weight_val = float(num_neg / num_pos)
        pos_weight = torch.tensor(
            [pos_weight_val], device=device, dtype=logits.dtype
        )
        return F.binary_cross_entropy_with_logits(
            logits, gt_mask, pos_weight=pos_weight, reduction="mean"
        )

    def _l_ext_region(
        self,
        x0_hat: torch.Tensor,
        x_start: torch.Tensor,
        labels_gt: ExtremeLabels,
    ) -> torch.Tensor:
        """
        Pointwise L1 on GT extreme regions (time mask derived from labels_gt).

        This provides "boundary" supervision implicitly by concentrating extra
        reconstruction pressure inside GT extreme intervals.
        """
        B, L, _ = x0_hat.shape
        K = labels_gt.m.shape[1]
        device = x0_hat.device

        # Build GT time mask (B, L) in {0,1}
        gt_mask = torch.zeros(B, L, device=device, dtype=x0_hat.dtype)
        for b in range(B):
            for e in range(K):
                if labels_gt.m[b, e]:
                    i_start = int(labels_gt.ts[b, e].clamp(0, L - 1).item())
                    i_end = int(labels_gt.te[b, e].clamp(0, L - 1).item())
                    gt_mask[b, i_start : i_end + 1] = 1.0

        # Pointwise L1, reduced over channels -> (B, L)
        abs_err = (x0_hat - x_start).abs().mean(dim=-1)
        masked = abs_err * gt_mask
        denom = gt_mask.sum().clamp(min=1.0)
        return masked.sum() / denom

    def _l_pmax(
        self,
        x0_hat: torch.Tensor,
        labels_gt: ExtremeLabels,
    ) -> torch.Tensor:
        """Weighted L1 on pmax over GT intervals."""
        return _pmax_loss_gt_intervals(
            x0_hat,
            labels_gt,
            threshold_value=self.threshold_norm,
            pmax_gt_std=self.pmax_gt_std,
            near_threshold_k=self.cfg.pmax_loss_near_threshold_k,
            near_over_weight=self.cfg.pmax_loss_near_over_weight,
            near_under_weight=self.cfg.pmax_loss_near_under_weight,
            magnitude_alpha=self.cfg.pmax_loss_magnitude_alpha,
            tail_under_weight=self.cfg.pmax_loss_tail_under_weight,
            tail_over_weight=self.cfg.pmax_loss_tail_over_weight,
            extreme_tail=self.cfg.extreme_tail,
        )

    def train(self, logger: Optional[Any] = None) -> None:
        """Unified training with (recon + extreme-region) losses (optionally log pmax)."""
        if logger is not None:
            self.logger = logger
        log = getattr(self, "logger", None)
        self.diffusion_model.train()

        seq_length = getattr(self.diffusion_model, "seq_length")
        t_min, t_max = self.cfg.t_min, self.cfg.t_max
        # Keep existing config shape: use total epochs = boundary_epochs + pmax_epochs.
        num_epochs = int(self.cfg.boundary_epochs) + int(self.cfg.pmax_epochs)
        lambda_ext = float(self.cfg.lambda_boundary)
        lambda_pmax = float(self.cfg.lambda_pmax)

        # Loss history for visualization/debugging
        self.loss_history: Dict[str, list] = {
            "total": [],
            "recon": [],
            "ext": [],
            "pmax": [],
            "grad_norm": [],
            "gate_mean": [],
            "gate_max": [],
        }

        total_steps = num_epochs * len(self.dataloader)
        if log is not None:
            tic = time.time()
            log.log_info(
                "ControlNet: unified training {} ep (lambda_ext={}, lambda_pmax(logged)={})".format(
                    num_epochs, lambda_ext, lambda_pmax
                )
            )
        pbar = tqdm(total=total_steps, desc="loss: ---")

        for epoch in range(num_epochs):
            for batch in self.dataloader:
                x0 = self._get_batch_x0(batch)
                # batch from build_dataloader is already normalized

                t = torch.randint(
                    t_min, t_max + 1, (x0.shape[0],), device=self.device
                ).long()
                labels_gt = self.semantic_extractor.extract_from_threshold(
                    x0, self.threshold_norm
                )
                C_control = build_extreme_control_fields(
                    labels_gt, seq_length, extreme_tail=self.cfg.extreme_tail
                )

                noise = torch.randn_like(x0, device=self.device)
                x_t = self.diffusion_model.q_sample(x0, t, noise=noise)
                x0_hat = self.diffusion_model.output(x_t, t, C_control=C_control)

                l_recon = self._l_recon(x0_hat, x0, t)
                l_ext = self._l_ext_region(x0_hat, x0, labels_gt)
                l_pmax = self._l_pmax(x0_hat, labels_gt)
                # For stability: do NOT optimize pmax here (keep it for monitoring only).
                loss = l_recon + lambda_ext * l_ext

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(self.control_net_params, max_norm=float(self.cfg.max_grad_norm))
                self.optimizer.step()

                # record
                self.loss_history["total"].append(float(loss.detach().item()))
                self.loss_history["recon"].append(float(l_recon.detach().item()))
                self.loss_history["ext"].append(float(l_ext.detach().item()))
                self.loss_history["pmax"].append(float(l_pmax.detach().item()))
                self.loss_history["grad_norm"].append(float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm))
                # ControlNet gate stats (effective gate determines injection strength)
                gate_mean, gate_max = 0.0, 0.0
                if getattr(self.diffusion_model, "control_net", None) is not None and hasattr(self.diffusion_model.control_net, "ctrl_gates"):
                    with torch.no_grad():
                        ctrl = self.diffusion_model.control_net
                        g = float(getattr(ctrl, "gate_max", 1.0)) * torch.sigmoid(ctrl.ctrl_gates.detach())
                        gate_mean = float(g.mean().item())
                        gate_max = float(g.abs().max().item())
                self.loss_history["gate_mean"].append(gate_mean)
                self.loss_history["gate_max"].append(gate_max)
                pbar.set_description(
                    f"[unified] loss: {loss.item():.6f} | recon: {l_recon.item():.4f} "
                    f"ext: {l_ext.item():.4f} pmax: {l_pmax.item():.4f} grad: {self.loss_history['grad_norm'][-1]:.3f} "
                    f"gate|mean:{gate_mean:.3f} max:{gate_max:.3f}"
                )
                pbar.update(1)

        pbar.close()
        if log is not None:
            log.log_info(
                "ControlNet training done, time: {:.2f}s".format(time.time() - tic)
            )


def run_stage3_train_controlnet(
    config: dict,
    args: Any,
    diffusion_model: nn.Module,
    device: torch.device,
    logger: Optional[Any] = None,
) -> nn.Module:
    """
    One-shot entry for ControlNet training.
    Requires diffusion checkpoint and optionally predictor (not used in stage A).
    """
    dataloader_info = build_dataloader(
        config,
        args,
        restrict_dataset_to_train_split=extreme_same_file_train_val_holdout(config),
    )
    base_dataset = dataloader_info["dataloader"].dataset
    normalizer = dataloader_info.get("normalizer")

    base_folder = config["solver"]["results_folder"]
    seq_len = config["model"]["params"]["seq_length"]
    results_folder = checkpoint_run_dir(base_folder, args.name, seq_len)
    ckpt_path = results_folder / "checkpoint_backbone.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {ckpt_path}. "
            "Train diffusion first, then run --train_controlnet."
        )

    # Load diffusion; create model with use_controlnet=True, then load with strict=False
    from Utils.io_utils import instantiate_from_config

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model_sd = ckpt.get("model", ckpt)
    cfg_model = dict(config["model"]["params"])
    cfg_model["use_controlnet"] = True
    model_cfg = {"target": config["model"]["target"], "params": cfg_model}
    diffusion_with_control = instantiate_from_config(model_cfg).to(device)
    diffusion_with_control.load_state_dict(model_sd, strict=False)
    diffusion_with_control.eval()
    for name, p in diffusion_with_control.named_parameters():
        if "control_net" not in name:
            p.requires_grad_(False)

    extreme_cfg = config.get("extreme_model", {})
    ctrl_cfg = extreme_cfg.get("controlnet", {})
    pred_cfg = extreme_cfg.get("prediction", {})
    extreme_tail = extreme_cfg.get("extreme_tail", "upper")
    ep_cfg = ExtremeControlNetConfig(
        K_max=int(ctrl_cfg.get("K_max", 5)),
        threshold=float(ctrl_cfg.get("threshold", 0.5)),
        # t_min / t_max always from prediction section for consistency with predictor
        t_min=int(pred_cfg.get("t_min", 0)),
        t_max=int(pred_cfg.get("t_max", 200)),
        batch_size=int(ctrl_cfg.get("batch_size", config["train"]["params"]["batch_size"])),
        boundary_epochs=int(ctrl_cfg.get("boundary_epochs", 30)),
        pmax_epochs=int(ctrl_cfg.get("pmax_epochs", 20)),
        lr=float(ctrl_cfg.get("lr", 1e-4)),
        max_grad_norm=float(ctrl_cfg.get("max_grad_norm", 1.0)),
        lambda_boundary=float(ctrl_cfg.get("lambda_boundary", 1.0)),
        lambda_pmax=float(ctrl_cfg.get("lambda_pmax", 1.0)),
        boundary_epsilon_stage2=float(ctrl_cfg.get("boundary_epsilon_stage2", 0.1)),
        soft_exceed_k=float(ctrl_cfg.get("soft_exceed_k", 10.0)),
        pmax_loss_near_threshold_k=float(ctrl_cfg.get("pmax_loss_near_threshold_k", 0.4)),
        pmax_loss_near_over_weight=float(ctrl_cfg.get("pmax_loss_near_over_weight", 1.5)),
        pmax_loss_near_under_weight=float(ctrl_cfg.get("pmax_loss_near_under_weight", 0.8)),
        pmax_loss_magnitude_alpha=float(ctrl_cfg.get("pmax_loss_magnitude_alpha", 0.35)),
        pmax_loss_tail_under_weight=float(ctrl_cfg.get("pmax_loss_tail_under_weight", 1.2)),
        pmax_loss_tail_over_weight=float(ctrl_cfg.get("pmax_loss_tail_over_weight", 1.0)),
        extreme_tail=str(extreme_tail),
    )

    if logger is not None:
        logger.log_info(
            "Stage-3: training ControlNet (diffusion ckpt: checkpoint_backbone.pt)"
        )

    solver = ExtremeControlNetSolver(
        cfg=ep_cfg,
        diffusion_model=diffusion_with_control,
        base_dataset=base_dataset,
        device=device,
        normalizer=normalizer,
    )
    solver.train(logger=logger)

    ctrl_ckpt_path = results_folder / "checkpoint_controlnet.pt"
    torch.save(
        {"control_net": diffusion_with_control.control_net.state_dict()},
        str(ctrl_ckpt_path),
    )
    if logger is not None:
        logger.log_info("Saved ControlNet to {}".format(ctrl_ckpt_path))

    return diffusion_with_control
