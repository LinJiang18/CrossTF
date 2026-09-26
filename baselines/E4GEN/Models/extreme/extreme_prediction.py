"""
Extreme prediction network module.
Used in diffusion training stage for extreme value prediction.
Content to be added step by step.
"""

from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from Data.build_dataloader import normalize_threshold
from Models.extreme.extreme_semantic import (
    ExtremeSemantic,
    ExtremeLabels,
    normalize_extreme_tail,
)


# This section prepares training data for extreme prediction.


def run_frozen_xpred_step(
    diffusion_model,
    x0: torch.Tensor,
    t_min: int,
    t_max: int,
    t: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
):
    """
    Single diffusion step with a frozen x-pred model to produce x0_hat.

    Args:
        diffusion_model:
            Trained diffusion model (e.g. E4GEN) already set to eval()
            and with gradients disabled externally. Must provide:
                - .num_timesteps (int)
                - .q_sample(x_start, t, noise)
                - .output(x, t, padding_masks=None, C_control=None)
        x0: Tensor of shape (B, L, C), real clean sequences in normalized space.
        t_min, t_max: Integers defining the inclusive sampling window for timesteps.
            Caller is responsible for ensuring 0 <= t_min <= t_max < diffusion_model.num_timesteps.
        t: Optional tensor of shape (B,) with pre-selected timesteps. If provided,
            it is used directly and t_min/t_max are ignored.
        noise: Optional tensor of same shape as x0 providing the noise eps.
            If None, standard Gaussian noise is sampled.

    Returns:
        x0_hat: Tensor of shape (B, L, C), x-pred estimate of x0 at timestep t.
        x_t:    Tensor of shape (B, L, C), noised input produced by q_sample.
        t:      Tensor of shape (B,), the timesteps actually used.
    """
    if x0.dim() != 3:
        raise ValueError(f"x0 must be 3D (B, L, C), got shape={tuple(x0.shape)}")

    B, L, C = x0.shape
    device = x0.device

    # Decide timesteps
    if t is None:
        if not (0 <= t_min <= t_max):
            raise ValueError(f"Invalid timestep window: t_min={t_min}, t_max={t_max}")
        # Uniform integer sampling in [t_min, t_max], per sample
        t = torch.randint(low=t_min, high=t_max + 1, size=(B,), device=device, dtype=torch.long)
    else:
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"t must have shape (B,), got shape={tuple(t.shape)}")
        t = t.to(device=device, dtype=torch.long)

    # Decide noise
    if noise is None:
        noise = torch.randn_like(x0)
    else:
        if noise.shape != x0.shape:
            raise ValueError(f"noise must have same shape as x0, got noise={tuple(noise.shape)}, x0={tuple(x0.shape)}")
        noise = noise.to(device=device, dtype=x0.dtype)

    # Forward diffusion: x0 -> x_t
    with torch.no_grad():
        x_t = diffusion_model.q_sample(x_start=x0, t=t, noise=noise)
        # x-pred network prediction of x0 from x_t at timestep t
        x0_hat = diffusion_model.output(x=x_t, t=t)

    return x0_hat, x_t, t


def build_extreme_labels_from_x0(
    x0: torch.Tensor,
    threshold: Union[torch.Tensor, float],
    semantic_extractor: ExtremeSemantic,
) -> ExtremeLabels:
    """
    Helper to build ExtremeLabels from real x0 and a threshold.

    Args:
        x0: (B, L, C) real clean sequences (typically normalized).
        threshold: per-feature threshold (C,) or scalar when C == 1.
        semantic_extractor: ExtremeSemantic instance configured with K_max.

    Returns:
        ExtremeLabels(ts, te, pmax, m) for the batch.
    """
    return semantic_extractor.extract_from_threshold(x0, threshold)


class ExtremeStage2Dataset(Dataset):
    """
    Dataset wrapper for stage-2 extreme prediction training.

    For each index, it:
      1) Reads a real clean window x0 from a base dataset.
      2) Builds ExtremeLabels on x0 using a threshold + ExtremeSemantic.
      3) Samples a small-noise timestep t in [t_min, t_max] and runs a frozen
         diffusion model to obtain x0_hat via x-pred.

    Returned item is a dict with:
        {
            "x0": x0,             # (1, L, C)
            "x0_hat": x0_hat,     # (1, L, C)
            "x_t": x_t,           # (1, L, C)
            "t": t,               # (1,)
            "extreme": labels,    # ExtremeLabels with B == 1
        }
    A DataLoader will stack these along batch dimension to form (B, L, C) etc.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        diffusion_model,
        semantic_extractor: ExtremeSemantic,
        threshold: Union[torch.Tensor, float],
        t_min: int,
        t_max: int,
        normalizer: Optional[Any] = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.diffusion_model = diffusion_model
        self.semantic_extractor = semantic_extractor
        self.threshold = threshold
        self.t_min = int(t_min)
        self.t_max = int(t_max)
        self.normalizer = normalizer
        self._feature_size = getattr(diffusion_model, "feature_size", 1)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _ensure_x0_shape(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Ensure x0 is shaped as (1, L, C) for a single sample.
        Base datasets typically return (L, C); this adds the batch dim and
        validates rank >= 2.
        """
        if x0.dim() == 2:
            # Assume (L, C)
            x0 = x0.unsqueeze(0)
        elif x0.dim() == 3:
            # Already (B, L, C); for safety require B == 1 here
            if x0.shape[0] != 1:
                raise ValueError(
                    f"Expected single-sample x0 with shape (1,L,C) or (L,C), got {tuple(x0.shape)}"
                )
        else:
            raise ValueError(f"x0 from base_dataset must be 2D or 3D, got shape={tuple(x0.shape)}")
        return x0

    def __getitem__(self, idx: int):
        # 1) Get real clean sequence x0 from base dataset
        x0 = self.base_dataset[idx]
        if not isinstance(x0, torch.Tensor):
            x0 = torch.as_tensor(x0)
        x0 = x0.float()
        x0 = self._ensure_x0_shape(x0)  # (1, L, C)

        # Move x0 to diffusion model device so q_sample/output stay on same device
        device = next(self.diffusion_model.parameters()).device
        x0 = x0.to(device)

        # 2) Build ExtremeLabels from x0 and threshold (threshold in raw space; normalize if normalizer given)
        if self.normalizer is not None:
            th = normalize_threshold(
                self.normalizer, self.threshold, self._feature_size
            )
            threshold_use = torch.as_tensor(
                np.atleast_1d(th), device=x0.device, dtype=x0.dtype
            )
        else:
            threshold_use = self.threshold
        labels = build_extreme_labels_from_x0(
            x0=x0, threshold=threshold_use, semantic_extractor=self.semantic_extractor
        )  # ExtremeLabels with B == 1

        # 3) Run frozen diffusion x-pred step to get x0_hat and x_t at small noise
        x0_hat, x_t, t = run_frozen_xpred_step(
            diffusion_model=self.diffusion_model,
            x0=x0,
            t_min=self.t_min,
            t_max=self.t_max,
        )

        return {
            "x0": x0,
            "x0_hat": x0_hat,
            "x_t": x_t,
            "t": t,
            "extreme": labels,
        }


class ExtremeSemanticPredictor(nn.Module):
    """
    Extreme semantic predictor r.

    High-level structure:
      - Backbone:  encode (x0_hat, t) into per-time-step features F (B, L, d).
      - Extreme head: produce per-point "is extreme" logits (B, L) from F.
      - Decode with threshold tau: binarize (sigmoid(logits) > tau), find contiguous runs → (ts_hat, te_hat, m_hat).
      - pmax head: for each predicted interval, pool F over [ts_hat, te_hat] (and use t) to regress segment extremum
        (upper: threshold + gap; lower: threshold - gap).

    The detailed architectures of backbone / heads are defined step by step.
    """

    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        K_max: int,
        num_timesteps: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        context_left: int = 0,
        context_right: int = 0,
        threshold: Optional[float] = None,
        extreme_tail: str = "upper",
    ):
        """
        Args:
            seq_length: length L of the time series.
            feature_size: number of channels C in x0_hat.
            K_max: maximum number of extreme events per sample.
            num_timesteps: total diffusion timesteps (for discrete t embedding).
            d_model: hidden feature dimension for backbone and heads.
            n_heads: number of attention heads in the Transformer encoder.
            n_layers: number of Transformer encoder layers.
            dropout: dropout rate in Transformer layers.
            extreme_tail: ``upper`` or ``lower``; must match training data / ControlNet semantics.
        """
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.K_max = K_max
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.context_left = max(int(context_left), 0)
        self.context_right = max(int(context_right), 0)

        # Not stored in state_dict; always reconstruct from YAML when loading ckpt (must match training).
        self._extreme_tail = normalize_extreme_tail(extreme_tail)

        # Store decoding threshold (in the same normalized space as x0_hat / pmax).
        # Upper tail: pmax_hat = threshold + gap (gap >= 0). Lower: pmax_hat = threshold - gap.
        threshold_val = 0.0 if threshold is None else float(threshold)
        self.register_buffer(
            "threshold_value",
            torch.tensor(threshold_val, dtype=torch.float32),
        )

        # ----- Backbone: (x0_hat, t) -> F (B, L, d_model) -----
        # Project input channels C -> d_model
        self.input_proj = nn.Linear(feature_size, d_model)

        # Learnable positional encoding over sequence length L
        self.pos_embed = nn.Embedding(seq_length, d_model)

        # Discrete timestep embedding: t in [0, num_timesteps-1]
        self.t_embed = nn.Embedding(num_timesteps, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ----- Extreme head: F -> per-point "is extreme" logits (B, L) -----
        self.extreme_head = nn.Linear(d_model, 1)  # (B,L,d) -> (B,L,1) extreme logits

        # ----- pmax regression head: pooled F + t_embed + segment extremum of x0_hat -> pmax_hat -----
        self.pmax_head = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    @property
    def extreme_tail(self) -> str:
        return self._extreme_tail

    def _encode_backbone(self, x0_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Encode (x0_hat, t) into per-time-step features F using a Transformer encoder.

        Args:
            x0_hat: (B, L, C) tensor, x-pred estimate of x0 at timestep t.
            t: (B,) or (B, 1) tensor of discrete timesteps.

        Returns:
            F: (B, L, d_model) backbone features.
        """
        if x0_hat.dim() != 3:
            raise ValueError(f"x0_hat must be 3D (B, L, C), got shape={tuple(x0_hat.shape)}")

        B, L, C = x0_hat.shape
        if L != self.seq_length or C != self.feature_size:
            raise ValueError(
                f"x0_hat shape (B,L,C)=({B},{L},{C}) inconsistent with "
                f"(seq_length={self.seq_length}, feature_size={self.feature_size})"
            )

        # Ensure t has shape (B,)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(-1)
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"t must have shape (B,) or (B,1), got shape={tuple(t.shape)}")

        t = t.long().clamp(min=0, max=self.num_timesteps - 1).to(device=x0_hat.device)

        # Input projection: (B,L,C) -> (B,L,d_model)
        h = self.input_proj(x0_hat)

        # Positional encoding: positions 0..L-1
        positions = torch.arange(self.seq_length, device=x0_hat.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(B, -1)  # (B,L)
        pos_emb = self.pos_embed(positions)  # (B,L,d_model)

        # Timestep embedding: (B,) -> (B,d_model) -> broadcast to (B,L,d_model)
        t_emb = self.t_embed(t)  # (B,d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, self.seq_length, -1)  # (B,L,d_model)

        # Combine and encode with Transformer
        h = h + pos_emb + t_emb  # (B,L,d_model)
        F = self.encoder(h)      # (B,L,d_model), batch_first=True
        return F

    def _extreme_logits(self, F: torch.Tensor) -> torch.Tensor:
        """From backbone features F (B, L, d_model) produce per-time-step extreme logits (B, L)."""
        return self.extreme_head(F).squeeze(-1)  # (B, L)

    @staticmethod
    def _find_event_runs_1d(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find contiguous runs of True in 1D boolean mask. Returns (starts, ends) inclusive, sorted by start."""
        L = mask.shape[0]
        if L == 0:
            return mask.new_empty(0, dtype=torch.long), mask.new_empty(0, dtype=torch.long)
        padded = F.pad(mask.float().unsqueeze(0), (1, 1), value=0.0).squeeze(0)
        diff = padded[1:] - padded[:-1]
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0] - 1
        return starts, ends

    @staticmethod
    def decode_intervals_from_logits(
        logits: torch.Tensor,
        tau: float,
        K_max: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode per-point extreme logits into at most K_max intervals per sample.
        Pipeline:
          1) Binarize with prob > tau (prob = sigmoid(logits)).
          2) Find contiguous runs of 1s.
          3) Optionally filter out very short runs and merge runs separated by very small gaps.

        Args:
            logits: (B, L) extreme logits per time step.
            tau: threshold in [0,1]; positions with sigmoid(logits) > tau are extreme.
            K_max: maximum number of intervals per sample.

        Returns:
            ts_hat: (B, K_max) long, predicted start indices.
            te_hat: (B, K_max) long, predicted end indices (inclusive).
            m_hat: (B, K_max) bool, True where a valid interval was produced.
        """
        B, L = logits.shape
        device = logits.device
        prob = torch.sigmoid(logits)
        y_hat = (prob > tau)  # (B, L) bool

        # Simple morphological rules to reduce over-fragmentation:
        # - Discard runs shorter than min_len.
        # - Merge neighbouring runs whose gap is <= min_gap.
        # These hyperparameters can be tuned; defaults chosen conservatively.
        min_len = 2
        min_gap = 1

        ts_hat = torch.zeros(B, K_max, dtype=torch.long, device=device)
        te_hat = torch.zeros(B, K_max, dtype=torch.long, device=device)
        m_hat = torch.zeros(B, K_max, dtype=torch.bool, device=device)

        for b in range(B):
            starts, ends = ExtremeSemanticPredictor._find_event_runs_1d(y_hat[b])
            if len(starts) == 0:
                continue

            # Filter out very short runs
            lengths = ends - starts + 1
            keep = lengths >= min_len
            starts = starts[keep]
            ends = ends[keep]
            if len(starts) == 0:
                continue

            # Merge runs that are separated by at most min_gap points
            if min_gap > 0 and len(starts) > 1:
                merged_starts = [starts[0].item()]
                merged_ends = [ends[0].item()]
                for i in range(1, len(starts)):
                    gap = int(starts[i].item()) - int(merged_ends[-1]) - 1
                    if gap <= min_gap:
                        # Extend current merged interval
                        merged_ends[-1] = int(ends[i].item())
                    else:
                        merged_starts.append(int(starts[i].item()))
                        merged_ends.append(int(ends[i].item()))
                starts = torch.tensor(merged_starts, device=device, dtype=torch.long)
                ends = torch.tensor(merged_ends, device=device, dtype=torch.long)

            nev = min(len(starts), K_max)
            if nev == 0:
                continue
            starts = starts[:nev]
            ends = ends[:nev]
            ts_hat[b, :nev] = starts
            te_hat[b, :nev] = ends
            m_hat[b, :nev] = True

        return ts_hat, te_hat, m_hat

    def _pool_event_features(
        self,
        F: torch.Tensor,
        ts_idx: torch.Tensor,
        te_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool backbone features over (optionally) expanded event intervals to obtain per-event vectors.

        For each event slot (b, e), we take the interval:
            s' = max(0, ts_idx[b,e] - context_left)
            e' = min(L-1, te_idx[b,e] + context_right)
        and apply mean pooling over F[b, s':e'+1, :].

        Args:
            F: (B, L, d_model) backbone features.
            ts_idx: (B, K_max) start indices.
            te_idx: (B, K_max) end indices (inclusive).

        Returns:
            z: (B, K_max, d_model) pooled event features.
        """
        if F.dim() != 3:
            raise ValueError(f"F must be 3D (B, L, d), got shape={tuple(F.shape)}")
        B, L, D = F.shape
        if ts_idx.shape != te_idx.shape:
            raise ValueError(f"ts_idx and te_idx must have same shape, got {tuple(ts_idx.shape)} vs {tuple(te_idx.shape)}")
        B2, K = ts_idx.shape
        if B2 != B:
            raise ValueError(f"Batch size mismatch: F has B={B}, indices have B={B2}")

        device = F.device
        z = torch.zeros(B, K, D, device=device, dtype=F.dtype)

        for b in range(B):
            for e in range(K):
                s = int(ts_idx[b, e].item())
                t_end = int(te_idx[b, e].item())
                if s < 0 or s >= L:
                    continue
                if t_end < 0 or t_end >= L:
                    continue
                s_exp = max(0, s - self.context_left)
                e_exp = min(L - 1, t_end + self.context_right)
                if e_exp < s_exp:
                    continue
                segment = F[b, s_exp : e_exp + 1, :]  # (len_seg, D)
                if segment.numel() == 0:
                    continue
                z[b, e] = segment.mean(dim=0)

        return z

    def _segment_max_x0_hat(
        self,
        x0_hat: torch.Tensor,
        ts_idx: torch.Tensor,
        te_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Max of x0_hat over each interval [ts_idx, te_idx]; returns (B, K_max)."""
        B, L, C = x0_hat.shape
        K = ts_idx.shape[1]
        device = x0_hat.device
        out = torch.zeros(B, K, device=device, dtype=x0_hat.dtype)
        for b in range(B):
            for e in range(K):
                s = int(ts_idx[b, e].item())
                e_end = int(te_idx[b, e].item())
                s = max(0, min(L - 1, s))
                e_end = max(s, min(L - 1, e_end))
                seg = x0_hat[b, s : e_end + 1, :]
                if seg.numel() > 0:
                    out[b, e] = seg.min() if self.extreme_tail == "lower" else seg.max()
        return out

    def _predict_pmax(
        self,
        F: torch.Tensor,
        x0_hat: torch.Tensor,
        ts_idx: torch.Tensor,
        te_idx: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict segment extremum: upper tail gap_hat >= 0 -> pmax_hat = thr + gap_hat;
        lower tail -> pmax_hat = thr - gap_hat.
        """
        z = self._pool_event_features(F, ts_idx, te_idx)  # (B,K,D)
        seg_max = self._segment_max_x0_hat(x0_hat, ts_idx, te_idx)  # (B,K) (max or min per tail)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(-1)
        t_clamped = t.long().clamp(min=0, max=self.num_timesteps - 1).to(device=F.device)
        t_emb = self.t_embed(t_clamped)  # (B,D)
        B, K, D = z.shape
        t_emb = t_emb.unsqueeze(1).expand(-1, K, -1)  # (B,K,D)
        seg_max_1 = seg_max.unsqueeze(-1)  # (B,K,1)
        h = torch.cat([z, t_emb, seg_max_1], dim=-1)  # (B,K,2D+1)
        delta = self.pmax_head(h).squeeze(-1)  # (B,K)

        # Non-negative gap relative to the (normalized) threshold.
        gap_hat = torch.nn.functional.softplus(delta)  # (B,K)

        thr = self.threshold_value.to(device=F.device, dtype=x0_hat.dtype)
        thr = thr.view(1, 1).expand(B, K)

        if self.extreme_tail == "lower":
            pmax_hat = thr - gap_hat
        else:
            pmax_hat = thr + gap_hat
        return pmax_hat

    def forward(
        self,
        x0_hat: torch.Tensor,
        t: torch.Tensor,
        tau: float = 0.5,
    ):
        """
        Forward pass of r.

        Args:
            x0_hat: (B, L, C) tensor, x-pred estimate of x0 at timestep t.
            t: (B,) or (B, 1) tensor of timesteps.
            tau: threshold in [0,1] for decoding extreme mask; positions with sigmoid(logits) > tau form intervals.

        Returns:
            Dict with:
              - "F": (B, L, d_model) backbone features.
              - "extreme_logits": (B, L) per-point extreme logits (sigmoid = P(extreme)).
              - "ts_hat", "te_hat", "m_hat", "pmax_hat": decoded with tau (only if tau is used).
        """
        F = self._encode_backbone(x0_hat, t)
        extreme_logits = self._extreme_logits(F)  # (B, L)
        ts_hat, te_hat, m_hat = self.decode_intervals_from_logits(extreme_logits, tau, self.K_max)
        pmax_hat = self._predict_pmax(F, x0_hat, ts_hat, te_hat, t)
        return {
            "F": F,
            "extreme_logits": extreme_logits,
            "ts_hat": ts_hat,
            "te_hat": te_hat,
            "m_hat": m_hat,
            "pmax_hat": pmax_hat,
        }




