"""
Extreme semantic label extraction from real x0 and binary event mask y.
Produces (ts, te, pmax, m) for up to K_max events per sample, sorted by start time.

Field ``pmax`` holds the segment extremum: max over the interval for upper-tail events,
min over the interval for lower-tail events (name kept for checkpoint/API stability).
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F


def normalize_extreme_tail(name: Optional[Union[str, object]]) -> str:
    """Return ``\"upper\"`` or ``\"lower\"``; default upper."""
    if name is None:
        return "upper"
    s = str(name).strip().lower()
    if s in ("upper", "up", "high", "max"):
        return "upper"
    if s in ("lower", "low", "min"):
        return "lower"
    raise ValueError(f"extreme_tail must be 'upper' or 'lower', got {name!r}")


@dataclass
class ExtremeLabels:
    """
    Container for per-sample extreme-event semantics.

    Shapes (batch-first):
        ts   : (B, K_max) long  - event start indices (0-based, inclusive)
        te   : (B, K_max) long  - event end indices (0-based, inclusive)
        pmax : (B, K_max) float - segment extremum: max x0 on interval (upper tail) or min (lower tail)
        m    : (B, K_max) bool  - validity mask, True for real events, False for padding
    """

    ts: torch.Tensor
    te: torch.Tensor
    pmax: torch.Tensor
    m: torch.Tensor


# Optional: threshold-based binary mask construction (x0 + threshold -> y)

def threshold_extreme_extraction(
    x0: torch.Tensor,
    threshold: torch.Tensor,
    extreme_tail: str = "upper",
) -> torch.Tensor:
    """
    Build binary extreme-event mask y from x0 by comparing with per-feature thresholds.

    x0: real clean sequence, shape (B, L, C) with
        B = batch, L = seq_length (time), C = feature_size (number of variables).
    threshold: shape (C,) or (1, 1, C); one threshold per feature (last dim of x0).
        Number of thresholds must equal x0.shape[-1].
    extreme_tail: ``upper`` -> y = (x0 > thr); ``lower`` -> y = (x0 < thr).

    Returns:
        y: binary mask same shape as x0, dtype torch.bool (or long for 0/1).
    """
    if x0.dim() != 3:
        raise ValueError("x0 must be 3D (B, L, C).")
    C = x0.shape[-1]
    t = threshold
    if t.dim() == 1:
        if t.shape[0] != C:
            raise ValueError(
                f"threshold length ({t.shape[0]}) must match x0 last dim (feature_size={C})."
            )
        # (C,) -> (1, 1, C) for broadcast
        t = t.view(1, 1, C).to(device=x0.device, dtype=x0.dtype)
    elif t.dim() == 3 and t.shape == (1, 1, C):
        t = t.to(device=x0.device, dtype=x0.dtype)
    else:
        raise ValueError(
            f"threshold must be 1D (C,) or 3D (1, 1, C), got shape {tuple(t.shape)}."
        )
    tail = normalize_extreme_tail(extreme_tail)
    if tail == "upper":
        y = x0 > t
    else:
        y = x0 < t
    return y


def _find_event_runs_1d(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find contiguous runs of 1s in a 1D boolean mask.
    Returns start and end indices (inclusive) for each run, sorted by start.
    mask: (L,) bool
    Returns: starts (N,), ends (N,) where N = number of runs.
    """
    L = mask.shape[0]
    if L == 0:
        return mask.new_empty(0, dtype=torch.long), mask.new_empty(0, dtype=torch.long)
    # Pad with zeros to detect boundaries
    padded = F.pad(mask.float().unsqueeze(0), (1, 1), value=0.0).squeeze(0)  # (L+2,)
    diff = padded[1:] - padded[:-1]  # (L+1,), 1 at run start, -1 at run end
    starts = (diff == 1).nonzero(as_tuple=True)[0]  # indices in diff: 0..L
    ends = (diff == -1).nonzero(as_tuple=True)[0] - 1  # inclusive end in original
    return starts, ends


class ExtremeSemantic:
    """
    Extracts extreme-event semantics (ts, te, pmax, m) from real x0 and binary mask y.
    Events are contiguous runs of 1s in y, sorted by start time, truncated to K_max per sample.
    """

    def __init__(
        self,
        K_max: int,
        seq_length: Optional[int] = None,
        extreme_tail: str = "upper",
    ):
        """
        Args:
            K_max: Maximum number of events per sample; output slots are (ts, te, pmax, m) for e=1..K_max.
            seq_length: Optional sequence length (for validation or padding); not strictly required for extract.
            extreme_tail: ``upper`` (segment max) or ``lower`` (segment min) for ``pmax`` and threshold mask.
        """
        self.K_max = int(K_max)
        self.seq_length = seq_length
        self.extreme_tail = normalize_extreme_tail(extreme_tail)
        if self.K_max <= 0:
            raise ValueError("K_max must be positive.")

    def extract(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
    ) -> ExtremeLabels:
        """
        From real x0 and binary event mask y, produce per-event semantics for up to K_max events.

        Args:
            x0: Real clean sequence, shape (B, L, C).
            y: Binary mask, shape (B, L, C) or (B, L). Value 1 indicates extreme at that time (and feature).
                If (B, L, C), any channel with 1 at (b, l) is treated as extreme at (b, l) (reduced to (B, L)).

        Returns:
            ts: (B, K_max) int64, start index of event e (0-indexed).
            te: (B, K_max) int64, end index of event e (inclusive).
            pmax: (B, K_max) float, segment extremum (max if upper tail, min if lower tail).
            m: (B, K_max) bool or float, validity mask; True/1 for real events, False/0 for padding slots.
        """
        B, L, C = x0.shape
        device = x0.device
        dtype_pmax = x0.dtype
        use_min = self.extreme_tail == "lower"

        if y.dim() == 3:
            if y.shape[0] != B or y.shape[1] != L or y.shape[2] != C:
                raise ValueError(f"y shape {tuple(y.shape)} inconsistent with x0 (B,L,C)=({B},{L},{C}).")
            # Any feature extreme at (b,l) -> event at (b,l)
            y_flat = y.any(dim=-1)  # (B, L)
        elif y.dim() == 2:
            if y.shape[0] != B or y.shape[1] != L:
                raise ValueError(f"y shape {tuple(y.shape)} inconsistent with x0 (B,L)=({B},{L}).")
            y_flat = y
        else:
            raise ValueError("y must be 2D (B, L) or 3D (B, L, C).")

        ts_out = torch.zeros(B, self.K_max, dtype=torch.long, device=device)
        te_out = torch.zeros(B, self.K_max, dtype=torch.long, device=device)
        pmax_out = torch.zeros(B, self.K_max, dtype=dtype_pmax, device=device)
        m_out = torch.zeros(B, self.K_max, dtype=torch.bool, device=device)

        for b in range(B):
            mask_b = y_flat[b]  # (L,)
            starts, ends = _find_event_runs_1d(mask_b)
            nev = min(len(starts), self.K_max)
            if nev == 0:
                continue
            # Sort by start (already in order from left to right)
            # Take first K_max
            starts = starts[: self.K_max]
            ends = ends[: self.K_max]
            ts_out[b, :nev] = starts
            te_out[b, :nev] = ends
            m_out[b, :nev] = True
            for e in range(nev):
                s, e_end = int(starts[e].item()), int(ends[e].item())
                segment = x0[b, s : e_end + 1, :]  # (length, C)
                pmax_out[b, e] = segment.min() if use_min else segment.max()

        return ExtremeLabels(ts=ts_out, te=te_out, pmax=pmax_out, m=m_out)

    def extract_from_threshold(
        self,
        x0: torch.Tensor,
        threshold: Union[torch.Tensor, float],
    ) -> ExtremeLabels:
        """
        Build y from x0 and threshold via threshold_extreme_extraction, then call extract.

        Args:
            x0: (B, L, C).
            threshold: Per-feature threshold, shape (C,) or scalar when C=1.

        Returns:
            Same as extract: (ts, te, pmax, m).
        """
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor([float(threshold)], device=x0.device, dtype=x0.dtype)
        elif isinstance(threshold, torch.Tensor) and threshold.dim() == 0:
            threshold = threshold.unsqueeze(0).to(device=x0.device, dtype=x0.dtype)
        y = threshold_extreme_extraction(x0, threshold, extreme_tail=self.extreme_tail)
        return self.extract(x0, y)


def build_extreme_control_fields(
    labels: ExtremeLabels,
    seq_length: int,
    extreme_tail: str = "upper",
) -> torch.Tensor:
    """
    Represent extreme semantics as a time-aligned control field for ControlNet-style conditioning.

    Args:
        labels: ExtremeLabels with shapes (B, K_max) for ts, te, pmax, m.
        seq_length: sequence length L (time dimension size).

    Returns:
        control: (B, L, 3) tensor, where channels are:
            [:, :, 0] - event mask (0/1) over time.
            [:, :, 1] - segment extremum field (same value along each event; overlaps merge
                        by max for upper tail, min for lower tail; 0 where no event).
            [:, :, 2] - relative position in event interval, mapped to [-1, 1],
                        0 where no event.
    """
    ts = labels.ts  # (B, K_max)
    te = labels.te  # (B, K_max)
    pmax = labels.pmax  # (B, K_max)
    m = labels.m  # (B, K_max)

    if ts.dim() != 2 or te.dim() != 2 or pmax.dim() != 2 or m.dim() != 2:
        raise ValueError("ExtremeLabels fields must all be 2D (B, K_max).")

    B, K_max = ts.shape
    device = ts.device
    dtype = pmax.dtype
    tail = normalize_extreme_tail(extreme_tail)
    merge_upper = tail == "upper"

    control = torch.zeros(B, seq_length, 3, device=device, dtype=dtype)
    strength_written = torch.zeros(B, seq_length, dtype=torch.bool, device=device)

    for b in range(B):
        for e in range(K_max):
            if not bool(m[b, e].item()):
                continue

            s = int(ts[b, e].item())
            e_end = int(te[b, e].item())

            s = max(0, min(seq_length - 1, s))
            e_end = max(0, min(seq_length - 1, e_end))
            if e_end < s:
                continue

            length = e_end - s + 1

            # Channel 0: event mask
            control[b, s : e_end + 1, 0] = 1.0

            # Channel 1: segment extremum; overlaps merge (max upper / min lower)
            new_val = pmax[b, e]
            for t_idx in range(s, e_end + 1):
                if not strength_written[b, t_idx]:
                    control[b, t_idx, 1] = new_val
                    strength_written[b, t_idx] = True
                else:
                    if merge_upper:
                        control[b, t_idx, 1] = torch.maximum(
                            control[b, t_idx, 1], new_val
                        )
                    else:
                        control[b, t_idx, 1] = torch.minimum(
                            control[b, t_idx, 1], new_val
                        )

            # Channel 2: relative position in interval, mapped to [-1, 1]
            t_idx = torch.arange(s, e_end + 1, device=device, dtype=dtype)
            center = (s + e_end) / 2.0
            denom = max(length / 2.0, 1.0)
            rel = (t_idx - center) / denom
            rel = torch.clamp(rel, -1.0, 1.0)
            control[b, s : e_end + 1, 2] = rel

    return control
