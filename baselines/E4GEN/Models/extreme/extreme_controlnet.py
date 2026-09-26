"""
ControlNet-style conditioning network for extreme semantics.

Takes per-time-step control fields (e.g. from build_extreme_control_fields)
and produces a list of residual feature maps aligned with the diffusion
backbone's decoder layers.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding


class ExtremeControlNet(nn.Module):
    """
    Lightweight 1D ControlNet for extreme semantics.

    Input:
        control_fields: (B, L, C_ctrl) control channels over time
                        (e.g. C_ctrl=3: mask, pmax field, relative position).

    Output:
        A list of length n_layers, where each element is a tensor of shape
        (B, L, d_model) to be injected as residuals into the corresponding
        decoder layers of the backbone.
    """

    def __init__(
        self,
        seq_length: int,
        d_model: int,
        n_layers: int,
        control_channels: int = 3,
        dropout: float = 0.1,
        gate_max: float = 0.5,
        gate_init: float = -2.2,
        mask_gate_radius: int = 3,
        mask_gate_strength: float = 0.7,
    ) -> None:
        super().__init__()
        self.seq_length = int(seq_length)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.control_channels = int(control_channels)
        self.gate_max = float(gate_max)
        self.mask_gate_radius = int(mask_gate_radius)
        self.mask_gate_strength = float(mask_gate_strength)

        # Project raw control channels to model dimension.
        self.ctrl_proj = nn.Linear(self.control_channels, self.d_model)

        # Time-axis positional encoding so ControlNet is aware of absolute time index.
        self.pos_enc = LearnablePositionalEncoding(
            self.d_model,
            dropout=dropout,
            max_len=self.seq_length,
        )

        # A shallow stack of residual MLP blocks; one per decoder layer.
        blocks: List[nn.Module] = []
        for _ in range(self.n_layers):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.d_model),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Soft-start gates for residual injection (ControlNet-style "zero-conv" effect).
        # We use a bounded non-negative gate:
        #   g_i = gate_max * sigmoid(raw_i)
        # Initialize raw gates to a negative value so initial injection is ~0.
        self.ctrl_gates = nn.Parameter(torch.full((self.n_layers,), float(gate_init)))

    def forward(
        self,
        control_fields: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            control_fields: (B, L, C_ctrl) control channels over time.
            padding_masks: optional (B, L) bool mask; currently unused but kept
                           for interface symmetry and future extension.

        Returns:
            List[Tensor]: length n_layers, each of shape (B, L, d_model).
        """
        if control_fields.dim() != 3:
            raise ValueError(
                f"control_fields must be 3D (B, L, C_ctrl), got shape={tuple(control_fields.shape)}"
            )

        B, L, C_ctrl = control_fields.shape
        if L != self.seq_length:
            raise ValueError(
                f"control_fields length L={L} inconsistent with seq_length={self.seq_length}"
            )
        if C_ctrl != self.control_channels:
            raise ValueError(
                f"control_fields last dim={C_ctrl} inconsistent with control_channels={self.control_channels}"
            )

        # (B, L, C_ctrl) -> (B, L, d_model)
        x = self.ctrl_proj(control_fields)
        x = self.pos_enc(x)

        # Soft mask gate w(t) in [0,1] derived from control_fields[:,:,0] (event mask).
        # Use AvgPool1d to smooth/dilate so ControlNet mainly affects extreme regions + neighbourhood.
        m = control_fields[..., 0].to(dtype=x.dtype)
        m = m.clamp(0.0, 1.0)  # (B, L)
        r = int(self.mask_gate_radius)
        if r > 0:
            k = 2 * r + 1
            w = F.avg_pool1d(m.unsqueeze(1), kernel_size=k, stride=1, padding=r).squeeze(1)
        else:
            w = m
        w = (self.mask_gate_strength * w).clamp(0.0, 1.0)  # (B, L)
        w = w.unsqueeze(-1)  # (B, L, 1) for broadcasting

        residuals: List[torch.Tensor] = []
        h = x
        for i, block in enumerate(self.blocks):
            h = h + block(h)
            g = self.gate_max * torch.sigmoid(self.ctrl_gates[i])
            residuals.append((g * h) * w)
        return residuals


__all__ = ["ExtremeControlNet"]

