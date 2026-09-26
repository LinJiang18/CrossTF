"""Per-step cross-view fusion of the two trained denoisers.

At every network evaluation the fused denoiser is

    D(x; sigma_i) = alpha_i * D_time(x; sigma_i) + (1 - alpha_i) * D_freq(x; sigma_i),

written below as d_freq + alpha * (d_time - d_freq). The 18-step EDM schedule
(rho = 7, sigma 0.002..80) is fixed by the config, so the sigma each step is evaluated at
is known in advance and alpha is a lookup by step index. Steps without an entry use
alpha = 1, the pure time branch; the caller decides the window by which steps it passes.

Sigmas are matched to the schedule by nearest value in log space with a small tolerance,
because the sampler passes sigma through float64 -> float32 casts.
"""

from __future__ import annotations

import numpy as np
import torch

NUM_STEPS, RHO, SIGMA_MIN, SIGMA_MAX = 18, 7, 0.002, 80.0


def schedule_sigmas(num_steps: int = NUM_STEPS) -> np.ndarray:
    i = np.arange(num_steps)
    return (SIGMA_MAX ** (1 / RHO)
            + i / (num_steps - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO


SIGMAS = schedule_sigmas()


class PerStepFusion(torch.nn.Module):
    """Wraps the two frozen branches; alpha is looked up per scheduled step.

    `alphas` maps a step index (0..17, 0 = highest noise) to its coefficient.
    """

    def __init__(self, net_time, net_freq, alphas: dict[int, float], rel_tol: float = 0.02,
                 skip_unfused: bool = False):
        super().__init__()
        # With `skip_unfused`, a step whose alpha is exactly 1 returns the time branch
        # without evaluating the frequency branch. `d_freq + 1 * (d_time - d_freq)` and
        # `d_time` differ in the last bits, so this is only used where every compared
        # configuration shares the same cached prefix (see evaluator.py).
        self.skip_unfused = bool(skip_unfused)
        self.net, self.net_freq = net_time, net_freq
        self.alphas = {int(k): float(v) for k, v in alphas.items()}
        self.rel_tol = float(rel_tol)
        self.sigma_min, self.sigma_max = net_time.sigma_min, net_time.sigma_max
        self._lut = torch.tensor([self.alphas.get(i, 1.0) for i in range(len(SIGMAS))],
                                 dtype=torch.float64)
        self._log_sig = torch.tensor(np.log(SIGMAS), dtype=torch.float64)

    def round_sigma(self, sigma):
        return self.net.round_sigma(sigma)

    def alpha_for(self, sigma: torch.Tensor) -> torch.Tensor:
        s = sigma.reshape(-1).to(torch.float64).clamp_min(1e-12)
        d = (s.log().unsqueeze(1) - self._log_sig.to(s.device).unsqueeze(0)).abs()
        j = d.argmin(dim=1)
        near = d.gather(1, j.unsqueeze(1)).squeeze(1) <= self.rel_tol
        return torch.where(near, self._lut.to(s.device)[j], torch.ones_like(s))

    def _alpha(self, d_time, sigma):
        """The coefficient for this sigma, shaped to broadcast over (batch, time, ch)."""
        s = torch.as_tensor(sigma, device=d_time.device, dtype=d_time.dtype).reshape(-1)
        if s.numel() == 1:
            s = s.expand(d_time.shape[0])
        return self.alpha_for(s).to(d_time.dtype).view(-1, 1, 1)

    @staticmethod
    def _combine(d_time, d_freq, a):
        return d_freq + a * (d_time - d_freq)

    def fuse(self, d_time, d_freq, sigma):
        """Combine two already-computed branch outputs exactly as `forward` would."""
        return self._combine(d_time, d_freq, self._alpha(d_time, sigma))

    def forward(self, x, sigma, class_labels=None, **kw):
        d_time = self.net(x, sigma, class_labels=class_labels, **kw)
        a = self._alpha(d_time, sigma)
        if self.skip_unfused and bool(torch.all(a == 1)):
            return d_time
        d_freq = self.net_freq(x, sigma, class_labels=class_labels, **kw)
        return self._combine(d_time, d_freq, a)

    __call__ = forward
