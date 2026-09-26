"""Sampling with per-step fusion coefficients, with the alpha-independent prefix cached.

The coefficients act only on the steps inside the cross-view window (sigma <= 0.35, the
last six of the 18 steps). Every step before the window is identical for every choice of
coefficients, so it is run once per (training seed, sampling seed) and cached; each
evaluation then resumes from the cached state. Outside the window only the time branch is
evaluated (`skip_unfused=True`). All coefficients compared within one solve share the
same prefix, so the comparison between them is paired and exact.
"""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import torch

from models.sampler import DiffusionProcess
from utils.utils_data import denormalize_npy, load_npy_dataset

from .branches import load_pair
from .fusion import SIGMAS, PerStepFusion

DEFAULT_WINDOW_SIGMA = 0.35


def window_steps(max_sigma: float = DEFAULT_WINDOW_SIGMA) -> list[int]:
    """Step indices whose noise level is at most `max_sigma` (12..17 at 0.35)."""
    return [i for i, s in enumerate(SIGMAS) if s <= max_sigma]


def t_steps_of(process, device):
    n, rho = process.num_steps, process.rho
    lo = max(process.sigma_min, process.net.sigma_min)
    hi = min(process.sigma_max, process.net.sigma_max)
    i = torch.arange(n, dtype=torch.float64, device=device)
    t = (hi ** (1 / rho) + i / (n - 1) * (lo ** (1 / rho) - hi ** (1 / rho))) ** rho
    return torch.cat([process.net.round_sigma(t), torch.zeros_like(t[:1])])


@torch.no_grad()
def run_steps(process, x, i0, i1, t, head=None):
    """Heun steps i0..i1-1, identical to `DiffusionProcess.sample` with S_churn = 0.

    `head` is the (time, frequency) branch pair already evaluated at (x, t[i0]). The first
    predictor call of the window always sees the cached prefix state, so these two
    network evaluations do not depend on the coefficients and are computed once.
    """
    net, n = process.net, process.num_steps
    x_next = x
    for i in range(i0, i1):
        t_cur, t_next = t[i], t[i + 1]
        x_hat = x_next
        t_hat = net.round_sigma(t_cur)
        d = net.fuse(head[0], head[1], t_hat) if (head is not None and i == i0) \
            else net(x_hat, t_hat)
        d_cur = (x_hat - d.to(torch.float64)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < n - 1:
            d_prime = (x_next - net(x_next, t_next).to(torch.float64)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


class Evaluator:
    """Loads both branches for each training seed, caches the prefix for each sampling
    seed, and generates a full set for any per-step coefficient map."""

    def __init__(self, config_file, name, seeds, ss_list, batch, device,
                 window_sigma=DEFAULT_WINDOW_SIGMA, log_root=None):
        self.config_file, self.name = config_file, name
        self.seeds, self.ss_list, self.batch = list(seeds), list(ss_list), int(batch)
        self.window = window_steps(window_sigma)
        self.split = self.window[0]
        self.models, self.args, self.prefix, self.head = {}, {}, {}, {}
        self._built, self._raw = set(), {}
        for sd in self.seeds:
            mt, mf, a = load_pair(config_file, name, sd, device, log_root)
            self.models[sd], self.args[sd] = (mt, mf), a
            _, raw, _, self.dmin, self.dmax = load_npy_dataset(a.data_path, seed=sd)
            self.n_gen = int(raw.shape[0])
        self.device = device

    def raw_real(self, sd):
        """The real dataset in its original units (identical for every seed)."""
        if sd not in self._raw:
            _, raw, _, _, _ = load_npy_dataset(self.args[sd].data_path, seed=sd)
            self._raw[sd] = raw.astype('float32')
        return self._raw[sd]

    def _process(self, sd, alphas, skip_unfused=False):
        mt, mf = self.models[sd]
        w = PerStepFusion(mt.net, mf.net, alphas, skip_unfused=skip_unfused)
        a = self.args[sd]
        return DiffusionProcess(a, w, (a.seq_len, a.ts_channels))

    @torch.no_grad()
    def build_prefix(self, seeds=None):
        """Run steps 0..split-1 (alpha = 1) once per (training seed, sampling seed)."""
        for sd in (self.seeds if seeds is None else seeds):
            if sd in self._built:
                continue
            mt, mf = self.models[sd]
            p = self._process(sd, {i: 1.0 for i in self.window}, skip_unfused=True)
            t = t_steps_of(p, self.device)
            with ExitStack() as st:
                for m in (mt, mf):
                    m.eval(); st.enter_context(m.ema_scope())
                for ss in self.ss_list:
                    torch.manual_seed(ss)
                    chunks, remaining = [], self.n_gen
                    while remaining > 0:
                        n = min(self.batch, remaining)
                        xT = torch.randn([n, *p.shape]).to(self.device)
                        chunks.append(run_steps(p, xT.to(torch.float64) * t[0],
                                                0, self.split, t))
                        remaining -= n
                    self.prefix[(sd, ss)] = chunks
                    # Both branches at the first window step; shared by every alpha.
                    t_hat = p.net.round_sigma(t[self.split])
                    self.head[(sd, ss)] = [(mt.net(x, t_hat, class_labels=None),
                                            mf.net(x, t_hat, class_labels=None))
                                           for x in chunks]
            self._built.add(sd)
            print(f'    prefix {self.name} seed {sd} done', flush=True)

    @torch.no_grad()
    def generate(self, sd, alphas: dict, ss):
        """The generated set for one (training seed, coefficient map, sampling seed),
        in the original units of the data. `alphas` maps window step -> coefficient."""
        mt, mf = self.models[sd]
        p = self._process(sd, {int(i): float(v) for i, v in alphas.items()})
        t = t_steps_of(p, self.device)
        with ExitStack() as st:
            for m in (mt, mf):
                m.eval(); st.enter_context(m.ema_scope())
            parts = [run_steps(p, x, self.split, p.num_steps, t, head=h)
                     .detach().cpu().numpy()
                     for x, h in zip(self.prefix[(sd, ss)], self.head[(sd, ss)])]
        return denormalize_npy(np.vstack(parts), self.dmin, self.dmax).astype(np.float32)
