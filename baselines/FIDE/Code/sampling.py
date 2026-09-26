import torch

import train_utilities as train_ut


@torch.no_grad()
def ddpm_sample(t: torch.Tensor, bm_sample: torch.Tensor) -> torch.Tensor:
    """
    Reverse diffusion sampling.

    t: [B, S, 1] time grid (same convention as training).
    bm_sample: [B, 1, 1] conditional block-maximum values (e.g. GEV draws).
    Returns x of shape [B, S, 1].
    """
    model = train_ut.model
    if model is None:
        raise RuntimeError("train_utilities.model must be set before sampling.")

    cov = train_ut.get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    x = L @ torch.randn_like(t)

    for diff_step in reversed(range(0, train_ut.diffusion_steps)):
        alpha = train_ut.alphas[diff_step].to(device=x.device, dtype=x.dtype)
        beta = train_ut.betas[diff_step].to(device=x.device, dtype=x.dtype)
        z = L @ torch.randn_like(t)
        i = torch.tensor([float(diff_step)], device=x.device, dtype=torch.float32).expand_as(x[..., :1])
        pred_noise = model(x, t, i, bm_sample)
        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z
    return x
