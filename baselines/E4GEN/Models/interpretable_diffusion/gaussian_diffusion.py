import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract
from Models.extreme.extreme_controlnet import ExtremeControlNet



# gaussian diffusion class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class E4GEN(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps = 50,
            fast_sampling=False,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            mode = True,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,

            heavy=False,
            heavy_nu=8.0,
            heavy_var_correction=True,

            # sanity check
            sanity_prob=0.002,
            sanity_quantile=0.999,
            sanity_compare_ref=True,
            sanity_check_xt=True,

            **kwargs
    ):
        super().__init__()


        self.heavy = bool(heavy)
        self.heavy_nu = float(heavy_nu)
        self.heavy_var_correction = bool(heavy_var_correction)
        self.sanity_prob = float(sanity_prob)
        self.sanity_quantile = float(sanity_quantile)
        self.sanity_compare_ref = bool(sanity_compare_ref)
        self.sanity_check_xt = bool(sanity_check_xt)


        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.fast_sampling = fast_sampling
        self.mode = mode
        self._trace = None
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)
        # Optional variance-matching loss to discourage over-smoothing.
        # When var_loss_weight == 0.0 (default), this term is disabled.
        self.var_loss_weight = float(kwargs.get("var_loss_weight", 0.0))
        _d_model = d_model if d_model is not None else 64

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        self.use_controlnet = bool(kwargs.get("use_controlnet", False))
        if self.use_controlnet:
            self.control_net = ExtremeControlNet(
                seq_length=seq_length,
                d_model=_d_model,
                n_layers=n_layer_dec,
                control_channels=3,
                dropout=resid_pd,
            )
        else:
            self.control_net = None

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

#        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        if self.fast_sampling:
            self.sampling_timesteps = sampling_timesteps
        else:
            self.sampling_timesteps = timesteps
        # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def set_trace(self, trace):
        self._trace = trace

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, padding_masks=None, C_control=None):
        ctrl_residuals = None
        if C_control is not None and getattr(self, "control_net", None) is not None:
            ctrl_residuals = self.control_net(C_control, padding_masks=None)
        trend, season, trend_cum, season_cum = self.model(
            x, t, padding_masks=padding_masks, label_emb=None,
            ctrl_residuals=ctrl_residuals,
        )
        model_output = trend + season

        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None, C_control=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks, C_control=C_control)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True, C_control=None):
        _, x_start = self.model_predictions(x, t, C_control=C_control)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def _student_t_noise_like(self, x, nu, heavy_var_correction):
        """Same GSM / Student-t construction as training forward (_train_loss_unconditional)."""
        if nu <= 2.0:
            raise ValueError(f"heavy_nu must be > 2, got {nu}")
        z = torch.randn_like(x)
        B = x.shape[0]
        kappa = torch.distributions.Chi2(df=nu).sample((B,)).to(
            device=x.device, dtype=x.dtype
        )
        scale = torch.sqrt(kappa / nu).view((B,) + (1,) * (x.ndim - 1))
        noise = z / scale
        if heavy_var_correction:
            noise = noise * math.sqrt((nu - 2.0) / nu)
        return noise

    def p_sample(self, x, t, clip_denoised=True, heavy=False, nu=4.0, heavy_var_correction=True,
                 C_control=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised,
                                C_control=C_control)

        if t > 0:
            if heavy:
                noise = self._student_t_noise_like(x, nu, heavy_var_correction)
            else:
                noise = torch.randn_like(x)
        else:
            noise = 0.0

        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, shape, predictor=None, t_min=None, t_max=None, tau=0.5):
        device = self.betas.device

        img = torch.randn(shape, device=device)

        total_T = self.num_timesteps

        # ControlNet: fixed t_control in [t_min, t_max]; C_control from predictor at t_control
        use_controlnet = (
            getattr(self, "use_controlnet", False)
            and predictor is not None
            and t_min is not None
            and t_max is not None
        )
        if use_controlnet:
            self._controlnet_imports()
            t_control = max(t_min, min(t_max, total_T - 1))
        else:
            t_control = None
        C_control = None
        base_C_control = None

        for t in tqdm(reversed(range(0, total_T)),
                      desc='sampling loop time step', total=total_T):
            # At t_control: get x0_hat without control, run predictor, build C_control
            if use_controlnet and t == t_control:
                x0_hat = self.output(img, torch.full((img.shape[0],), t, device=device, dtype=torch.long)
                                     , C_control=None)
                out = predictor(x0_hat, torch.full((img.shape[0],), t, device=device, dtype=torch.long), tau=tau)
                labels = self._extreme_labels_from_predictor(out)
                tail = getattr(predictor, "extreme_tail", "upper")
                base_C_control = self._build_extreme_control_fields(labels, extreme_tail=tail)

            if use_controlnet and base_C_control is not None and t <= t_control:
                # Time gate: strong control around t_control, fade out towards t=0.
                # Use a simple linear schedule: g(t)=1 for t>=t_f, g(t)=t/t_f for t<t_f.
                t_f = max(1, t_control // 2)
                if t >= t_f:
                    g_t = 1.0
                else:
                    g_t = float(t) / float(t_f)
                C_step = base_C_control * g_t
            else:
                C_step = None

            heavy = bool(getattr(self, "heavy", False))
            nu = float(getattr(self, "heavy_nu", 8.0))
            hvc = bool(getattr(self, "heavy_var_correction", True))
            img, x_start = self.p_sample(
                img, t, C_control=C_step,
                heavy=heavy, nu=nu, heavy_var_correction=hvc,
            )

            if (
                getattr(self, "_trace", None) is not None
                and (not self.mode)
                and (t % 10 == 0)
            ):
                self._trace.log(t, img, x_start)
        return img

    def _controlnet_imports(self):
        if not hasattr(self, "_controlnet_imports_done"):
            from Models.extreme.extreme_semantic import ExtremeLabels, build_extreme_control_fields
            self._ExtremeLabels = ExtremeLabels
            self._build_extreme_control_fields_fn = build_extreme_control_fields
            self._controlnet_imports_done = True

    def _extreme_labels_from_predictor(self, out):
        """Build ExtremeLabels from predictor output dict."""
        self._controlnet_imports()
        return self._ExtremeLabels(
            ts=out["ts_hat"],
            te=out["te_hat"],
            pmax=out["pmax_hat"],
            m=out["m_hat"],
        )

    def _build_extreme_control_fields(self, labels, extreme_tail: str = "upper"):
        self._controlnet_imports()
        return self._build_extreme_control_fields_fn(
            labels, self.seq_length, extreme_tail=extreme_tail
        )

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True,
                    predictor=None, t_min=None, t_max=None, tau=0.5):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)

        use_controlnet = (
            getattr(self, "use_controlnet", False)
            and predictor is not None
            and t_min is not None
            and t_max is not None
        )
        if use_controlnet:
            self._controlnet_imports()
            t_control = max(t_min, min(t_max, total_timesteps - 1))
        else:
            t_control = None
        C_control = None
        base_C_control = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            # At first step with time <= t_control: build C_control from predictor
            if use_controlnet and time <= t_control and base_C_control is None:
                x0_hat = self.output(img, time_cond, C_control=None)
                out = predictor(x0_hat, time_cond, tau=tau)
                labels = self._extreme_labels_from_predictor(out)
                tail = getattr(predictor, "extreme_tail", "upper")
                base_C_control = self._build_extreme_control_fields(labels, extreme_tail=tail)

            if use_controlnet and base_C_control is not None and time <= t_control:
                t_f = max(1, t_control // 2)
                if time >= t_f:
                    g_t = 1.0
                else:
                    g_t = float(time) / float(t_f)
                C_control = base_C_control * g_t
            else:
                C_control = None

            pred_noise, x_start = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised, C_control=C_control
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c_coef = (1 - alpha_next - sigma ** 2).sqrt()
            if bool(getattr(self, "heavy", False)):
                nu = float(getattr(self, "heavy_nu", 8.0))
                hvc = bool(getattr(self, "heavy_var_correction", True))
                noise = self._student_t_noise_like(img, nu, hvc)
            else:
                noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c_coef * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, batch_size=16,
                     predictor=None, t_min=None, t_max=None, tau=0.5):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn(
            (batch_size, seq_length, feature_size),
            predictor=predictor,
            t_min=t_min,
            t_max=t_max,
            tau=tau,
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss_unconditional(self, x_start, t, target=None, noise=None, padding_masks=None):
        """
        Unconditional training loss.
        Heavy-tailed (optional) DDPM, x0-prediction + optional Fourier loss.
        """

        # -------------------------
        # Config knobs
        # -------------------------
        heavy = bool(getattr(self, "heavy", False))
        nu = float(getattr(self, "heavy_nu", 8.0))
        heavy_var_correction = bool(getattr(self, "heavy_var_correction", True))

        # Sanity-check knobs (apply to BOTH heavy and gaussian)
        sanity_prob = float(getattr(self, "sanity_prob", 0.002))  # printing probability
        sanity_q = float(getattr(self, "sanity_quantile", 0.999))  # tail quantile
        sanity_compare_ref = bool(getattr(self, "sanity_compare_ref", True))  # compare with gaussian reference
        sanity_check_xt = bool(getattr(self, "sanity_check_xt", True))  # also check x_t tails

        # -------------------------
        # (A) Sample noise for q_sample
        # -------------------------
        if noise is None:
            if heavy:
                noise = self._student_t_noise_like(x_start, nu, heavy_var_correction)
            else:
                noise = torch.randn_like(x_start)

        # Target is x0 for x0-prediction
        if target is None:
            target = x_start

        # -------------------------
        # (B) Create x_t via q_sample (unchanged)
        # -------------------------
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # =========================
        # SANITY CHECK (DEBUG ONLY) - DISABLED
        # Apply to BOTH heavy and gaussian, so you can directly compare runs.
        # You can delete this whole block later.
        # =========================
        # if sanity_prob > 0 and torch.rand((), device=x_start.device) < sanity_prob:
        #     with torch.no_grad():
        #         mode = "HEAVY" if heavy else "GAUSS"
        #
        #         # (1) Element-wise tail of the actual noise used (global across batch)
        #         abs_noise_all = noise.detach().abs().reshape(-1)
        #         noise_tail = abs_noise_all.quantile(sanity_q)
        #
        #         msg = f"[sanity:{mode}] q={sanity_q:.3f} |noise|_q(all)={float(noise_tail):.6g}"
        #         if heavy:
        #             msg += f" nu={nu:.2f} var_corr={int(heavy_var_correction)}"
        #
        #         # (2) Optional baseline Gaussian reference tail (same shape, global)
        #         if sanity_compare_ref:
        #             g_ref = torch.randn_like(noise)
        #             ref_tail = g_ref.detach().abs().reshape(-1).quantile(sanity_q)
        #             ratio = (noise_tail / (ref_tail + 1e-12)).item()
        #             msg += f" | ref_gauss(all)={float(ref_tail):.6g} | ratio={ratio:.3f}"
        #
        #         # (3) Confirm q_sample used provided noise:
        #         # x = sqrt_ab*x0 + sqrt_1mab*noise => recon_noise ≈ noise
        #         sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        #         sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        #         recon_noise = (x.detach() - sqrt_ab * x_start.detach()) / (sqrt_1mab + 1e-12)
        #         recon_diff = (recon_noise - noise.detach()).abs().mean()
        #         msg += f" | recon_diff={float(recon_diff):.3e}"
        #
        #         # (4) Optional: tail of x_t itself (this is what actually hits the network)
        #         if sanity_check_xt:
        #             abs_xt_all = x.detach().abs().reshape(-1)
        #             xt_tail = abs_xt_all.quantile(sanity_q)
        #             msg += f" | |x_t|_q(all)={float(xt_tail):.6g}"
        #
        #             if sanity_compare_ref:
        #                 # Construct a "gaussian x_t reference" with the SAME coefficients but gaussian noise
        #                 g = torch.randn_like(noise)
        #                 x_gauss = sqrt_ab * x_start.detach() + sqrt_1mab * g
        #                 xt_ref_tail = x_gauss.abs().reshape(-1).quantile(sanity_q)
        #                 xt_ratio = (xt_tail / (xt_ref_tail + 1e-12)).item()
        #                 msg += f" | x_t_ref_gauss_q={float(xt_ref_tail):.6g} | xt_ratio={xt_ratio:.3f}"
        #
        #         print(msg)
        # =========================
        # END SANITY CHECK
        # =========================

        # -------------------------
        # (D) Model forward + losses
        # -------------------------
        model_out = self.output(x, t, padding_masks)
        train_loss = self.loss_fn(model_out, target, reduction='none')

        # Optional Fourier loss (unchanged)
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)

            fourier_loss = (
                    self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')
                    + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            )
            train_loss = train_loss + self.ff_weight * fourier_loss

        # Reduce to per-sample scalar
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        B = x_start.shape[0]
        train_loss = train_loss.view(B, -1).mean(dim=1)  # (B,)

        # Optional: match per-sample overall variance of x0_hat to x0 to reduce over-smoothing.
        if getattr(self, "var_loss_weight", 0.0) > 0.0:
            x_hat_flat = model_out.view(B, -1)
            x_gt_flat = x_start.view(B, -1)
            std_hat = x_hat_flat.std(dim=1, unbiased=False)
            std_gt = x_gt_flat.std(dim=1, unbiased=False)
            l_var = (std_hat - std_gt).abs()
            train_loss = train_loss + self.var_loss_weight * l_var

        # Apply per-timestep weighting (broadcasts over batch dimension)
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)

        return train_loss.mean()



    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss_unconditional(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x


if __name__ == '__main__':
    pass
