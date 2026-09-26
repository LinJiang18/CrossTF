import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from models.networks import DhariwalUNet, MultiViewPredictor, OneDEDMPrecond
from models.ema import LitEma
from models.img_transformations import STFTEmbedder, DelayEmbedder
from utils.utils_config import infer_attn_resolution


_VALID_MODES = ('time', 'frequency')


class CrossTFBranch(nn.Module):
    """One CrossTF branch: 1D EDM diffusion with a 2D (delay or STFT) denoiser.

    Diffusion state is always the 1D series x_t. Delay / STFT are only internal
    representations inside MultiViewPredictor. Outer EDM preconditioning is 1D.
    """

    def __init__(self, args, device):
        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps
        self.device = device

        mode = getattr(args, 'prediction_mode', None)
        if mode is None:
            raise ValueError('prediction_mode is unset: pass --time or --frequency')
        self.prediction_mode = str(mode).lower()
        if self.prediction_mode not in _VALID_MODES:
            raise ValueError(
                f'prediction_mode must be time|frequency, got {self.prediction_mode}'
            )

        self.seq_len = int(args.seq_len)
        self.ts_channels = int(getattr(args, 'ts_channels', 1))

        n_fft = args.n_fft if getattr(args, 'n_fft', None) is not None else min(63, args.seq_len - 1)
        hop_length = args.hop_length if getattr(args, 'hop_length', None) is not None else max(1, n_fft // 3)

        need_delay = self.prediction_mode == 'time'
        need_stft = self.prediction_mode == 'frequency'

        self.delay_embedder = None
        self.stft_embedder = None
        if need_delay:
            self.delay = args.delay
            self.embedding = args.embedding
            self.delay_embedder = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding)
        if need_stft:
            self.stft_embedder = STFTEmbedder(self.device, args.seq_len, n_fft, hop_length)

        time_net = None
        freq_net = None
        if need_delay:
            time_img_resolution = int(
                getattr(args, 'time_img_resolution', None)
                or getattr(args, 'img_resolution', None)
            )
            time_attn = list(args.attn_resolution) if getattr(args, 'attn_resolution', None) is not None else infer_attn_resolution(time_img_resolution)
            time_attn = [r for r in time_attn if r <= time_img_resolution] or infer_attn_resolution(time_img_resolution)
            time_unet_kwargs = dict(
                img_resolution=time_img_resolution,
                in_channels=self.ts_channels,
                out_channels=self.ts_channels,
                label_dim=0,
                model_channels=args.unet_channels,
                channel_mult=args.ch_mult,
                attn_resolutions=time_attn,
            )
            if getattr(args, 'num_blocks', None) is not None:
                time_unet_kwargs['num_blocks'] = args.num_blocks
            time_net = DhariwalUNet(**time_unet_kwargs)
            self.time_img_resolution = time_img_resolution

        if need_stft:
            freq_img_resolution = int(
                getattr(args, 'freq_img_resolution', None)
                or max(self.stft_embedder.stft_hw())
            )
            freq_channels = 2 * self.ts_channels
            freq_attn = list(getattr(args, 'freq_attn_resolution', None) or [])
            if not freq_attn:
                base_attn = list(args.attn_resolution) if getattr(args, 'attn_resolution', None) is not None else []
                freq_attn = [r for r in base_attn if r <= freq_img_resolution] or infer_attn_resolution(freq_img_resolution)
            freq_unet_kwargs = dict(
                img_resolution=freq_img_resolution,
                in_channels=freq_channels,
                out_channels=freq_channels,
                label_dim=0,
                model_channels=args.unet_channels,
                channel_mult=args.ch_mult,
                attn_resolutions=freq_attn,
            )
            if getattr(args, 'num_blocks', None) is not None:
                freq_unet_kwargs['num_blocks'] = args.num_blocks
            freq_net = DhariwalUNet(**freq_unet_kwargs)
            self.freq_img_resolution = freq_img_resolution
            self.freq_input_channels = freq_channels

        predictor = MultiViewPredictor(
            prediction_mode=self.prediction_mode,
            delay_embedder=self.delay_embedder,
            stft_embedder=self.stft_embedder,
            time_net=time_net,
            freq_net=freq_net,
            ts_channels=self.ts_channels,
        )
        self.net = OneDEDMPrecond(
            predictor,
            sigma_data=self.sigma_data,
            sigma_min=0,
            sigma_max=float('inf'),
        )

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def init_stft_embedder(self, train_loader):
        """Cache dataset-level STFT min/max once (not sample-dependent)."""
        assert self.stft_embedder is not None, "STFTEmbedder was not created"
        data = []
        for data_batch in train_loader:
            data.append(data_batch[0])
        self.stft_embedder.cache_min_max_params(torch.cat(data, dim=0))
        assert self.stft_embedder.min_real is not None
        assert self.stft_embedder.max_real is not None
        assert self.stft_embedder.min_imag is not None
        assert self.stft_embedder.max_imag is not None

    def loss_fn(self, x):
        """x: clean 1D series [B, T, C]."""
        to_log = {}
        output, weight = self.forward(x)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return loss, to_log

    def forward(self, x, labels=None, augment_pipe=None):
        # Corruption always in 1D: x_t = x_0 + sigma * eps
        rnd_normal = torch.randn([x.shape[0], 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        noisy = y + torch.randn_like(y) * sigma
        out = self.net(noisy, sigma, labels, augment_labels=augment_labels)
        return out, weight

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        if self.use_ema:
            self.model_ema(self.net)
