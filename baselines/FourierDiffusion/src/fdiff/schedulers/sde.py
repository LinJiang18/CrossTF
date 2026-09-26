"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs. Adapted from https://github.com/yang-song/score_sde."""

import abc
import math
from collections import namedtuple
from typing import Optional

import torch

SamplingOutput = namedtuple("SamplingOutput", ["prev_sample"])


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, fourier_noise_scaling: bool = False, eps: float = 1e-5):
        """Construct an SDE.
        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.noise_scaling = fourier_noise_scaling
        self.eps = eps
        self.G: Optional[torch.Tensor] = None

    @property
    def T(self) -> float:
        """End time of the SDE."""
        return 1.0

    @abc.abstractmethod
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""

    @abc.abstractmethod
    def step(
        self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput: ...

    @abc.abstractmethod
    def x0_from_score(
        self, x_t: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Recover the model-space denoised sample from x_t and the score
        s_theta(x_t, t), without running any additional reverse steps.

        Under the Gaussian marginal p_t(x_t | x_0) = N(mean(t), Sigma_t) used by
        `marginal_prob`, score = grad log p_t(x_t|x_0) = -Sigma_t^{-1} (x_t - mean(t)),
        so mean(t) = x_t + Sigma_t * score. `mean(t)` equals x_0 directly for VE
        (marginal_prob's mean is the identity), or alpha(t) * x_0 for VP, hence the
        different final division in each subclass.
        """

    def set_noise_scaling(self, max_len: int) -> None:
        """Finish the initialization of the scheduler by setting G (scaling diagonal)

        Args:
            max_len (int): number of time steps of the time series
        """

        G = torch.ones(max_len)
        if self.noise_scaling:
            G = 1 / (math.sqrt(2)) * G
            # Double the variance for the first component
            G[0] *= math.sqrt(2)
            # Double the variance for the middle component if max_len is even
            if max_len % 2 == 0:
                G[max_len // 2] *= math.sqrt(2)

        self.G = G  # Tensor of size (max_len)
        self.G_matrix = torch.diag(G)  # Tensor of size (max_len, max_len)
        assert G.shape[0] == max_len

    def set_timesteps(self, num_diffusion_steps: int) -> None:
        self.timesteps = torch.linspace(1.0, self.eps, num_diffusion_steps)
        self.step_size = self.timesteps[0] - self.timesteps[1]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x0 = original_samples
        mean, _ = self.marginal_prob(x0, timesteps)

        # Note that the std is not used here because the noise has been scaled prior to calling the function
        sample = mean + noise
        return sample

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        # Reshape the G matrix to be (1, max_len, max_len)
        scaling_matrix = self.G_matrix.view(
            -1, self.G_matrix.shape[0], self.G_matrix.shape[1]
        )

        z = torch.randn(*shape)
        # Return G@z where z \sim N(0,I)
        return torch.matmul(scaling_matrix, z)


class VEScheduler(SDE):
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
    ):
        """Construct a Variance Exploding SDE.
        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(fourier_noise_scaling=fourier_noise_scaling, eps=eps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # perturbation kernel P(X(t)|X(0)) parameters
        if self.G is None:
            self.set_noise_scaling(x.shape[1])
        assert self.G is not None

        sigma_min = torch.tensor(self.sigma_min).type_as(t)
        sigma_max = torch.tensor(self.sigma_max).type_as(t)
        std = (sigma_min * (sigma_max / sigma_min) ** t).view(-1, 1) * self.G.to(
            x.device
        )
        mean = x
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        # In the case of VESDE, the prior is scaled by the maximum noise std
        return self.sigma_max * super().prior_sampling(shape)

    def x0_from_score(
        self, x_t: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        # marginal_prob's mean is the identity for VE (mean = x_0), so
        # x0_hat = x_t + Sigma_t * score directly. `std` only depends on t and
        # G (not on the values of x_t), so reusing marginal_prob here is exact.
        _, std = self.marginal_prob(x_t, t)  # std: (batch_size, max_len)
        var = (std**2).unsqueeze(-1)  # (batch_size, max_len, 1), broadcasts over channels
        return x_t + var * score

    def step(
        self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput:
        """Single denoising step, used for sampling.

        Args:
            model_output (torch.Tensor): output of the score model
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): current sample to be denoised

        Returns:
            SamplingOutput: _description_
        """

        sqrt_derivative = (
            self.sigma_min
            * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))
            * (self.sigma_max / self.sigma_min) ** (timestep)
        )

        diffusion = torch.diag_embed(sqrt_derivative * self.G).to(device=sample.device)

        # Compute drift for the reverse: f(x,t) - G(x,t)G(x,t)^{T}*score
        drift = -(
            torch.matmul(diffusion * diffusion, model_output)
        )  # Notice that the drift of the forward is 0

        # Sample noise
        z = torch.randn_like(sample)
        assert self.step_size > 0
        x = (
            sample
            - drift * self.step_size  # - sign because of reverse time
            + torch.sqrt(self.step_size) * torch.matmul(diffusion, z)
        )
        output = SamplingOutput(prev_sample=x)
        return output


class VPScheduler(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
    ):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
          G: tensor of size max_len
        """
        super().__init__(fourier_noise_scaling=fourier_noise_scaling, eps=eps)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """-1/2*\\int_0^t \\beta(s) ds, i.e. log(alpha(t))."""
        return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # first check if G has been init.
        if self.G is None:
            self.set_noise_scaling(x.shape[1])
        assert self.G is not None

        log_mean_coeff = self.log_mean_coeff(t)

        mean = (
            torch.exp(log_mean_coeff[(...,) + (None,) * len(x.shape[1:])]) * x
        )  # mean: (batch_size, max_len, n_channels)

        std = torch.sqrt(
            (1.0 - torch.exp(2.0 * log_mean_coeff.view(-1, 1)))
        ) * self.G.to(
            x.device
        )  # std: (batch_size, max_len)

        return mean, std

    def x0_from_score(
        self, x_t: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        # marginal_prob's mean is alpha(t) * x_0 for VP, so
        # x0_hat = (x_t + Sigma_t * score) / alpha(t). `alpha_t` is recomputed
        # with the exact same `log_mean_coeff` used by marginal_prob, and `std`
        # only depends on t and G (not on the values of x_t), so reusing
        # marginal_prob here for Sigma_t is exact.
        _, std = self.marginal_prob(x_t, t)  # std: (batch_size, max_len)
        var = (std**2).unsqueeze(-1)  # (batch_size, max_len, 1), broadcasts over channels
        alpha_t = torch.exp(self.log_mean_coeff(t)).view(-1, 1, 1)
        return (x_t + var * score) / alpha_t

    def get_beta(self, timestep: float) -> float:
        return self.beta_0 + timestep * (self.beta_1 - self.beta_0)

    def step(
        self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput:
        """Single denoising step, used for sampling.

        Args:
            model_output (torch.Tensor): output of the score model
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): current sample to be denoised

        Returns:
            SamplingOutput: _description_
        """
        beta = self.get_beta(timestep)
        assert self.G is not None
        diffusion = torch.diag_embed(math.sqrt(beta) * self.G).to(device=sample.device)

        # Compute drift
        drift = -0.5 * beta * sample - (
            torch.matmul(diffusion * diffusion, model_output)
        )

        # Sample noise
        z = torch.randn_like(sample)
        assert self.step_size > 0
        x = (
            sample
            - drift * self.step_size
            + torch.sqrt(self.step_size) * torch.matmul(diffusion, z)
        )
        output = SamplingOutput(prev_sample=x)
        return output
