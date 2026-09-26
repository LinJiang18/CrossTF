from typing import Optional

import torch
from tqdm import tqdm

from fdiff.models.score_models import ScoreModule
from fdiff.schedulers.sde import SDE
from fdiff.utils.dataclasses import DiffusableBatch


class DiffusionSampler:
    def __init__(
        self,
        score_model: ScoreModule,
        sample_batch_size: int,
    ) -> None:
        self.score_model = score_model
        self.noise_scheduler = score_model.noise_scheduler

        self.sample_batch_size = sample_batch_size
        self.n_channels = score_model.n_channels
        self.max_len = score_model.max_len

    def reverse_diffusion_step(
        self, batch: DiffusableBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get X and timesteps
        X = batch.X
        timesteps = batch.timesteps

        # Check the validity of the timestep (current implementation assumes same time for all samples)
        assert timesteps is not None and timesteps.size(0) == len(batch)
        assert torch.min(timesteps) == torch.max(timesteps)

        # Predict score for the current batch
        score = self.score_model(batch)
        # Apply a step of reverse diffusion
        output = self.noise_scheduler.step(
            model_output=score, timestep=timesteps[0].item(), sample=X
        )

        X_prev = output.prev_sample
        assert isinstance(X_prev, torch.Tensor)

        return X_prev, score

    def sample(
        self,
        num_samples: int,
        num_diffusion_steps: Optional[int] = None,
        x0_pred_every: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, dict[str, object]]]:
        """Run the reverse SDE from X_T down to X_0.

        Args:
            num_samples: number of samples to generate.
            num_diffusion_steps: number of Euler-Maruyama discretization steps.
            x0_pred_every: if set, additionally records, every `x0_pred_every`
                steps (step indices 0, x0_pred_every, 2*x0_pred_every, ...), the
                model-space denoised prediction x0_hat computed from the current
                x_t and the score just evaluated for the update at that step
                (no extra forward passes or additional trajectories). Returns
                `(samples, x0_predictions)` in that case, where `x0_predictions`
                maps step_idx -> {"timestep": float, "x0_model_space": Tensor of
                shape (num_samples, max_len, n_channels)}.
        """
        # Set the score model in eval mode and move it to GPU
        self.score_model.eval()

        # If the number of diffusion steps is not provided, use the number of training steps
        num_diffusion_steps = (
            self.score_model.num_training_steps
            if num_diffusion_steps is None
            else num_diffusion_steps
        )
        self.noise_scheduler.set_timesteps(num_diffusion_steps)

        # Create the list that will store the samples
        all_samples = []
        x0_by_step: dict[int, list[torch.Tensor]] = {}
        x0_timestep: dict[int, float] = {}

        # Compute the required amount of batches (round up so that the last,
        # partial batch is not silently dropped when num_samples is not a
        # multiple of sample_batch_size).
        num_batches = max(1, -(-num_samples // self.sample_batch_size))

        # No need to track gradients when sampling
        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc="Sampling",
                unit="batch",
                leave=False,
                colour="blue",
            ):
                # Compute the batch size
                batch_size = min(
                    num_samples - batch_idx * self.sample_batch_size,
                    self.sample_batch_size,
                )
                # Sample from noise distribution
                X = self.sample_prior(batch_size)

                # Perform the diffusion step by step
                for step_idx, t in enumerate(
                    tqdm(
                        self.noise_scheduler.timesteps,
                        desc="Diffusion",
                        unit="step",
                        leave=False,
                        colour="green",
                    )
                ):
                    # Define timesteps for the batch
                    timesteps = torch.full(
                        (batch_size,),
                        t,
                        dtype=(
                            torch.long if isinstance(t.item(), int) else torch.float32
                        ),
                        device=self.score_model.device,
                        requires_grad=False,
                    )
                    # Create diffusable batch
                    batch = DiffusableBatch(X=X, y=None, timesteps=timesteps)
                    # Return denoised X and the score used to compute it

                    X_prev, score = self.reverse_diffusion_step(batch)

                    if x0_pred_every is not None and step_idx % x0_pred_every == 0:
                        # Reuse X (=x_t) and the score just evaluated above: no
                        # extra forward pass, no extra trajectory.
                        x0_hat = self.noise_scheduler.x0_from_score(
                            x_t=X, t=timesteps, score=score
                        )
                        x0_by_step.setdefault(step_idx, []).append(
                            x0_hat.detach().cpu()
                        )
                        x0_timestep[step_idx] = float(t.item())

                    X = X_prev

                # Add the samples to the list
                all_samples.append(X.cpu())

        samples = torch.cat(all_samples, dim=0)

        if x0_pred_every is None:
            return samples

        x0_predictions = {
            step_idx: {
                "timestep": x0_timestep[step_idx],
                "x0_model_space": torch.cat(tensors, dim=0),
            }
            for step_idx, tensors in x0_by_step.items()
        }
        return samples, x0_predictions

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        # Sample from the prior distribution
        if isinstance(self.noise_scheduler, SDE):
            X = self.noise_scheduler.prior_sampling(
                (batch_size, self.max_len, self.n_channels)
            ).to(device=self.score_model.device)

        else:
            raise NotImplementedError("Scheduler not recognized.")

        assert isinstance(X, torch.Tensor)
        return X
