import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.metrics import MetricCollection
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.extraction import dict_to_str, get_best_checkpoint, get_model_type
from fdiff.utils.fourier import dft, idft


class SamplingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        self.random_seed: int = cfg.random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        self.unified_output = cfg.get("project_name") is not None

        if self.unified_output:
            self._init_unified(cfg)
        else:
            self._init_legacy(cfg)

    def _init_unified(self, cfg: DictConfig) -> None:
        """project_name mode: reads <output_dir>/<project_name>/ directly
        (checkpoint + <project_name>_origindata.npy saved by train.py), with
        no train_config.yaml / metrics / results.yaml."""
        self.project_name: str = cfg.project_name
        self.fourier_transform: bool = bool(cfg.fourier_transform)
        self.save_dir = Path(cfg.output_dir) / self.project_name

        logging.info(
            "Welcome in the sampling script (project_name mode)! "
            f"project_name={self.project_name}, fourier_transform={self.fourier_transform}"
        )

        origin_path = self.save_dir / f"{self.project_name}_origindata.npy"
        assert origin_path.exists(), (
            f"Origin data not found at {origin_path}. Run train.py with the same "
            f"project_name first."
        )
        origin = np.load(origin_path)
        origin_t = torch.tensor(origin, dtype=torch.float32)
        # Standardization stats must be computed in the same space the model
        # was trained on: DiffusionDataset applies dft() before standardizing
        # whenever fourier_transform=True, so the mean/std used to destandardize
        # here have to be taken over the DFT coefficients too, not over the raw
        # time-domain series (using raw-domain stats corrupts every frequency
        # coefficient's scale/offset before the inverse DFT).
        stats_basis = dft(origin_t) if self.fourier_transform else origin_t
        self.feature_mean = stats_basis.mean(dim=0)
        self.feature_std = stats_basis.std(dim=0)

        # Default: generate as many samples as the original dataset size
        self.num_samples = (
            origin.shape[0] if cfg.num_samples is None else cfg.num_samples
        )
        self.num_diffusion_steps: int = cfg.num_diffusion_steps

        # Load score model from the single checkpoint in the project folder
        ckpts = list(self.save_dir.glob("*.ckpt"))
        assert len(ckpts) == 1, (
            f"Expected exactly one checkpoint in {self.save_dir}, found {len(ckpts)}. "
            "Re-run train.py with the same project_name."
        )
        self.score_model = ScoreModule.load_from_checkpoint(
            checkpoint_path=ckpts[0],
            weights_only=False,  # checkpoint contains custom fdiff scheduler classes
        )
        if torch.cuda.is_available():
            self.score_model.to(device=torch.device("cuda"))

        # Instantiate sampler
        sampler_partial = instantiate(cfg.sampler)
        self.sampler: DiffusionSampler = sampler_partial(score_model=self.score_model)

    def _init_legacy(self, cfg: DictConfig) -> None:
        """Upstream behaviour (reads train_config.yaml and computes upstream's
        metrics), used when project_name is not set. Not used for the reported
        results."""
        # Read out the config
        logging.info(
            f"Welcome in the sampling script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        # Get model path and id
        self.model_path = Path(cfg.model_path)
        self.model_id = cfg.model_id

        # Save sampling config to model directory
        self.save_dir = self.model_path / self.model_id
        OmegaConf.save(config=cfg, f=self.save_dir / "sample_config.yaml")

        # Read training config from model directory and instantiate the right datamodule
        train_cfg = OmegaConf.load(self.save_dir / "train_config.yaml")
        self.datamodule: Datamodule = instantiate(train_cfg.datamodule)
        self.fourier_transform = self.datamodule.fourier_transform
        self.datamodule.prepare_data()
        self.datamodule.setup()
        # Default: generate as many samples as the input dataset size
        if cfg.num_samples is None:
            self.num_samples = getattr(
                self.datamodule,
                "num_dataset_samples",
                len(self.datamodule.X_train),
            )
        else:
            self.num_samples = cfg.num_samples
        self.num_diffusion_steps = cfg.num_diffusion_steps

        # Load score model from checkpoint
        best_checkpoint_path = get_best_checkpoint(self.save_dir / "checkpoints")
        model_type = get_model_type(train_cfg)
        self.score_model = model_type.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path,
            weights_only=False,  # checkpoint contains custom fdiff scheduler classes
        )
        if torch.cuda.is_available():
            self.score_model.to(device=torch.device("cuda"))

        # Instantiate sampler
        sampler_partial = instantiate(cfg.sampler)
        self.sampler: DiffusionSampler = sampler_partial(score_model=self.score_model)

        # Instantiate metrics
        metrics_partial = instantiate(cfg.metrics)
        self.metrics: MetricCollection = metrics_partial(
            original_samples=self.datamodule.X_train
        )

    def sample(self) -> None:
        # Sample from score model
        X = self.sampler.sample(
            num_samples=self.num_samples,
            num_diffusion_steps=self.num_diffusion_steps,
        )

        if self.unified_output:
            self._sample_unified(X)
        else:
            self._sample_legacy(X)

    def _to_raw_scale(self, X: torch.Tensor) -> torch.Tensor:
        """Destandardize, then inverse-DFT if trained in the frequency domain."""
        X = X * self.feature_std + self.feature_mean
        if self.fourier_transform:
            X = idft(X)
        return X

    def _sample_unified(self, X: torch.Tensor) -> None:
        X = self._to_raw_scale(X)

        out_path = self.save_dir / f"{self.project_name}_generatedata.npy"
        np.save(out_path, X.numpy())
        logging.info(f"Saved generated data to {out_path} (shape={tuple(X.shape)}).")

    def _sample_legacy(self, X: torch.Tensor) -> None:
        # Map to the original scale if the input was standardized
        if self.datamodule.standardize:
            feature_mean, feature_std = self.datamodule.feature_mean_and_std
            X = X * feature_std + feature_mean

        # If sampling in frequency domain, bring back the sample to time domain
        if self.fourier_transform:
            X = idft(X)

        # Compute metrics
        results = self.metrics(X)
        logging.info(f"Metrics:\n{dict_to_str(results)}")

        # Save everything
        logging.info(f"Saving samples ands metrics to {self.save_dir}.")
        yaml.dump(
            data=results,
            stream=open(self.save_dir / "results.yaml", "w"),
        )
        torch.save(X, self.save_dir / "samples.pt")


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    runner = SamplingRunner(cfg)
    runner.sample()


if __name__ == "__main__":
    main()
