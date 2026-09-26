import logging
import os
import sys
import shutil
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fdiff.dataloaders.datamodules import Datamodule, NpyDatamodule
from fdiff.models.score_models import ScoreModule
from fdiff.utils.callbacks import SamplingCallback
from fdiff.utils.extraction import dict_to_str, get_training_params

# Shared dataset directory of the release (<repo>/data/), via baselines/common.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common  # noqa: E402

DATA_DIR = common.DATA_DIR


class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        self.unified_output = cfg.get("project_name") is not None

        if self.unified_output:
            self._init_unified(cfg)
        else:
            self._init_legacy(cfg)

        # Finish instantiation of the model if necessary
        if isinstance(self.score_model, partial):
            training_params = get_training_params(self.datamodule, self.trainer)
            self.score_model = self.score_model(**training_params)

        # Possibly setup the datamodule in the sampling callback
        for callback in self.trainer.callbacks:  # type: ignore
            if isinstance(callback, SamplingCallback):
                callback.setup_datamodule(datamodule=self.datamodule)

    def _init_unified(self, cfg: DictConfig) -> None:
        """project_name mode: one flat <output_dir>/<project_name>/ folder
        holding only the checkpoint and the origin/generated data arrays."""
        assert cfg.get("config") is not None, (
            "project_name was set but config=<dataset> was not. "
            "Example: config=ecg project_name=FourierDiffusion_ecg_e1000"
        )
        self.project_name: str = cfg.project_name

        self.score_model: ScoreModule = instantiate(cfg.score_model)
        self.datamodule: Datamodule = NpyDatamodule(
            data_dir=DATA_DIR,
            filename=cfg.config.filename,
            num_dataset_samples=cfg.config.num_dataset_samples,
            random_seed=cfg.random_seed,
            batch_size=64,
            fourier_transform=cfg.fourier_transform,
            standardize=cfg.standardize,
        )

        self.save_dir = Path(cfg.output_dir) / self.project_name
        os.makedirs(self.save_dir, exist_ok=True)
        logging.info(f"[unified] Saving outputs to {self.save_dir}.")

        # Clear out any checkpoint left over from a previous run that reused
        # this same project_name, so the folder only ever contains the model
        # just trained.
        for stale_ckpt in self.save_dir.glob("*.ckpt"):
            logging.info(f"Removing stale checkpoint {stale_ckpt}")
            stale_ckpt.unlink()

        self.trainer: pl.Trainer = instantiate(
            cfg.trainer, default_root_dir=str(self.save_dir)
        )
        # Make the ModelCheckpoint callback write directly into save_dir,
        # instead of the default <project_name>/checkpoints/ subfolder.
        for callback in self.trainer.callbacks:  # type: ignore
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                callback.dirpath = str(self.save_dir)

        logging.info(
            "Welcome in the training script (project_name mode)! "
            f"project_name={self.project_name}, fourier_transform={cfg.fourier_transform}"
        )

        # Set-up dataset
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

        # Save the full original data (train+test, raw scale) for sample.py to
        # use later, without needing to re-instantiate the datamodule.
        origin_path = DATA_DIR / cfg.config.filename
        origin = np.load(origin_path)
        np.save(self.save_dir / f"{self.project_name}_origindata.npy", origin)

    def _init_legacy(self, cfg: DictConfig) -> None:
        """Upstream behaviour (datamodule from cfg.datamodule), used when
        project_name is not set. Not used for the reported results."""
        run_id = f"run_{cfg.random_seed}"

        # Instantiate score model and datamodule
        self.score_model: ScoreModule = instantiate(cfg.score_model)
        self.datamodule: Datamodule = instantiate(cfg.datamodule)

        # All outputs go to <output_dir>/{run_id}/
        self.save_dir = Path(cfg.output_dir) / run_id
        os.makedirs(self.save_dir, exist_ok=True)
        logging.info(f"Saving outputs to {self.save_dir}.")

        # Clear out any checkpoints left over from a previous training run that
        # reused this same output_dir/run_id, so that after this run finishes,
        # the checkpoints directory only ever contains the model just trained
        # (get_best_checkpoint would otherwise pick the lowest val_loss across
        # both the stale and the new checkpoints, not necessarily the new one).
        checkpoints_dir = self.save_dir / "checkpoints"
        if checkpoints_dir.exists():
            logging.info(
                f"Removing pre-existing checkpoints in {checkpoints_dir} before training."
            )
            shutil.rmtree(checkpoints_dir)

        self.trainer: pl.Trainer = instantiate(
            cfg.trainer, default_root_dir=str(self.save_dir)
        )

        logging.info(
            f"Welcome in the training script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        logging.info(f"Saving the config into {self.save_dir}.")
        OmegaConf.save(config=cfg, f=self.save_dir / "train_config.yaml")

        # Set-up dataset
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

    def train(self) -> None:
        assert not (
            self.score_model.scale_noise and not self.datamodule.fourier_transform
        ), "You cannot use noise scaling without the Fourier transform."
        self.trainer.fit(model=self.score_model, datamodule=self.datamodule)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    runner = TrainingRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
