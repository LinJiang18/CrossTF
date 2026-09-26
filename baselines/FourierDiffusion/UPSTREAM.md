# FourierDiffusion

**Time Series Diffusion in the Frequency Domain** — Crabbé, Huynh, Stanczuk, van der
Schaar ([arXiv:2402.05933](https://arxiv.org/abs/2402.05933)).

- Upstream: <https://github.com/JonathanCrabbe/FourierDiffusion>, commit
  `e60d532c0d7e304fae7e23beab7de3b8bf023051`
- License: MIT (`LICENSE`, upstream's, unchanged). `README.md` and `pyproject.toml` are
  upstream's (with the wandb setup step / dependency removed); `README.md` describes
  upstream's own datasets, notebooks and the LSTM backbone, which are not included here.

We run the **frequency-domain** model (`fourier_transform=true`, which also turns on the
score model's `fourier_noise_scaling`): a transformer score model trained with a VP-SDE
on the DFT coefficients of each series, sampled with 1000 Euler–Maruyama steps and mapped
back with an inverse DFT.

## What is here

| path | |
|---|---|
| `src/fdiff/` | upstream library (see changes below; `utils/wandb.py` removed) |
| `cmd/train.py`, `cmd/sample.py` | upstream entry points plus a `project_name` mode (below) |
| `cmd/conf/` | upstream Hydra configs actually composed by the two entry points; `cmd/conf/config` is a symlink to `config/` |
| `config/<dataset>.yaml` | ours: one per dataset, `filename` + `num_dataset_samples` |
| `collect.py` | ours: copies the generated array into `<repo>/outputs/` and writes `summary.json` |
| `run.sh` | train + sample + collect for all seven datasets |

Not included: upstream's `notebooks/`, `tests/`, other datamodule configs, the LSTM/MLP
score-model configs.

## Deviations from upstream

1. **Frequency domain.** `fourier_transform=true` is passed on the command line (upstream's
   `train.yaml` default is `false`, the time-domain variant).
2. **`max_epochs = 1000`** instead of upstream's 200 (hard-coded in `run.sh`; the YAML
   still says 200). At 200 epochs the model was clearly under-trained on our small
   datasets (N = 1000–1838; e.g. ECG samples had ~9x the real standard deviation and the
   validation loss was still improving far past epoch 200). One value for all seven
   datasets.
3. **`project_name` mode** in `cmd/train.py` / `cmd/sample.py` (used for all results):
   trains on `<repo>/data/<filename>` through `NpyDatamodule` (ours, added to
   `datamodules.py`: seeded random 90/10 train/validation split, `batch_size` 64), keeps the
   single best checkpoint (lowest `val/loss`) in `runs/<project_name>/`, and `sample.py`
   generates **N = dataset size** series (upstream: 10000) and writes them destandardised
   and inverse-DFT'd, i.e. in original units. The destandardisation statistics in
   `sample.py` are computed on the full dataset, whereas training standardises with the
   90 % training split (a small difference in mean/std).
4. **No wandb, no `SamplingCallback` / `LearningRateMonitor`.** The trainer uses
   `logger: false` and only a `ModelCheckpoint` callback (`trainer/callbacks/no_logger.yaml`).
   None of these enter the loss, but upstream's mid-training sampling consumed the torch
   RNG, so runs are not bit-identical to upstream.
5. **`DiffusionSampler.sample`** rounds the number of batches up, so the last partial
   batch is no longer dropped (upstream returns `floor(N/200)*200` samples). It also
   accepts an optional `x0_pred_every` argument (and the SDE schedulers gained
   `x0_from_score`), unused here.
6. Minor: `ECGDatamodule` accepts `max_train_samples` / `max_test_samples` (upstream ECG
   config only, not used here); `get_best_checkpoint` raises a clear error when no
   checkpoint exists.

## Hyperparameters used for the reported results

Upstream defaults except items 1–2: `random_seed 42`, `d_model 72`, `num_layers 10`,
`n_head 12`, `lr_max 1e-3` (cosine schedule with warm-up), VP-SDE (`beta_min 0.1`,
`beta_max 20`, `eps 1e-5`), `likelihood_weighting false`, `standardize true`,
`gradient_clip_val 1.0`, `batch_size 64`, `train_ratio 0.9`, **`max_epochs 1000`**,
`num_diffusion_steps 1000`, `sample_batch_size 200`, `fourier_transform true`.
One run (seed 42) per dataset.

## Running

```bash
pip install -r baselines/FourierDiffusion/requirements.txt
bash baselines/FourierDiffusion/run.sh
```

which, per dataset `<ds>`, runs from inside `baselines/FourierDiffusion/` with
`PYTHONPATH=src`:

```bash
python cmd/train.py  config=<ds> project_name=FourierDiffusion_<ds>_e1000 fourier_transform=true trainer.max_epochs=1000
python cmd/sample.py config=<ds> project_name=FourierDiffusion_<ds>_e1000 fourier_transform=true
python collect.py --dataset <ds> --run-name FourierDiffusion_<ds>_e1000 --max-epochs 1000
```

Final output: `<repo>/outputs/FourierDiffusion/<ds>/FourierDiffusion_<ds>_generatedata.npy`
(float32, shape (N, T, 1)). Checkpoints and Hydra logs are written to
`baselines/FourierDiffusion/runs/` (git-ignored; delete after collecting).
