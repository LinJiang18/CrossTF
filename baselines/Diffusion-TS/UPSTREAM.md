# Diffusion-TS (baseline)

Yuan & Qiao, *Diffusion-TS: Interpretable Diffusion for General Time Series Generation*,
ICLR 2024.

- Upstream: https://github.com/Y-debug-sys/Diffusion-TS
- Upstream commit: `566307e`
- License: MIT (`LICENSE`, unchanged). `README.md` is upstream's README, unchanged; it
  describes the full upstream repository, of which only the files below are vendored.

## What is here

| path | origin |
|---|---|
| `engine/solver.py` | upstream, one change (see below) |
| `engine/lr_sch.py`, `Data/build_dataloader.py`, `Utils/io_utils.py`, `Utils/masking_utils.py`, `Models/interpretable_diffusion/{gaussian_diffusion,transformer,model_utils}.py` | upstream, unchanged |
| `Utils/Data_utils/preprocessed_dataset.py` | ours: loader for pre-segmented `(N, T, C)` `.npy` arrays, same MinMax + `[-1, 1]` scaling as upstream's loaders |
| `run_diffusion_ts.py`, `config.yaml`, `run.sh`, `requirements.txt` | ours |

Only files on the import path of `run_diffusion_ts.py` are included (no `main.py`,
configs, notebooks, metrics, `ts2vec`, or the classifier-guidance code).

## Running

```bash
pip install -r baselines/Diffusion-TS/requirements.txt
bash baselines/Diffusion-TS/run.sh                       # 7 datasets x 3 seeds
python baselines/Diffusion-TS/run_diffusion_ts.py --dataset ecg --seed-index 2   # one run
```

Output: `outputs/Diffusion-TS/seed<index>/<dataset>/Diffusion-TS_<dataset>_generatedata.npy`
(float32, `(N, T, 1)`, original units, N = number of real series) plus a small
`summary.json`. Checkpoints (every 2000 steps) and `train_metrics.jsonl` go to
`baselines/Diffusion-TS/work/seed<index>/<dataset>/` (override with `--work-dir` or
`DTS_WORK`). One GPU per run.

## Hyperparameters and seeds used for the paper

Shared by all seven datasets (`config.yaml`; the derivation from upstream's own configs is
written out there): `n_layer_enc 3`, `n_layer_dec 3`, `d_model 64`, `n_heads 4`,
`mlp_hidden_times 4`, `timesteps = sampling_timesteps = 1000`, cosine schedule, L1 loss,
`attn_pd = resid_pd = 0`, `kernel_size 1`; 20000 gradient steps, batch 64,
`gradient_accumulate_every 2`, `base_lr 1e-5`, warmup 500 steps to `8e-4`,
ReduceLROnPlateau (factor 0.5, patience 4000), EMA 0.995 every 10 steps; sampling in
chunks of 256 with the EMA model, full 1000-step reverse process. `seq_length` is the
dataset's T, `feature_size 1`.

Three runs per dataset:

| seed-index (output dir) | numeric seed |
|---|---|
| 1 (`seed1/`) | 12345 (upstream `main.py` default) |
| 2 (`seed2/`) | 2 |
| 3 (`seed3/`) | 3 |

## Deviations from upstream

- **Output in original units**: upstream `main.py` stops at `[0, 1]` (the MinMax inverse is commented out); we call `dataset.unnormalize()`.
- **Capacity derived, not copied**: upstream ships no config for univariate series of length 90-250; values are extrapolated from upstream's configs (see `config.yaml`).
- **Exactly N samples**: `Trainer.sample` in `engine/solver.py` is changed to draw exactly `num` samples; upstream draws `num // size_every + 1` full batches and never trims.
- **Sampling chunk 256 instead of 2001**: upstream's 2001 is sized for length 24 and does not fit in memory at our lengths; chunking does not change the distribution.
- **No train/test split** (`proportion: 1.0`): same as upstream for unconditional generation.
- **Data loader**: `PreprocessedNpyDataset` reads our `.npy` files; normalisation matches upstream (MinMax per feature, then `[-1, 1]`).
- **Logging**: training loss is written to jsonl via the logger hook `Trainer` already calls, instead of tensorboard.
- **Environment**: torch 2.11 instead of the pinned 2.0.1; mujoco / dm-control / gluonts dropped from `requirements.txt` (only used by upstream loaders that are not vendored).
