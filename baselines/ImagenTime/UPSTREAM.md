# ImagenTime baseline

- Paper: Naiman et al., "Utilizing Image Transforms and Diffusion Models for Generative
  Modeling of Short and Long Time Series", NeurIPS 2024 (arXiv:2410.19538).
- Upstream code: https://github.com/azencot-group/ImagenTime, commit `f372626`
  (the vendored model code is identical in every upstream commit since `c40a61b`).
- License: the upstream repository ships **no LICENSE file**, so none is included here.
  `README.md` is the upstream README, unchanged.

Only the files on the unconditional training/sampling path are vendored. The metric modules
(`metrics/`, `models/testing_models/s4d.py`) are imported by `run_unconditional.py` but not
executed, because mid-training evaluation is off (see below).

## How to run

Environment: upstream `requirements.yaml` (conda env with PyTorch + torchaudio, omegaconf,
einops, opt_einsum, tqdm, scikit-learn). A CUDA GPU is required: upstream's
`DelayEmbedder.img_to_ts` moves its result to the GPU with `.cuda()`.

```bash
bash baselines/ImagenTime/run.sh                          # 7 datasets x seeds 1 2 3
SEEDS="1" DATASETS="ecg" bash baselines/ImagenTime/run.sh  # subset
# one run by hand:
cd baselines/ImagenTime
python -u run_unconditional.py --config configs/ecg.yaml --seed 1 --log_dir ../../checkpoints/ImagenTime/seed1/ecg
```

Each run trains for 200 epochs and then samples N series (N = size of the real dataset)
with the EMA weights of the last epoch, written in original units as float32 `(N, T, 1)` to
`outputs/ImagenTime/seed<N>/<dataset>/ImagenTime_<dataset>_generatedata.npy`.
The config key `benchmark_dataset` makes the runner take the data file and the output path
from `baselines/common.py`.

## Settings used for the paper

Seeds 1, 2, 3 (passed as `--seed`; the seed also fixes the data shuffle in the loader).
The seven configs in `configs/` differ only in the dataset, `num_samples`, `seq_len` and
`delay`; the three seeds use the same config.

| setting | value |
|---|---|
| epochs | 200 |
| batch size | 8 (also the sampling batch size) |
| optimizer | AdamW, lr 1e-4, weight decay 1e-5, grad-norm clip 1.0 |
| transform | delay embedding, `embedding` 32 (image 32x32, 1 channel) |
| delay | 2 for T=90 (precipitation), 4 for T=144, 8 for T=250 (ecg) |
| network | EDM-preconditioned U-Net, 128 channels, ch_mult [1,2,4,4], attention at [32,16,8] |
| diffusion | 18 sampling steps (EDM sampler), EMA with warmup 100 |
| mid-training evaluation | off (`eval_during_training: false`) |

`delay` is chosen so the `embedding` columns of the delay image cover the series: column i
holds `x[i*delay : i*delay+32]`, giving 31/30/29 filled columns out of 32 for T=90/144/250.
For T=250 the formula would give 7, but delays 5-7 make the special-case branch in
`DelayEmbedder` write one column past the image and fail, so 8 is used. The round trip
series -> image -> series is exact for all three settings.

## Changes to upstream

- `utils/utils_data.py`: the `toyexample` dataset branch loads an `(N, T, C)` `.npy` from
  `data_path`, min-max scales it to [0, 1] with one global min/max, shuffles it with
  `RandomState(seed)`, and keeps the min/max to undo the scaling on the samples. The eager
  imports of `data.long_range` / `data.data_provider` (not shipped, not used here) are moved
  into the branches that need them.
- `utils/utils_args.py`: new options `data_path`, `project_name`, `output_dir`, `num_samples`,
  `gen_batch_size`, `num_blocks`, `eval_during_training`, `benchmark_dataset`; `toyexample`
  added to the dataset choices.
- `models/model.py`, `utils/utils.py`: optional `num_blocks` pass-through to the U-Net
  (unset in our configs, so the upstream default is used).
- `run_unconditional.py`:
  - `eval_during_training: false` skips upstream's periodic evaluation. Upstream picks the
    checkpoint by a generation score computed against its `test_loader`, which for this
    loader branch is the training set itself; with evaluation off the last epoch is used and
    the checkpoint is overwritten every epoch.
  - after training, N samples are generated with the EMA weights, de-normalised and saved
    (only the generated set, to the path given by `baselines/common.py`).
- Configs: written for our seven datasets (hyperparameters in the table above); upstream's
  configs are not shipped. Upstream trains for 1000 epochs; we use 200.
