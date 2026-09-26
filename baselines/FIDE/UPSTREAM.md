# FIDE

**FIDE: Frequency-Inflated Conditional Diffusion Model for Extreme-Aware Time Series
Generation** — Galib, Tan, Luo, NeurIPS 2024.

- Upstream: <https://github.com/galib19/FIDE>, commit `0922b651314f9087751e3245bec4c905c099a001`
- License: **the upstream repository has no LICENSE file.** None is included here.

## What is here

| file | origin |
|---|---|
| `Code/model.py` | upstream; imports added, `seq_len` read from env var `FIDE_SEQ_LEN` (upstream relies on a global defined by its main script) |
| `Code/train_utilities.py` | upstream; imports and module globals (`device`, `model`, `is_regularizer`, `gev_model`) added so it is importable, `reg_loss` initialised in `get_loss` (upstream raises `UnboundLocalError` when the regulariser is off). `run_fide.py` does not call `get_loss`. |
| `Code/sampling.py` | upstream's `sample()` wrapped as an importable `ddpm_sample(t, bm_sample)`; the reverse-diffusion update is unchanged. Upstream's trailing notebook/plotting code removed. |
| `Code/temperature_inverse.py` | ours: inverse of upstream's high-frequency boost |
| `run_fide.py` | ours: replaces upstream's `main_temperature_example.py` (data loading, training loop, sampling, output) |
| `config.yaml` | hyperparameters used for the reported results |
| `run.sh` | runs all seven datasets |

Not included: upstream's data, evaluation scripts (`distribution_evaluation.py`,
`evaluation_*.py`, `TSNE_evaluation.py`, `general_utilities.py`), `data_process.py` and
`main_temperature_example.py`.

## Deviations from upstream

The model, the noise schedule, `add_noise` and the sampler are used unchanged.

1. **GEV regulariser made differentiable.** Upstream computes it through
   `.detach().cpu().numpy()` and scipy, so it is a constant with zero gradient and the
   released code trains a plain conditional DDPM. We evaluate the GEV log-density in torch.
2. **x0 estimate.** DDPM estimate `(x_t - sqrt(1-a)*eps)/sqrt(a)` instead of upstream's
   `x_noisy - pred_noise` (which omits the noise-schedule scaling).
3. **`lambda_1 = linear_decay(i)` applied** as a per-sample weight on the regulariser
   (upstream computes it and never uses it).
4. **GEV log-density floored at -50** (and its support argument clamped at 1e-6):
   without the floor, heavy-tailed datasets (GEV shape c < 0) produce exploding
   penalties that dominate the loss.
5. **2000 epochs instead of 400.** `batch_size` 2000 exceeds every N here, so an epoch is
   one full-batch gradient step; our sequences (T = 90-250) are much longer than
   upstream's example (T = 30).
6. **`drop_last=False`, `shuffle=True`.** Upstream's `batch_size=2000, drop_last=True`
   yields no batches at all when N < 2000 (all our datasets).
7. **Seed 10** for python/numpy/torch and the GEV draws (upstream sets no seed).
8. **Data and output.** Reads `<repo>/data/<file>.npy` (N, T, 1); the global
   StandardScaler and high-frequency boost are upstream's; samples are mapped back to
   original units and written to `<repo>/outputs/FIDE/<dataset>/FIDE_<dataset>_generatedata.npy`
   (float32, N = dataset size), plus `summary.json` and per-epoch `train_metrics.jsonl`.

Left as upstream has it: the 8 `i_proj` layers in `model.py` are built but never used
(their parameters receive no gradient); the GEV is fitted on block maxima of the scaled
data before the high-frequency boost.

`--no-regularizer` trains the as-released model (plain conditional DDPM); it was not used
for the reported results.

## Hyperparameters used for the reported results

`hidden_dim 64`, `batch_size 2000` (full batch), `n_epochs 2000`, `freq_enhance_c 1.1`,
`percentage_of_freq_enhanced 20`, GEV weight `0.05`, Adam defaults, `diffusion_steps 100`,
linear beta schedule 1e-4 -> 0.2, `gp_sigma 0.05`, seed 10, one run per dataset.

## Running

```bash
pip install -r baselines/FIDE/requirements.txt
bash baselines/FIDE/run.sh                          # all seven datasets
python baselines/FIDE/run_fide.py --dataset ecg     # one dataset
```

A GPU is strongly recommended (full-batch attention over all N series).
