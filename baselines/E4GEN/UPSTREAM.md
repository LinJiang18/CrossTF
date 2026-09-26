# E4GEN and HeavyDiff baselines

This directory vendors the code of **E4GEN** (*E4GEN: Event-level Explainable
Extreme-Enhanced Time-series Generation*, arXiv:2606.01634), a third-party baseline.
E4GEN's backbone is derived from **Diffusion-TS** (https://github.com/Y-debug-sys/Diffusion-TS);
the `Models/interpretable_diffusion/`, `engine/` and `Utils/io_utils.py` files follow that
code base. `LICENSE` is the MIT license shipped with the E4GEN code, unchanged.

The **HeavyDiff** baseline uses the same code: E4GEN's own `--heavy` option replaces the
Gaussian diffusion noise with Student-t noise (training and sampling), and only the two
backbone stages (`--train`, `--sample`) are run. There is no separate HeavyDiff directory;
its launcher is `run_heavydiff.sh` here.

Only the files on the run path are included (the upstream analysis notebooks, figures,
metrics, TS2Vec, classifier, toy data/configs and bundled data copies are omitted).

## Layout

| path | |
|---|---|
| `main.py` | upstream entry point (one stage per process) |
| `engine/`, `Data/`, `Utils/io_utils.py`, `Models/` | upstream model / training code |
| `Config/<Cfg>_E4GEN.yaml` | E4GEN configs (five-stage pipeline) |
| `Config/<Cfg>.yaml` | base configs, used by HeavyDiff |
| `collect.py` | ours: copies `OUTPUT/<run>/generate_data_unnorm_<run>.npy` to the shared `outputs/` layout |
| `run.sh` | ours: E4GEN, 7 datasets x 3 seeds |
| `run_heavydiff.sh` | ours: HeavyDiff, 7 datasets x 1 run |

`<Cfg>` is `Building, ECG, Electricity, precipitation, temperature, Traffic, transportation`
for the datasets `building, ecg, electricity, precipitation, temperature, traffic,
transportation`. All scripts must be run from this directory (the launchers `cd` here);
config `data_path`s are relative to it (`../../data/<file>.npy`). `main.py` writes run
artifacts to `OUTPUT/<run>/` and checkpoints to `Checkpoint/<run>_<seq_len>/` (both
created at run time, git-ignored).

## Running

Environment: Python 3.10, `pip install -r requirements.txt`. One GPU; each E4GEN
(dataset, seed) takes roughly 15-25 minutes on an A100, each HeavyDiff dataset 9-12 minutes.

```bash
cd baselines/E4GEN
bash run.sh              # E4GEN     -> outputs/E4GEN/<ds>/ (seed 1), outputs/E4GEN/seed{2,3}/<ds>/
bash run_heavydiff.sh    # HeavyDiff -> outputs/HeavyDiff/<ds>/
# overrides: DATASETS="ecg traffic" SEEDS="2 3" PYTHON=python3 bash run.sh
```

Per (dataset, seed) `run.sh` executes

```bash
python main.py --name <run> --config_file Config/<Cfg>_E4GEN.yaml [--seed N] --train
python main.py --name <run> --config_file Config/<Cfg>_E4GEN.yaml [--seed N] --train_extreme
python main.py --name <run> --config_file Config/<Cfg>_E4GEN.yaml [--seed N] --validate_extreme
python main.py --name <run> --config_file Config/<Cfg>_E4GEN.yaml [--seed N] --train_controlnet
python main.py --name <run> --config_file Config/<Cfg>_E4GEN.yaml [--seed N] --sample --use_controlnet
python collect.py --method E4GEN --dataset <ds> --run_name <run> [--seed N]
```

and per dataset `run_heavydiff.sh` executes

```bash
python main.py --name <ds>_heavydiff --config_file Config/<Cfg>.yaml --train  --heavy --var_loss_weight 3
python main.py --name <ds>_heavydiff --config_file Config/<Cfg>.yaml --sample --heavy --var_loss_weight 3
python collect.py --method HeavyDiff --dataset <ds> --run_name <ds>_heavydiff
```

Output: `outputs/<Method>/[seed<N>/]<ds>/<Method>_<ds>_generatedata.npy`, float32,
shape (N, T, 1) equal to the real dataset, original units (`main.py`'s `_unnorm` file;
no rescaling). Both methods sample exactly N sequences.

## Seeds used for the reported results

* **E4GEN**: three training seeds.
  * seed 1: run name `<Cfg>_E4GEN`, **no `--seed` flag** (so `seed_everything` is never
    called and this run is not bit-reproducible); flat output `outputs/E4GEN/<ds>/`.
  * seeds 2, 3: run name `<Cfg>_E4GEN_s<N>`, `--seed N` on **every** stage (each stage is a
    separate process that seeds itself); output `outputs/E4GEN/seed<N>/<ds>/`.
* **HeavyDiff**: one run per dataset, no `--seed`; flat output `outputs/HeavyDiff/<ds>/`.

## Hyperparameters (as run)

Shared by all 14 configs: Diffusion-TS/E4GEN transformer backbone with `n_layer_enc = 2`,
`n_layer_dec = 2`, `d_model = 64`, `n_heads = 4`, `mlp_hidden_times = 4`; `timesteps =
sampling_timesteps = 1000` (no fast sampling), cosine beta schedule, L1 loss;
`max_epochs = 10000` optimiser steps, `batch_size = 64`, `gradient_accumulate_every = 2`,
`base_lr = 1e-5` with 500-step warm-up to 8e-4 and ReduceLROnPlateau, EMA 0.995;
`val_holdout_fraction = 0.8` (only stages 2-3 use the split); `heavy_nu = 2.5`,
`heavy_var_correction = True` (read only when `--heavy` is set);
`num_samples = N` of the dataset. Extreme predictor / ControlNet settings
(`K_max = 8`, `t_min = 0`, `t_max = 100`, epochs, loss weights) are as shipped upstream.

| dataset | T | N | E4GEN `var_loss_weight` | E4GEN extreme threshold (raw units, tail) | HeavyDiff `var_loss_weight` |
|---|---:|---:|---:|---|---:|
| building | 144 | 1838 | 0.175 | 513.2, upper | 3 |
| ecg | 250 | 1000 | 0.175 | 1.22, upper | 3 |
| electricity | 144 | 1096 | 0.175 | 3.1794, upper | 3 |
| precipitation | 90 | 1056 | 0.175 | 2030.0, upper | 3 |
| temperature | 144 | 1520 | 0.175 | 8.3, lower | 3 |
| traffic | 144 | 1790 | 0.175 | 69.0, upper | 3 |
| transportation | 144 | 1320 | 0.175 | 0.1147, upper | 3 |

HeavyDiff's `var_loss_weight = 3` is always passed on the command line; it overrides the
base config value (3 for five datasets, -1 = term disabled for precipitation and
temperature).

## Deviations from upstream

* **Extreme threshold**: `extreme_model.prediction.threshold` and
  `extreme_model.controlnet.threshold` in the `_E4GEN` configs set to the top 5% of the raw
  data (upstream: roughly the 90th percentile), to match the 5% tail used by our
  evaluation. temperature is a lower-tail dataset whose upstream value (8.3) already is
  its 5th percentile, so it is unchanged.
* **`var_loss_weight = 0.175` in `precipitation_E4GEN.yaml` and `temperature_E4GEN.yaml`**
  (upstream: 1 and 0), the value every other upstream `_E4GEN` config uses.
* **New configs for building and traffic** (`Building[_E4GEN].yaml`,
  `Traffic[_E4GEN].yaml`): cloned from the Electricity configs (also T = 144), changing
  only N, `data_path` and the threshold.
* **Data paths** point to the shared `../../data/*.npy` instead of upstream's bundled
  `./Data/Extreme_Data/...` copies (identical files for the five upstream datasets).
* **`main.py`**: `model.to(device)` instead of `model.cuda()` (identical on a GPU; also
  allows CPU runs). No other code change.
* **HeavyDiff**: `var_loss_weight` pinned to 3 on the command line for all datasets
  (see above), so all seven datasets are trained with the same loss.
* **Added** `collect.py`, `run.sh`, `run_heavydiff.sh`; configs moved from
  `Config/Extreme/` to `Config/`; `requirements.txt` trimmed to the run-time dependencies.
