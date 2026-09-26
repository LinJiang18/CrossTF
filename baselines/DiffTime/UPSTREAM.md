# DiffTime baseline

- Paper: Coletta, Gopalakrishnan, Borrajo, Vyetrenko, "On the Constrained Time-Series
  Generation Problem", NeurIPS 2023 (arXiv:2307.01717). We run its unconstrained generator
  (DiffTime with no constraint: Algorithm 3 for training, Algorithm 4 for sampling).
- DiffTime has **no public code release**. The paper states its architecture is based on
  CSDI, so the code here is CSDI: https://github.com/ermongroup/CSDI, commit `7f24a43`.
- License: `LICENSE` is CSDI's MIT license, unchanged. `README.md` is CSDI's README,
  unchanged (it describes CSDI's own imputation scripts, which are not shipped here).

## Files

| file | origin |
|---|---|
| `diff_models.py` | CSDI, byte-identical |
| `main_model.py` | CSDI, unchanged, plus `import copy` and an appended `CSDI_Unconditional` class (univariate unconditional training loss and DDPM sampler from pure noise) |
| `dataset_npy.py` | new: `(N, T, 1)` array dataset and collate function for `CSDI_Unconditional` |
| `run_difftime.py` | new: training loop, sampling, output writing |
| `config.yaml` | new: DiffTime Table 5 hyperparameters |
| `requirements.txt` | CSDI, unchanged |

`diff_models.py` imports `linear_attention_transformer` at module level, so the package must
be installed, although it is never executed (`is_linear: false`).

## How to run

Environment: Python 3 with PyTorch, numpy, pyyaml and `linear_attention_transformer`
(e.g. `pip install torch numpy pyyaml linear_attention_transformer`). A GPU is expected.

```bash
bash baselines/DiffTime/run.sh                            # 7 datasets x seeds 1 2 3
SEEDS="1" DATASETS="ecg" bash baselines/DiffTime/run.sh    # subset
python -u baselines/DiffTime/run_difftime.py --dataset ecg --seed 1   # one run
```

Each run writes `outputs/DiffTime/seed<N>/<dataset>/DiffTime_<dataset>_generatedata.npy`
(float32 `(N, T, 1)`, original units, N = size of the real dataset) and a `summary.json`
next to it; the trained weights (`model.pth`) and the loss log (`train_metrics.jsonl`) go to
`checkpoints/DiffTime/seed<N>/<dataset>/` (override with `--run-dir`).

## Settings used for the paper

Seeds 1, 2, 3 (`--seed`; `run.seed` in the config is only the default). The seed seeds
Python, NumPy and PyTorch, the data-loader shuffle, and is reset before sampling.

All values are from Table 5 of arXiv:2307.01717:

| setting | value |
|---|---|
| batch size | 16 |
| optimizer | Adam, lr 1e-4, weight decay 1e-6, no lr schedule |
| epochs | 10000 passes over the data = 10000 x ceil(N/16) gradient steps (e.g. 630,000 for ecg) |
| denoiser | 4 residual layers, 64 channels, 8 heads, diffusion embedding 128 |
| noise | 50 steps, quadratic schedule, beta_1 1e-6, beta_T 0.5 |

Normalisation: z-score with one global mean/std over the whole array, inverted before the
samples are written. Training uses the whole dataset (no split). Sampling draws exactly N
series in batches of 64.

## Deviations from the paper / from CSDI

- **Code is CSDI, not DiffTime**: no DiffTime code exists; CSDI is the implementation
  DiffTime says it builds on.
- **Side information kept**: DiffTime removes CSDI's side information (appendix E.4);
  `CSDI_Unconditional` keeps CSDI's time embedding (128) and feature embedding (16), since the
  CSDI model is used unchanged.
- **Convolution kernel size 1** (CSDI's hard-coded value) instead of Table 5's 2, for the same
  reason; the paper also tunes this per dataset and series length.
- **No lr schedule**: CSDI's own training loop decays the lr (MultiStepLR at 75% / 90%);
  DiffTime's Algorithm 3 has a single lr, which is what is used.
- **Normalisation**: not stated in the paper; global z-score as above.
- **Training loop**: CSDI's `train()` / `exe_*.py` are replaced by `run_difftime.py`, which
  implements Algorithm 3 with the Table 5 values; model, loss and sampler come unchanged from
  `main_model.py` / `diff_models.py`.
