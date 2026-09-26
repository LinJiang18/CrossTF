# Upstream and deviations

- Upstream: https://github.com/mobicom24/RF-Diffusion @ `eb872b0c4543da65424f5598ae40826e76e7edea`
  (2024-06-17), GPL v3 -- `LICENSE` is upstream's, unchanged.
- Vendored (only what the run path imports): `tfdiff/{__init__,params,diffusion,wifi_model,learner,dataset}.py`,
  `complex/{__init__,complex_module}.py`. Everything else upstream (`train.py`,
  `inference.py`, the FMCW/MIMO/EEG models, `complex_layers.py`/`complex_functions.py`,
  plots, images) is not used and not included.
- Ours: `prepare_data.py`, `run_rfdiff.py`, `run.sh`.

## Change to vendored code

All vendored files are byte-identical to upstream except one:

- `tfdiff/diffusion.py`, `SignalDiffusion.degrade_fn`: removed `torch.manual_seed(11)`.
  It reset the global torch RNG on every call (twice per reverse step), so every sampling
  batch started from the same initial noise and the generated set contained many duplicate
  series; it also re-seeded the RNG on every training step. Seeding is done once in
  `run_rfdiff.py`.
  *Reproducibility note:* for the reported results the models were trained with this line
  still in place; it was then removed and generation was re-run from the same final
  checkpoints. A fresh run of this release therefore trains without the per-step reset.

## Deviations from upstream usage

| # | what | why |
|---|---|---|
| 1 | Condition held constant (zeros, `cond_dim = 6`) | All upstream tasks are conditional (`sampling()` takes its batch size from `cond.shape[0]`); there is no unconditional path. The condition only enters as an embedding added to the step embedding, so a constant condition gives `p(x | c0) = p(x)` with the architecture untouched. |
| 2 | Real series fed as complex with zero imaginary part | The whole stack is complex-valued. |
| 3 | `blur_schedule` = 0.01 per step instead of `params_wifi`'s 1e-10 | With 1e-10 the Gaussian blur kernel is flat and normalises to exactly 1 in every bin, i.e. a plain Gaussian DDPM. 0.01 is the value upstream uses in `params_mimo`/`params_eeg`. |
| 4 | Own runner instead of `train.py` | `train.py::train` (single-GPU path) maps task ids to the wrong models (0 -> EEG, 1 -> MIMO); `run_rfdiff.py` builds `tfdiff_WiFi` + `SignalDiffusion` directly and trains with upstream's `tfdiffLearner`. It sets `model.device`, which the learner expects from DistributedDataParallel. |
| 5 | Per-series amplitude restored from the real set | The collator normalises each series by its own mean/std, so generated series have no scale. Each generated series (re-standardised) gets a (mean, std) pair drawn at random from the real per-series pairs (`stats.npz`). |
| 6 | LR schedule `StepLR(step_size = epochs // 4 = 50, gamma = 0.5)` | Upstream's learner uses `StepLR(opt, 1, gamma=0.5)` stepped each epoch (commented `# mimo`), i.e. lr = 1e-3 * 0.5^epoch, dead after ~15 epochs. Replaced at runtime from the runner; `learner.py` is unchanged. |
| 7 | Sampler: `robust_sampling` from pure noise | Upstream `inference.py` uses `native_sampling`, which degrades real data and restores in one step (reconstruction, not generation). |
| 8 | All-zero series excluded from training | The collator's per-sample std would be 0 (NaN loss); precipitation has one such series. |

Upstream observations (not changed): `learner.py` defines `tfdiffLoss` (time + frequency
MSE) but never uses it -- training uses `nn.MSELoss()` as upstream does. Pre-trained
upstream weights (Wi-Fi/FMCW/MIMO/EEG) are not used; every dataset is trained from scratch.

## Settings used for the paper

From `params_wifi` except where noted above.

| | |
|---|---|
| model | `tfdiff_WiFi` (complex DiT), 19.4M parameters |
| embed_dim / hidden_dim / num_block / num_heads | 256 / 128 / 32 / 8 |
| optimizer | AdamW, lr 1e-3, StepLR(50 epochs, 0.5) |
| batch size | 32 (drop_last, as upstream's dataloader) |
| training | 200 epochs (`max_iter = 200 * floor(N_train / 32)`) |
| diffusion | 100 steps, `noise_schedule = linspace(1e-4, 0.003, 100)`, `blur_schedule = 0.01` |
| sampling | `robust_sampling`, batch 32, N series |
| seed | 1 (single seed) |
