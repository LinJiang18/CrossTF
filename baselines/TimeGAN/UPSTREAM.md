# Upstream and deviations

- Paper code: https://github.com/jsyoon0823/TimeGAN (TensorFlow 1.x). Not used.
- Implementation used: tsgm, https://github.com/AlexanderVNikitin/tsgm, installed from
  PyPI as `tsgm==0.1.0`; model class `tsgm.models.timeGAN.TimeGAN`. No tsgm code is
  vendored, so no upstream license file is included here (tsgm is Apache-2.0).
- Ours: `run_timegan.py` (data loading, scaling, budget, output), `config.yaml`,
  `patch_tsgm_torch.py`, `run.sh`.

## Deviations

| what | why |
|---|---|
| tsgm's TimeGAN port instead of the paper's code | The original is TensorFlow 1.x and does not run on current GPUs / CUDA 12. Results describe tsgm's TimeGAN. |
| 10000 iterations per phase instead of 50000 | Compute budget (upstream's 50000 per phase is several days per dataset). tsgm's `fit()` has no default length. |
| Budget in iterations, converted per dataset to `epochs = ceil(10000 / ceil(N / 128))` | Every dataset gets the same number of updates regardless of N. `summary.json` records epochs and the actual iteration count. |
| batch_size 128 | Upstream's value; tsgm's default is 256. |
| Keras torch backend + `patch_tsgm_torch.py` | tsgm 0.1.0 pins TimeGAN to the CPU on the torch backend and crashes logging losses / in `generate()` on CUDA. The patch changes device placement and two numpy conversions only; model, losses and training schedule are untouched. |
| Seed 10 | Upstream sets none; python / numpy / backend are seeded via `keras.utils.set_random_seed`. |
| Min-max scaling to [0, 1] over the whole array, inverted after sampling | As in the paper's code (global per-channel min/max rather than per-sequence). |
| No train/test split | Trains on the whole array and samples N series, as for the other baselines. |

## Settings used for the paper

| | |
|---|---|
| hidden_dim / n_layers / module / gamma | 24 / 3 / gru / 1.0 (upstream defaults; tsgm agrees) |
| batch_size | 128 |
| optimizers | Adam, lr 1e-3, all five tsgm optimizers |
| losses | embedding MSE, classification binary cross-entropy |
| training | 10000 iterations per phase (e.g. building: 667 epochs x 15 steps = 10005) |
| sampling | `model.generate(N)` |
| seed | 10 (single seed) |
| software | tsgm 0.1.0, keras 3.15.1 (torch backend), torch 2.11.0+cu128, numpy 2.4.6, Python 3.11 |
