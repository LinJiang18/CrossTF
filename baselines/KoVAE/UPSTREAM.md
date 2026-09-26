# KoVAE (baseline)

Naiman, Erichson, Ren, Mahoney, Azencot, *Generative Modeling of Regular and Irregular
Time Series Data via Koopman VAEs*, ICLR 2024
([arXiv:2310.02619](https://arxiv.org/abs/2310.02619)).

- Upstream: https://github.com/azencot-group/KoVAE
- Upstream commit: `bf8cc95` (the vendored files match it apart from the changes listed
  below)
- License: the upstream repository contains **no LICENSE file**, so none is included here.
  The vendored files remain the work of their authors; please cite the paper if you use
  them.

```bibtex
@inproceedings{naiman2024generative,
  title={Generative Modeling of Regular and Irregular Time Series Data via Koopman {VAE}s},
  author={Ilan Naiman and N. Benjamin Erichson and Pu Ren and Michael W. Mahoney and Omri Azencot},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=eY7sLb0dVF}
}
```

## What is here

| path | origin |
|---|---|
| `models/losses.py`, `models/modules.py`, `models/neuralCDE.py`, `controldiffeq/*` | upstream, unchanged |
| `models/kovae.py` | upstream; hardcoded `.cuda()` replaced by the model's device |
| `utils/utils.py` | upstream; `import tensorflow` moved inside `set_seed_device` |
| `utils/utils_data.py` | upstream plus two added functions, `extreme_npy_loading` and `inverse_MinMax_NTC` |
| `run_kovae.py`, `config.yaml`, `run.sh`, `requirements.txt` | ours |

Only files on the import path of `run_kovae.py` are included (no `run_regular.py`,
`run_irregular.py`, `TorchDiffEqPack/`, `metrics/`, `utils/utils_model.py`, datasets or
figures).

## Running

```bash
pip install -r baselines/KoVAE/requirements.txt
bash baselines/KoVAE/run.sh                                  # all 7 datasets
python baselines/KoVAE/run_kovae.py --dataset electricity    # one dataset
```

Output: `outputs/KoVAE/<dataset>/KoVAE_<dataset>_generatedata.npy` (float32, `(N, T, 1)`,
original units, N = number of real series) plus a small `summary.json`. Per-epoch losses
go to `baselines/KoVAE/work/<dataset>/train_metrics.jsonl` (override with `--work-dir` or
`KOVAE_WORK`). No checkpoints are saved. One GPU (or CPU) per run.

## Hyperparameters and seed used for the paper

All seven datasets use upstream `run_regular.py`'s argparse defaults (`config.yaml`):
600 epochs, Adam lr 7e-4, weight decay 0, batch 64, `num_layers 3`, `z_dim 16`,
`hidden_dim 20`, `num_steps 1`, loss weights `w_rec 1.0 / w_kl 0.007 / w_pred_prior 0.005`,
batch norm on, `pinv_solver false`, regular sampling (`missing_value 0`), DataLoader
`num_workers 4`. `seq_len` and `inp_dim` are taken from the data.

**One run per dataset, seed 10** (upstream's default), hence no `seed<N>/` level in the
output path.

## Deviations from upstream

- **Data**: upstream loads sine/stock/energy/mujoco; we read our `(N, T, C)` `.npy` files with `extreme_npy_loading` (per-channel min-max to `[0, 1]` over the whole array) and train on all N series, no held-out split.
- **Output in original units**: generated samples are mapped back with `inverse_MinMax_NTC` and written to `outputs/KoVAE/`; upstream only keeps them in memory for its metric.
- **Exactly N samples**: as upstream, one prior sample batch per training batch after re-seeding, which gives exactly N.
- **No post-hoc discriminative score**: upstream trains an RNN discriminator 10x after every run; skipped, since generated sets are scored by the shared evaluation code.
- **tensorflow / neptune removed from the run path**: upstream seeds TF (`tf.keras.utils.set_random_seed`) and creates log dirs with `tf.io.gfile`; TF touches no computation, so the runner seeds torch/numpy exactly as upstream does and skips TF. In `utils/utils.py` the module-level `import tensorflow` is moved into `set_seed_device` (the only function using it, never called here); this does not change any computation.
- **Device handling**: `models/kovae.py` allocates its sampling buffers on the model's device instead of `.cuda()`, so the model also runs on CPU; identical on GPU.
- **Environment**: Python 3.10, torch 2.8 instead of upstream's pinned Python 3.9 / torch 2.0 / tensorflow 2.12 (`environment.yml` not included).
