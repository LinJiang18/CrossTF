# TimeGAN baseline

**Time-series Generative Adversarial Networks** (Yoon, Jarrett, van der Schaar,
NeurIPS 2019). The model that runs is the TimeGAN port in the pip package
[tsgm](https://github.com/AlexanderVNikitin/tsgm) (`tsgm.models.timeGAN.TimeGAN`,
version 0.1.0); no model code is vendored here. See `UPSTREAM.md` for provenance and
deviations.

## Environment

```bash
pip install -r baselines/TimeGAN/requirements.txt      # plus a CUDA build of torch
python baselines/TimeGAN/patch_tsgm_torch.py           # once per environment
```

`patch_tsgm_torch.py` edits the installed `tsgm/models/timeGAN.py` so that TimeGAN uses
the GPU under the Keras torch backend (tsgm 0.1.0 forces the CPU) and fixes two crashes on
that path; see its docstring. Re-run it after reinstalling tsgm; `--check` reports status.

## Run

```bash
bash baselines/TimeGAN/run.sh                          # all seven datasets
```

which runs, for each dataset (building, ecg, electricity, precipitation, temperature,
traffic, transportation),

```bash
KERAS_BACKEND=torch python baselines/TimeGAN/run_timegan.py --dataset <dataset> --iterations 10000 --seed 10
```

- Output: `outputs/TimeGAN/<dataset>/TimeGAN_<dataset>_generatedata.npy`, float32,
  `(N, T, 1)`, original units, N = size of the real dataset, plus `summary.json`.
- Work files: `baselines/TimeGAN/work/<dataset>/` (override with `TIMEGAN_WORK`) holds the
  run log and `train_metrics.jsonl` (tsgm's per-epoch losses).
- One GPU per run; 10000 iterations per phase took roughly 2-6 h per dataset on one GPU.

**Check the losses.** tsgm catches exceptions inside its training steps, prints
`Warning: Skipping ... batch due to error` and returns a zero loss, so a broken run can
look healthy. `run.sh` fails if the log contains `Skipping`; also check that the losses in
`train_metrics.jsonl` are not pinned at 0.0.
