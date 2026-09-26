# RF-Diffusion baseline

**RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion** (Chi et al.,
MobiCom 2024), run unconditionally on the seven univariate datasets. The model code in
`tfdiff/` and `complex/` is vendored from upstream (see `UPSTREAM.md` for the source
commit, the one-line change to `tfdiff/diffusion.py`, and all deviations).

**License.** The vendored upstream code is GPL v3 (`LICENSE`); the GPL applies to this
directory (`baselines/RF-Diffusion/`).

## Run

```bash
pip install -r baselines/RF-Diffusion/requirements.txt   # plus a CUDA build of torch
bash baselines/RF-Diffusion/run.sh                        # all seven datasets
```

which is equivalent to

```bash
cd baselines/RF-Diffusion
python prepare_data.py                                   # npy -> per-series .mat, once
python run_rfdiff.py --dataset <dataset> --epochs 200 --batch-size 32 --seed 1
```

for each `<dataset>` in building, ecg, electricity, precipitation, temperature, traffic,
transportation. One GPU per run; the model has 19.4M parameters.

- Output: `outputs/RF-Diffusion/<dataset>/RF-Diffusion_<dataset>_generatedata.npy`,
  float32, `(N, T, 1)`, original units, N = size of the real dataset, plus a
  `summary.json` with the run settings.
- Work files: `baselines/RF-Diffusion/work/` (override with `RFDIFF_WORK`) holds
  `dataset/<ds>/raw/user_*.mat` + `stats.npz`, `model/<ds>/weights*.pt` (one checkpoint per
  epoch, as upstream's learner writes them) and `log/<ds>/` (tensorboard). Training resumes
  from `model/<ds>/weights.pt` if present -- delete it for a fresh run.

## Shape contract

    sample_rate = T        (90 / 144 / 250, per dataset)
    input_dim   = 1
    extra_dim   = [1]
    cond_dim    = 6        (constant zeros)
    data        = (B, T, 1, 2)     ->  degrade_fn  ->  model  ->  (B, T, 1, 2)

`input_dim` is read by two consumers on different axes: `PositionEmbedding` projects the
last complex axis with `ComplexLinear(input_dim, hidden)`, while `SignalDiffusion` builds
its blur kernel along axis 1 of length `input_dim`. With `input_dim = 1` the kernel is a
single bin (see UPSTREAM.md, deviation 3).
