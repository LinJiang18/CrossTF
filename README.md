# CrossTF

Code, data and generated samples for **CrossTF**, a time-series diffusion model that
trains a time-view and a frequency-view denoiser separately and fuses them, step by
step, inside a cross-view window at the end of sampling. The per-step fusion coefficients
are solved to improve extreme-value fidelity under a constraint on overall fidelity.

This repository contains:

* the CrossTF implementation (training, coefficient solve, sampling);
* the seven datasets used in the paper;
* the coefficients and the generated samples behind the reported CrossTF results;
* the evaluation code for the ten reported metrics;
* minimal runnable implementations of the eleven baselines, adapted to the same data.

## Layout

```
main.py              entry point: --train/--sample  x  --time/--frequency/--crosstf
run_crosstf.sh       full pipeline for one or more datasets
configs/             one config per dataset (both branches + fusion settings)
models/              branch model: 1D EDM diffusion with a delay-embedding (time) or
                     STFT (frequency) 2D U-Net denoiser
crosstf/             cross-view fusion, prefix-cached sampler, coefficient solver
utils/               config/data helpers
evaluation/          the ten metrics (evaluate.py is the entry point)
data/                the seven datasets, (N, T, 1) float32, original units
results/CrossTF/     solved coefficients (alpha.json) and generated samples per dataset
baselines/           the eleven baselines (see baselines/README.md)
```

## Setup

```bash
conda create -n crosstf python=3.10 -y && conda activate crosstf
pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -r requirements.txt
```

Keep the torchaudio version: the frequency view uses `torchaudio.transforms.Spectrogram`,
and the configs' `n_fft`/`hop_length` fix the image size of the frequency U-Net.
Each baseline has its own environment; see its README.

## Data

| name | file | N | T |
|---|---|---|---|
| building | `data/Building_seq_144_sample_1838.npy` | 1838 | 144 |
| ecg | `data/ECG_seq_250_sample_1000.npy` | 1000 | 250 |
| electricity | `data/Electricity_seq_144_sample_1096.npy` | 1096 | 144 |
| precipitation | `data/precipitation_seq_90_sample_1056.npy` | 1056 | 90 |
| temperature | `data/temperature_seq_144_sample_1520.npy` | 1520 | 144 |
| traffic | `data/Traffic_seq_144_sample_1790.npy` | 1790 | 144 |
| transportation | `data/transportation_seq_144_sample_1320.npy` | 1320 | 144 |

All arrays are univariate, in original units. Min-max normalisation is applied at load
time. Dataset sources are described in the paper.

## Running CrossTF

```bash
bash run_crosstf.sh ecg          # or no argument for all seven datasets
```

which runs, for dataset `ecg`:

```bash
# 1. train both branches for each training seed (4, 5, 6)
python main.py --config_file configs/ecg.yaml --seed 4 --train --time
python main.py --config_file configs/ecg.yaml --seed 4 --train --frequency
# ... seeds 5 and 6

# 2. solve the per-step fusion coefficients -> logs/CrossTF_ecg/alpha.json
python main.py --config_file configs/ecg.yaml --train --crosstf

# 3. sample with them -> outputs/CrossTF_ecg4/CrossTF_ecg4_generatedata.npy
python main.py --config_file configs/ecg.yaml --seed 4 --sample --crosstf
```

To sample with the released coefficients instead of solving them, add
`--alpha_file results/CrossTF/ecg/alpha.json` to step 3. Single-branch samples are
obtained with `--sample --time` / `--sample --frequency`.

**Settings.** The 18-step deterministic Heun sampler of EDM is used throughout. The
cross-view window is sigma <= 0.35, i.e. the last six steps (step indices 12..17 in
the code, 0 being the highest noise level); outside it the time branch is used alone.
Each training seed's samples are drawn with a fixed sampling seed (4 -> 303, 5 -> 404,
6 -> 505). The coefficient solve (`crosstf/solver.py`) uses the branch pairs of all
three training seeds and sampling seeds 101 and 202:

1. one coefficient shared by the window, minimising a hierarchical peaks-over-threshold
   divergence subject to an energy-distance constraint (SLSQP), accepted only if its gain
   over the time branch exceeds one paired standard error;
2. one coefficient per window step, starting from the stage-1 value, with the same
   objective and constraint plus a smoothness cap sum_k (a_k - a_{k-1})^2 <= 0.2.

One branch trains in roughly 1-3 hours on a single recent GPU; the solve takes 1-3 hours
per dataset.

## Released results

`results/CrossTF/<dataset>/` holds `alpha.json` (the per-step coefficients used for the
paper) and `CrossTF_<dataset><seed>_generatedata.npy` for training seeds 4, 5 and 6,
each with the same shape as the real data. Model checkpoints are not included (about
2.7 GB for the 42 branch models); they are reproduced by step 1 above. GPU
non-determinism means retrained branches and re-solved coefficients will not match the
released ones bit for bit.

## Evaluation

```bash
# the released CrossTF samples and every generated set found under outputs/
python evaluation/evaluate.py --dataset ecg --all --out scores/ecg.json
# specific files; --cheap-only skips the three learned metrics (no GPU needed)
python evaluation/evaluate.py --dataset ecg --gen path/to/generated.npy --cheap-only
```

The ten metrics (all lower is better):

| group | metrics | code |
|---|---|---|
| extreme fidelity | `lev_rate`, `lev_excess`, `jmp_rate`, `jmp_excess` | `evaluation/extreme_quantities.py` |
| whole-series fidelity | `wasserstein`, `acd`, `marginal` | `evaluation/trunk.py` |
| learned fidelity | `discriminative`, `predictive`, `context_fid` | `evaluation/fidelity.py` |

The four extreme quantities score each series on its level (|x - median|) and jump
(|dx|) sides against thresholds fixed at the 95% quantile of the pooled real scores; the
metric is the W1 distance between the generated and real per-series values, divided by
the std of the real values.

## Baselines

See [`baselines/README.md`](baselines/README.md). Every baseline reads `data/` and
writes `outputs/<Method>/[seed<N>/]<dataset>/<Method>_<dataset>_generatedata.npy`, which
`evaluation/evaluate.py --all` picks up.

## Third-party code

* `models/networks.py`: U-Net and preconditioning code from EDM
  (https://github.com/NVlabs/edm, CC BY-NC-SA 4.0); `OneDEDMPrecond` and
  `MultiViewPredictor` are ours.
* `models/`, `utils/`: the delay-embedding / STFT image transforms and the training
  scaffolding are adapted from ImagenTime (https://github.com/azencot-group/ImagenTime).
* `evaluation/ts2vec/`: TS2Vec (https://github.com/zhihanyue/ts2vec, MIT).
* `baselines/`: see each baseline's `UPSTREAM.md` and license file.
