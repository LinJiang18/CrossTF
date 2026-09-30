<h1 align="center">CrossTF</h1>

<p align="center">
  <b>Cross-View Time-Frequency Diffusion for Extreme-Aware Time-Series Generation</b>
</p>

<p align="center">
  Anonymous authors &nbsp;·&nbsp; Paper under double-blind review
</p>

<p align="center">
  <img src="assets/overview.png" width="100%" alt="CrossTF sampling step: the intermediate sample is lifted into a 2D time view (delay embedding) and a 2D frequency view (STFT), each denoised by its own network, and the two clean-signal estimates are fused along the cross-view direction.">
</p>

**CrossTF** is a time-series diffusion model that trains a time-view and a frequency-view
denoiser separately and fuses them, step by step, inside a cross-view window at the end of
sampling. At each step the noisy series is lifted into a 2D time view (delay embedding) and
a 2D frequency view (STFT); each view's denoiser gives a clean-signal estimate, and the two
are combined along the cross-view direction,
$\hat{x}_{0,i} = \hat{x}^F_{0,i} + \alpha_i\,(\hat{x}^T_{0,i} - \hat{x}^F_{0,i})$.
The per-step coefficients $\alpha_i$ are solved to improve extreme-value fidelity under a
constraint on overall fidelity.

## Main results

Summary of comparisons on seven datasets. *Best* denotes the strongest baseline. *W–L*
counts wins–losses against 11 baselines across 10 metrics, yielding 110 pairwise
comparisons per dataset. Lower is better.

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Dataset</th>
      <th colspan="4">Extreme Fidelity</th>
      <th colspan="4">Overall Performance</th>
      <th rowspan="2">W–L</th>
    </tr>
    <tr>
      <th>Best</th><th>Ours</th><th>Baseline</th><th>Improve</th>
      <th>Best</th><th>Ours</th><th>Baseline</th><th>Improve</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Traffic</td>
      <td>ImagenTime</td><td align="right"><b>0.0343</b></td><td align="right">0.0591</td><td align="right"><b>+42.0%</b></td>
      <td>FourierDiff</td><td align="right"><b>0.0083</b></td><td align="right">0.0138</td><td align="right"><b>+39.9%</b></td>
      <td align="center">110–0</td></tr>
    <tr><td>Precipitation</td>
      <td>ImagenTime</td><td align="right"><b>0.0755</b></td><td align="right">0.1086</td><td align="right"><b>+30.5%</b></td>
      <td>ImagenTime</td><td align="right"><b>0.0050</b></td><td align="right">0.0051</td><td align="right"><b>+2.0%</b></td>
      <td align="center">103–7</td></tr>
    <tr><td>Building</td>
      <td>ImagenTime</td><td align="right"><b>0.0762</b></td><td align="right">0.1012</td><td align="right"><b>+24.7%</b></td>
      <td>ImagenTime</td><td align="right"><b>0.0040</b></td><td align="right">0.0046</td><td align="right"><b>+13.0%</b></td>
      <td align="center">102–8</td></tr>
    <tr><td>Temperature</td>
      <td>ImagenTime</td><td align="right"><b>0.0884</b></td><td align="right">0.1222</td><td align="right"><b>+27.7%</b></td>
      <td>FIDE</td><td align="right"><b>0.0131</b></td><td align="right">0.0165</td><td align="right"><b>+20.6%</b></td>
      <td align="center">101–9</td></tr>
    <tr><td>Electricity</td>
      <td>ImagenTime</td><td align="right"><b>0.0691</b></td><td align="right">0.0734</td><td align="right"><b>+5.9%</b></td>
      <td>ImagenTime</td><td align="right">0.0220</td><td align="right"><b>0.0193</b></td><td align="right">−14.0%</td>
      <td align="center">102–8</td></tr>
    <tr><td>Transportation</td>
      <td>ImagenTime</td><td align="right"><b>0.0477</b></td><td align="right">0.0691</td><td align="right"><b>+31.0%</b></td>
      <td>ImagenTime</td><td align="right">0.0120</td><td align="right"><b>0.0114</b></td><td align="right">−5.3%</td>
      <td align="center">107–3</td></tr>
    <tr><td>ECG</td>
      <td>ImagenTime</td><td align="right"><b>0.1421</b></td><td align="right">0.1807</td><td align="right"><b>+21.4%</b></td>
      <td>ImagenTime</td><td align="right">0.0201</td><td align="right"><b>0.0193</b></td><td align="right">−4.1%</td>
      <td align="center">105–5</td></tr>
  </tbody>
  <tfoot>
    <tr><td><b>Summary</b></td>
      <td align="center">–</td><td align="center">–</td><td align="center">–</td><td align="right"><b>+26.1%</b></td>
      <td align="center">–</td><td align="center">–</td><td align="center">–</td><td align="right"><b>+7.4%</b></td>
      <td align="center"><b>730–40</b></td></tr>
  </tfoot>
</table>

## What is in this repository

* the CrossTF implementation (training, coefficient solve, sampling);
* the seven datasets used in the paper;
* the coefficients and the generated samples behind the reported CrossTF results, and the
  trained Precipitation branches as a sample checkpoint;
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
evaluation/          metrics; make_table.py reproduces the result tables
data/                the seven datasets, (N, T, 1) float32, original units
results/             generated sets behind the reported numbers (CrossTF + baselines)
pretrained/          the six Precipitation branch checkpoints behind the released results
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

These commands set up CrossTF only. Each baseline runs in its own environment, defined in
its directory (see [`baselines/README.md`](baselines/README.md)).

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

Two ways to run the pipeline end to end on Precipitation, the fastest dataset: a quick
check that starts from the released branches, and a run from scratch. They show that the
code works; the reported numbers are reproduced from the released samples (see
[Evaluation](#evaluation)).

### Quick check: released branches

`pretrained/` holds the six Precipitation branches behind the released results (time and
frequency view, training seeds 4, 5, 6), so no training is needed.

```bash
# 1. solve the per-step fusion coefficients    -> pretrained/CrossTF_precipitation/alpha.json
python main.py --config_file configs/precipitation.yaml --train --crosstf --log_dir pretrained

# 2. sample with them                          -> outputs/CrossTF_precipitation4/
python main.py --config_file configs/precipitation.yaml --seed 4 --sample --crosstf --log_dir pretrained
```

### From scratch

```bash
# 1. train the time branch                     -> logs/Time_precipitation1/
python main.py --config_file configs/precipitation.yaml --seed 1 --train --time

# 2. train the frequency branch                -> logs/Frequency_precipitation1/
python main.py --config_file configs/precipitation.yaml --seed 1 --train --frequency

# 3. solve the per-step fusion coefficients    -> logs/CrossTF_precipitation/alpha.json
python main.py --config_file configs/precipitation.yaml --train --crosstf --solve_seeds 1

# 4. sample with them                          -> outputs/CrossTF_precipitation1/
python main.py --config_file configs/precipitation.yaml --seed 1 --sample --crosstf
```

`--solve_seeds 1` solves on this single branch pair; the paper solves on three (seeds 4, 5,
6). Any other dataset runs the same way with its config. Single-branch samples are obtained
with `--sample --time` / `--sample --frequency`.

### Retraining the paper's models

`bash run_crosstf.sh [dataset ...]` (all seven datasets by default) trains both branches
for training seeds 4, 5 and 6, solves the coefficients on all three pairs and samples each
seed. Branches whose checkpoint already exists are not retrained.

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

## Released results

`results/` holds the generated sets behind the reported numbers, one file per method,
dataset and training seed, each with the same shape as the real data:

```
results/CrossTF/<dataset>/alpha.json                          solved per-step coefficients
results/CrossTF/<dataset>/CrossTF_<dataset><seed>_generatedata.npy        seeds 4, 5, 6
results/<Baseline>/[seed<N>/]<dataset>/<Baseline>_<dataset>_generatedata.npy
```

Baselines with three training seeds (DiffTime, Diffusion-TS, E4GEN, ImagenTime) have
`seed1/`..`seed3/`; the others were run once.

Because of space limits, `pretrained/` includes a sample of CrossTF's 42 branch models
(about 2.6 GB): the six Precipitation ones. All checkpoints needed for reproduction will be
released upon acceptance; until then the others are reproduced by `run_crosstf.sh`. GPU
non-determinism means retrained models will not match the released samples bit for bit.

## Evaluation

The per-dataset result tables (every method, seven metrics, mean over training seeds,
best in bold and second best underlined) are reproduced from the released samples by

```bash
python evaluation/make_table.py --out tables/          # all seven datasets, CPU only
python evaluation/make_table.py --root outputs         # the same for regenerated samples
```

which prints one Markdown table per dataset and writes `tables/<dataset>.{md,tex}` and
`tables/scores.json` (including per-seed values).

| column | metric | code |
|---|---|---|
| Mag-Rate, Mag-Excess | `lev_rate`, `lev_excess` | `evaluation/extreme_quantities.py` |
| Chg-Rate, Chg-Excess | `jmp_rate`, `jmp_excess` | `evaluation/extreme_quantities.py` |
| Wass., Marg., ACD | `wasserstein`, `marginal`, `acd` | `evaluation/trunk.py` |

All are lower-is-better. The extreme quantities score each series on its level
(|x - median|, "Mag") and change (|dx|, "Chg") sides against thresholds fixed at the 95%
quantile of the pooled real scores: the rate is how often a series exceeds the threshold,
the excess how far it goes past it. Each metric is the W1 distance between the generated
and real per-series values, divided by the std of the real values.

`evaluation/evaluate.py` scores individual files and additionally computes three learned
metrics (discriminative, predictive, Context-FID; GPU required):

```bash
python evaluation/evaluate.py --dataset ecg --all                  # everything in results/ and outputs/
python evaluation/evaluate.py --dataset ecg --gen my_samples.npy --cheap-only
```

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
