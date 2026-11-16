# Reproducing Table 1

This folder reproduces Table 1 (extreme-sensitive cross-view asymmetry) from the released
time and frequency branches. All numbers are recomputed on one GPU from the checkpoints in
`rebuttal/pretrained/`.

## Contents

| Path | What it is |
|---|---|
| `reproduce_Table1.py` | recomputes every number of Table 1 and prints the table |
| `pretrained/{Time,Frequency}_<dataset><seed>/` | time and frequency branches of all seven datasets, training seeds 4 and 5 (28 checkpoints) |

The script also uses files that are already in the repository: the data (`data/`), the
dataset configs (`configs/`), the code (`crosstf/`, `evaluation/`, `models/`, `utils/`)
and our solved coefficient alpha* (`results/CrossTF/<dataset>/alpha.json`).

## Commands

Set up the environment as in the main [README](../README.md#setup), then run from the
repository root:

```bash
# Table 1, all seven datasets (one GPU, about 15 minutes)
python rebuttal/reproduce_Table1.py

# a subset of the datasets
python rebuttal/reproduce_Table1.py --datasets precipitation traffic
```

Per-dataset results are cached in `rebuttal/outputs/Table1/<dataset>.json`. A second run
prints the table from the cache; `--force` recomputes everything.

## Expected output

```
Table 1: Extreme-sensitive cross-view asymmetry (alpha in [0, 1])
Relative max-min range: Loss, Overall, Extreme; Ratio: Ext./Loss, Ext./Overall
-----------------------------------------------------------------------
Dataset             Loss   Overall   Extreme |  Ext./Loss  Ext./Overall
-----------------------------------------------------------------------
Traffic             4.5%      8.4%    290.4% |     65.22x        34.44x
Transportation      7.5%     51.1%    254.2% |     34.11x         4.97x
Temperature         7.3%     12.6%    145.6% |     19.91x        11.52x
Electricity         8.7%     78.4%     85.2% |      9.82x         1.09x
Precipitation      33.3%    339.9%    160.3% |      4.81x         0.47x
ECG                16.7%      3.9%     38.2% |      2.29x         9.93x
Building           12.9%     10.8%     22.8% |      1.77x         2.10x
-----------------------------------------------------------------------
Median              8.7%     12.6%    145.6% |     16.74x        11.56x
-----------------------------------------------------------------------
```

Sampling is deterministic (fixed noise, sampling seed 101), so a rerun gives the same
numbers; on a different GPU type the last digit can differ through floating-point
arithmetic.

## What is measured

One global fusion coefficient alpha is applied at every step of the fusion window
(sigma <= 0.35, the last 6 of the 18 sampling steps); before the window the time branch
runs alone. alpha = 0 is the frequency branch, alpha = 1 the time branch (Time-Only).
Each quantity q is summarised by its relative range over alpha in {0, 0.25, 0.5, 0.75,
0.9, 1}: (max q - min q) / |q(alpha = 1)|.

- **Loss**: one-step denoising loss of the fused denoiser on the real data.
- **Extreme**: our four extreme quantities (Mag-Rate, Mag-Excess, Chg-Rate, Chg-Excess),
  each the W1 distance to the real data.
- **Overall**: Wasserstein distance to the real data.

Each range is averaged over the two training seeds; see the docstring of
`reproduce_Table1.py` for the exact definitions.
