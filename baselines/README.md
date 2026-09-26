# Baselines

Minimal implementations of the eleven baselines, each adapted to read the seven datasets
in `../data/` and to write its generated set to

    outputs/<Method>/[seed<N>/]<dataset>/<Method>_<dataset>_generatedata.npy

(float32, same shape as the real data, original units), which
`python evaluation/evaluate.py --dataset <dataset> --all` scores. Paths are shared through
`common.py`. Each directory contains only the code on the run path, a `run.sh` launcher
covering all seven datasets and the seeds used in the paper, and an `UPSTREAM.md` with the
upstream source, our changes, and the exact settings. Every baseline has its own Python
environment (see its `requirements*` file); all need one CUDA GPU.

| Method | Directory | Upstream | License | Seeds in the paper | Launcher |
|---|---|---|---|---|---|
| DiffTime | `DiffTime/` | CSDI code (ermongroup/CSDI) with DiffTime's settings | MIT | 1, 2, 3 | `bash DiffTime/run.sh` |
| Diffusion-TS | `Diffusion-TS/` | Y-debug-sys/Diffusion-TS | MIT | 12345, 2, 3 | `bash Diffusion-TS/run.sh` |
| E4GEN | `E4GEN/` | arXiv:2606.01634 (Diffusion-TS backbone) | MIT | 1, 2, 3 | `bash E4GEN/run.sh` |
| HeavyDiff | `E4GEN/` (`--heavy`) | Student-t noise variant of the same code | MIT | 1 run | `bash E4GEN/run_heavydiff.sh` |
| FIDE | `FIDE/` | galib19/FIDE | none upstream | 10 | `bash FIDE/run.sh` |
| FourierDiffusion | `FourierDiffusion/` | JonathanCrabbe/FourierDiffusion | MIT | 42 | `bash FourierDiffusion/run.sh` |
| ImagenTime | `ImagenTime/` | azencot-group/ImagenTime | none upstream | 1, 2, 3 | `bash ImagenTime/run.sh` |
| KoVAE | `KoVAE/` | azencot-group/KoVAE | none upstream | 10 | `bash KoVAE/run.sh` |
| RF-Diffusion | `RF-Diffusion/` | mobicom24/RF-Diffusion | GPL-3.0 (this directory) | 1 | `bash RF-Diffusion/run.sh` |
| SDForger | `SDForger/` | IBM/fms-dgt (time_series databuilder) | Apache-2.0 | 42 | `bash SDForger/run.sh` |
| TimeGAN | `TimeGAN/` | tsgm package (pip) | Apache-2.0 (tsgm) | 10 | `bash TimeGAN/run.sh` |

Notes:

* SDForger de-duplicates and filters its samples, so it returns slightly more series
  than the real data (100-103%); the evaluator draws a random subset of the real size.
* TimeGAN needs a one-off patch of the installed tsgm package
  (`python TimeGAN/patch_tsgm_torch.py`) before the first run.
* Intermediate files (checkpoints, prepared data) go to per-baseline work directories
  documented in each README/UPSTREAM.md; they are git-ignored.
