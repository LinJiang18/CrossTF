#!/usr/bin/env bash
# FourierDiffusion (frequency-domain model) on the seven datasets: train 1000 epochs,
# sample N series (N = real dataset size), copy the result to
# <repo>/outputs/FourierDiffusion/<dataset>/FourierDiffusion_<dataset>_generatedata.npy.
# Seed 42 (cmd/conf/train.yaml, cmd/conf/sample.yaml).
#
# Checkpoints and Hydra logs go to baselines/FourierDiffusion/runs/ (safe to delete
# after collect.py has run).
#
# Usage:  bash baselines/FourierDiffusion/run.sh
#         DATASETS="ecg traffic" bash baselines/FourierDiffusion/run.sh
#         PYTHON=/path/to/python bash baselines/FourierDiffusion/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"
EPOCHS=1000   # upstream default is 200; all reported results use 1000

cd "${HERE}"
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:${PYTHONPATH}}"

for DS in ${DATASETS}; do
  NAME="FourierDiffusion_${DS}_e${EPOCHS}"
  echo "=========== FourierDiffusion ${DS}: train"
  "${PYTHON}" -u cmd/train.py config="${DS}" project_name="${NAME}" \
    fourier_transform=true trainer.max_epochs="${EPOCHS}"
  echo "=========== FourierDiffusion ${DS}: sample"
  "${PYTHON}" -u cmd/sample.py config="${DS}" project_name="${NAME}" fourier_transform=true
  "${PYTHON}" -u collect.py --dataset "${DS}" --run-name "${NAME}" --max-epochs "${EPOCHS}"
done
