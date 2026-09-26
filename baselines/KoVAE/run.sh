#!/usr/bin/env bash
# KoVAE on the seven datasets, one run each (seed 10, upstream's default).
#
# Usage
#   bash baselines/KoVAE/run.sh
#   DATASETS="ecg building" bash baselines/KoVAE/run.sh
#   PYTHON=/path/to/python KOVAE_WORK=/scratch/kovae bash baselines/KoVAE/run.sh
#
# Output: outputs/KoVAE/<dataset>/KoVAE_<dataset>_generatedata.npy
# Training logs: $KOVAE_WORK (default baselines/KoVAE/work/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"

for DS in ${DATASETS}; do
  echo "=== KoVAE | ${DS}"
  "${PYTHON}" -u "${HERE}/run_kovae.py" --dataset "${DS}"
done
