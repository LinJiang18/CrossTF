#!/usr/bin/env bash
# FIDE (repaired GEV regulariser) on the seven datasets, seed 10, 2000 epochs.
# Output: <repo>/outputs/FIDE/<dataset>/FIDE_<dataset>_generatedata.npy
#
# Usage:  bash baselines/FIDE/run.sh                    # all seven datasets
#         DATASETS="ecg traffic" bash baselines/FIDE/run.sh
#         PYTHON=/path/to/python bash baselines/FIDE/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"
SEED="${SEED:-10}"

for DS in ${DATASETS}; do
  echo "=========== FIDE ${DS} (seed ${SEED})"
  "${PYTHON}" -u "${HERE}/run_fide.py" --dataset "${DS}" --seed "${SEED}"
done
