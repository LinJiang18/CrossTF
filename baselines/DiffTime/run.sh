#!/usr/bin/env bash
# Train DiffTime and sample one full-size synthetic set for every dataset and seed.
#
#   bash baselines/DiffTime/run.sh
#   SEEDS="1" DATASETS="ecg" bash baselines/DiffTime/run.sh      # a subset
#
# Each run writes
#   outputs/DiffTime/seed<N>/<dataset>/DiffTime_<dataset>_generatedata.npy  (+ summary.json)
# and keeps the trained weights and loss log under checkpoints/DiffTime/seed<N>/<dataset>/.
# Existing generated sets are skipped, so an interrupted sweep can simply be restarted.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-1 2 3}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"

for SEED in ${SEEDS}; do
  for DS in ${DATASETS}; do
    OUT="${REPO}/outputs/DiffTime/seed${SEED}/${DS}/DiffTime_${DS}_generatedata.npy"
    if [[ -f "${OUT}" ]]; then
      echo "skip ${DS} seed ${SEED}: ${OUT} exists"
      continue
    fi
    echo "=== DiffTime  dataset=${DS}  seed=${SEED}"
    "${PYTHON}" -u "${HERE}/run_difftime.py" --dataset "${DS}" --seed "${SEED}"
  done
done
