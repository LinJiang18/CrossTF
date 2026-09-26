#!/usr/bin/env bash
# Train ImagenTime and sample one full-size synthetic set for every dataset and seed.
#
#   bash baselines/ImagenTime/run.sh
#   SEEDS="1" DATASETS="ecg" bash baselines/ImagenTime/run.sh     # a subset
#
# Each run trains for 200 epochs and then writes
#   outputs/ImagenTime/seed<N>/<dataset>/ImagenTime_<dataset>_generatedata.npy
# The single checkpoint of a run (overwritten every epoch) goes to
#   ${CKPT_ROOT}/seed<N>/<dataset>/
# Existing generated sets are skipped, so an interrupted sweep can simply be restarted.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-1 2 3}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"
CKPT_ROOT="${CKPT_ROOT:-${REPO}/checkpoints/ImagenTime}"

cd "${HERE}"
for SEED in ${SEEDS}; do
  for DS in ${DATASETS}; do
    OUT="${REPO}/outputs/ImagenTime/seed${SEED}/${DS}/ImagenTime_${DS}_generatedata.npy"
    if [[ -f "${OUT}" ]]; then
      echo "skip ${DS} seed ${SEED}: ${OUT} exists"
      continue
    fi
    echo "=== ImagenTime  dataset=${DS}  seed=${SEED}"
    CKPT="${CKPT_ROOT}/seed${SEED}/${DS}"
    mkdir -p "${CKPT}"
    "${PYTHON}" -u run_unconditional.py --config "configs/${DS}.yaml" --seed "${SEED}" --log_dir "${CKPT}"
  done
done
