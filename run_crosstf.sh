#!/usr/bin/env bash
# Full CrossTF pipeline for one or more datasets:
#   1. train the time and the frequency branch for every seed in the config's train.seeds
#   2. solve the per-step fusion coefficients (logs/CrossTF_<name>/alpha.json)
#   3. sample with them for every seed (outputs/CrossTF_<name><seed>/)
#
#   bash run_crosstf.sh                    # all seven datasets
#   bash run_crosstf.sh ecg electricity    # a subset
#   SEEDS="4" STAGES="crosstf" bash run_crosstf.sh ecg
#
# STAGES: any subset of "time frequency solve sample" (default: all four).
# A branch whose checkpoint already exists is not retrained.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(building ecg electricity precipitation temperature traffic transportation)
STAGES="${STAGES:-time frequency solve sample}"
has() { [[ " ${STAGES} " == *" $1 "* ]]; }

for NAME in "${DATASETS[@]}"; do
  CFG="configs/${NAME}.yaml"
  read -r -a SEEDS <<< "${SEEDS:-4 5 6}"
  for VIEW in time frequency; do
    has "${VIEW}" || continue
    LABEL=$([[ ${VIEW} == time ]] && echo Time || echo Frequency)
    for SEED in "${SEEDS[@]}"; do
      if compgen -G "logs/${LABEL}_${NAME}${SEED}/pred=${VIEW}-*" > /dev/null; then
        echo "[${NAME}] ${LABEL} seed ${SEED}: checkpoint exists, skipping training"
      else
        "${PYTHON}" main.py --config_file "${CFG}" --name "${NAME}" --seed "${SEED}" --train --"${VIEW}"
      fi
    done
  done
  if has solve; then
    "${PYTHON}" main.py --config_file "${CFG}" --name "${NAME}" --train --crosstf
  fi
  if has sample; then
    for SEED in "${SEEDS[@]}"; do
      "${PYTHON}" main.py --config_file "${CFG}" --name "${NAME}" --seed "${SEED}" --sample --crosstf
    done
  fi
done
