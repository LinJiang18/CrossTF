#!/usr/bin/env bash
# E4GEN on the seven datasets, three training seeds.
# Per (dataset, seed): five main.py processes -- backbone -> extreme predictor ->
# validate (tunes the decoding tau) -> ControlNet -> sample with ControlNet -- then
# collect.py copies the generated set to outputs/E4GEN/[seed<N>/]<dataset>/.
#
# Seed 1 is run WITHOUT --seed (run name <Cfg>_E4GEN, flat output layout), exactly as in
# the reported runs; it is therefore not bit-reproducible. Seeds 2 and 3 pass --seed N to
# every stage (each stage is a separate process that seeds itself) and use the run name
# <Cfg>_E4GEN_s<N>, which also keeps OUTPUT/ and Checkpoint/ apart.
#
#   bash run.sh                                   # all 7 datasets x seeds 1 2 3
#   DATASETS="ecg traffic" SEEDS="2" bash run.sh  # a subset
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"
SEEDS="${SEEDS:-1 2 3}"

declare -A CFG=( [building]=Building [ecg]=ECG [electricity]=Electricity
                 [precipitation]=precipitation [temperature]=temperature
                 [traffic]=Traffic [transportation]=transportation )

for DS in ${DATASETS}; do
  BASE="${CFG[$DS]}_E4GEN"
  CONFIG="Config/${BASE}.yaml"
  for SEED in ${SEEDS}; do
    if [[ "${SEED}" == "1" ]]; then
      NAME="${BASE}";         SEED_ARGS=();              OUT_SEED=()
    else
      NAME="${BASE}_s${SEED}"; SEED_ARGS=(--seed "${SEED}"); OUT_SEED=(--seed "${SEED}")
    fi
    echo "=== E4GEN ${DS} seed ${SEED} (run ${NAME})"
    for STAGE in "--train" "--train_extreme" "--validate_extreme" "--train_controlnet" \
                 "--sample --use_controlnet"; do
      echo "--- ${STAGE}"
      # shellcheck disable=SC2086
      "${PYTHON}" main.py --name "${NAME}" --config_file "${CONFIG}" "${SEED_ARGS[@]}" ${STAGE}
    done
    "${PYTHON}" collect.py --method E4GEN --dataset "${DS}" --run_name "${NAME}" "${OUT_SEED[@]}"
  done
done
