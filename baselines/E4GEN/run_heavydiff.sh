#!/usr/bin/env bash
# HeavyDiff on the seven datasets: the E4GEN/Diffusion-TS backbone trained and sampled
# with Student-t noise (--heavy; heavy_nu = 2.5, heavy_var_correction = True in the
# configs). Two stages only (no extreme predictor, no ControlNet), base configs
# Config/<Cfg>.yaml (not *_E4GEN.yaml).
#
# var_loss_weight is fixed to 3 on the command line for every dataset: the base configs
# ship 3 except precipitation/temperature, which ship -1 (the term is off for any value
# <= 0). One run per dataset, no --seed (as in the reported runs), flat output layout
# outputs/HeavyDiff/<dataset>/.
#
#   bash run_heavydiff.sh
#   DATASETS="ecg" bash run_heavydiff.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"
VAR_LOSS_WEIGHT="${VAR_LOSS_WEIGHT:-3}"

declare -A CFG=( [building]=Building [ecg]=ECG [electricity]=Electricity
                 [precipitation]=precipitation [temperature]=temperature
                 [traffic]=Traffic [transportation]=transportation )

for DS in ${DATASETS}; do
  NAME="${DS}_heavydiff"
  CONFIG="Config/${CFG[$DS]}.yaml"
  echo "=== HeavyDiff ${DS} (run ${NAME})"
  "${PYTHON}" main.py --name "${NAME}" --config_file "${CONFIG}" --train  --heavy --var_loss_weight "${VAR_LOSS_WEIGHT}"
  "${PYTHON}" main.py --name "${NAME}" --config_file "${CONFIG}" --sample --heavy --var_loss_weight "${VAR_LOSS_WEIGHT}"
  "${PYTHON}" collect.py --method HeavyDiff --dataset "${DS}" --run_name "${NAME}"
done
