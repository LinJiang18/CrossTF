#!/usr/bin/env bash
# SDForger (fms-dgt time_series databuilder) on the seven datasets, one after another.
#
#   bash baselines/SDForger/run.sh                        # all seven
#   DATASETS="ecg traffic" bash baselines/SDForger/run.sh # a subset
#
# Needs one CUDA GPU and the environment described in README.md. The seed (42) is fixed in
# fms-dgt/fms_dgt/public/databuilders/time_series/time_series.yaml.
#
# Environment overrides:
#   PYTHON          python interpreter                     (default: python)
#   SDFORGER_WORK   work dir for parquets, fine-tuned GPT-2 checkpoints and jsonl
#                   (~500 MB per dataset)                  (default: baselines/SDForger/work)
#   HF_OFFLINE=1    set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE (gpt2 must already be cached)
#   DATASETS        space-separated subset of the seven dataset names
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FMS="${HERE}/fms-dgt"
PYTHON="${PYTHON:-python}"
WORK="${SDFORGER_WORK:-${HERE}/work}"
OUT_ROOT="$(cd "${HERE}/../.." && pwd)/outputs/SDForger"

export DGT_DATA_DIR="${WORK}/data"
export DGT_OUTPUT_DIR="${WORK}/dgt_output"
export DGT_TELEMETRY_DIR="${WORK}/telemetry"
export PYTHONPATH="${FMS}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${HF_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
mkdir -p "${DGT_DATA_DIR}" "${DGT_OUTPUT_DIR}"

# dataset -> fms-dgt task stem (leading token of the .npy file name)
declare -A TASK_STEM=(
  [building]=Building [ecg]=ECG [electricity]=Electricity [precipitation]=precipitation
  [temperature]=temperature [traffic]=Traffic [transportation]=transportation
)
read -r -a DS_LIST <<< "${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"

"${PYTHON}" "${HERE}/prepare_data.py" --datasets "${DS_LIST[@]}"

cd "${FMS}"
for DS in "${DS_LIST[@]}"; do
  TASK="${TASK_STEM[${DS}]}_multisample"
  # Resumable: a dataset whose generated .npy already exists is skipped.
  if [[ -f "${OUT_ROOT}/${DS}/SDForger_${DS}_generatedata.npy" ]]; then
    echo "=========== SKIP ${DS} (output exists)"; continue
  fi
  echo "=========== ${DS}  (task ${TASK})"
  T0=$(date +%s)
  "${PYTHON}" -m fms_dgt.public --task-paths "./tasks/public/time_series/${TASK}/task.yaml"
  MIN=$(( ($(date +%s) - T0) / 60 ))
  "${PYTHON}" "${HERE}/collect_outputs.py" --dataset "${DS}" \
    --task-out-dir "${DGT_OUTPUT_DIR}/time_series/${TASK}" \
    --extra "{\"train_minutes\": ${MIN}, \"task\": \"${TASK}\", \"seed\": 42}"
done
