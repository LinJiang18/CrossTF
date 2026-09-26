#!/usr/bin/env bash
# TimeGAN (tsgm) on the seven datasets, one after another: 10000 iterations per phase,
# seed 10, Keras torch backend -- the configuration reported (see config.yaml).
#
#   python baselines/TimeGAN/patch_tsgm_torch.py        # once per environment
#   bash baselines/TimeGAN/run.sh
#   DATASETS="ecg traffic" bash baselines/TimeGAN/run.sh # a subset
#
# Environment overrides:
#   PYTHON         python interpreter                              (default: python)
#   TIMEGAN_WORK   work dir for per-run logs and train_metrics.jsonl
#                                                     (default: baselines/TimeGAN/work)
#   DATASETS       space-separated subset of the seven dataset names
#   ITERATIONS     iterations per phase                            (default: 10000)
#   SEED           (default: 10)
#
# Writes outputs/TimeGAN/<dataset>/TimeGAN_<dataset>_generatedata.npy (+ summary.json).
# Datasets whose generated set already exists are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
WORK="${TIMEGAN_WORK:-${HERE}/work}"
ITERATIONS="${ITERATIONS:-10000}"
SEED="${SEED:-10}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"
export KERAS_BACKEND=torch
export TIMEGAN_WORK="${WORK}"

# Refuse to train with an unpatched tsgm (it would silently run on the CPU).
"${PYTHON}" "${HERE}/patch_tsgm_torch.py" --check

cd "${HERE}"
for DS in ${DATASETS}; do
  OUT="${REPO}/outputs/TimeGAN/${DS}/TimeGAN_${DS}_generatedata.npy"
  if [[ -f "${OUT}" ]]; then
    echo "skip ${DS}: ${OUT} exists"
    continue
  fi
  echo "=== TimeGAN  dataset=${DS}  seed=${SEED}  iterations/phase=${ITERATIONS}"
  mkdir -p "${WORK}/${DS}"
  LOG="${WORK}/${DS}/train.log"
  "${PYTHON}" -u run_timegan.py --dataset "${DS}" --iterations "${ITERATIONS}" --seed "${SEED}" 2>&1 | tee "${LOG}"
  # tsgm catches exceptions inside its training steps, prints "Skipping ... batch" and
  # returns a zero loss, so a broken run can look healthy. Fail loudly instead.
  SKIPPED=$(grep -c "Skipping" "${LOG}" || true)
  if [[ "${SKIPPED}" -ne 0 ]]; then
    echo "ERROR: ${DS}: tsgm skipped ${SKIPPED} batches after swallowing errors; see ${LOG}" >&2
    exit 1
  fi
done
