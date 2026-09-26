#!/usr/bin/env bash
# RF-Diffusion on the seven datasets: prepare the .mat files once, then train and sample
# each dataset in turn (200 epochs, batch size 32, seed 1 -- the configuration reported).
#
#   bash baselines/RF-Diffusion/run.sh
#   DATASETS="ecg traffic" bash baselines/RF-Diffusion/run.sh     # a subset
#
# Environment overrides:
#   PYTHON        python interpreter                                  (default: python)
#   RFDIFF_WORK   work dir for the .mat files, checkpoints and tensorboard logs
#                                                   (default: baselines/RF-Diffusion/work)
#   DATASETS      space-separated subset of the seven dataset names
#   SEED          (default: 1)
#
# Writes outputs/RF-Diffusion/<dataset>/RF-Diffusion_<dataset>_generatedata.npy.
# Datasets whose generated set already exists are skipped, so a sweep can be restarted;
# training itself also resumes from the last checkpoint in the work dir.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-1}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"

cd "${HERE}"
"${PYTHON}" prepare_data.py --datasets ${DATASETS}

for DS in ${DATASETS}; do
  OUT="${REPO}/outputs/RF-Diffusion/${DS}/RF-Diffusion_${DS}_generatedata.npy"
  if [[ -f "${OUT}" ]]; then
    echo "skip ${DS}: ${OUT} exists"
    continue
  fi
  echo "=== RF-Diffusion  dataset=${DS}  seed=${SEED}"
  "${PYTHON}" -u run_rfdiff.py --dataset "${DS}" --epochs 200 --batch-size 32 --seed "${SEED}"
done
