#!/usr/bin/env bash
# Diffusion-TS on the seven datasets, three runs each (seed-index 1/2/3 = seeds 12345/2/3).
#
# Usage
#   bash baselines/Diffusion-TS/run.sh
#   DATASETS="ecg building" SEED_INDICES="1" bash baselines/Diffusion-TS/run.sh
#   PYTHON=/path/to/python DTS_WORK=/scratch/dts bash baselines/Diffusion-TS/run.sh
#
# Output: outputs/Diffusion-TS/seed<index>/<dataset>/Diffusion-TS_<dataset>_generatedata.npy
# Checkpoints and training logs: $DTS_WORK (default baselines/Diffusion-TS/work/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-building ecg electricity precipitation temperature traffic transportation}"
SEED_INDICES="${SEED_INDICES:-1 2 3}"
# Upstream wraps its 1000-step reverse loop in tqdm; display only.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

for S in ${SEED_INDICES}; do
  for DS in ${DATASETS}; do
    echo "=== Diffusion-TS | ${DS} | seed-index ${S}"
    "${PYTHON}" -u "${HERE}/run_diffusion_ts.py" --dataset "${DS}" --seed-index "${S}"
  done
done
