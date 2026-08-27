#!/usr/bin/env bash
set -euo pipefail

# DLC training entrypoint for the TFCR comparison matrix.
#
# Example DLC command:
#   bash scripts/dlc_train_tfcr.sh a5_tfcr
#
# Useful controls:
#   bash scripts/dlc_train_tfcr.sh a3_two_view
#   bash scripts/dlc_train_tfcr.sh a4_inv_only

EXP="${1:-${EXP:-a5_tfcr}}"
REPO_DIR="${REPO_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT}"
TRAIN_ENV="${TRAIN_ENV:-/root/anaconda3/envs/repa}"
SNAPSHOT_CODE="${SNAPSHOT_CODE:-0}"

export DATA_DIR="${DATA_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models}"
export OUTPUT_DIR="${OUTPUT_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/results}"
export STEPS="${STEPS:-400000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export SEED="${SEED:-0}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export REPORT_TO="${REPORT_TO:-none}"
export FACTOR_BATCH_RATIO="${FACTOR_BATCH_RATIO:-0.5}"
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_WARMUP_STEPS="${FACTOR_WARMUP_STEPS:-10000}"
export FACTOR_DECAY_START="${FACTOR_DECAY_START:-250000}"
export FACTOR_DECAY_END="${FACTOR_DECAY_END:-400000}"
export FACTOR_MIN_LOSS_SCALE="${FACTOR_MIN_LOSS_SCALE:-0}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

export NUM_PROCESSES="${NUM_PROCESSES:-${NPROC:-1}}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-${RANK:-0}}"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"

if [[ "$SNAPSHOT_CODE" == "1" ]]; then
  snapshot_parent="${CODE_SNAPSHOT_PARENT:-/tmp}"
  job_id="${DLC_JOB_ID:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${HOSTNAME:-job}}}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  slug="$(printf '%s_%s_%s' "$EXP" "${RUN_SUFFIX:-run}" "$job_id" | tr -c 'A-Za-z0-9_.-' '_')"
  snapshot_dir="${CODE_SNAPSHOT_DIR:-$snapshot_parent/diversedit_tfcr_${slug}_${timestamp}}"
  mkdir -p "$snapshot_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude .git \
      --exclude __pycache__ \
      --exclude '*/__pycache__' \
      --exclude results \
      --exclude sampled_images \
      --exclude wandb \
      "$REPO_DIR"/ "$snapshot_dir"/
  else
    (
      cd "$REPO_DIR"
      tar \
        --exclude='./.git' \
        --exclude='./__pycache__' \
        --exclude='*/__pycache__' \
        --exclude='./results' \
        --exclude='./sampled_images' \
        --exclude='./wandb' \
        -cf - .
    ) | (
      cd "$snapshot_dir"
      tar -xf -
    )
  fi
  export REPO_DIR="$snapshot_dir"
  export SNAPSHOT_CODE=0
  echo "Code snapshot: $snapshot_dir"
  exec bash "$snapshot_dir/scripts/dlc_train_tfcr.sh" "$EXP"
fi

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate "$TRAIN_ENV"

cd "$REPO_DIR"
echo "Launching $EXP on $NUM_PROCESSES process(es), output=$OUTPUT_DIR, data=$DATA_DIR"
bash scripts/tfcr_ablation.sh "$EXP"
