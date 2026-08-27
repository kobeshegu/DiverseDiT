#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/tfcr_ablation.sh a5_tfcr
# Defaults mirror the local trajectory-dino-implementation training setup.
# Optional overrides: MODEL, STEPS, BATCH_SIZE, SEED, OUTPUT_DIR, NUM_PROCESSES.

EXP="${1:-a5_tfcr}"
MODEL="${MODEL:-SiT-B/2}"
STEPS="${STEPS:-400000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/results}"
NUM_PROCESSES="${NUM_PROCESSES:-${NPROC:-1}}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-${RANK:-0}}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MASTER_PORT="${MASTER_PORT:-29501}"
REPORT_TO="${REPORT_TO:-none}"
FACTOR_DIM="${FACTOR_DIM:-256}"
FACTOR_PROJECTOR_DIM="${FACTOR_PROJECTOR_DIM:-1024}"
FACTOR_SOURCE_DEPTH="${FACTOR_SOURCE_DEPTH:-8}"
FACTOR_TARGET_DEPTH="${FACTOR_TARGET_DEPTH:-}"
FACTOR_BATCH_RATIO="${FACTOR_BATCH_RATIO:-0.5}"
CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
FACTOR_MIN_DELTA_T="${FACTOR_MIN_DELTA_T:-0.15}"
FACTOR_MAX_DELTA_T="${FACTOR_MAX_DELTA_T:-0.7}"
FACTOR_INV_COEFF="${FACTOR_INV_COEFF:-0.1}"
FACTOR_PERSISTENT_COEFF="${FACTOR_PERSISTENT_COEFF:-0.05}"
FACTOR_EVOLVING_COEFF="${FACTOR_EVOLVING_COEFF:-0.05}"
FACTOR_RECOM_COEFF="${FACTOR_RECOM_COEFF:-0.1}"
FACTOR_TRANSITION_COEFF="${FACTOR_TRANSITION_COEFF:-0.05}"
FACTOR_WARMUP_STEPS="${FACTOR_WARMUP_STEPS:-10000}"
FACTOR_DECAY_START="${FACTOR_DECAY_START:-250000}"
FACTOR_DECAY_END="${FACTOR_DECAY_END:-400000}"
FACTOR_MIN_LOSS_SCALE="${FACTOR_MIN_LOSS_SCALE:-0}"
BLOCK_DIVERSITY_LOSS_COEFF="${BLOCK_DIVERSITY_LOSS_COEFF:-0.001}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
TRAIN_ENV="${TRAIN_ENV:-/root/anaconda3/envs/repa}"

DATA_DIR="${DATA_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models}"

activate_env() {
  local env_name="$1"
  if [[ -z "$env_name" ]]; then
    return
  fi
  if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
  fi
  conda activate "$env_name"
}

COMMON=(
  --report-to "$REPORT_TO"
  --allow-tf32
  --mixed-precision fp16
  --seed "$SEED"
  --path-type linear
  --prediction v
  --weighting uniform
  --model "$MODEL"
  --encoder-depth 8
  --output-dir "$OUTPUT_DIR"
  --data-dir "$DATA_DIR"
  --pretrained-model-path "$PRETRAINED_MODEL_PATH"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --max-train-steps "$STEPS"
  --checkpointing-steps 5000
  --skip-training-samples
  --enc-type none
  --proj-coeff 0
  --factor-dim "$FACTOR_DIM"
  --factor-projector-dim "$FACTOR_PROJECTOR_DIM"
  --factor-source-depth "$FACTOR_SOURCE_DEPTH"
  --factor-min-delta-t "$FACTOR_MIN_DELTA_T"
  --factor-max-delta-t "$FACTOR_MAX_DELTA_T"
  --factor-batch-ratio "$FACTOR_BATCH_RATIO"
  --factor-warmup-steps "$FACTOR_WARMUP_STEPS"
  --factor-decay-start "$FACTOR_DECAY_START"
  --factor-decay-end "$FACTOR_DECAY_END"
  --factor-min-loss-scale "$FACTOR_MIN_LOSS_SCALE"
)
if [[ -n "$FACTOR_TARGET_DEPTH" ]]; then
  COMMON+=(--factor-target-depth "$FACTOR_TARGET_DEPTH")
fi

EXTRA=()
case "$EXP" in
  a0_sit)
    ;;
  a1_diversedit)
    EXTRA+=(
      --skip-layer-connection
      --block-diversity-loss
      --block-diversity-loss-coeff "$BLOCK_DIVERSITY_LOSS_COEFF"
    )
    ;;
  a2_repa)
    COMMON+=(--enc-type dinov2-vit-b --proj-coeff 0.5)
    ;;
  a3_two_view)
    EXTRA+=(
      --trajectory-factorization
      --factor-paired-view-only
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff 0
      --factor-persistent-coeff 0
      --factor-evolving-coeff 0
      --factor-recom-coeff 0
      --factor-transition-coeff 0
    )
    ;;
  a4_inv_only)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff 0
      --factor-evolving-coeff 0
      --factor-recom-coeff 0
      --factor-transition-coeff 0
    )
    ;;
  a5_tfcr)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  a6_tfcr_transition)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-transition
      --factor-transition-coeff "$FACTOR_TRANSITION_COEFF"
    )
    ;;
  a7_tfcr_diversedit)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --skip-layer-connection
      --block-diversity-loss
      --block-diversity-loss-coeff "$BLOCK_DIVERSITY_LOSS_COEFF"
    )
    ;;
  a8_tfcr_repa)
    COMMON+=(--enc-type dinov2-vit-b --proj-coeff 0.5)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
    )
    ;;
  *)
    echo "Unknown experiment: $EXP" >&2
    exit 2
    ;;
esac

EXP_NAME="${EXP}-${MODEL//\//-}-s${SEED}"
case "$EXP" in
  a3_two_view|a4_inv_only|a5_tfcr|a6_tfcr_transition|a7_tfcr_diversedit|a8_tfcr_repa)
    EXP_NAME+="-r${FACTOR_BATCH_RATIO}-x${CROSS_NOISE_PROB}"
    ;;
esac
if [[ -n "$RUN_SUFFIX" ]]; then
  EXP_NAME+="-${RUN_SUFFIX}"
fi
COMMON+=(--exp-name "$EXP_NAME")

export MASTER_PORT
activate_env "$TRAIN_ENV"

ACCELERATE_ARGS=(
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --machine_rank "$MACHINE_RANK"
  --main_process_port "$MASTER_PORT"
)
if [[ "$NUM_MACHINES" != "1" ]]; then
  ACCELERATE_ARGS+=(--main_process_ip "$MAIN_PROCESS_IP")
fi

accelerate launch "${ACCELERATE_ARGS[@]}" train.py "${COMMON[@]}" "${EXTRA[@]}"
