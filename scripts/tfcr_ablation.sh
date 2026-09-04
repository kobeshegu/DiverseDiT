#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/tfcr_ablation.sh a5_tfcr
# Defaults mirror the local trajectory-dino-implementation training setup.
# Optional overrides: MODEL, STEPS, BATCH_SIZE, SEED, OUTPUT_DIR, NUM_PROCESSES.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tfcr_common.sh"

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
INVARIANT_DIM="${INVARIANT_DIM:-256}"
INVARIANT_PROJECTOR_DIM="${INVARIANT_PROJECTOR_DIM:-1024}"
INVARIANT_PROJECTOR_TYPE="${INVARIANT_PROJECTOR_TYPE:-linear}"
INVARIANT_SOURCE_DEPTH="${INVARIANT_SOURCE_DEPTH:-4}"
INVARIANT_BATCH_RATIO="${INVARIANT_BATCH_RATIO:-0.375}"
INVARIANT_MIN_DELTA_T="${INVARIANT_MIN_DELTA_T:-0.05}"
INVARIANT_MAX_DELTA_T="${INVARIANT_MAX_DELTA_T:-0.2}"
INVARIANT_MAX_T="${INVARIANT_MAX_T:-0.8}"
INVARIANT_SNR_POWER="${INVARIANT_SNR_POWER:-1.0}"
INVARIANT_TIME_COEFF="${INVARIANT_TIME_COEFF:-0.1}"
INVARIANT_NOISE_COEFF="${INVARIANT_NOISE_COEFF:-0.1}"
INVARIANT_IMAGE_VARIANCE_COEFF="${INVARIANT_IMAGE_VARIANCE_COEFF:-0.02}"
INVARIANT_SPATIAL_VARIANCE_COEFF="${INVARIANT_SPATIAL_VARIANCE_COEFF:-0.02}"
INVARIANT_COVARIANCE_COEFF="${INVARIANT_COVARIANCE_COEFF:-0.001}"
INVARIANT_BASIS_COEFF="${INVARIANT_BASIS_COEFF:-0.01}"
INVARIANT_RELATION_COEFF="${INVARIANT_RELATION_COEFF:-0.05}"
INVARIANT_VARIANCE_TARGET="${INVARIANT_VARIANCE_TARGET:-1.0}"
INVARIANT_SPATIAL_VARIANCE_TARGET="${INVARIANT_SPATIAL_VARIANCE_TARGET:-0.5}"
INVARIANT_WARMUP_STEPS="${INVARIANT_WARMUP_STEPS:-10000}"
INVARIANT_DECAY_START="${INVARIANT_DECAY_START:--1}"
INVARIANT_DECAY_END="${INVARIANT_DECAY_END:--1}"
INVARIANT_MIN_LOSS_SCALE="${INVARIANT_MIN_LOSS_SCALE:-0}"
BLOCK_DIVERSITY_LOSS_COEFF="${BLOCK_DIVERSITY_LOSS_COEFF:-0.001}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
TRAIN_ENV="${TRAIN_ENV:-/root/anaconda3/envs/repa}"
DRY_RUN="${DRY_RUN:-0}"

DATA_DIR="${DATA_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models}"

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
  --invariant-dim "$INVARIANT_DIM"
  --invariant-projector-dim "$INVARIANT_PROJECTOR_DIM"
  --invariant-projector-type "$INVARIANT_PROJECTOR_TYPE"
  --invariant-source-depth "$INVARIANT_SOURCE_DEPTH"
  --invariant-min-delta-t "$INVARIANT_MIN_DELTA_T"
  --invariant-max-delta-t "$INVARIANT_MAX_DELTA_T"
  --invariant-batch-ratio "$INVARIANT_BATCH_RATIO"
  --invariant-max-t "$INVARIANT_MAX_T"
  --invariant-snr-power "$INVARIANT_SNR_POWER"
  --invariant-warmup-steps "$INVARIANT_WARMUP_STEPS"
  --invariant-decay-start "$INVARIANT_DECAY_START"
  --invariant-decay-end "$INVARIANT_DECAY_END"
  --invariant-min-loss-scale "$INVARIANT_MIN_LOSS_SCALE"
  --invariant-variance-target "$INVARIANT_VARIANCE_TARGET"
  --invariant-spatial-variance-target "$INVARIANT_SPATIAL_VARIANCE_TARGET"
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
  q0_invariant_three_view)
    EXTRA+=(
      --trajectory-invariance
      --invariant-view-control-only
      --invariant-time-coeff 0
      --invariant-noise-coeff 0
      --invariant-image-variance-coeff 0
      --invariant-spatial-variance-coeff 0
      --invariant-covariance-coeff 0
      --invariant-basis-coeff 0
      --invariant-relation-coeff 0
    )
    ;;
  q1_orbit_consistency)
    EXTRA+=(
      --trajectory-invariance
      --invariant-time-coeff "$INVARIANT_TIME_COEFF"
      --invariant-noise-coeff "$INVARIANT_NOISE_COEFF"
      --invariant-image-variance-coeff 0
      --invariant-spatial-variance-coeff 0
      --invariant-covariance-coeff 0
      --invariant-basis-coeff 0
      --invariant-relation-coeff 0
    )
    ;;
  q2_orbit_spread)
    EXTRA+=(
      --trajectory-invariance
      --invariant-time-coeff "$INVARIANT_TIME_COEFF"
      --invariant-noise-coeff "$INVARIANT_NOISE_COEFF"
      --invariant-image-variance-coeff "$INVARIANT_IMAGE_VARIANCE_COEFF"
      --invariant-spatial-variance-coeff "$INVARIANT_SPATIAL_VARIANCE_COEFF"
      --invariant-covariance-coeff "$INVARIANT_COVARIANCE_COEFF"
      --invariant-basis-coeff "$INVARIANT_BASIS_COEFF"
      --invariant-relation-coeff 0
    )
    ;;
  q3_orbit_full)
    EXTRA+=(
      --trajectory-invariance
      --invariant-time-coeff "$INVARIANT_TIME_COEFF"
      --invariant-noise-coeff "$INVARIANT_NOISE_COEFF"
      --invariant-image-variance-coeff "$INVARIANT_IMAGE_VARIANCE_COEFF"
      --invariant-spatial-variance-coeff "$INVARIANT_SPATIAL_VARIANCE_COEFF"
      --invariant-covariance-coeff "$INVARIANT_COVARIANCE_COEFF"
      --invariant-basis-coeff "$INVARIANT_BASIS_COEFF"
      --invariant-relation-coeff "$INVARIANT_RELATION_COEFF"
    )
    ;;
  *)
    echo "Unknown experiment: $EXP" >&2
    exit 2
    ;;
esac

EXP_NAME="$(tfcr_experiment_name "$EXP")"
COMMON+=(--exp-name "$EXP_NAME")

export MASTER_PORT

ACCELERATE_ARGS=(
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --machine_rank "$MACHINE_RANK"
  --main_process_port "$MASTER_PORT"
)
if [[ "$NUM_MACHINES" != "1" ]]; then
  ACCELERATE_ARGS+=(--main_process_ip "$MAIN_PROCESS_IP")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'accelerate launch'
  printf ' %q' "${ACCELERATE_ARGS[@]}"
  printf ' train.py'
  printf ' %q' "${COMMON[@]}" "${EXTRA[@]}"
  printf '\n'
  exit 0
fi

tfcr_activate_env "$TRAIN_ENV"
accelerate launch "${ACCELERATE_ARGS[@]}" train.py "${COMMON[@]}" "${EXTRA[@]}"
