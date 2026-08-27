#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/tfcr_ablation.sh a5_tfcr
# Required environment: DATA_DIR, PRETRAINED_MODEL_PATH.
# Optional: MODEL, STEPS, BATCH_SIZE, SEED, OUTPUT_DIR, NUM_PROCESSES.

EXP="${1:-a5_tfcr}"
MODEL="${MODEL:-SiT-B/2}"
STEPS="${STEPS:-450000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-exps/tfcr}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MASTER_PORT="${MASTER_PORT:-29501}"
REPORT_TO="${REPORT_TO:-none}"
FACTOR_BATCH_RATIO="${FACTOR_BATCH_RATIO:-0.5}"
CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

: "${DATA_DIR:?Set DATA_DIR to the ImageNet latent dataset}"
: "${PRETRAINED_MODEL_PATH:?Set PRETRAINED_MODEL_PATH to the local VAE root}"

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
  --factor-batch-ratio "$FACTOR_BATCH_RATIO"
  --factor-warmup-steps 10000
  --factor-decay-start 250000
  --factor-decay-end 400000
  --factor-min-loss-scale 0
)

EXTRA=()
case "$EXP" in
  a0_sit)
    ;;
  a1_diversedit)
    EXTRA+=(--skip-layer-connection --block-diversity-loss)
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
      --factor-inv-coeff 0.1
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
      --factor-inv-coeff 0.1
      --factor-persistent-coeff 0.05
      --factor-evolving-coeff 0.05
      --factor-recom-coeff 0.1
      --factor-transition-coeff 0
    )
    ;;
  a6_tfcr_transition)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff 0.1
      --factor-persistent-coeff 0.05
      --factor-evolving-coeff 0.05
      --factor-recom-coeff 0.1
      --factor-transition
      --factor-transition-coeff 0.05
    )
    ;;
  a7_tfcr_diversedit)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff 0.1
      --factor-persistent-coeff 0.05
      --factor-evolving-coeff 0.05
      --factor-recom-coeff 0.1
      --skip-layer-connection
      --block-diversity-loss
    )
    ;;
  a8_tfcr_repa)
    COMMON+=(--enc-type dinov2-vit-b --proj-coeff 0.5)
    EXTRA+=(
      --trajectory-factorization
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff 0.1
      --factor-persistent-coeff 0.05
      --factor-evolving-coeff 0.05
      --factor-recom-coeff 0.1
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
accelerate launch --num_processes "$NUM_PROCESSES" train.py "${COMMON[@]}" "${EXTRA[@]}"
