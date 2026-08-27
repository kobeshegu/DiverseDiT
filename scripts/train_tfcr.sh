#!/usr/bin/env bash
set -euo pipefail

# End-to-end TFCR pipeline, following the train -> sample -> npz -> evaluate
# organization used by the feature/trajectory branch.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
#   bash scripts/train_tfcr.sh all

RUN_STAGE="${1:-all}"  # train | sample | package | evaluate | all
REPO_DIR="${REPO_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT}"
TRAIN_ENV="${TRAIN_ENV:-/root/anaconda3/envs/repa}"
FID_ENV="${FID_ENV:-/root/anaconda3/envs/scale_rae}"

DATA_DIR="${DATA_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models}"
REF_NPZ="${REF_NPZ:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/datasets/VIRTUAL_imagenet256_labeled.npz}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-tfcr_sit_b2_no_repa_seed0}"
MODEL="${MODEL:-SiT-B/2}"
OUTPUT_DIR="${OUTPUT_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/results}"
SAMPLE_DIR="${SAMPLE_DIR:-sampled_images/$EXPERIMENT_NAME}"
RESOLUTION="${RESOLUTION:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-400000}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-29501}"
NPROC="${NPROC:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
ENABLE_TRANSITION="${ENABLE_TRANSITION:-0}"
ENABLE_DIVERSEDIT="${ENABLE_DIVERSEDIT:-0}"

NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-32}"
MODE="${MODE:-sde}"
NUM_STEPS="${NUM_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.0}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-1.0}"
VAE="${VAE:-mse}"

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

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
}

cd "$REPO_DIR"
export MASTER_PORT

TRAIN_EXTRA=()
SAMPLE_EXTRA=()
if [[ "$ENABLE_TRANSITION" == "1" ]]; then
  TRAIN_EXTRA+=(--factor-transition --factor-transition-coeff "$FACTOR_TRANSITION_COEFF")
  SAMPLE_EXTRA+=(--factor-transition)
fi
if [[ "$ENABLE_DIVERSEDIT" == "1" ]]; then
  TRAIN_EXTRA+=(
    --skip-layer-connection
    --block-diversity-loss
    --block-diversity-loss-coeff "$BLOCK_DIVERSITY_LOSS_COEFF"
  )
  SAMPLE_EXTRA+=(--skip-layer-connection --block-diversity-loss)
fi
if [[ -n "$FACTOR_TARGET_DEPTH" ]]; then
  TRAIN_EXTRA+=(--factor-target-depth "$FACTOR_TARGET_DEPTH")
fi

run_train() {
  require_value DATA_DIR "$DATA_DIR"
  require_value PRETRAINED_MODEL_PATH "$PRETRAINED_MODEL_PATH"
  activate_env "$TRAIN_ENV"
  accelerate launch --num_processes "$NPROC" train.py \
    --resolution "$RESOLUTION" \
    --batch-size "$BATCH_SIZE" \
    --report-to none \
    --allow-tf32 \
    --num-workers "$NUM_WORKERS" \
    --mixed-precision fp16 \
    --seed "$SEED" \
    --path-type linear \
    --prediction v \
    --weighting uniform \
    --model "$MODEL" \
    --enc-type none \
    --proj-coeff 0 \
    --encoder-depth 8 \
    --output-dir "$OUTPUT_DIR" \
    --exp-name "$EXPERIMENT_NAME" \
    --data-dir "$DATA_DIR" \
    --pretrained-model-path "$PRETRAINED_MODEL_PATH" \
    --max-train-steps "$MAX_TRAIN_STEPS" \
    --checkpointing-steps 5000 \
    --skip-training-samples \
    --trajectory-factorization \
    --factor-dim "$FACTOR_DIM" \
    --factor-projector-dim "$FACTOR_PROJECTOR_DIM" \
    --factor-source-depth "$FACTOR_SOURCE_DEPTH" \
    --factor-batch-ratio "$FACTOR_BATCH_RATIO" \
    --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB" \
    --factor-min-delta-t "$FACTOR_MIN_DELTA_T" \
    --factor-max-delta-t "$FACTOR_MAX_DELTA_T" \
    --factor-inv-coeff "$FACTOR_INV_COEFF" \
    --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF" \
    --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF" \
    --factor-recom-coeff "$FACTOR_RECOM_COEFF" \
    --factor-warmup-steps "$FACTOR_WARMUP_STEPS" \
    --factor-decay-start "$FACTOR_DECAY_START" \
    --factor-decay-end "$FACTOR_DECAY_END" \
    --factor-min-loss-scale "$FACTOR_MIN_LOSS_SCALE" \
    "${TRAIN_EXTRA[@]}"
}

CKPT_PADDED="$(printf '%07d' "$CHECKPOINT_STEP")"
CKPT="$OUTPUT_DIR/$EXPERIMENT_NAME/checkpoints/$CKPT_PADDED.pt"
MODEL_TAG="${MODEL//\//-}"
FOLDER_NAME="$MODEL_TAG-$CKPT_PADDED-size-$RESOLUTION-vae-$VAE-cfg-$CFG_SCALE-seed-$SEED-$MODE"
SAMPLE_NPZ="$SAMPLE_DIR/$FOLDER_NAME.npz"

run_sample() {
  require_value PRETRAINED_MODEL_PATH "$PRETRAINED_MODEL_PATH"
  activate_env "$TRAIN_ENV"
  torchrun --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" generate.py \
    --model "$MODEL" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --ckpt "$CKPT" \
    --path-type linear \
    --encoder-depth 8 \
    --projector-embed-dims none \
    --no-projection \
    --trajectory-factorization \
    --factor-dim "$FACTOR_DIM" \
    --factor-projector-dim "$FACTOR_PROJECTOR_DIM" \
    --factor-source-depth "$FACTOR_SOURCE_DEPTH" \
    --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
    --mode "$MODE" \
    --num-steps "$NUM_STEPS" \
    --cfg-scale "$CFG_SCALE" \
    --guidance-high "$GUIDANCE_HIGH" \
    --sample-dir "$SAMPLE_DIR" \
    --resolution "$RESOLUTION" \
    --vae "$VAE" \
    --global-seed "$SEED" \
    --pretrained-model-path "$PRETRAINED_MODEL_PATH" \
    "${SAMPLE_EXTRA[@]}"
}

run_package() {
  activate_env "$TRAIN_ENV"
  python npz_convert.py \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --sample-dir "$SAMPLE_DIR" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --resolution "$RESOLUTION" \
    --vae "$VAE" \
    --cfg-scale "$CFG_SCALE" \
    --global-seed "$SEED" \
    --mode "$MODE"
}

run_evaluate() {
  require_value REF_NPZ "$REF_NPZ"
  activate_env "$FID_ENV"
  python evaluator.py "$REF_NPZ" "$SAMPLE_NPZ"
}

case "$RUN_STAGE" in
  train) run_train ;;
  sample) run_sample ;;
  package) run_package ;;
  evaluate) run_evaluate ;;
  all)
    run_train
    run_sample
    run_package
    run_evaluate
    ;;
  *)
    echo "Unknown stage: $RUN_STAGE" >&2
    exit 2
    ;;
esac
