#!/usr/bin/env bash
set -euo pipefail

# DLC train -> sample -> npz -> metrics entrypoint for the TFCR comparison matrix.
#
# Example DLC command:
#   bash scripts/dlc_train_tfcr.sh a5_tfcr
#
# Useful controls:
#   RUN_STAGE=train bash scripts/dlc_train_tfcr.sh a5_tfcr
#   RUN_STAGE=sample bash scripts/dlc_train_tfcr.sh a5_tfcr
#   RUN_STAGE=fid bash scripts/dlc_train_tfcr.sh a5_tfcr

EXP="${1:-${EXP:-a5_tfcr}}"
RUN_STAGE="${2:-${RUN_STAGE:-all}}"  # train | sample | package | fid | evaluate | all
REPO_DIR="${REPO_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT}"
TRAIN_ENV="${TRAIN_ENV:-/root/anaconda3/envs/repa}"
FID_ENV="${FID_ENV:-/root/anaconda3/envs/scale_rae}"
ARCHIVE_CODE="${ARCHIVE_CODE:-1}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export DATA_DIR="${DATA_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models}"
export OUTPUT_DIR="${OUTPUT_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT/results}"
export SAMPLE_ROOT="${SAMPLE_ROOT:-$REPO_DIR/sampled_images}"
export REF_NPZ="${REF_NPZ:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/datasets/VIRTUAL_imagenet256_labeled.npz}"
export INCEPTION_V3_PATH="${INCEPTION_V3_PATH:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/REPA/classify_image_graph_def.pb}"
export MODEL="${MODEL:-SiT-B/2}"
export STEPS="${STEPS:-400000}"
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-$STEPS}"
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
export FACTOR_DIM="${FACTOR_DIM:-256}"
export FACTOR_PROJECTOR_DIM="${FACTOR_PROJECTOR_DIM:-1024}"
export FACTOR_SOURCE_DEPTH="${FACTOR_SOURCE_DEPTH:-8}"
export FACTOR_TARGET_DEPTH="${FACTOR_TARGET_DEPTH:-}"
export NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
export PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-32}"
export RESOLUTION="${RESOLUTION:-256}"
export MODE="${MODE:-sde}"
export NUM_STEPS="${NUM_STEPS:-250}"
export CFG_SCALE="${CFG_SCALE:-1.0}"
export GUIDANCE_HIGH="${GUIDANCE_HIGH:-1.0}"
export VAE="${VAE:-mse}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export NUM_PROCESSES="${NUM_PROCESSES:-${NPROC:-1}}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export MACHINE_RANK="${MACHINE_RANK:-${RANK:-0}}"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"

source "$REPO_DIR/scripts/tfcr_common.sh"
EXP_NAME="$(tfcr_experiment_name "$EXP")"

CKPT_PADDED="$(printf '%07d' "$CHECKPOINT_STEP")"
CKPT="${CKPT:-$OUTPUT_DIR/$EXP_NAME/checkpoints/$CKPT_PADDED.pt}"
MODEL_TAG="${MODEL//\//-}"
CKPT_TAG="$(basename "$CKPT" .pt)"
FOLDER_NAME="$MODEL_TAG-$CKPT_TAG-size-$RESOLUTION-vae-$VAE-cfg-$CFG_SCALE-seed-$SEED-$MODE"
SAMPLE_DIR="${SAMPLE_DIR:-$SAMPLE_ROOT/$EXP_NAME}"
SAMPLE_NPZ="${SAMPLE_NPZ:-$SAMPLE_DIR/$FOLDER_NAME.npz}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $name: $path" >&2
    exit 2
  fi
}

PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-none}"
PROJECTOR_ARGS=(--projector-embed-dims "$PROJECTOR_EMBED_DIMS")
projector_spec="$(printf '%s' "$PROJECTOR_EMBED_DIMS" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$projector_spec" || "$projector_spec" == "none" || "$projector_spec" == "null" ]]; then
  PROJECTOR_ARGS+=(--no-projection)
fi

MODEL_EXTRA=()
case "$EXP" in
  a0_sit)
    ;;
  a1_diversedit)
    MODEL_EXTRA+=(--skip-layer-connection --block-diversity-loss)
    ;;
  a2_repa)
    if [[ "$projector_spec" == "none" || "$projector_spec" == "null" ]]; then
      echo "a2_repa eval needs PROJECTOR_EMBED_DIMS to match the trained REPA checkpoint" >&2
      exit 2
    fi
    ;;
  a3_two_view|a4_inv_only|a5_tfcr)
    MODEL_EXTRA+=(--trajectory-factorization)
    ;;
  a6_tfcr_transition)
    MODEL_EXTRA+=(--trajectory-factorization --factor-transition)
    ;;
  a7_tfcr_diversedit)
    MODEL_EXTRA+=(
      --trajectory-factorization
      --skip-layer-connection
      --block-diversity-loss
    )
    ;;
  a8_tfcr_repa)
    if [[ "$projector_spec" == "none" || "$projector_spec" == "null" ]]; then
      echo "a8_tfcr_repa eval needs PROJECTOR_EMBED_DIMS to match the trained REPA checkpoint" >&2
      exit 2
    fi
    MODEL_EXTRA+=(--trajectory-factorization)
    ;;
  *)
    echo "Unknown experiment: $EXP" >&2
    exit 2
    ;;
esac

SAMPLE_CMD=(
  torchrun
  --nnodes=1
  --nproc_per_node="$NUM_PROCESSES"
  --master_port="$MASTER_PORT"
  generate.py
  --model "$MODEL"
  --num-fid-samples "$NUM_FID_SAMPLES"
  --ckpt "$CKPT"
  --path-type linear
  --encoder-depth 8
  "${PROJECTOR_ARGS[@]}"
  --factor-dim "$FACTOR_DIM"
  --factor-projector-dim "$FACTOR_PROJECTOR_DIM"
  --factor-source-depth "$FACTOR_SOURCE_DEPTH"
  --per-proc-batch-size "$PER_PROC_BATCH_SIZE"
  --mode "$MODE"
  --num-steps "$NUM_STEPS"
  --cfg-scale "$CFG_SCALE"
  --guidance-high "$GUIDANCE_HIGH"
  --sample-dir "$SAMPLE_DIR"
  --resolution "$RESOLUTION"
  --vae "$VAE"
  --global-seed "$SEED"
  --pretrained-model-path "$PRETRAINED_MODEL_PATH"
  "${MODEL_EXTRA[@]}"
)
if [[ -n "$FACTOR_TARGET_DEPTH" ]]; then
  SAMPLE_CMD+=(--factor-target-depth "$FACTOR_TARGET_DEPTH")
fi

PACKAGE_CMD=(
  python npz_convert.py
  --model "$MODEL"
  --ckpt "$CKPT"
  --sample-dir "$SAMPLE_DIR"
  --num-fid-samples "$NUM_FID_SAMPLES"
  --resolution "$RESOLUTION"
  --vae "$VAE"
  --cfg-scale "$CFG_SCALE"
  --global-seed "$SEED"
  --mode "$MODE"
)

FID_CMD=(python evaluator_tf.py "$REF_NPZ" "$SAMPLE_NPZ")

run_train() {
  tfcr_activate_env "$TRAIN_ENV"
  cd "$REPO_DIR"
  echo "Training $EXP_NAME on $NUM_PROCESSES process(es), output=$OUTPUT_DIR, data=$DATA_DIR"
  bash scripts/tfcr_ablation.sh "$EXP"
}

run_sample() {
  require_file checkpoint "$CKPT"
  tfcr_activate_env "$TRAIN_ENV"
  cd "$REPO_DIR"
  echo "Generating samples for $EXP_NAME from $CKPT"
  "${SAMPLE_CMD[@]}"
}

run_package() {
  require_file checkpoint "$CKPT"
  tfcr_activate_env "$TRAIN_ENV"
  cd "$REPO_DIR"
  echo "Packing samples from $SAMPLE_DIR/$FOLDER_NAME/images"
  "${PACKAGE_CMD[@]}"
}

run_fid() {
  require_file reference_npz "$REF_NPZ"
  require_file sample_npz "$SAMPLE_NPZ"
  require_file inception_graph "$INCEPTION_V3_PATH"
  tfcr_activate_env "$FID_ENV"
  cd "$REPO_DIR"
  echo "Computing metrics for $SAMPLE_NPZ"
  "${FID_CMD[@]}"
}

print_dry_run() {
  echo "EXP_NAME=$EXP_NAME"
  echo "CKPT=$CKPT"
  echo "SAMPLE_DIR=$SAMPLE_DIR"
  echo "SAMPLE_NPZ=$SAMPLE_NPZ"
  echo "INCEPTION_V3_PATH=$INCEPTION_V3_PATH"
  case "$RUN_STAGE" in
    train)
      DRY_RUN=1 bash "$REPO_DIR/scripts/tfcr_ablation.sh" "$EXP"
      ;;
    sample)
      tfcr_print_cmd "${SAMPLE_CMD[@]}"
      ;;
    package)
      tfcr_print_cmd "${PACKAGE_CMD[@]}"
      ;;
    fid|evaluate)
      tfcr_print_cmd "${FID_CMD[@]}"
      ;;
    all)
      DRY_RUN=1 bash "$REPO_DIR/scripts/tfcr_ablation.sh" "$EXP"
      tfcr_print_cmd "${SAMPLE_CMD[@]}"
      tfcr_print_cmd "${PACKAGE_CMD[@]}"
      tfcr_print_cmd "${FID_CMD[@]}"
      ;;
    *)
      echo "Unknown stage: $RUN_STAGE" >&2
      exit 2
      ;;
  esac
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  print_dry_run
  exit 0
fi

tfcr_archive_code "$REPO_DIR" "$OUTPUT_DIR" "$EXP_NAME" "$RUN_STAGE"

case "$RUN_STAGE" in
  train)
    run_train
    ;;
  sample)
    run_sample
    ;;
  package)
    run_package
    ;;
  fid|evaluate)
    run_fid
    ;;
  all)
    run_train
    run_sample
    run_package
    run_fid
    ;;
  *)
    echo "Unknown stage: $RUN_STAGE" >&2
    exit 2
    ;;
esac
