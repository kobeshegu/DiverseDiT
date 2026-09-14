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
FID_FALLBACK_ENV="${FID_FALLBACK_ENV:-}"
AUTO_FIX_FID_ENV="${AUTO_FIX_FID_ENV:-1}"
FID_ENV_REPAIR_PACKAGES="${FID_ENV_REPAIR_PACKAGES:-tensorflow-cpu==2.15.1 numpy<2 protobuf<4 scipy tqdm}"

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
FACTOR_ORBIT_MODE="${FACTOR_ORBIT_MODE:-legacy}"
FACTOR_ORBIT_NOISE_ONLY_PROB="${FACTOR_ORBIT_NOISE_ONLY_PROB:-0.5}"
FACTOR_MIN_DELTA_T="${FACTOR_MIN_DELTA_T:-0.15}"
FACTOR_MAX_DELTA_T="${FACTOR_MAX_DELTA_T:-0.7}"
FACTOR_INV_COEFF="${FACTOR_INV_COEFF:-0.1}"
FACTOR_PERSISTENT_COEFF="${FACTOR_PERSISTENT_COEFF:-0.05}"
FACTOR_EVOLVING_COEFF="${FACTOR_EVOLVING_COEFF:-0.05}"
FACTOR_RECOM_COEFF="${FACTOR_RECOM_COEFF:-0.1}"
FACTOR_RELIABILITY_KEEP_RATIO="${FACTOR_RELIABILITY_KEEP_RATIO:-0.75}"
FACTOR_RELIABILITY_FLOOR="${FACTOR_RELIABILITY_FLOOR:-0.0}"
FACTOR_VELOCITY_RECOM_COEFF="${FACTOR_VELOCITY_RECOM_COEFF:-0.05}"
FACTOR_ADVERSARIAL_TIMESTEP_BINS="${FACTOR_ADVERSARIAL_TIMESTEP_BINS:-8}"
FACTOR_ADVERSARIAL_GRL_SCALE="${FACTOR_ADVERSARIAL_GRL_SCALE:-0.1}"
FACTOR_ADVERSARIAL_START_STEPS="${FACTOR_ADVERSARIAL_START_STEPS:-20000}"
FACTOR_ADVERSARIAL_WARMUP_STEPS="${FACTOR_ADVERSARIAL_WARMUP_STEPS:-30000}"
FACTOR_ADV_PERSISTENT_TIME_COEFF="${FACTOR_ADV_PERSISTENT_TIME_COEFF:-0.05}"
FACTOR_ADV_PERSISTENT_ORBIT_COEFF="${FACTOR_ADV_PERSISTENT_ORBIT_COEFF:-0.05}"
FACTOR_PROBE_EVOLVING_TIME_COEFF="${FACTOR_PROBE_EVOLVING_TIME_COEFF:-0.05}"
FACTOR_PROBE_EVOLVING_ORBIT_COEFF="${FACTOR_PROBE_EVOLVING_ORBIT_COEFF:-0.05}"
FACTOR_TRANSITION_COEFF="${FACTOR_TRANSITION_COEFF:-0.05}"
FACTOR_WARMUP_STEPS="${FACTOR_WARMUP_STEPS:-10000}"
FACTOR_DECAY_START="${FACTOR_DECAY_START:-250000}"
FACTOR_DECAY_END="${FACTOR_DECAY_END:-400000}"
FACTOR_MIN_LOSS_SCALE="${FACTOR_MIN_LOSS_SCALE:-0}"
BLOCK_DIVERSITY_LOSS_COEFF="${BLOCK_DIVERSITY_LOSS_COEFF:-0.001}"
ENABLE_TRANSITION="${ENABLE_TRANSITION:-0}"
ENABLE_DIVERSEDIT="${ENABLE_DIVERSEDIT:-0}"
ENABLE_ORBIT_CONSENSUS="${ENABLE_ORBIT_CONSENSUS:-0}"
ENABLE_ADVERSARIAL="${ENABLE_ADVERSARIAL:-0}"
ADVERSARIAL_SHUFFLE_LABELS="${ADVERSARIAL_SHUFFLE_LABELS:-0}"

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

  local conda_candidates=()
  if [[ "$env_name" == */envs/* ]]; then
    conda_candidates+=("${env_name%%/envs/*}/etc/profile.d/conda.sh")
  fi
  conda_candidates+=(
    /opt/conda/etc/profile.d/conda.sh
    /root/anaconda3/etc/profile.d/conda.sh
    /root/miniconda3/etc/profile.d/conda.sh
  )

  local conda_sh
  for conda_sh in "${conda_candidates[@]}"; do
    if [[ -f "$conda_sh" ]]; then
      source "$conda_sh"
      break
    fi
  done

  if command -v conda >/dev/null 2>&1 && conda activate "$env_name"; then
    return
  fi

  if [[ -f "$env_name/bin/activate" ]]; then
    source "$env_name/bin/activate"
  else
    echo "Cannot activate environment: $env_name" >&2
    echo "Expected conda or $env_name/bin/activate to be available." >&2
    exit 2
  fi
}

activate_fid_env() {
  local primary_env="$FID_ENV"
  local fallback_env="$FID_FALLBACK_ENV"
  local active_env="$primary_env"
  local last_log
  local fid_probe="import tensorflow.compat.v1; import scipy.linalg; import tqdm.auto"
  last_log="$(mktemp)"
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

  activate_env "$primary_env"
  if python -c "$fid_probe" >"$last_log" 2>&1; then
    echo "Using FID env: $primary_env"
    rm -f "$last_log"
    return
  fi

  if [[ "$fallback_env" != "$primary_env" && -d "$fallback_env" ]]; then
    echo "FID env $primary_env cannot import required metric dependencies; falling back to $fallback_env"
    activate_env "$fallback_env"
    active_env="$fallback_env"
    if python -c "$fid_probe" >"$last_log" 2>&1; then
      echo "Using FID env: $fallback_env"
      rm -f "$last_log"
      return
    fi
  fi

  if [[ "$AUTO_FIX_FID_ENV" == "1" ]]; then
    local repair_lock="${FID_ENV_REPAIR_LOCK:-$active_env/.tfcr_fid_env_repair.lock}"
    echo "Attempting to repair current FID env with: $FID_ENV_REPAIR_PACKAGES"
    if command -v flock >/dev/null 2>&1; then
      (
        flock 9
        if ! python -c "$fid_probe" >"$last_log" 2>&1; then
          python -m pip install $FID_ENV_REPAIR_PACKAGES
        fi
      ) 9>"$repair_lock"
    else
      python -m pip install $FID_ENV_REPAIR_PACKAGES
    fi
    if python -c "$fid_probe" >"$last_log" 2>&1; then
      echo "Using repaired FID env: $(python -c 'import sys; print(sys.prefix)')"
      rm -f "$last_log"
      return
    fi
  fi

  echo "No usable FID environment found. Tried: $primary_env and $fallback_env" >&2
  echo "The FID environment must import tensorflow.compat.v1, scipy, and tqdm." >&2
  echo "Last metric dependency import error:" >&2
  tail -40 "$last_log" >&2
  rm -f "$last_log"
  exit 2
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
if [[ "$ENABLE_ORBIT_CONSENSUS" == "1" ]]; then
  if [[ "$FACTOR_ORBIT_MODE" == "legacy" ]]; then
    FACTOR_ORBIT_MODE=orthogonal
  fi
  TRAIN_EXTRA+=(
    --factor-share-cfg-dropout
    --factor-reliable-target
    --factor-reliability-keep-ratio "$FACTOR_RELIABILITY_KEEP_RATIO"
    --factor-reliability-floor "$FACTOR_RELIABILITY_FLOOR"
    --factor-velocity-recomposition
    --factor-velocity-recom-coeff "$FACTOR_VELOCITY_RECOM_COEFF"
  )
  SAMPLE_EXTRA+=(--factor-velocity-recomposition)
fi
if [[ "$ENABLE_ADVERSARIAL" == "1" ]]; then
  if [[ "$FACTOR_ORBIT_MODE" != "orthogonal" ]]; then
    echo "ENABLE_ADVERSARIAL requires FACTOR_ORBIT_MODE=orthogonal" >&2
    exit 2
  fi
  TRAIN_EXTRA+=(
    --factor-adversarial
    --factor-share-cfg-dropout
    --factor-adversarial-timestep-bins "$FACTOR_ADVERSARIAL_TIMESTEP_BINS"
    --factor-adversarial-grl-scale "$FACTOR_ADVERSARIAL_GRL_SCALE"
    --factor-adversarial-start-steps "$FACTOR_ADVERSARIAL_START_STEPS"
    --factor-adversarial-warmup-steps "$FACTOR_ADVERSARIAL_WARMUP_STEPS"
    --factor-adv-persistent-time-coeff "$FACTOR_ADV_PERSISTENT_TIME_COEFF"
    --factor-adv-persistent-orbit-coeff "$FACTOR_ADV_PERSISTENT_ORBIT_COEFF"
    --factor-probe-evolving-time-coeff "$FACTOR_PROBE_EVOLVING_TIME_COEFF"
    --factor-probe-evolving-orbit-coeff "$FACTOR_PROBE_EVOLVING_ORBIT_COEFF"
  )
  SAMPLE_EXTRA+=(
    --factor-adversarial
    --factor-adversarial-timestep-bins "$FACTOR_ADVERSARIAL_TIMESTEP_BINS"
  )
  if [[ "$ADVERSARIAL_SHUFFLE_LABELS" == "1" ]]; then
    TRAIN_EXTRA+=(--factor-adversarial-shuffle-labels)
  fi
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
    --factor-orbit-mode "$FACTOR_ORBIT_MODE" \
    --factor-orbit-noise-only-prob "$FACTOR_ORBIT_NOISE_ONLY_PROB" \
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
METRICS_TXT="${METRICS_TXT:-$SAMPLE_DIR/${FOLDER_NAME}_metrics.txt}"

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
  if [[ "${FORCE_EVALUATE:-0}" != "1" && -f "$METRICS_TXT" ]]; then
    echo "Metrics already exist: $METRICS_TXT"
    echo "Set FORCE_EVALUATE=1 to recompute."
    return
  fi
  require_value REF_NPZ "$REF_NPZ"
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
  activate_fid_env
  python evaluator_tf.py "$REF_NPZ" "$SAMPLE_NPZ"
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
