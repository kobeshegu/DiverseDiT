#!/usr/bin/env bash
set -euo pipefail

# DLC train -> sample -> npz -> metrics entrypoint for the TFCR comparison matrix.
#
# Example DLC command:
#   bash scripts/dlc_train_tfcr.sh a5_tfcr
#
# Useful controls:
#   RUN_STAGE=train bash scripts/dlc_train_tfcr.sh a5_tfcr
#   RUN_STAGE=eval bash scripts/dlc_train_tfcr.sh a5_tfcr
#   RUN_STAGE=fid bash scripts/dlc_train_tfcr.sh a5_tfcr

EXP="${1:-${EXP:-a5_tfcr}}"
RUN_STAGE="${2:-${RUN_STAGE:-all}}"  # train | sample | package | eval | fid | all
REPO_DIR="${REPO_DIR:-/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT}"
TRAIN_ENV="${TRAIN_ENV:-/root/anaconda3/envs/repa}"
FID_ENV="${FID_ENV:-/root/anaconda3/envs/scale_rae}"
FID_FALLBACK_ENV="${FID_FALLBACK_ENV:-}"
AUTO_FIX_FID_ENV="${AUTO_FIX_FID_ENV:-1}"
FID_ENV_REPAIR_PACKAGES="${FID_ENV_REPAIR_PACKAGES:-tensorflow-cpu==2.15.1 numpy<2 protobuf<4 scipy tqdm}"
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
export FACTOR_ADVERSARIAL_TIMESTEP_BINS="${FACTOR_ADVERSARIAL_TIMESTEP_BINS:-8}"
export FACTOR_ADVERSARIAL_GRL_SCALE="${FACTOR_ADVERSARIAL_GRL_SCALE:-0.1}"
export FACTOR_ADVERSARIAL_START_STEPS="${FACTOR_ADVERSARIAL_START_STEPS:-20000}"
export FACTOR_ADVERSARIAL_WARMUP_STEPS="${FACTOR_ADVERSARIAL_WARMUP_STEPS:-30000}"
export FACTOR_ADV_PERSISTENT_TIME_COEFF="${FACTOR_ADV_PERSISTENT_TIME_COEFF:-0.05}"
export FACTOR_ADV_PERSISTENT_ORBIT_COEFF="${FACTOR_ADV_PERSISTENT_ORBIT_COEFF:-0.05}"
export FACTOR_PROBE_EVOLVING_TIME_COEFF="${FACTOR_PROBE_EVOLVING_TIME_COEFF:-0.05}"
export FACTOR_PROBE_EVOLVING_ORBIT_COEFF="${FACTOR_PROBE_EVOLVING_ORBIT_COEFF:-0.05}"
export FACTOR_CLEAN_CONSENSUS_TEMPERATURE="${FACTOR_CLEAN_CONSENSUS_TEMPERATURE:-0.25}"
export FACTOR_CLEAN_CONSENSUS_COEFF="${FACTOR_CLEAN_CONSENSUS_COEFF:-0.05}"
export FACTOR_SHARED_REPA_COEFF="${FACTOR_SHARED_REPA_COEFF:-0.5}"
export FACTOR_NATIVE_SOURCE_COEFF="${FACTOR_NATIVE_SOURCE_COEFF:-0.1}"
export FACTOR_NATIVE_NOISE_COEFF="${FACTOR_NATIVE_NOISE_COEFF:-0.1}"
export FACTOR_NATIVE_ANTITHETIC_COEFF="${FACTOR_NATIVE_ANTITHETIC_COEFF:-0.05}"
export FACTOR_NATIVE_BASE_COEFF="${FACTOR_NATIVE_BASE_COEFF:-0.0}"
export FACTOR_SEMANTIC_REPA_COEFF="${FACTOR_SEMANTIC_REPA_COEFF:-0.5}"
export FACTOR_SEMANTIC_SOURCE_CONSISTENCY_COEFF="${FACTOR_SEMANTIC_SOURCE_CONSISTENCY_COEFF:-0.0}"
export FACTOR_SEMANTIC_DECORRELATION_COEFF="${FACTOR_SEMANTIC_DECORRELATION_COEFF:-0.005}"
export FACTOR_SEMANTIC_INJECTION_SCALE="${FACTOR_SEMANTIC_INJECTION_SCALE:-1.0}"
export FACTOR_SHARED_SELF_DISTILL_COEFF="${FACTOR_SHARED_SELF_DISTILL_COEFF:-0.05}"
export FACTOR_SHARED_VARIANCE_COEFF="${FACTOR_SHARED_VARIANCE_COEFF:-0.01}"
export FACTOR_SHARED_VARIANCE_TARGET="${FACTOR_SHARED_VARIANCE_TARGET:-1.0}"
export FACTOR_SHARED_SOURCE_DEPTH="${FACTOR_SHARED_SOURCE_DEPTH:-8}"
export FACTOR_SHARED_TARGET_TEMPERATURE="${FACTOR_SHARED_TARGET_TEMPERATURE:-0.25}"
export FACTOR_SHARED_SNR_POWER="${FACTOR_SHARED_SNR_POWER:-1.0}"
export FACTOR_SHARED_CONTRASTIVE_COEFF="${FACTOR_SHARED_CONTRASTIVE_COEFF:-0.05}"
export FACTOR_SHARED_CONTRASTIVE_TEMPERATURE="${FACTOR_SHARED_CONTRASTIVE_TEMPERATURE:-0.2}"
export FACTOR_SHARED_RELATION_COEFF="${FACTOR_SHARED_RELATION_COEFF:-0.05}"
export FACTOR_EVOLVING_SEPARATION_COEFF="${FACTOR_EVOLVING_SEPARATION_COEFF:-0.05}"
export FACTOR_EVOLVING_SEPARATION_MARGIN="${FACTOR_EVOLVING_SEPARATION_MARGIN:-0.5}"
export FACTOR_SELECTIVE_DIM="${FACTOR_SELECTIVE_DIM:-128}"
export FACTOR_SELECTIVE_SOURCE_DEPTH="${FACTOR_SELECTIVE_SOURCE_DEPTH:-8}"
export FACTOR_SELECTIVE_COEFF="${FACTOR_SELECTIVE_COEFF:-0.1}"
export FACTOR_SELECTIVE_ORTH_COEFF="${FACTOR_SELECTIVE_ORTH_COEFF:-0.01}"
export FACTOR_SELECTIVE_VARIANCE_COEFF="${FACTOR_SELECTIVE_VARIANCE_COEFF:-0.02}"
export FACTOR_SELECTIVE_VARIANCE_TARGET="${FACTOR_SELECTIVE_VARIANCE_TARGET:-1.0}"
export FACTOR_DIM="${FACTOR_DIM:-256}"
export FACTOR_PROJECTOR_DIM="${FACTOR_PROJECTOR_DIM:-1024}"
export FACTOR_SOURCE_DEPTH="${FACTOR_SOURCE_DEPTH:-8}"
export FACTOR_TARGET_DEPTH="${FACTOR_TARGET_DEPTH:-}"
export INVARIANT_DIM="${INVARIANT_DIM:-256}"
export INVARIANT_PROJECTOR_DIM="${INVARIANT_PROJECTOR_DIM:-1024}"
export INVARIANT_PROJECTOR_TYPE="${INVARIANT_PROJECTOR_TYPE:-linear}"
export INVARIANT_SOURCE_DEPTH="${INVARIANT_SOURCE_DEPTH:-4}"
export INVARIANT_BATCH_RATIO="${INVARIANT_BATCH_RATIO:-0.375}"
export INVARIANT_MIN_DELTA_T="${INVARIANT_MIN_DELTA_T:-0.05}"
export INVARIANT_MAX_DELTA_T="${INVARIANT_MAX_DELTA_T:-0.2}"
export INVARIANT_MAX_T="${INVARIANT_MAX_T:-0.8}"
export INVARIANT_SNR_POWER="${INVARIANT_SNR_POWER:-1.0}"
export INVARIANT_TIME_COEFF="${INVARIANT_TIME_COEFF:-0.1}"
export INVARIANT_NOISE_COEFF="${INVARIANT_NOISE_COEFF:-0.1}"
export INVARIANT_IMAGE_VARIANCE_COEFF="${INVARIANT_IMAGE_VARIANCE_COEFF:-0.02}"
export INVARIANT_SPATIAL_VARIANCE_COEFF="${INVARIANT_SPATIAL_VARIANCE_COEFF:-0.02}"
export INVARIANT_COVARIANCE_COEFF="${INVARIANT_COVARIANCE_COEFF:-0.001}"
export INVARIANT_BASIS_COEFF="${INVARIANT_BASIS_COEFF:-0.01}"
export INVARIANT_RELATION_COEFF="${INVARIANT_RELATION_COEFF:-0.05}"
export INVARIANT_VARIANCE_TARGET="${INVARIANT_VARIANCE_TARGET:-1.0}"
export INVARIANT_SPATIAL_VARIANCE_TARGET="${INVARIANT_SPATIAL_VARIANCE_TARGET:-0.5}"
export INVARIANT_WARMUP_STEPS="${INVARIANT_WARMUP_STEPS:-10000}"
export INVARIANT_DECAY_START="${INVARIANT_DECAY_START:--1}"
export INVARIANT_DECAY_END="${INVARIANT_DECAY_END:--1}"
export INVARIANT_MIN_LOSS_SCALE="${INVARIANT_MIN_LOSS_SCALE:-0}"
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
METRICS_TXT="${METRICS_TXT:-$SAMPLE_DIR/${FOLDER_NAME}_metrics.txt}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $name: $path" >&2
    exit 2
  fi
}

PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-none}"
if [[ "$EXP" == "s1_a5_shared_repa" ]]; then
  case "$(printf '%s' "$PROJECTOR_EMBED_DIMS" | tr '[:upper:]' '[:lower:]')" in
    ""|none|null)
      PROJECTOR_EMBED_DIMS=768
      ;;
  esac
fi
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
  s1_a5_shared_repa)
    if [[ "$projector_spec" == "none" || "$projector_spec" == "null" ]]; then
      echo "s1_a5_shared_repa eval needs PROJECTOR_EMBED_DIMS to match the trained REPA checkpoint" >&2
      exit 2
    fi
    MODEL_EXTRA+=(--trajectory-factorization)
    ;;
  s2_a5_shared_clean|s3_a5_shared_self_distill|s4_a5_shared_contrastive|s5_a5_shared_relation|s6_a5_private_separation|s7_a5_contrastive_private)
    MODEL_EXTRA+=(--trajectory-factorization)
    ;;
  t1_antithetic_pair)
    MODEL_EXTRA+=(--trajectory-factorization)
    ;;
  t0_native_fm_only|t2_native_source|t3_native_noise|t4_native_recomposition|t5_native_shuffled_source)
    MODEL_EXTRA+=(
      --trajectory-factorization
      --factor-native-parameterization
    )
    ;;
  u0_paired_repa)
    if [[ "$projector_spec" == "none" || "$projector_spec" == "null" ]]; then
      echo "u0_paired_repa eval needs PROJECTOR_EMBED_DIMS=768" >&2
      exit 2
    fi
    MODEL_EXTRA+=(--trajectory-factorization)
    ;;
  u1_scheduled_repa)
    if [[ "$projector_spec" == "none" || "$projector_spec" == "null" ]]; then
      echo "u1_scheduled_repa eval needs PROJECTOR_EMBED_DIMS=768" >&2
      exit 2
    fi
    ;;
  u2_selective_semantic|u3_semantic_no_injection|u4_semantic_no_a5|u5_semantic_shuffled_source)
    if [[ "$projector_spec" == "none" || "$projector_spec" == "null" ]]; then
      echo "$EXP eval needs PROJECTOR_EMBED_DIMS=768" >&2
      exit 2
    fi
    MODEL_EXTRA+=(
      --trajectory-factorization
      --factor-semantic-conditioning
      --factor-semantic-injection-scale "$FACTOR_SEMANTIC_INJECTION_SCALE"
    )
    ;;
  a9_orbit_consensus)
    MODEL_EXTRA+=(
      --trajectory-factorization
      --factor-velocity-recomposition
    )
    ;;
  a10_adv_time|a11_adv_orbit|a12_adv_purification|a13_adv_shuffled|a14_critic_only)
    MODEL_EXTRA+=(
      --trajectory-factorization
      --factor-velocity-recomposition
      --factor-adversarial
      --factor-adversarial-timestep-bins "$FACTOR_ADVERSARIAL_TIMESTEP_BINS"
    )
    ;;
  v0_a3_shared|v1_clean_consensus)
    MODEL_EXTRA+=(--trajectory-factorization)
    ;;
  v2_selective_uniform|v3_selective_stability|v4_vgsc|v5_vgsc_shuffled_source|v6_vgsc_shuffled_utility|v7_selective_task_only|v8_vgsc_weak)
    MODEL_EXTRA+=(
      --trajectory-factorization
      --factor-selective-invariance
    )
    ;;
  q0_invariant_three_view|q1_orbit_consistency|q2_orbit_spread|q3_orbit_full)
    MODEL_EXTRA+=(--trajectory-invariance)
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
  --factor-selective-dim "$FACTOR_SELECTIVE_DIM"
  --factor-selective-source-depth "$FACTOR_SELECTIVE_SOURCE_DEPTH"
  --invariant-dim "$INVARIANT_DIM"
  --invariant-projector-dim "$INVARIANT_PROJECTOR_DIM"
  --invariant-source-depth "$INVARIANT_SOURCE_DEPTH"
  --invariant-projector-type "$INVARIANT_PROJECTOR_TYPE"
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

activate_fid_env() {
  local primary_env="$FID_ENV"
  local fallback_env="$FID_FALLBACK_ENV"
  local active_env="$primary_env"
  local last_log
  local fid_probe="import tensorflow.compat.v1; import scipy.linalg; import tqdm.auto"
  last_log="$(mktemp)"
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

  tfcr_activate_env "$primary_env"
  if python -c "$fid_probe" >"$last_log" 2>&1; then
    echo "Using FID env: $primary_env"
    rm -f "$last_log"
    return
  fi

  if [[ "$fallback_env" != "$primary_env" && -d "$fallback_env" ]]; then
    echo "FID env $primary_env cannot import required metric dependencies; falling back to $fallback_env"
    tfcr_activate_env "$fallback_env"
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

run_fid() {
  if [[ "${FORCE_EVALUATE:-0}" != "1" && -f "$METRICS_TXT" ]]; then
    echo "Metrics already exist: $METRICS_TXT"
    echo "Set FORCE_EVALUATE=1 to recompute."
    return
  fi
  require_file reference_npz "$REF_NPZ"
  require_file sample_npz "$SAMPLE_NPZ"
  require_file inception_graph "$INCEPTION_V3_PATH"
  activate_fid_env
  cd "$REPO_DIR"
  echo "Computing metrics for $SAMPLE_NPZ"
  "${FID_CMD[@]}"
}

print_dry_run() {
  echo "EXP_NAME=$EXP_NAME"
  echo "CKPT=$CKPT"
  echo "SAMPLE_DIR=$SAMPLE_DIR"
  echo "SAMPLE_NPZ=$SAMPLE_NPZ"
  echo "METRICS_TXT=$METRICS_TXT"
  echo "INCEPTION_V3_PATH=$INCEPTION_V3_PATH"
  echo "FID_ENV=$FID_ENV"
  echo "FID_FALLBACK_ENV=$FID_FALLBACK_ENV"
  echo "AUTO_FIX_FID_ENV=$AUTO_FIX_FID_ENV"
  echo "FID_ENV_REPAIR_PACKAGES=$FID_ENV_REPAIR_PACKAGES"
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
    eval|test|posttrain)
      tfcr_print_cmd "${SAMPLE_CMD[@]}"
      tfcr_print_cmd "${PACKAGE_CMD[@]}"
      tfcr_print_cmd "${FID_CMD[@]}"
      ;;
    fid|metrics)
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
  eval|test|posttrain)
    require_file checkpoint "$CKPT"
    echo "Preflighting FID environment before sampling"
    activate_fid_env
    run_sample
    run_package
    run_fid
    ;;
  fid|metrics)
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
