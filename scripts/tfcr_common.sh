#!/usr/bin/env bash

tfcr_activate_env() {
  local env_name="$1"
  if [[ -z "$env_name" ]]; then
    return
  fi
  if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
  fi
  conda activate "$env_name"
}

tfcr_experiment_name() {
  local exp="$1"
  local model="${MODEL:-SiT-B/2}"
  local seed="${SEED:-0}"
  local run_suffix="${RUN_SUFFIX:-}"
  local exp_name="${exp}-${model//\//-}-s${seed}"

  case "$exp" in
    a3_two_view|a4_inv_only|a5_tfcr|a6_tfcr_transition|a7_tfcr_diversedit|a8_tfcr_repa|a9_orbit_consensus|a10_adv_time|a11_adv_orbit|a12_adv_purification|a13_adv_shuffled|a14_critic_only|v0_a3_shared|v1_clean_consensus|v2_selective_uniform|v3_selective_stability|v4_vgsc|v5_vgsc_shuffled_source|v6_vgsc_shuffled_utility)
      exp_name+="-r${FACTOR_BATCH_RATIO:-0.5}-x${CROSS_NOISE_PROB:-0.5}"
      ;;
    q0_invariant_three_view|q1_orbit_consistency|q2_orbit_spread|q3_orbit_full)
      exp_name+="-r${INVARIANT_BATCH_RATIO:-0.375}-${INVARIANT_PROJECTOR_TYPE:-linear}"
      ;;
  esac
  if [[ -n "$run_suffix" ]]; then
    exp_name+="-${run_suffix}"
  fi

  printf '%s\n' "$exp_name"
}

tfcr_archive_code() {
  local repo_dir="$1"
  local output_dir="$2"
  local exp_name="$3"
  local phase="${4:-run}"

  if [[ "${ARCHIVE_CODE:-1}" != "1" || "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi

  local timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
  local archive_dir="${CODE_ARCHIVE_DIR:-$output_dir/$exp_name/code_snapshots/${timestamp}_${phase}}"
  mkdir -p "$archive_dir"

  local items=(
    train.py
    generate.py
    npz_convert.py
    evaluator.py
    evaluator_tf.py
    dataset.py
    loss.py
    samplers.py
    samplers_t2i.py
    utils.py
    requirements.txt
    README.md
    AGENTS.md
    models
    dinov2
    preprocessing
    analysis
    docs
    scripts
  )

  if command -v rsync >/dev/null 2>&1; then
    for item in "${items[@]}"; do
      if [[ -e "$repo_dir/$item" ]]; then
        rsync -a \
          --exclude __pycache__ \
          --exclude '*/__pycache__' \
          --exclude '*.pyc' \
          "$repo_dir/$item" "$archive_dir"/
      fi
    done
  else
    for item in "${items[@]}"; do
      if [[ -e "$repo_dir/$item" ]]; then
        cp -a "$repo_dir/$item" "$archive_dir"/
      fi
    done
  fi

  {
    printf 'Repository: %s\n' "$repo_dir"
    printf 'Experiment: %s\n' "$exp_name"
    printf 'Phase: %s\n' "$phase"
    printf 'Timestamp: %s\n' "$timestamp"
    printf 'Git commit: '
    git -C "$repo_dir" rev-parse HEAD 2>/dev/null || true
    printf '\nGit status --short:\n'
    git -C "$repo_dir" status --short 2>/dev/null || true
  } > "$archive_dir/RUN_CONTEXT.txt"

  echo "Archived code: $archive_dir"
}

tfcr_print_cmd() {
  printf '%q' "$1"
  shift
  printf ' %q' "$@"
  printf '\n'
}
