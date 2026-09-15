#!/usr/bin/env bash

tfcr_activate_env() {
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

fcr_activate_env() {
  tfcr_activate_env "$@"
}

tfcr_experiment_name() {
  local exp="$1"
  local model="${MODEL:-SiT-B/2}"
  local seed="${SEED:-0}"
  local run_suffix="${RUN_SUFFIX:-}"
  local exp_name="${exp}-${model//\//-}-s${seed}"

  case "$exp" in
    a3_two_view|a4_inv_only|a5_tfcr|a6_tfcr_transition|a7_tfcr_diversedit|a8_tfcr_repa|a9_orbit_consensus|a10_adv_time|a11_adv_orbit|a12_adv_purification|a13_adv_shuffled|a14_critic_only|s1_a5_shared_repa|s2_a5_shared_clean|s3_a5_shared_self_distill|s4_a5_shared_contrastive|s5_a5_shared_relation|s6_a5_private_separation|s7_a5_contrastive_private|v0_a3_shared|v1_clean_consensus|v2_selective_uniform|v3_selective_stability|v4_vgsc|v5_vgsc_shuffled_source|v6_vgsc_shuffled_utility|v7_selective_task_only|v8_vgsc_weak)
      exp_name+="-r${FACTOR_BATCH_RATIO:-0.5}-x${CROSS_NOISE_PROB:-0.5}"
      ;;
    t0_native_fm_only|t1_antithetic_pair|t2_native_source|t3_native_noise|t4_native_recomposition|t5_native_shuffled_source)
      exp_name+="-r${FACTOR_BATCH_RATIO:-1.0}-antithetic"
      ;;
    u0_paired_repa)
      exp_name+="-r${FACTOR_BATCH_RATIO:-1.0}-paired-repa"
      ;;
    u2_selective_semantic|u3_semantic_no_injection|u4_semantic_no_a5|u5_semantic_shuffled_source)
      exp_name+="-r${FACTOR_BATCH_RATIO:-1.0}-semantic"
      ;;
    u1_scheduled_repa)
      exp_name+="-scheduled"
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
