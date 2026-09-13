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
FACTOR_CLEAN_CONSENSUS_TEMPERATURE="${FACTOR_CLEAN_CONSENSUS_TEMPERATURE:-0.25}"
FACTOR_CLEAN_CONSENSUS_COEFF="${FACTOR_CLEAN_CONSENSUS_COEFF:-0.05}"
FACTOR_SHARED_REPA_COEFF="${FACTOR_SHARED_REPA_COEFF:-0.5}"
FACTOR_SHARED_SELF_DISTILL_COEFF="${FACTOR_SHARED_SELF_DISTILL_COEFF:-0.05}"
FACTOR_SHARED_VARIANCE_COEFF="${FACTOR_SHARED_VARIANCE_COEFF:-0.01}"
FACTOR_SHARED_VARIANCE_TARGET="${FACTOR_SHARED_VARIANCE_TARGET:-1.0}"
FACTOR_SHARED_SOURCE_DEPTH="${FACTOR_SHARED_SOURCE_DEPTH:-8}"
FACTOR_SHARED_TARGET_TEMPERATURE="${FACTOR_SHARED_TARGET_TEMPERATURE:-0.25}"
FACTOR_SHARED_SNR_POWER="${FACTOR_SHARED_SNR_POWER:-1.0}"
FACTOR_SHARED_CONTRASTIVE_COEFF="${FACTOR_SHARED_CONTRASTIVE_COEFF:-0.05}"
FACTOR_SHARED_CONTRASTIVE_TEMPERATURE="${FACTOR_SHARED_CONTRASTIVE_TEMPERATURE:-0.2}"
FACTOR_SHARED_RELATION_COEFF="${FACTOR_SHARED_RELATION_COEFF:-0.05}"
FACTOR_EVOLVING_SEPARATION_COEFF="${FACTOR_EVOLVING_SEPARATION_COEFF:-0.05}"
FACTOR_EVOLVING_SEPARATION_MARGIN="${FACTOR_EVOLVING_SEPARATION_MARGIN:-0.5}"
FACTOR_SELECTIVE_DIM="${FACTOR_SELECTIVE_DIM:-128}"
FACTOR_SELECTIVE_SOURCE_DEPTH="${FACTOR_SELECTIVE_SOURCE_DEPTH:-8}"
FACTOR_SELECTIVE_COEFF="${FACTOR_SELECTIVE_COEFF:-0.1}"
FACTOR_SELECTIVE_ORTH_COEFF="${FACTOR_SELECTIVE_ORTH_COEFF:-0.01}"
FACTOR_SELECTIVE_VARIANCE_COEFF="${FACTOR_SELECTIVE_VARIANCE_COEFF:-0.02}"
FACTOR_SELECTIVE_VARIANCE_TARGET="${FACTOR_SELECTIVE_VARIANCE_TARGET:-1.0}"
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
  --factor-orbit-mode "$FACTOR_ORBIT_MODE"
  --factor-orbit-noise-only-prob "$FACTOR_ORBIT_NOISE_ONLY_PROB"
  --factor-reliability-keep-ratio "$FACTOR_RELIABILITY_KEEP_RATIO"
  --factor-reliability-floor "$FACTOR_RELIABILITY_FLOOR"
  --factor-adversarial-timestep-bins "$FACTOR_ADVERSARIAL_TIMESTEP_BINS"
  --factor-adversarial-grl-scale "$FACTOR_ADVERSARIAL_GRL_SCALE"
  --factor-adversarial-start-steps "$FACTOR_ADVERSARIAL_START_STEPS"
  --factor-adversarial-warmup-steps "$FACTOR_ADVERSARIAL_WARMUP_STEPS"
  --factor-clean-consensus-temperature "$FACTOR_CLEAN_CONSENSUS_TEMPERATURE"
  --factor-shared-source-depth "$FACTOR_SHARED_SOURCE_DEPTH"
  --factor-shared-target-temperature "$FACTOR_SHARED_TARGET_TEMPERATURE"
  --factor-shared-snr-power "$FACTOR_SHARED_SNR_POWER"
  --factor-shared-variance-target "$FACTOR_SHARED_VARIANCE_TARGET"
  --factor-shared-contrastive-temperature "$FACTOR_SHARED_CONTRASTIVE_TEMPERATURE"
  --factor-evolving-separation-margin "$FACTOR_EVOLVING_SEPARATION_MARGIN"
  --factor-selective-dim "$FACTOR_SELECTIVE_DIM"
  --factor-selective-source-depth "$FACTOR_SELECTIVE_SOURCE_DEPTH"
  --factor-selective-variance-target "$FACTOR_SELECTIVE_VARIANCE_TARGET"
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
  a9_orbit_consensus)
    EXTRA+=(
      --trajectory-factorization
      --factor-orbit-mode orthogonal
      --factor-share-cfg-dropout
      --factor-reliable-target
      --factor-velocity-recomposition
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-velocity-recom-coeff "$FACTOR_VELOCITY_RECOM_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  a10_adv_time)
    EXTRA+=(
      --trajectory-factorization
      --factor-orbit-mode orthogonal
      --factor-share-cfg-dropout
      --factor-reliable-target
      --factor-velocity-recomposition
      --factor-adversarial
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-velocity-recom-coeff "$FACTOR_VELOCITY_RECOM_COEFF"
      --factor-adv-persistent-time-coeff "$FACTOR_ADV_PERSISTENT_TIME_COEFF"
      --factor-adv-persistent-orbit-coeff 0
      --factor-probe-evolving-time-coeff "$FACTOR_PROBE_EVOLVING_TIME_COEFF"
      --factor-probe-evolving-orbit-coeff 0
    )
    ;;
  a11_adv_orbit)
    EXTRA+=(
      --trajectory-factorization
      --factor-orbit-mode orthogonal
      --factor-share-cfg-dropout
      --factor-reliable-target
      --factor-velocity-recomposition
      --factor-adversarial
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-velocity-recom-coeff "$FACTOR_VELOCITY_RECOM_COEFF"
      --factor-adv-persistent-time-coeff 0
      --factor-adv-persistent-orbit-coeff "$FACTOR_ADV_PERSISTENT_ORBIT_COEFF"
      --factor-probe-evolving-time-coeff 0
      --factor-probe-evolving-orbit-coeff "$FACTOR_PROBE_EVOLVING_ORBIT_COEFF"
    )
    ;;
  a12_adv_purification|a13_adv_shuffled|a14_critic_only)
    EXTRA+=(
      --trajectory-factorization
      --factor-orbit-mode orthogonal
      --factor-share-cfg-dropout
      --factor-reliable-target
      --factor-velocity-recomposition
      --factor-adversarial
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-velocity-recom-coeff "$FACTOR_VELOCITY_RECOM_COEFF"
      --factor-adv-persistent-time-coeff "$FACTOR_ADV_PERSISTENT_TIME_COEFF"
      --factor-adv-persistent-orbit-coeff "$FACTOR_ADV_PERSISTENT_ORBIT_COEFF"
      --factor-probe-evolving-time-coeff "$FACTOR_PROBE_EVOLVING_TIME_COEFF"
      --factor-probe-evolving-orbit-coeff "$FACTOR_PROBE_EVOLVING_ORBIT_COEFF"
    )
    if [[ "$EXP" == "a13_adv_shuffled" ]]; then
      EXTRA+=(--factor-adversarial-shuffle-labels)
    elif [[ "$EXP" == "a14_critic_only" ]]; then
      EXTRA+=(
        --factor-adversarial-grl-scale 0
        --factor-probe-evolving-time-coeff 0
        --factor-probe-evolving-orbit-coeff 0
      )
    fi
    ;;
  s1_a5_shared_repa)
    COMMON+=(--enc-type dinov2-vit-b --proj-coeff 0)
    EXTRA+=(
      --trajectory-factorization
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-shared-repa
      --factor-shared-repa-coeff "$FACTOR_SHARED_REPA_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  s2_a5_shared_clean)
    EXTRA+=(
      --trajectory-factorization
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-shared-clean
      --factor-shared-clean-coeff "$FACTOR_CLEAN_CONSENSUS_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  s3_a5_shared_self_distill)
    EXTRA+=(
      --trajectory-factorization
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-shared-self-distill
      --factor-shared-self-distill-coeff "$FACTOR_SHARED_SELF_DISTILL_COEFF"
      --factor-shared-variance-coeff "$FACTOR_SHARED_VARIANCE_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  s4_a5_shared_contrastive)
    EXTRA+=(
      --trajectory-factorization
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-shared-contrastive
      --factor-shared-contrastive-coeff "$FACTOR_SHARED_CONTRASTIVE_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  s5_a5_shared_relation)
    EXTRA+=(
      --trajectory-factorization
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-shared-relation
      --factor-shared-relation-coeff "$FACTOR_SHARED_RELATION_COEFF"
      --factor-shared-variance-coeff "$FACTOR_SHARED_VARIANCE_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  s6_a5_private_separation)
    EXTRA+=(
      --trajectory-factorization
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-evolving-separation
      --factor-evolving-separation-coeff "$FACTOR_EVOLVING_SEPARATION_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  s7_a5_contrastive_private)
    EXTRA+=(
      --trajectory-factorization
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff "$FACTOR_INV_COEFF"
      --factor-persistent-coeff "$FACTOR_PERSISTENT_COEFF"
      --factor-evolving-coeff "$FACTOR_EVOLVING_COEFF"
      --factor-recom-coeff "$FACTOR_RECOM_COEFF"
      --factor-shared-contrastive
      --factor-shared-contrastive-coeff "$FACTOR_SHARED_CONTRASTIVE_COEFF"
      --factor-evolving-separation
      --factor-evolving-separation-coeff "$FACTOR_EVOLVING_SEPARATION_COEFF"
      --factor-transition-coeff 0
    )
    ;;
  v0_a3_shared|v1_clean_consensus|v2_selective_uniform|v3_selective_stability|v4_vgsc|v5_vgsc_shuffled_source|v6_vgsc_shuffled_utility|v7_selective_task_only|v8_vgsc_weak)
    # All V-series experiments preserve A3's legacy paired-view sampler and
    # disable every historical persistent/evolving objective.  V0 controls
    # only for the shared CFG decision required by cross-view consistency.
    EXTRA+=(
      --trajectory-factorization
      --factor-paired-view-only
      --factor-share-cfg-dropout
      --factor-pair-cross-noise-prob "$CROSS_NOISE_PROB"
      --factor-inv-coeff 0
      --factor-persistent-coeff 0
      --factor-evolving-coeff 0
      --factor-recom-coeff 0
      --factor-velocity-recom-coeff 0
      --factor-adv-persistent-time-coeff 0
      --factor-adv-persistent-orbit-coeff 0
      --factor-probe-evolving-time-coeff 0
      --factor-probe-evolving-orbit-coeff 0
      --factor-transition-coeff 0
      --factor-decorrelation-coeff 0
      --factor-variance-coeff 0
    )
    if [[ "$EXP" != "v0_a3_shared" && "$EXP" != "v7_selective_task_only" ]]; then
      EXTRA+=(
        --factor-clean-consensus
        --factor-clean-consensus-coeff "$FACTOR_CLEAN_CONSENSUS_COEFF"
      )
    fi
    if [[ "$EXP" != "v0_a3_shared" && "$EXP" != "v1_clean_consensus" ]]; then
      EXTRA+=(
        --factor-selective-invariance
        --factor-selective-coeff "$FACTOR_SELECTIVE_COEFF"
        --factor-selective-orth-coeff "$FACTOR_SELECTIVE_ORTH_COEFF"
        --factor-selective-variance-coeff "$FACTOR_SELECTIVE_VARIANCE_COEFF"
      )
      case "$EXP" in
        v2_selective_uniform)
          EXTRA+=(--factor-selective-weighting uniform)
          ;;
        v3_selective_stability)
          EXTRA+=(--factor-selective-weighting stability)
          ;;
        v4_vgsc|v7_selective_task_only|v8_vgsc_weak)
          EXTRA+=(--factor-selective-weighting task)
          ;;
        v5_vgsc_shuffled_source)
          EXTRA+=(
            --factor-selective-weighting task
            --factor-selective-shuffle-targets
          )
          ;;
        v6_vgsc_shuffled_utility)
          EXTRA+=(
            --factor-selective-weighting task
            --factor-selective-shuffle-utility
          )
          ;;
      esac
    fi
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
