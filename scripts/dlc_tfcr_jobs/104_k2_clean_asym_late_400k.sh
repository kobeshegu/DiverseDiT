#!/usr/bin/env bash
set -euo pipefail

# K2: no external REPA target; only late clean-space asymmetric consistency.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29604
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${K2_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export PROJECTOR_EMBED_DIMS=none
export FACTOR_DYNAMIC_CLEAN_COEFF="${K2_FACTOR_DYNAMIC_CLEAN_COEFF:-0.025}"
export FACTOR_DYNAMIC_CLEAN_TARGET_MODE=hard
export FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE="${K2_FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE:-0.65}"
export FACTOR_REGULARIZATION_START_STEPS="${K2_FACTOR_REGULARIZATION_START_STEPS:-100000}"
export RUN_SUFFIX="${RUN_SUFFIX:-j104-k2-clean-asym-late-ratio1-${STEPS}}"

run_tfcr_job k2_clean_asym_late
