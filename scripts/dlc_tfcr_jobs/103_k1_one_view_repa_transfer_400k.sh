#!/usr/bin/env bash
set -euo pipefail

# K1: only one paired trajectory view receives the external DINO REPA target;
# the other view learns through FM plus late asymmetric clean-space transfer.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29603
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${K1_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export FACTOR_PROJ_TARGET_VIEW_MODE="${K1_FACTOR_PROJ_TARGET_VIEW_MODE:-first}"
export FACTOR_DYNAMIC_CLEAN_COEFF="${K1_FACTOR_DYNAMIC_CLEAN_COEFF:-0.025}"
export FACTOR_DYNAMIC_CLEAN_TARGET_MODE=hard
export FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE="${K1_FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE:-0.65}"
export FACTOR_REGULARIZATION_START_STEPS="${K1_FACTOR_REGULARIZATION_START_STEPS:-100000}"
export RUN_SUFFIX="${RUN_SUFFIX:-j103-k1-one-view-repa-transfer-ratio1-${STEPS}}"

run_tfcr_job k1_one_view_repa_transfer
