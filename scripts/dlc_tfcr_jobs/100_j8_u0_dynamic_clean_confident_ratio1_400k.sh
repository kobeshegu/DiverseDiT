#!/usr/bin/env bash
set -euo pipefail

# J8: J7 plus confidence filtering; only reliable trajectory pairs carry the auxiliary loss.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29600
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${J8_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export FACTOR_DYNAMIC_CLEAN_COEFF="${J8_FACTOR_DYNAMIC_CLEAN_COEFF:-0.025}"
export FACTOR_DYNAMIC_CLEAN_TARGET_MODE=hard
export FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE="${J8_FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE:-0.65}"
export RUN_SUFFIX="${RUN_SUFFIX:-j100-j8-u0-dynamic-clean-confident-ratio1-${STEPS}}"

run_tfcr_job j8_u0_dynamic_clean_confident
