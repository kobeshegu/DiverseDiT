#!/usr/bin/env bash
set -euo pipefail

# J9: J6 with late-start dynamic clean consistency to avoid early self-target noise.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29601
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${J9_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export FACTOR_DYNAMIC_CLEAN_COEFF="${J9_FACTOR_DYNAMIC_CLEAN_COEFF:-0.025}"
export FACTOR_DYNAMIC_CLEAN_TARGET_MODE=soft
export FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE=0.0
export FACTOR_REGULARIZATION_START_STEPS="${J9_FACTOR_REGULARIZATION_START_STEPS:-100000}"
export RUN_SUFFIX="${RUN_SUFFIX:-j101-j9-u0-dynamic-clean-late100k-ratio1-${STEPS}}"

run_tfcr_job j9_u0_dynamic_clean_late
