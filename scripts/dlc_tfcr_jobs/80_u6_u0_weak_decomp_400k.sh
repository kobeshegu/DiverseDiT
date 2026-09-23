#!/usr/bin/env bash
set -euo pipefail

# U6: U0 paired REPA plus weak decomposition regularization.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29580
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${U6_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export FACTOR_INV_COEFF=0.02
export FACTOR_PERSISTENT_COEFF=0.01
export FACTOR_EVOLVING_COEFF=0.01
export FACTOR_RECOM_COEFF=0.02
export FACTOR_REGULARIZATION_START_STEPS=0
export RUN_SUFFIX="${RUN_SUFFIX:-j80-u6-u0-weak-decomp-ratio1-${STEPS}}"

run_tfcr_job u6_u0_weak_decomp
