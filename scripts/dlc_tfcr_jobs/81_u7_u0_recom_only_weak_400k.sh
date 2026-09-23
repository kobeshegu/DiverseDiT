#!/usr/bin/env bash
set -euo pipefail

# U7: U0 paired REPA plus weak recomposition only; no explicit split targets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29581
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${U7_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export FACTOR_RECOM_COEFF=0.03
export FACTOR_REGULARIZATION_START_STEPS=0
export RUN_SUFFIX="${RUN_SUFFIX:-j81-u7-u0-recom-only-weak-ratio1-${STEPS}}"

run_tfcr_job u7_u0_recom_only_weak
