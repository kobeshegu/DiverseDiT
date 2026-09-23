#!/usr/bin/env bash
set -euo pipefail

# U0 view-diversity probe: always pair different noise views.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29578
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${U0_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB=1.0
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j78-u0-paired-repa-seed0-ratio1-x1-coeff${REPA_PROJ_COEFF}-${STEPS}}"

run_tfcr_job u0_paired_repa
