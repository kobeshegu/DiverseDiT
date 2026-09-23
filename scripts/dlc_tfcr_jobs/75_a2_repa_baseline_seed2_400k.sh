#!/usr/bin/env bash
set -euo pipefail

# A2 seed repeat: standard REPA baseline matched to U0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29575
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=2
export STEPS="${A2_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j75-repa-baseline-seed2-coeff${REPA_PROJ_COEFF}-${STEPS}}"

run_tfcr_job a2_repa
