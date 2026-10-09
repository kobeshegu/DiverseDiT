#!/usr/bin/env bash
set -euo pipefail

# J3: cross-view transport supervision using one view's recovered clean state.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29595
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${J3_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DYNAMIC_TRANSPORT_COEFF="${FACTOR_DYNAMIC_TRANSPORT_COEFF:-0.05}"
export RUN_SUFFIX="${RUN_SUFFIX:-j95-j3-dynamic-transport-ratio1-${STEPS}}"

run_tfcr_job j3_dynamic_transport
