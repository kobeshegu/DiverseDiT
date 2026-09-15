#!/usr/bin/env bash
set -euo pipefail

# T2: native velocity parameterization with only exact x0 supervision.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${T2_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=antithetic
export FACTOR_MIN_LOSS_SCALE=1.0
export RUN_SUFFIX="${RUN_SUFFIX:-j53-t2-native-source-ratio1-${STEPS}}"

run_tfcr_job t2_native_source
