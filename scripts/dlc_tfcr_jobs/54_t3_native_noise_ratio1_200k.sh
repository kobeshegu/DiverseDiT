#!/usr/bin/env bash
set -euo pipefail

# T3: native velocity parameterization with epsilon/antisymmetry supervision.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${T3_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=antithetic
export FACTOR_MIN_LOSS_SCALE=1.0
export RUN_SUFFIX="${RUN_SUFFIX:-j54-t3-native-noise-ratio1-${STEPS}}"

run_tfcr_job t3_native_noise
