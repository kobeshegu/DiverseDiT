#!/usr/bin/env bash
set -euo pipefail

# T0: native two-head architecture trained only by recomposed flow matching.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${T0_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=antithetic
export FACTOR_MIN_LOSS_SCALE=1.0
export RUN_SUFFIX="${RUN_SUFFIX:-j57-t0-native-fm-only-ratio1-${STEPS}}"

run_tfcr_job t0_native_fm_only
