#!/usr/bin/env bash
set -euo pipefail

# T4 main: exact x0/epsilon factors directly recompose the sampled velocity.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${T4_STEPS:-400000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=antithetic
export FACTOR_MIN_LOSS_SCALE=1.0
export RUN_SUFFIX="${RUN_SUFFIX:-j55-t4-native-recomposition-ratio1-${STEPS}}"

run_tfcr_job t4_native_recomposition
