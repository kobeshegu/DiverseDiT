#!/usr/bin/env bash
set -euo pipefail

# T5: causal control that shuffles only the exact source-branch target.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${T5_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=antithetic
export FACTOR_MIN_LOSS_SCALE=1.0
export RUN_SUFFIX="${RUN_SUFFIX:-j56-t5-native-shuffled-source-ratio1-${STEPS}}"

run_tfcr_job t5_native_shuffled_source
