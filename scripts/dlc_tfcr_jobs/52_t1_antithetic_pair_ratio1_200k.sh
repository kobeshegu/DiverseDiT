#!/usr/bin/env bash
set -euo pipefail

# T1: compute-matched antithetic two-view control without native heads.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${T1_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=antithetic
# Keep the paired-view data construction active for the whole control run.
export FACTOR_WARMUP_STEPS=0
export FACTOR_MIN_LOSS_SCALE=1.0
export RUN_SUFFIX="${RUN_SUFFIX:-j52-t1-antithetic-pair-ratio1-${STEPS}}"

run_tfcr_job t1_antithetic_pair
