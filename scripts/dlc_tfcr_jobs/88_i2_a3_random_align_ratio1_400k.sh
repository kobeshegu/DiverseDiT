#!/usr/bin/env bash
set -euo pipefail

# I2: no-REPA paired trajectory with fixed random-projection readout alignment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29588
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${I2_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_PAIR_RANDOM_ALIGN_COEFF="${FACTOR_PAIR_RANDOM_ALIGN_COEFF:-0.05}"
export FACTOR_PAIR_RANDOM_VARIANCE_COEFF="${FACTOR_PAIR_RANDOM_VARIANCE_COEFF:-0.01}"
export FACTOR_PAIR_RANDOM_ALIGN_DIM="${FACTOR_PAIR_RANDOM_ALIGN_DIM:-256}"
export RUN_SUFFIX="${RUN_SUFFIX:-j88-i2-a3-random-align-ratio1-${STEPS}}"

run_tfcr_job i2_a3_random_align
