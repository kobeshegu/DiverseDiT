#!/usr/bin/env bash
set -euo pipefail

# I3: no-REPA paired trajectory with BYOL-style projected readout alignment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29589
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${I3_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_PAIR_BYOL_ALIGN_COEFF="${FACTOR_PAIR_BYOL_ALIGN_COEFF:-0.05}"
export FACTOR_PAIR_BYOL_VARIANCE_COEFF="${FACTOR_PAIR_BYOL_VARIANCE_COEFF:-0.01}"
export FACTOR_PAIR_ALIGNMENT_DIM="${FACTOR_PAIR_ALIGNMENT_DIM:-256}"
export FACTOR_PAIR_ALIGNMENT_PREDICTOR_DIM="${FACTOR_PAIR_ALIGNMENT_PREDICTOR_DIM:-1024}"
export RUN_SUFFIX="${RUN_SUFFIX:-j89-i3-a3-byol-align-ratio1-${STEPS}}"

run_tfcr_job i3_a3_byol_align
