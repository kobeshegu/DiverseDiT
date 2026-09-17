#!/usr/bin/env bash
set -euo pipefail

# SF1: Self-Flow/SRA-style baseline. Align online full hidden features to
# a reliability-weighted EMA teacher consensus; no explicit source injection.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${SF1_STEPS:-400000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SELF_FLOW_FULL_COEFF="${FACTOR_SELF_FLOW_FULL_COEFF:-0.05}"
export RUN_SUFFIX="${RUN_SUFFIX:-j66-sf1-ema-full-align-ratio1-400k}"

run_tfcr_job sf1_ema_full_align
