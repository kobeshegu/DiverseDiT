#!/usr/bin/env bash
set -euo pipefail

# SF2: main EMA source-branch method. Only the source branch aligns to the EMA
# teacher consensus; the evolving branch remains governed by the trajectory task.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${SF2_STEPS:-400000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SELF_FLOW_SOURCE_COEFF="${FACTOR_SELF_FLOW_SOURCE_COEFF:-0.05}"
export FACTOR_SEMANTIC_INJECTION_SCALE="${FACTOR_SEMANTIC_INJECTION_SCALE:-1.0}"
export RUN_SUFFIX="${RUN_SUFFIX:-j67-sf2-ema-source-align-ratio1-400k}"

run_tfcr_job sf2_ema_source_align
