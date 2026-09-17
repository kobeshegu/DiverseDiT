#!/usr/bin/env bash
set -euo pipefail

# SF3: source EMA alignment without late-block source injection.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${SF3_STEPS:-400000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SELF_FLOW_SOURCE_COEFF="${FACTOR_SELF_FLOW_SOURCE_COEFF:-0.05}"
export FACTOR_SEMANTIC_INJECTION_SCALE=0
export RUN_SUFFIX="${RUN_SUFFIX:-j68-sf3-no-injection-ratio1-400k}"

run_tfcr_job sf3_no_injection
