#!/usr/bin/env bash
set -euo pipefail

# J5: train and sample with a self-predicted clean-state guidance token.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29597
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${J5_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DYNAMIC_GUIDANCE_COEFF="${FACTOR_DYNAMIC_GUIDANCE_COEFF:-0.05}"
export FACTOR_DYNAMIC_GUIDANCE_SCALE="${FACTOR_DYNAMIC_GUIDANCE_SCALE:-1.0}"
export RUN_SUFFIX="${RUN_SUFFIX:-j97-j5-dynamic-guidance-ratio1-${STEPS}}"

run_tfcr_job j5_dynamic_guidance
