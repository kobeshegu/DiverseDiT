#!/usr/bin/env bash
set -euo pipefail

# J2: orbit-aware dynamic consistency over recovered x0, epsilon, and velocity.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29594
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${J2_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=orthogonal
export FACTOR_ORBIT_NOISE_ONLY_PROB="${FACTOR_ORBIT_NOISE_ONLY_PROB:-0.5}"
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DYNAMIC_ORBIT_CLEAN_COEFF="${FACTOR_DYNAMIC_ORBIT_CLEAN_COEFF:-0.05}"
export FACTOR_DYNAMIC_ORBIT_NOISE_COEFF="${FACTOR_DYNAMIC_ORBIT_NOISE_COEFF:-0.025}"
export FACTOR_DYNAMIC_ORBIT_VELOCITY_COEFF="${FACTOR_DYNAMIC_ORBIT_VELOCITY_COEFF:-0.025}"
export RUN_SUFFIX="${RUN_SUFFIX:-j94-j2-dynamic-orbit-ratio1-${STEPS}}"

run_tfcr_job j2_dynamic_orbit
