#!/usr/bin/env bash
set -euo pipefail

# Priority P0c / main: source stability x velocity utility selective invariance.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SELECTIVE_DIM="${FACTOR_SELECTIVE_DIM:-128}"
export FACTOR_SELECTIVE_SOURCE_DEPTH="${FACTOR_SELECTIVE_SOURCE_DEPTH:-8}"
export RUN_SUFFIX="${RUN_SUFFIX:-j33-v4-vgsc-ratio1-400k}"

run_tfcr_job v4_vgsc
