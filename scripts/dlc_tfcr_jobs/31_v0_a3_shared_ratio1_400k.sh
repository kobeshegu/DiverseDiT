#!/usr/bin/env bash
set -euo pipefail

# Priority P0a: exact A3 data path plus shared classifier-free dropout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j31-v0-a3-shared-ratio1-400k}"

run_tfcr_job v0_a3_shared
