#!/usr/bin/env bash
set -euo pipefail

# Priority P2a: causal negative control for correct source correspondence.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j36-v5-vgsc-shuffled-source-ratio1-400k}"

run_tfcr_job v5_vgsc_shuffled_source
