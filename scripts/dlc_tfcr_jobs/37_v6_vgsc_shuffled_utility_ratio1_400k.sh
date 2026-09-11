#!/usr/bin/env bash
set -euo pipefail

# Priority P2b: causal negative control for task-relevance assignment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j37-v6-vgsc-shuffled-utility-ratio1-400k}"

run_tfcr_job v6_vgsc_shuffled_utility
