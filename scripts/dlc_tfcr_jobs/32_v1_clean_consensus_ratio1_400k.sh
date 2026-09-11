#!/usr/bin/env bash
set -euo pipefail

# Priority P0b: test whether an analytic clean-state consensus improves A3.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_CLEAN_CONSENSUS_COEFF="${FACTOR_CLEAN_CONSENSUS_COEFF:-0.05}"
export RUN_SUFFIX="${RUN_SUFFIX:-j32-v1-clean-consensus-ratio1-400k}"

run_tfcr_job v1_clean_consensus
