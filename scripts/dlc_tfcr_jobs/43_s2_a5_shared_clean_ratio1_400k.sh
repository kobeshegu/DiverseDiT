#!/usr/bin/env bash
set -euo pipefail

# Shared target S2: A5 TFCR plus confidence-weighted analytic clean consensus.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_CLEAN_CONSENSUS_COEFF="${FACTOR_CLEAN_CONSENSUS_COEFF:-0.05}"
export FACTOR_CLEAN_CONSENSUS_TEMPERATURE="${FACTOR_CLEAN_CONSENSUS_TEMPERATURE:-0.25}"
export RUN_SUFFIX="${RUN_SUFFIX:-j43-s2-a5-shared-clean-ratio1-400k}"

run_tfcr_job s2_a5_shared_clean
