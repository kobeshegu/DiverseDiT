#!/usr/bin/env bash
set -euo pipefail

# Shared target S5: A5 TFCR plus cross-view source/spatial relation matching.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SHARED_RELATION_COEFF="${FACTOR_SHARED_RELATION_COEFF:-0.05}"
export FACTOR_SHARED_VARIANCE_COEFF="${FACTOR_SHARED_VARIANCE_COEFF:-0.01}"
export FACTOR_SHARED_SOURCE_DEPTH="${FACTOR_SHARED_SOURCE_DEPTH:-8}"
export RUN_SUFFIX="${RUN_SUFFIX:-j46-s5-a5-shared-relation-ratio1-400k}"

run_tfcr_job s5_a5_shared_relation
