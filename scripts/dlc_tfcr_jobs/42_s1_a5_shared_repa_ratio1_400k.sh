#!/usr/bin/env bash
set -euo pipefail

# Shared target S1: A5 TFCR plus a factor-scheduled external REPA target.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SHARED_REPA_COEFF="${FACTOR_SHARED_REPA_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j42-s1-a5-shared-repa-ratio1-400k}"

run_tfcr_job s1_a5_shared_repa
