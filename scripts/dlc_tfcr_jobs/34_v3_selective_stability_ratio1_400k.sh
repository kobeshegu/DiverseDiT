#!/usr/bin/env bash
set -euo pipefail

# Priority P1a: ablate FM-gradient utility while retaining source stability.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j34-v3-selective-stability-ratio1-400k}"

run_tfcr_job v3_selective_stability
