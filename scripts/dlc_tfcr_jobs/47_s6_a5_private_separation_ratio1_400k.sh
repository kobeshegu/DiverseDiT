#!/usr/bin/env bash
set -euo pipefail

# Shared-private S6: A5 TFCR plus a margin that keeps evolving codes view-specific.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_EVOLVING_SEPARATION_COEFF="${FACTOR_EVOLVING_SEPARATION_COEFF:-0.05}"
export FACTOR_EVOLVING_SEPARATION_MARGIN="${FACTOR_EVOLVING_SEPARATION_MARGIN:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j47-s6-a5-private-separation-ratio1-400k}"

run_tfcr_job s6_a5_private_separation
