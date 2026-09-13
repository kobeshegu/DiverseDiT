#!/usr/bin/env bash
set -euo pipefail

# S8: weaker S4 contrastive source-identity objective.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SHARED_CONTRASTIVE_COEFF="${S8_SHARED_CONTRASTIVE_COEFF:-0.025}"
export FACTOR_SHARED_CONTRASTIVE_TEMPERATURE="${S8_SHARED_CONTRASTIVE_TEMPERATURE:-0.2}"
export FACTOR_SHARED_SOURCE_DEPTH="${FACTOR_SHARED_SOURCE_DEPTH:-8}"
export RUN_SUFFIX="${RUN_SUFFIX:-j49-s8-s4-contrastive-weak-ratio1-400k}"

run_tfcr_job s4_a5_shared_contrastive
