#!/usr/bin/env bash
set -euo pipefail

# Shared target S4: A5 TFCR plus trajectory-view contrastive source identity.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SHARED_CONTRASTIVE_COEFF="${FACTOR_SHARED_CONTRASTIVE_COEFF:-0.05}"
export FACTOR_SHARED_CONTRASTIVE_TEMPERATURE="${FACTOR_SHARED_CONTRASTIVE_TEMPERATURE:-0.2}"
export FACTOR_SHARED_SOURCE_DEPTH="${FACTOR_SHARED_SOURCE_DEPTH:-8}"
export RUN_SUFFIX="${RUN_SUFFIX:-j45-s4-a5-shared-contrastive-ratio1-400k}"

run_tfcr_job s4_a5_shared_contrastive
