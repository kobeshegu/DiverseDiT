#!/usr/bin/env bash
set -euo pipefail

# Proposal 4 + REPA: semantic-only DINO alignment plus cross-covariance,
# anti-collapse, and zero/swap necessity.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29610
source "$SCRIPT_DIR/common.sh"
export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export STEPS="${P4_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DIM="${P4_FACTOR_DIM:-128}"
export FACTOR_SEMANTIC_SOURCE_CONSISTENCY_COEFF="${P4_CONSISTENCY_COEFF:-0.1}"
export PROJECTOR_EMBED_DIMS=768
export RUN_SUFFIX="${RUN_SUFFIX:-j110-p4-identifiable-repa-400k}"
run_tfcr_job p4_identifiable_repa
