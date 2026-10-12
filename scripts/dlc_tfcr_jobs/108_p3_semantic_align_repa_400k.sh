#!/usr/bin/env bash
set -euo pipefail

# Proposal 3 + REPA: DINO aligns only the semantic bottleneck.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29608
source "$SCRIPT_DIR/common.sh"
export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export STEPS="${P3_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DIM="${P3_FACTOR_DIM:-128}"
export FACTOR_SEMANTIC_SOURCE_CONSISTENCY_COEFF="${P3_CONSISTENCY_COEFF:-0.1}"
export PROJECTOR_EMBED_DIMS=768
export RUN_SUFFIX="${RUN_SUFFIX:-j108-p3-semantic-align-repa-400k}"
run_tfcr_job p3_semantic_align_repa
