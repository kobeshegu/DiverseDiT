#!/usr/bin/env bash
set -euo pipefail

# Proposal 2 + REPA: full-hidden standard REPA plus semantic/state readouts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29606
source "$SCRIPT_DIR/common.sh"
export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export STEPS="${P2_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DIM="${P2_FACTOR_DIM:-128}"
export PROJECTOR_EMBED_DIMS=768
export RUN_SUFFIX="${RUN_SUFFIX:-j106-p2-split-repa-400k}"
run_tfcr_job p2_split_repa
