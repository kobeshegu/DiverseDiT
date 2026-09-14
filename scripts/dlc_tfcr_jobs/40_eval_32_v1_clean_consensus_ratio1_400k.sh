#!/usr/bin/env bash
set -euo pipefail

# Post-train eval for Job 32: sample -> npz -> FID from the 400k checkpoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE=eval
export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_CLEAN_CONSENSUS_COEFF="${FACTOR_CLEAN_CONSENSUS_COEFF:-0.05}"
export RUN_SUFFIX="${RUN_SUFFIX:-j32-v1-clean-consensus-ratio1-400k}"

export FID_ENV="${FID_ENV:-/root/anaconda3/envs/scale_rae}"
export FID_FALLBACK_ENV="${FID_FALLBACK_ENV:-}"
export AUTO_FIX_FID_ENV="${AUTO_FIX_FID_ENV:-1}"

run_tfcr_job v1_clean_consensus
