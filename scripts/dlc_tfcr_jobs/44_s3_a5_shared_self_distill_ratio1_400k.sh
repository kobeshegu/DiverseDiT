#!/usr/bin/env bash
set -euo pipefail

# Shared target S3: A5 TFCR plus full-feature reliable self-distillation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_SHARED_SELF_DISTILL_COEFF="${FACTOR_SHARED_SELF_DISTILL_COEFF:-0.05}"
export FACTOR_SHARED_VARIANCE_COEFF="${FACTOR_SHARED_VARIANCE_COEFF:-0.01}"
export FACTOR_SHARED_SOURCE_DEPTH="${FACTOR_SHARED_SOURCE_DEPTH:-8}"
export FACTOR_SHARED_TARGET_TEMPERATURE="${FACTOR_SHARED_TARGET_TEMPERATURE:-0.25}"
export FACTOR_SHARED_SNR_POWER="${FACTOR_SHARED_SNR_POWER:-1.0}"
export RUN_SUFFIX="${RUN_SUFFIX:-j44-s3-a5-shared-self-distill-ratio1-400k}"

run_tfcr_job s3_a5_shared_self_distill
