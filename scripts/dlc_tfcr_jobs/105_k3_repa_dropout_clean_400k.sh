#!/usr/bin/env bash
set -euo pipefail

# K3: bootstrap with paired REPA, then progressively drop the external target
# while late asymmetric clean-space consistency remains active.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29605
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${K3_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export FACTOR_DYNAMIC_CLEAN_COEFF="${K3_FACTOR_DYNAMIC_CLEAN_COEFF:-0.025}"
export FACTOR_DYNAMIC_CLEAN_TARGET_MODE=hard
export FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE="${K3_FACTOR_DYNAMIC_CLEAN_MIN_CONFIDENCE:-0.65}"
export FACTOR_REGULARIZATION_START_STEPS="${K3_FACTOR_REGULARIZATION_START_STEPS:-100000}"
export PROJ_TARGET_DROPOUT_START="${K3_PROJ_TARGET_DROPOUT_START:-100000}"
export PROJ_TARGET_DROPOUT_END="${K3_PROJ_TARGET_DROPOUT_END:-250000}"
export PROJ_TARGET_DROPOUT_PROB="${K3_PROJ_TARGET_DROPOUT_PROB:-1.0}"
export RUN_SUFFIX="${RUN_SUFFIX:-j105-k3-repa-dropout-clean-ratio1-${STEPS}}"

run_tfcr_job k3_repa_dropout_clean
