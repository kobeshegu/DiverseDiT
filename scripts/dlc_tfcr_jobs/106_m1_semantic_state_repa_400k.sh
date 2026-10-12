#!/usr/bin/env bash
set -euo pipefail

# M1-REPA: DINO supervises only the low-dimensional semantic subspace. The
# complementary state subspace remains trajectory-specific; cross-covariance,
# anti-collapse, and zero/swap recomposition make the split operational.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29606
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED="${SEED:-0}"
export STEPS="${M1_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DIM="${M1_FACTOR_DIM:-128}"
export FACTOR_PERSISTENT_COEFF="${M1_PERSISTENT_COEFF:-0.05}"
export FACTOR_EVOLVING_COEFF="${M1_EVOLVING_COEFF:-0.05}"
export FACTOR_RECOM_COEFF="${M1_RECOM_COEFF:-0.1}"
export FACTOR_SEMANTIC_REPA_COEFF="${M1_REPA_COEFF:-0.5}"
export FACTOR_SEMANTIC_SOURCE_CONSISTENCY_COEFF="${M1_CONSISTENCY_COEFF:-0.1}"
export FACTOR_SEMANTIC_CROSS_COVARIANCE_COEFF="${M1_CROSS_COV_COEFF:-0.005}"
export FACTOR_SEMANTIC_INTERVENTION_COEFF="${M1_INTERVENTION_COEFF:-0.05}"
export FACTOR_SEMANTIC_INTERVENTION_MARGIN="${M1_INTERVENTION_MARGIN:-0.1}"
export FACTOR_SEMANTIC_VARIANCE_COEFF="${M1_VARIANCE_COEFF:-0.02}"
export FACTOR_SEMANTIC_INJECTION_SCALE=0
export PROJECTOR_EMBED_DIMS=768
export RUN_SUFFIX="${RUN_SUFFIX:-j106-m1-semantic-state-repa-400k}"

run_tfcr_job m1_semantic_state_repa
