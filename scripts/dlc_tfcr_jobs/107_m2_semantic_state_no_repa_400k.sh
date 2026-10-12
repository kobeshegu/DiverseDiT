#!/usr/bin/env bash
set -euo pipefail

# M2-no-REPA: teacher-free counterpart. Same-source trajectory stability
# defines the semantic bottleneck; state sufficiency comes from recomposition
# and the zero/swap intervention objective.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29607
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED="${SEED:-0}"
export STEPS="${M2_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_DIM="${M2_FACTOR_DIM:-128}"
export FACTOR_PERSISTENT_COEFF="${M2_PERSISTENT_COEFF:-0.05}"
export FACTOR_EVOLVING_COEFF="${M2_EVOLVING_COEFF:-0.05}"
export FACTOR_RECOM_COEFF="${M2_RECOM_COEFF:-0.1}"
export FACTOR_SEMANTIC_REPA_COEFF=0
export FACTOR_SEMANTIC_SOURCE_CONSISTENCY_COEFF="${M2_CONSISTENCY_COEFF:-0.1}"
export FACTOR_SEMANTIC_CROSS_COVARIANCE_COEFF="${M2_CROSS_COV_COEFF:-0.005}"
export FACTOR_SEMANTIC_INTERVENTION_COEFF="${M2_INTERVENTION_COEFF:-0.05}"
export FACTOR_SEMANTIC_INTERVENTION_MARGIN="${M2_INTERVENTION_MARGIN:-0.1}"
export FACTOR_SEMANTIC_VARIANCE_COEFF="${M2_VARIANCE_COEFF:-0.02}"
export FACTOR_SEMANTIC_INJECTION_SCALE=0
export PROJECTOR_EMBED_DIMS=none
export RUN_SUFFIX="${RUN_SUFFIX:-j107-m2-semantic-state-no-repa-400k}"

run_tfcr_job m2_semantic_state_no_repa
