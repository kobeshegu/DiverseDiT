#!/usr/bin/env bash
set -euo pipefail

# I1: U0 paired REPA plus hidden pair-consensus interaction.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29587
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${I1_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export REPA_PROJ_COEFF="${REPA_PROJ_COEFF:-0.5}"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export FACTOR_PAIR_INTERACTION_SCALE="${FACTOR_PAIR_INTERACTION_SCALE:-1.0}"
export FACTOR_PAIR_INTERACTION_SELF_PROB="${FACTOR_PAIR_INTERACTION_SELF_PROB:-0.25}"
export RUN_SUFFIX="${RUN_SUFFIX:-j87-i1-u0-pair-interaction-repa-ratio1-coeff${REPA_PROJ_COEFF}-${STEPS}}"

run_tfcr_job i1_u0_pair_interaction_repa
