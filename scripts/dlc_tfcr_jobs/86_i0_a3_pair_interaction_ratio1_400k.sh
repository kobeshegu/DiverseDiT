#!/usr/bin/env bash
set -euo pipefail

# I0: no-REPA paired trajectory with hidden pair-consensus interaction.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEFAULT_MASTER_PORT=29586
source "$SCRIPT_DIR/common.sh"

export RUN_STAGE="${TFCR_JOB_RUN_STAGE:-all}"
export SEED=0
export STEPS="${I0_STEPS:-${STEPS:-400000}}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export FACTOR_PAIR_INTERACTION_SCALE="${FACTOR_PAIR_INTERACTION_SCALE:-1.0}"
export FACTOR_PAIR_INTERACTION_SELF_PROB="${FACTOR_PAIR_INTERACTION_SELF_PROB:-0.25}"
export RUN_SUFFIX="${RUN_SUFFIX:-j86-i0-a3-pair-interaction-ratio1-${STEPS}}"

run_tfcr_job i0_a3_pair_interaction
