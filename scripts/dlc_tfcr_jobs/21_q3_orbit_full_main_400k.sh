#!/usr/bin/env bash
set -euo pipefail

# Launch this long run only after the 100k Q0--Q3 gate and seed check pass.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export INVARIANT_DECAY_START=300000
export INVARIANT_DECAY_END=400000
export INVARIANT_MIN_LOSS_SCALE=0.1
export RUN_SUFFIX="${RUN_SUFFIX:-j21-main400k}"

run_tfcr_job q3_orbit_full
