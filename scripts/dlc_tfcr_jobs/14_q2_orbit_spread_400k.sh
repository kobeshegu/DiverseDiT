#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export RUN_SUFFIX="${RUN_SUFFIX:-j14-full400k}"

run_tfcr_job q2_orbit_spread
