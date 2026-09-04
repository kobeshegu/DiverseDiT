#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=100000
export CHECKPOINT_STEP=100000
export INVARIANT_NOISE_COEFF=0
export RUN_SUFFIX="${RUN_SUFFIX:-j16-time-only-screen100k}"

run_tfcr_job q1_orbit_consistency
