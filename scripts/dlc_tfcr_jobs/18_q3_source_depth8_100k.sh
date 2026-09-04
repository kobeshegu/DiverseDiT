#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=100000
export CHECKPOINT_STEP=100000
export INVARIANT_SOURCE_DEPTH=8
export RUN_SUFFIX="${RUN_SUFFIX:-j18-depth8-screen100k}"

run_tfcr_job q3_orbit_full
