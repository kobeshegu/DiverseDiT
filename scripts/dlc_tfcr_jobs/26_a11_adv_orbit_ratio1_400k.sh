#!/usr/bin/env bash
set -euo pipefail

# A9 + orbit-type purification from persistent and positive orbit probe on evolving.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=orthogonal
export FACTOR_ADV_PERSISTENT_TIME_COEFF=0
export FACTOR_PROBE_EVOLVING_TIME_COEFF=0
export RUN_SUFFIX="${RUN_SUFFIX:-j26-adv-orbit-ratio1-400k}"

run_tfcr_job a11_adv_orbit
