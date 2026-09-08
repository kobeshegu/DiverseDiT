#!/usr/bin/env bash
set -euo pipefail

# A9 + timestep purification from persistent and positive timestep probe on evolving.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=orthogonal
export FACTOR_ADV_PERSISTENT_ORBIT_COEFF=0
export FACTOR_PROBE_EVOLVING_ORBIT_COEFF=0
export RUN_SUFFIX="${RUN_SUFFIX:-j25-adv-time-ratio100}"

run_tfcr_job a10_adv_time
