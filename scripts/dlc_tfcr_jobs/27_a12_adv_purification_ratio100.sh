#!/usr/bin/env bash
set -euo pipefail

# Main adversarial experiment: remove timestep/orbit leakage from persistent,
# relocate both signals to evolving, and retain A9 task sufficiency.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=orthogonal
export RUN_SUFFIX="${RUN_SUFFIX:-j27-adv-purification-ratio100}"

run_tfcr_job a12_adv_purification
