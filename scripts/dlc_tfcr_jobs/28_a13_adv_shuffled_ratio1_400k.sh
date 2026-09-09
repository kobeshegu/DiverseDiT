#!/usr/bin/env bash
set -euo pipefail

# Random-label control: preserves heads, losses, and compute without meaningful
# nuisance supervision.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=orthogonal
export RUN_SUFFIX="${RUN_SUFFIX:-j28-adv-shuffled-ratio1-400k}"

run_tfcr_job a13_adv_shuffled
