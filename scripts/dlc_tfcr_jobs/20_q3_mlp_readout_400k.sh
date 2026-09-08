#!/usr/bin/env bash
set -euo pipefail

# Expressive-readout control: if only this setting aligns while the linear
# subspace does not, the auxiliary head is absorbing the objective.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export INVARIANT_PROJECTOR_TYPE=mlp
export INVARIANT_BASIS_COEFF=0
export RUN_SUFFIX="${RUN_SUFFIX:-j20-mlp-readout-full400k}"

run_tfcr_job q3_orbit_full
