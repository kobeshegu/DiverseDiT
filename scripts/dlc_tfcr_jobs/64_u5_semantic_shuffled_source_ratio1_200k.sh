#!/usr/bin/env bash
set -euo pipefail

# U5: causal control with incorrect clean-source semantic identities.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${U5_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j64-u5-semantic-shuffled-source-ratio1-${STEPS}}"

run_tfcr_job u5_semantic_shuffled_source
