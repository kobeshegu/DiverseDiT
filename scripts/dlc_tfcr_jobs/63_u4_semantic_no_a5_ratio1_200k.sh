#!/usr/bin/env bash
set -euo pipefail

# U4: semantic source FiLM without historical A5 decomposition losses.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${U4_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j63-u4-semantic-no-a5-ratio1-${STEPS}}"

run_tfcr_job u4_semantic_no_a5
