#!/usr/bin/env bash
set -euo pipefail

# U3: source-subspace REPA is auxiliary only; no FiLM injection.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${U3_STEPS:-200000}"
export CHECKPOINT_STEP="$STEPS"
export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=legacy
export FACTOR_SEMANTIC_INJECTION_SCALE=0
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j62-u3-semantic-no-injection-ratio1-${STEPS}}"

run_tfcr_job u3_semantic_no_injection
