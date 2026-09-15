#!/usr/bin/env bash
set -euo pipefail

# U1: single-view REPA with S1's warmup/decay schedule.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS="${U1_STEPS:-400000}"
export CHECKPOINT_STEP="$STEPS"
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j59-u1-scheduled-repa-${STEPS}}"

run_tfcr_job u1_scheduled_repa
