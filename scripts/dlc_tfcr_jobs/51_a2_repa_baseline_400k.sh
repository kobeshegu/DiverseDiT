#!/usr/bin/env bash
set -euo pipefail

# REPA baseline: standard SiT training with a DINOv2-B projection target.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export STEPS=400000
export CHECKPOINT_STEP=400000
export PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
export RUN_SUFFIX="${RUN_SUFFIX:-j51-repa-baseline-400k}"

run_tfcr_job a2_repa
