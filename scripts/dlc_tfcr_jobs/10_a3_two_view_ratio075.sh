#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export SEED="${SEED:-0}"
export FACTOR_BATCH_RATIO=0.75
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j10-two-view-ratio075}"

run_tfcr_job a3_two_view
