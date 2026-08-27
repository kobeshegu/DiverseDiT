#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_BATCH_RATIO="${FACTOR_BATCH_RATIO:-0.25}"
export RUN_SUFFIX="${RUN_SUFFIX:-j07-ratio025}"

run_tfcr_job a5_tfcr
