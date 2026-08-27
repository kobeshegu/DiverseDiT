#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_BATCH_RATIO=0.75
export RUN_SUFFIX="${RUN_SUFFIX:-j09-ratio075}"

run_tfcr_job a5_tfcr
