#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-1.0}"
export RUN_SUFFIX="${RUN_SUFFIX:-j06-cross-noise}"

run_tfcr_job a5_tfcr
