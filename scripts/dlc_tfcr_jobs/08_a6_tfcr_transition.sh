#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_TRANSITION_COEFF="${FACTOR_TRANSITION_COEFF:-0.05}"
export RUN_SUFFIX="${RUN_SUFFIX:-j08-transition}"

run_tfcr_job a6_tfcr_transition
