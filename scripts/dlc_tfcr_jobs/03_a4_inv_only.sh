#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_INV_COEFF="${FACTOR_INV_COEFF:-0.1}"
export FACTOR_BATCH_RATIO="${FACTOR_BATCH_RATIO:-0.5}"
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j03-inv-only}"

run_tfcr_job a4_inv_only
