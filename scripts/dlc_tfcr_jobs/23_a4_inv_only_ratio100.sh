#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export SEED="${SEED:-0}"
export FACTOR_INV_COEFF="${FACTOR_INV_COEFF:-0.1}"
export FACTOR_BATCH_RATIO=1.0
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${RUN_SUFFIX:-j23-inv-only-ratio100}"

run_tfcr_job a4_inv_only
