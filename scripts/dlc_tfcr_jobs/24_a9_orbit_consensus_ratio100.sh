#!/usr/bin/env bash
set -euo pipefail

# TFCR v2 main experiment: orthogonal two-view orbit, reliability-selected
# invariant target, and task-sufficient swapped velocity reconstruction.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=orthogonal
export FACTOR_ORBIT_NOISE_ONLY_PROB="${FACTOR_ORBIT_NOISE_ONLY_PROB:-0.5}"
export FACTOR_RELIABILITY_KEEP_RATIO="${FACTOR_RELIABILITY_KEEP_RATIO:-0.75}"
export FACTOR_VELOCITY_RECOM_COEFF="${FACTOR_VELOCITY_RECOM_COEFF:-0.05}"
export RUN_SUFFIX="${RUN_SUFFIX:-j24-orbit-consensus-ratio100}"

run_tfcr_job a9_orbit_consensus
