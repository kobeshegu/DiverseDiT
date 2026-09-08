#!/usr/bin/env bash
set -euo pipefail

# Diagnostic control: persistent critics learn real labels but GRL=0 blocks
# their gradients from changing persistent features; evolving probes are off.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export FACTOR_BATCH_RATIO=1.0
export FACTOR_ORBIT_MODE=orthogonal
export FACTOR_ADVERSARIAL_GRL_SCALE=0
export RUN_SUFFIX="${RUN_SUFFIX:-j29-critic-only-ratio100}"

run_tfcr_job a14_critic_only
