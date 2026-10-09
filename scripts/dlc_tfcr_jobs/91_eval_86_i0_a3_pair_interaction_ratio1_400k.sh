#!/usr/bin/env bash
set -euo pipefail

# Post-train evaluation for Job 86. Reuses the existing 400k checkpoint and
# runs sample -> package npz -> compute metrics without restarting training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TFCR_JOB_RUN_STAGE=eval
export MASTER_PORT="${MASTER_PORT:-29591}"
export FID_ENV="${FID_ENV:-/root/anaconda3/envs/scale_rae}"
export AUTO_FIX_FID_ENV="${AUTO_FIX_FID_ENV:-1}"
export FORCE_PACKAGE="${FORCE_PACKAGE:-1}"
export FORCE_EVALUATE="${FORCE_EVALUATE:-1}"

exec bash "$SCRIPT_DIR/86_i0_a3_pair_interaction_ratio1_400k.sh"
