#!/usr/bin/env bash
set -euo pipefail

# Post-train evaluation for Job 87. Reuses the existing 400k checkpoint and
# runs sample -> package npz -> compute metrics without restarting training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TFCR_JOB_RUN_STAGE=eval
export MASTER_PORT="${MASTER_PORT:-29592}"
export FID_ENV="${FID_ENV:-/root/anaconda3/envs/scale_rae}"
export AUTO_FIX_FID_ENV="${AUTO_FIX_FID_ENV:-1}"
export FORCE_PACKAGE="${FORCE_PACKAGE:-1}"
export FORCE_EVALUATE="${FORCE_EVALUATE:-1}"

exec bash "$SCRIPT_DIR/87_i1_u0_pair_interaction_repa_ratio1_400k.sh"
