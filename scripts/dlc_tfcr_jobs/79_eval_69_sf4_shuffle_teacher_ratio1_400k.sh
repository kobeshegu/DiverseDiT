#!/usr/bin/env bash
set -euo pipefail

# Post-train only: sample -> package npz -> compute metrics for Job 69.
# Reuses the SF4 checkpoint and does not restart training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_STAGE=eval
export DEFAULT_MASTER_PORT="${DEFAULT_MASTER_PORT:-29579}"
export FID_ENV="${FID_ENV:-/root/anaconda3/envs/scale_rae}"
export AUTO_FIX_FID_ENV="${AUTO_FIX_FID_ENV:-1}"
export FORCE_PACKAGE="${FORCE_PACKAGE:-1}"
export FORCE_EVALUATE="${FORCE_EVALUATE:-1}"

exec bash "$SCRIPT_DIR/69_sf4_shuffle_teacher_ratio1_400k.sh"
