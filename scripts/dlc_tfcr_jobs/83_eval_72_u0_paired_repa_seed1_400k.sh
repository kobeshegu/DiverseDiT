#!/usr/bin/env bash
set -euo pipefail

# Post-train evaluation for Job 72. Reuses the existing 400k checkpoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TFCR_JOB_RUN_STAGE=eval

exec bash "$SCRIPT_DIR/72_u0_paired_repa_seed1_400k.sh"
