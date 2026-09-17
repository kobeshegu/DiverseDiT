#!/usr/bin/env bash
set -euo pipefail

# Post-train only: sample -> package npz -> compute metrics for Job 61.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_STAGE=eval
exec bash "$SCRIPT_DIR/61_u2_selective_semantic_ratio1_400k.sh"
