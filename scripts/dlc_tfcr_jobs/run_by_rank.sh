#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 11))"

# These jobs are independent single-card experiments, not one distributed run.
export NUM_PROCESSES=1
export NPROC=1
export NUM_MACHINES=1
export MACHINE_RANK=0

case "$TASK_INDEX" in
  0) exec bash "$SCRIPT_DIR/01_a0_sit_baseline.sh" ;;
  1) exec bash "$SCRIPT_DIR/02_a3_two_view_control.sh" ;;
  2) exec bash "$SCRIPT_DIR/03_a4_inv_only.sh" ;;
  3) exec bash "$SCRIPT_DIR/04_a5_tfcr_default.sh" ;;
  4) exec bash "$SCRIPT_DIR/05_a5_tfcr_same_noise.sh" ;;
  5) exec bash "$SCRIPT_DIR/06_a5_tfcr_cross_noise.sh" ;;
  6) exec bash "$SCRIPT_DIR/07_a5_tfcr_ratio025.sh" ;;
  7) exec bash "$SCRIPT_DIR/08_a6_tfcr_transition.sh" ;;
  8) exec bash "$SCRIPT_DIR/09_a5_tfcr_ratio075.sh" ;;
  9) exec bash "$SCRIPT_DIR/10_a3_two_view_ratio075.sh" ;;
  10) exec bash "$SCRIPT_DIR/11_a4_inv_only_ratio075.sh" ;;
  *)
    echo "Unreachable task index: $TASK_INDEX" >&2
    exit 2
    ;;
esac
