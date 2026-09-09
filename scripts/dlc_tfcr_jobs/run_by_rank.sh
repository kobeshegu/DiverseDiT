#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 20))"

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
  11) exec bash "$SCRIPT_DIR/22_a5_tfcr_ratio1_400k.sh" ;;
  12) exec bash "$SCRIPT_DIR/23_a4_inv_only_ratio1_400k.sh" ;;
  13) exec bash "$SCRIPT_DIR/24_a9_orbit_consensus_ratio1_400k.sh" ;;
  14) exec bash "$SCRIPT_DIR/25_a10_adv_time_ratio1_400k.sh" ;;
  15) exec bash "$SCRIPT_DIR/26_a11_adv_orbit_ratio1_400k.sh" ;;
  16) exec bash "$SCRIPT_DIR/27_a12_adv_purification_ratio1_400k.sh" ;;
  17) exec bash "$SCRIPT_DIR/28_a13_adv_shuffled_ratio1_400k.sh" ;;
  18) exec bash "$SCRIPT_DIR/29_a14_critic_only_ratio1_400k.sh" ;;
  19) exec bash "$SCRIPT_DIR/30_a3_two_view_ratio1_400k.sh" ;;
  *)
    echo "Unreachable task index: $TASK_INDEX" >&2
    exit 2
    ;;
esac
