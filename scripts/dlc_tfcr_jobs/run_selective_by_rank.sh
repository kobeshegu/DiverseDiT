#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 9))"

# This order is the experimental priority for the nine-run VGSC matrix.
export NUM_PROCESSES=1
export NPROC=1
export NUM_MACHINES=1
export MACHINE_RANK=0

case "$TASK_INDEX" in
  0) exec bash "$SCRIPT_DIR/31_v0_a3_shared_ratio1_400k.sh" ;;
  1) exec bash "$SCRIPT_DIR/32_v1_clean_consensus_ratio1_400k.sh" ;;
  2) exec bash "$SCRIPT_DIR/38_v7_selective_task_only_ratio1_400k.sh" ;;
  3) exec bash "$SCRIPT_DIR/33_v4_vgsc_ratio1_400k.sh" ;;
  4) exec bash "$SCRIPT_DIR/39_v8_vgsc_weak_ratio1_400k.sh" ;;
  5) exec bash "$SCRIPT_DIR/34_v3_selective_stability_ratio1_400k.sh" ;;
  6) exec bash "$SCRIPT_DIR/35_v2_selective_uniform_ratio1_400k.sh" ;;
  7) exec bash "$SCRIPT_DIR/36_v5_vgsc_shuffled_source_ratio1_400k.sh" ;;
  8) exec bash "$SCRIPT_DIR/37_v6_vgsc_shuffled_utility_ratio1_400k.sh" ;;
  *)
    echo "Unreachable task index: $TASK_INDEX" >&2
    exit 2
    ;;
esac
