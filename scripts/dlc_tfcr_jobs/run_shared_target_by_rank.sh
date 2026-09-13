#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 7))"

export NUM_PROCESSES=1
export NPROC=1
export NUM_MACHINES=1
export MACHINE_RANK=0

case "$TASK_INDEX" in
  0) exec bash "$SCRIPT_DIR/42_s1_a5_shared_repa_ratio1_400k.sh" ;;
  1) exec bash "$SCRIPT_DIR/43_s2_a5_shared_clean_ratio1_400k.sh" ;;
  2) exec bash "$SCRIPT_DIR/44_s3_a5_shared_self_distill_ratio1_400k.sh" ;;
  3) exec bash "$SCRIPT_DIR/45_s4_a5_shared_contrastive_ratio1_400k.sh" ;;
  4) exec bash "$SCRIPT_DIR/46_s5_a5_shared_relation_ratio1_400k.sh" ;;
  5) exec bash "$SCRIPT_DIR/47_s6_a5_private_separation_ratio1_400k.sh" ;;
  6) exec bash "$SCRIPT_DIR/48_s7_a5_contrastive_private_ratio1_400k.sh" ;;
  *)
    echo "Unreachable task index: $TASK_INDEX" >&2
    exit 2
    ;;
esac
