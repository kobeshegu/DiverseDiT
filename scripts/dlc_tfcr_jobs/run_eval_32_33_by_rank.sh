#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 2))"

export NUM_PROCESSES=1
export NPROC=1
export NUM_MACHINES=1
export MACHINE_RANK=0

case "$TASK_INDEX" in
  0) exec bash "$SCRIPT_DIR/40_eval_32_v1_clean_consensus_ratio1_400k.sh" ;;
  1) exec bash "$SCRIPT_DIR/41_eval_33_v4_vgsc_ratio1_400k.sh" ;;
  *)
    echo "Unreachable task index: $TASK_INDEX" >&2
    exit 2
    ;;
esac
