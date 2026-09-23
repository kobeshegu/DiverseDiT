#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}}"

jobs=(
  80_u6_u0_weak_decomp_400k.sh
  81_u7_u0_recom_only_weak_400k.sh
  82_u8_u0_late_weak_decomp_400k.sh
)

if (( rank < 0 || rank >= ${#jobs[@]} )); then
  echo "rank must be in [0, $((${#jobs[@]} - 1))]" >&2
  exit 2
fi

export NUM_PROCESSES=1
export NPROC=1
export NUM_MACHINES=1
export MACHINE_RANK=0

exec bash "$SCRIPT_DIR/${jobs[$rank]}"
