#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}}"

jobs=(
  66_sf1_ema_full_align_ratio1_400k.sh
  67_sf2_ema_source_align_ratio1_400k.sh
  68_sf3_no_injection_ratio1_400k.sh
  69_sf4_shuffle_teacher_ratio1_400k.sh
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
