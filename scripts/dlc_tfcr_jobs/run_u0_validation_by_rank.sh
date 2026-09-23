#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}}"

jobs=(
  72_u0_paired_repa_seed1_400k.sh
  73_u0_paired_repa_seed2_400k.sh
  74_a2_repa_baseline_seed1_400k.sh
  75_a2_repa_baseline_seed2_400k.sh
  76_u0_paired_repa_coeff025_seed0_400k.sh
  77_u0_paired_repa_coeff100_seed0_400k.sh
  78_u0_paired_repa_cross_noise1_seed0_400k.sh
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
