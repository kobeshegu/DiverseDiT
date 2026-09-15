#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-${RANK:-0}}"

jobs=(
  52_t1_antithetic_pair_ratio1_200k.sh
  57_t0_native_fm_only_ratio1_200k.sh
  55_t4_native_recomposition_ratio1_400k.sh
  53_t2_native_source_ratio1_200k.sh
  54_t3_native_noise_ratio1_200k.sh
  56_t5_native_shuffled_source_ratio1_200k.sh
)

if (( rank < 0 || rank >= ${#jobs[@]} )); then
  echo "rank must be in [0, $((${#jobs[@]} - 1))]" >&2
  exit 2
fi

bash "$SCRIPT_DIR/${jobs[$rank]}"
