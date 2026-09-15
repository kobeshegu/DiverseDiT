#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-${RANK:-0}}"

jobs=(
  58_u0_paired_repa_ratio1_200k.sh
  60_u2_selective_semantic_ratio1_200k.sh
  62_u3_semantic_no_injection_ratio1_200k.sh
  64_u5_semantic_shuffled_source_ratio1_200k.sh
  63_u4_semantic_no_a5_ratio1_200k.sh
  59_u1_scheduled_repa_400k.sh
  65_u0_paired_repa_ratio1_400k.sh
  61_u2_selective_semantic_ratio1_400k.sh
)

if (( rank < 0 || rank >= ${#jobs[@]} )); then
  echo "rank must be in [0, $((${#jobs[@]} - 1))]" >&2
  exit 2
fi

bash "$SCRIPT_DIR/${jobs[$rank]}"
