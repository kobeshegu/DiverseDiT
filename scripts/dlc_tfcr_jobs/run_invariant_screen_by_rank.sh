#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 9))"

# Nine independent one-card 100k screening experiments.  The 400k main run
# is intentionally excluded and must be launched with script 21 explicitly.
export NUM_PROCESSES=1
export NPROC=1
export NUM_MACHINES=1
export MACHINE_RANK=0

case "$TASK_INDEX" in
  0) exec bash "$SCRIPT_DIR/12_q0_invariant_three_view_100k.sh" ;;
  1) exec bash "$SCRIPT_DIR/13_q1_orbit_consistency_100k.sh" ;;
  2) exec bash "$SCRIPT_DIR/14_q2_orbit_spread_100k.sh" ;;
  3) exec bash "$SCRIPT_DIR/15_q3_orbit_full_100k.sh" ;;
  4) exec bash "$SCRIPT_DIR/16_q1_time_only_100k.sh" ;;
  5) exec bash "$SCRIPT_DIR/17_q1_noise_only_100k.sh" ;;
  6) exec bash "$SCRIPT_DIR/18_q3_source_depth8_100k.sh" ;;
  7) exec bash "$SCRIPT_DIR/19_q3_dim128_100k.sh" ;;
  8) exec bash "$SCRIPT_DIR/20_q3_mlp_readout_100k.sh" ;;
  *)
    echo "Unreachable task index: $TASK_INDEX" >&2
    exit 2
    ;;
esac
