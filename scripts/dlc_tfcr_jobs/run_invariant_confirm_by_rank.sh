#!/usr/bin/env bash
set -euo pipefail

# Seed confirmation array for selected 400k full-run settings.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 4))"

export NUM_PROCESSES=1
export NPROC=1
export NUM_MACHINES=1
export MACHINE_RANK=0

case "$TASK_INDEX" in
  0)
    export SEED=1 RUN_SUFFIX=confirm-seed1-full400k
    exec bash "$SCRIPT_DIR/12_q0_invariant_three_view_400k.sh"
    ;;
  1)
    export SEED=1 RUN_SUFFIX=confirm-seed1-full400k
    exec bash "$SCRIPT_DIR/15_q3_orbit_full_400k.sh"
    ;;
  2)
    export SEED=2 RUN_SUFFIX=confirm-seed2-full400k
    exec bash "$SCRIPT_DIR/12_q0_invariant_three_view_400k.sh"
    ;;
  3)
    export SEED=2 RUN_SUFFIX=confirm-seed2-full400k
    exec bash "$SCRIPT_DIR/15_q3_orbit_full_400k.sh"
    ;;
  *)
    echo "Unreachable task index: $TASK_INDEX" >&2
    exit 2
    ;;
esac
