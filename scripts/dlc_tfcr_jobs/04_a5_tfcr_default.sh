#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export RUN_SUFFIX="${RUN_SUFFIX:-j04-default}"

run_tfcr_job a5_tfcr
