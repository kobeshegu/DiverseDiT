#!/usr/bin/env bash
set -euo pipefail

# Run one controlled representation evaluation per DLC task. Metrics include
# variance decomposition, class/timestep probes, source/noise retrieval,
# semantic/state effective rank, and layer/timestep CKA.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
source "$REPO_DIR/scripts/tfcr_common.sh"

TASK_INDEX="${TASK_INDEX:-${DLC_TASK_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}"
TASK_INDEX="$((TASK_INDEX % 6))"

experiments=(
  p2_split_repa
  p2_split_no_repa
  p3_semantic_align_repa
  p3_semantic_align_no_repa
  p4_identifiable_repa
  p4_identifiable_no_repa
)
suffixes=(
  j106-p2-split-repa-400k
  j107-p2-split-no-repa-400k
  j108-p3-semantic-align-repa-400k
  j109-p3-semantic-align-no-repa-400k
  j110-p4-identifiable-repa-400k
  j111-p4-identifiable-no-repa-400k
)
methods=(P2-REPA P2-no-REPA P3-REPA P3-no-REPA P4-REPA P4-no-REPA)

export FACTOR_BATCH_RATIO=1.0
export CROSS_NOISE_PROB="${CROSS_NOISE_PROB:-0.5}"
export RUN_SUFFIX="${suffixes[$TASK_INDEX]}"
experiment="${experiments[$TASK_INDEX]}"
method="${methods[$TASK_INDEX]}"
experiment_name="$(tfcr_experiment_name "$experiment")"

checkpoint_step="${REPRESENTATION_CHECKPOINT_STEP:-400000}"
checkpoint_padded="$(printf '%07d' "$checkpoint_step")"
checkpoint="${CKPT:-$OUTPUT_DIR/$experiment_name/checkpoints/$checkpoint_padded.pt}"
analysis_root="${REPRESENTATION_OUTPUT_DIR:-$REPO_DIR/analysis/results/semantic_state_106_111}"
feature_file="$analysis_root/features/${experiment_name}.npz"
report_file="$analysis_root/reports/${experiment_name}.json"

if [[ ! -f "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 2
fi

mkdir -p "$(dirname "$feature_file")" "$(dirname "$report_file")"
tfcr_activate_env "$TRAIN_ENV"
cd "$REPO_DIR"

python analysis/extract_trajectory_features.py \
  --ckpt "$checkpoint" \
  --data-dir "$DATA_DIR" \
  --method "$method" \
  --num-samples "${REPRESENTATION_NUM_SAMPLES:-512}" \
  --analysis-class-count "${REPRESENTATION_CLASS_COUNT:-64}" \
  --num-noises "${REPRESENTATION_NUM_NOISES:-4}" \
  --batch-size "${REPRESENTATION_BATCH_SIZE:-16}" \
  --num-workers "${REPRESENTATION_NUM_WORKERS:-4}" \
  --device "${REPRESENTATION_DEVICE:-cuda}" \
  --output "$feature_file"

python analysis/trajectory_metrics.py "$feature_file" --output "$report_file"
echo "Representation report: $report_file"
