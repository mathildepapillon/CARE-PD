#!/usr/bin/env bash
# Parallel evaluate_shap.py across GPUs FIRST_GPU .. FIRST_GPU+NUM_SHARDS-1, then merge_shap_eval.py.
#
# Usage (from anywhere):
#   bash scripts/run_evaluate_shap_parallel.sh
# Or:
#   chmod +x scripts/run_evaluate_shap_parallel.sh
#   ./scripts/run_evaluate_shap_parallel.sh
#
# Override paths by exporting env vars before running, e.g.:
#   OUT_DIR=results/my_run FIRST_GPU=5 bash scripts/run_evaluate_shap_parallel.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults (edit or override with env vars) ---
ACTOR_SHAP_CKPT="${ACTOR_SHAP_CKPT:-experiment_outs/actor_shap/actor_shap_BMCLab_fold1_20260409_152427/actor_shap_last.ckpt}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr}"
OUT_DIR="${OUT_DIR:-results/shap_actor_potr_bmclab_fold1_full}"
CACHE_DIR="${CACHE_DIR:-results/shap_cache_fold1}"
BACKBONE="${BACKBONE:-potr}"
CONFIG="${CONFIG:-BMCLab.json}"
NUM_FOLDS="${NUM_FOLDS:-23}"
FOLD="${FOLD:-1}"
N_KERNEL_SAMPLES="${N_KERNEL_SAMPLES:-200}"
N_COMPLETION_SAMPLES="${N_COMPLETION_SAMPLES:-20}"
NUM_SHARDS="${NUM_SHARDS:-4}"
FIRST_GPU="${FIRST_GPU:-4}"
SKIP_MERGE="${SKIP_MERGE:-0}"
MERGE_GPU="${MERGE_GPU:-$FIRST_GPU}"

if ! [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SHARDS must be a positive integer" >&2
  exit 1
fi

echo "Repo: $REPO_ROOT"
echo "Output: $OUT_DIR"
echo "Shards: $NUM_SHARDS on GPUs $FIRST_GPU..$((FIRST_GPU + NUM_SHARDS - 1))"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((FIRST_GPU + i))
  echo "Starting shard $i/$NUM_SHARDS on CUDA device $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python evaluate_shap.py \
    --actor_shap_ckpt "$ACTOR_SHAP_CKPT" \
    --actor_shap_cache_dir "$CACHE_DIR" \
    --backbone "$BACKBONE" \
    --config "$CONFIG" \
    --num_folds "$NUM_FOLDS" \
    --fold "$FOLD" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --output_dir "$OUT_DIR" \
    --n_kernel_samples "$N_KERNEL_SAMPLES" \
    --n_completion_samples "$N_COMPLETION_SAMPLES" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$i" \
    --device cuda:0 &
done

wait
echo "All evaluate_shap shards finished."

if [[ "$SKIP_MERGE" != "0" ]]; then
  echo "SKIP_MERGE=$SKIP_MERGE — skipping merge_shap_eval.py"
  exit 0
fi

echo "Running merge_shap_eval.py on GPU $MERGE_GPU …"
CUDA_VISIBLE_DEVICES="$MERGE_GPU" python merge_shap_eval.py \
  --output_dir "$OUT_DIR" \
  --actor_shap_ckpt "$ACTOR_SHAP_CKPT" \
  --backbone "$BACKBONE" \
  --config "$CONFIG" \
  --num_folds "$NUM_FOLDS" \
  --fold "$FOLD" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --n_completion_samples "$N_COMPLETION_SAMPLES" \
  --device cuda:0

echo "Done. See $OUT_DIR/per_sequence.jsonl and $OUT_DIR/aggregate.json"
