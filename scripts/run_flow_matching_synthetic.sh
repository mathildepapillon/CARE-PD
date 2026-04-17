#!/usr/bin/env bash
# run_flow_matching_synthetic.sh — end-to-end synthetic Gaussian pipeline.
#
# One GPU, four phases, no branching. Produces:
#   1. Synthetic Gaussian data + classifier (build_synthetic_gaussian_data.py)
#   2. Flow-matching training cache (generate_velocity_synthetic.py)
#   3. Trained velocity net (train_flow_matching.py)
#   4. OTFlow-SHAP attributions on the Gaussian classifier
#      (scripts/compute_flow_shap_synthetic.py)
#
# Override GPU and run tag via env vars:
#     GPU=0 TAG=v1 bash scripts/run_flow_matching_synthetic.sh

set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
TAG="${TAG:-synthetic_gaussian}"

DATA_DIR="experiment_outs/actor_shap_synthetic/${TAG}"
CACHE_DIR="cache/flow_matching/synthetic_gaussian"
FLOW_CKPT_DIR="experiment_outs/flow_matching_synthetic/synthetic_gaussian"
SHAP_OUT_DIR="${FLOW_CKPT_DIR}/ig_synthetic"
LOG_DIR="experiment_outs/flow_matching_synthetic/logs"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[synthetic] PHASE 1 — build Gaussian data + classifier"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/build_synthetic_gaussian_data.py \
    --out_dir  "$DATA_DIR" \
    --rho 0.5 --alpha 0.8 \
    --n_train 2000 --n_val 500 --n_test 100 \
    --clf_epochs 80 --seed 0 \
    > "$LOG_DIR/01_build_data.log" 2>&1

echo "[synthetic] PHASE 2 — flow-matching cache"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/generate_velocity_synthetic.py \
    --ckpt_dir "$DATA_DIR" \
    --out_dir  "$CACHE_DIR" \
    > "$LOG_DIR/02_cache.log" 2>&1

echo "[synthetic] PHASE 3 — flow-matching training"
CUDA_VISIBLE_DEVICES="$GPU" python train_flow_matching.py \
    --config configs/flow_matching/synthetic_gaussian.json --no_wandb \
    > "$LOG_DIR/03_train_flow.log" 2>&1

echo "[synthetic] PHASE 4 — OTFlow-SHAP on Gaussian classifier"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/compute_flow_shap_synthetic.py \
    --ckpt_dir      "$DATA_DIR" \
    --flow_ckpt_dir "$FLOW_CKPT_DIR" \
    --flow_config   configs/flow_matching/synthetic_gaussian.json \
    --data_mode     synthetic_gaussian \
    --n_clips       100 --class_idx 0 \
    --output_dir    "$SHAP_OUT_DIR" \
    > "$LOG_DIR/04_flow_shap.log" 2>&1

echo "[synthetic] DONE"
echo "  data:      $DATA_DIR"
echo "  cache:     $CACHE_DIR"
echo "  flow ckpt: $FLOW_CKPT_DIR"
echo "  psi.npz:   $SHAP_OUT_DIR"
