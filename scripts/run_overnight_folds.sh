#!/usr/bin/env bash
# Full overnight pipeline for folds 2 and 8:
#   1. Train actor_cvae (fold 2 on GPU 4, fold 8 on GPU 5)  — in parallel
#   2. Train actor_shap (fold 2 on GPU 4, fold 8 on GPU 5)  — in parallel
#   3. Evaluate actor SHAP  (fold 2 then fold 8, 4 shards each on GPUs 4-7)
#   4. Merge evaluate_shap shards for each fold
#   5. Run multi-group consistency analysis with all available data
#
# Usage (from repo root):
#   bash scripts/run_overnight_folds.sh
#
# Logs are written to logs/overnight_*.log

set -euo pipefail
cd "$(dirname "$0")/.."

# ── GPU assignment ────────────────────────────────────────────────────────────
# GPUs 4 and 5 are busy with baseline runs for folds 2 and 8 (they will finish
# before training is done).  Use 6 and 7 for actor_cvae / actor_shap training;
# then use all four (4 5 6 7) for the 4-shard evaluate_shap.py runs.
GPU_FOLD2_TRAIN=6          # for actor_cvae + actor_shap training, fold 2
GPU_FOLD8_TRAIN=7          # for actor_cvae + actor_shap training, fold 8
EVAL_GPUS=(4 5 6 7)        # 4 shards for evaluate_shap.py
CLASSIFIER_BASE="experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold"

# ── Hyper-parameters (mirror fold-1 setup) ───────────────────────────────────
CVAE_EPOCHS=500
SHAP_EPOCHS=150
N_FOLDS=23
BACKBONE="potr"
CONFIG="BMCLab.json"

# ── Paths ─────────────────────────────────────────────────────────────────────
PKL_1="assets/preprocessed_data/potr_processing/Hypertune/BMCLab_center_True_zscore/23fold/BMCLab_eval_1.pkl"
PKL_2="assets/preprocessed_data/potr_processing/Hypertune/BMCLab_center_True_zscore/23fold/BMCLab_eval_2.pkl"
PKL_8="assets/preprocessed_data/potr_processing/Hypertune/BMCLab_center_True_zscore/23fold/BMCLab_eval_8.pkl"

FOLD1_BASELINE_DIR="results/shap_baselines_potr_bmclab_fold1_full"
FOLD2_BASELINE_DIR="results/shap_baselines_potr_bmclab_fold2"
FOLD8_BASELINE_DIR="results/shap_baselines_potr_bmclab_fold8"
ACTOR_FOLD1_DIR="results/shap_actor_potr_bmclab_fold1_full"
ACTOR_FOLD2_DIR="results/shap_actor_potr_bmclab_fold2"
ACTOR_FOLD8_DIR="results/shap_actor_potr_bmclab_fold8"

mkdir -p logs

# ─────────────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a logs/overnight_pipeline.log; }

source carepd/bin/activate

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Train actor_cvae for folds 2 and 8 in parallel
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Phase 1: Training actor_cvae (fold 2 on GPU ${GPU_FOLD2_TRAIN}, fold 8 on GPU ${GPU_FOLD8_TRAIN}) ==="

TIMESTAMP_START_CVAE=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=${GPU_FOLD2_TRAIN} \
python3 train_actor_cvae.py \
    --config configs/actor_cvae_bmclab_best.json \
    --dataset BMCLab \
    --num_folds ${N_FOLDS} \
    --fold 2 \
    --epochs ${CVAE_EPOCHS} \
    > logs/overnight_cvae_fold2.log 2>&1 &
PID_CVAE2=$!
log "  fold 2 actor_cvae launched (PID ${PID_CVAE2})"

CUDA_VISIBLE_DEVICES=${GPU_FOLD8_TRAIN} \
python3 train_actor_cvae.py \
    --config configs/actor_cvae_bmclab_best.json \
    --dataset BMCLab \
    --num_folds ${N_FOLDS} \
    --fold 8 \
    --epochs ${CVAE_EPOCHS} \
    > logs/overnight_cvae_fold8.log 2>&1 &
PID_CVAE8=$!
log "  fold 8 actor_cvae launched (PID ${PID_CVAE8})"

wait ${PID_CVAE2} && log "  fold 2 actor_cvae DONE" || log "  ERROR: fold 2 actor_cvae failed — check logs/overnight_cvae_fold2.log"
wait ${PID_CVAE8} && log "  fold 8 actor_cvae DONE" || log "  ERROR: fold 8 actor_cvae failed — check logs/overnight_cvae_fold8.log"

# Resolve the most-recently-created actor_cvae best checkpoint for each fold
CVAE_CKPT_2=$(ls -t experiment_outs/actor_cvae/actor_carepd_BMCLab_fold2_*/actor_cvae_best.ckpt 2>/dev/null | head -1)
CVAE_CKPT_8=$(ls -t experiment_outs/actor_cvae/actor_carepd_BMCLab_fold8_*/actor_cvae_best.ckpt 2>/dev/null | head -1)

if [[ -z "${CVAE_CKPT_2}" || -z "${CVAE_CKPT_8}" ]]; then
    log "ERROR: Could not find actor_cvae checkpoints for one or both folds. Aborting."
    exit 1
fi

log "  fold 2 cvae ckpt: ${CVAE_CKPT_2}"
log "  fold 8 cvae ckpt: ${CVAE_CKPT_8}"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Train actor_shap for folds 2 and 8 in parallel
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Phase 2: Training actor_shap ==="

CUDA_VISIBLE_DEVICES=${GPU_FOLD2_TRAIN} \
python3 train_actor_shap.py \
    --dataset BMCLab \
    --num_folds ${N_FOLDS} \
    --fold 2 \
    --actor_cvae_ckpt "${CVAE_CKPT_2}" \
    --epochs ${SHAP_EPOCHS} \
    --batch_size 8 \
    --lr 1e-4 \
    --lr_decoder 1e-5 \
    --latent_dim 256 \
    --ff_size 1024 \
    --num_layers 8 \
    --num_heads 4 \
    --dropout 0.0 \
    --lambda_kl 1.0 \
    --kl_warmup_epochs 20 \
    --mask_axis spatial \
    > logs/overnight_shap_fold2.log 2>&1 &
PID_SHAP2=$!
log "  fold 2 actor_shap launched (PID ${PID_SHAP2})"

CUDA_VISIBLE_DEVICES=${GPU_FOLD8_TRAIN} \
python3 train_actor_shap.py \
    --dataset BMCLab \
    --num_folds ${N_FOLDS} \
    --fold 8 \
    --actor_cvae_ckpt "${CVAE_CKPT_8}" \
    --epochs ${SHAP_EPOCHS} \
    --batch_size 8 \
    --lr 1e-4 \
    --lr_decoder 1e-5 \
    --latent_dim 256 \
    --ff_size 1024 \
    --num_layers 8 \
    --num_heads 4 \
    --dropout 0.0 \
    --lambda_kl 1.0 \
    --kl_warmup_epochs 20 \
    --mask_axis spatial \
    > logs/overnight_shap_fold8.log 2>&1 &
PID_SHAP8=$!
log "  fold 8 actor_shap launched (PID ${PID_SHAP8})"

wait ${PID_SHAP2} && log "  fold 2 actor_shap DONE" || log "  ERROR: fold 2 actor_shap failed — check logs/overnight_shap_fold2.log"
wait ${PID_SHAP8} && log "  fold 8 actor_shap DONE" || log "  ERROR: fold 8 actor_shap failed — check logs/overnight_shap_fold8.log"

# Resolve actor_shap checkpoints — prefer best_diversity, fall back to last
SHAP_CKPT_2=$(ls -t experiment_outs/actor_shap/actor_shap_BMCLab_fold2_*/actor_shap_best_diversity.ckpt 2>/dev/null | head -1)
[[ -z "${SHAP_CKPT_2}" ]] && SHAP_CKPT_2=$(ls -t experiment_outs/actor_shap/actor_shap_BMCLab_fold2_*/actor_shap_last.ckpt 2>/dev/null | head -1)

SHAP_CKPT_8=$(ls -t experiment_outs/actor_shap/actor_shap_BMCLab_fold8_*/actor_shap_best_diversity.ckpt 2>/dev/null | head -1)
[[ -z "${SHAP_CKPT_8}" ]] && SHAP_CKPT_8=$(ls -t experiment_outs/actor_shap/actor_shap_BMCLab_fold8_*/actor_shap_last.ckpt 2>/dev/null | head -1)

if [[ -z "${SHAP_CKPT_2}" || -z "${SHAP_CKPT_8}" ]]; then
    log "ERROR: Could not find actor_shap checkpoints. Aborting."
    exit 1
fi

log "  fold 2 shap ckpt: ${SHAP_CKPT_2}"
log "  fold 8 shap ckpt: ${SHAP_CKPT_8}"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Evaluate actor SHAP on fold 2 (4 shards, GPUs 4-7)
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Phase 3a: Evaluating actor SHAP — fold 2 ==="
mkdir -p "${ACTOR_FOLD2_DIR}"

EVAL_PIDS=()
for SHARD_ID in 0 1 2 3; do
    GPU=${EVAL_GPUS[$SHARD_ID]}
    CUDA_VISIBLE_DEVICES=${GPU} \
    python3 evaluate_shap.py \
        --actor_shap_ckpt "${SHAP_CKPT_2}" \
        --backbone "${BACKBONE}" \
        --config "${CONFIG}" \
        --classifier_ckpt "${CLASSIFIER_BASE}/fold2/latest_epoch.pth.tr" \
        --fold 2 \
        --num_folds ${N_FOLDS} \
        --output_dir "${ACTOR_FOLD2_DIR}" \
        --num_shards 4 \
        --shard_id ${SHARD_ID} \
        > "logs/overnight_eval_fold2_shard${SHARD_ID}.log" 2>&1 &
    EVAL_PIDS+=($!)
    log "  fold 2 shard ${SHARD_ID} on GPU ${GPU} (PID ${EVAL_PIDS[-1]})"
done

for PID in "${EVAL_PIDS[@]}"; do
    wait ${PID} || log "  WARNING: a fold 2 eval shard returned non-zero exit"
done
log "  fold 2 evaluate_shap DONE"

log "  Merging fold 2 shards..."
python3 merge_shap_eval.py \
    --output_dir "${ACTOR_FOLD2_DIR}" \
    --actor_shap_ckpt "${SHAP_CKPT_2}" \
    --backbone "${BACKBONE}" \
    --config "${CONFIG}" \
    --classifier_ckpt "${CLASSIFIER_BASE}/fold2/latest_epoch.pth.tr" \
    --fold 2 \
    --num_folds ${N_FOLDS} \
    --skip_fid \
    >> logs/overnight_pipeline.log 2>&1
log "  fold 2 merge DONE"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3b — Evaluate actor SHAP on fold 8 (4 shards, GPUs 4-7)
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Phase 3b: Evaluating actor SHAP — fold 8 ==="
mkdir -p "${ACTOR_FOLD8_DIR}"

EVAL_PIDS=()
for SHARD_ID in 0 1 2 3; do
    GPU=${EVAL_GPUS[$SHARD_ID]}
    CUDA_VISIBLE_DEVICES=${GPU} \
    python3 evaluate_shap.py \
        --actor_shap_ckpt "${SHAP_CKPT_8}" \
        --backbone "${BACKBONE}" \
        --config "${CONFIG}" \
        --classifier_ckpt "${CLASSIFIER_BASE}/fold8/latest_epoch.pth.tr" \
        --fold 8 \
        --num_folds ${N_FOLDS} \
        --output_dir "${ACTOR_FOLD8_DIR}" \
        --num_shards 4 \
        --shard_id ${SHARD_ID} \
        > "logs/overnight_eval_fold8_shard${SHARD_ID}.log" 2>&1 &
    EVAL_PIDS+=($!)
    log "  fold 8 shard ${SHARD_ID} on GPU ${GPU} (PID ${EVAL_PIDS[-1]})"
done

for PID in "${EVAL_PIDS[@]}"; do
    wait ${PID} || log "  WARNING: a fold 8 eval shard returned non-zero exit"
done
log "  fold 8 evaluate_shap DONE"

log "  Merging fold 8 shards..."
python3 merge_shap_eval.py \
    --output_dir "${ACTOR_FOLD8_DIR}" \
    --actor_shap_ckpt "${SHAP_CKPT_8}" \
    --backbone "${BACKBONE}" \
    --config "${CONFIG}" \
    --classifier_ckpt "${CLASSIFIER_BASE}/fold8/latest_epoch.pth.tr" \
    --fold 8 \
    --num_folds ${N_FOLDS} \
    --skip_fid \
    >> logs/overnight_pipeline.log 2>&1
log "  fold 8 merge DONE"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Multi-group consistency analysis with all folds
# ═══════════════════════════════════════════════════════════════════════════════
log "=== Phase 4: Running multi-group consistency analysis ==="

# Include fold1 actor SHAP if available
ACTOR_DIRS=("${ACTOR_FOLD2_DIR}" "${ACTOR_FOLD8_DIR}")
[[ -d "${ACTOR_FOLD1_DIR}/shards" ]] && ACTOR_DIRS=("${ACTOR_FOLD1_DIR}" "${ACTOR_DIRS[@]}")

# Include fold2 and fold8 baselines even if they finished early
BASELINE_DIRS=("${FOLD1_BASELINE_DIR}")
[[ -d "${FOLD2_BASELINE_DIR}" ]] && BASELINE_DIRS+=("${FOLD2_BASELINE_DIR}")
[[ -d "${FOLD8_BASELINE_DIR}" ]] && BASELINE_DIRS+=("${FOLD8_BASELINE_DIR}")

python3 evaluate_shap_consistency.py \
    --actor_shap_dirs "${ACTOR_DIRS[@]}" \
    --baseline_dirs   "${BASELINE_DIRS[@]}" \
    --eval_pkls       "${PKL_1}" "${PKL_2}" "${PKL_8}" \
    --output_dir      "results/shap_consistency_multigroup_full"

log "=== All done. Results in results/shap_consistency_multigroup_full/ ==="
