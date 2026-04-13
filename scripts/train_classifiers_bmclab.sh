#!/usr/bin/env bash
# train_classifiers_bmclab.sh
#
# Train LOSO (23-fold) BMCLab classifier checkpoints for all CARE-PD
# feature-encoder models.  Uses pre-tuned hyperparameters from
# configs/best_configs_augmented/Hypertune/ so no Optuna run is required.
#
# Resulting checkpoints land at:
#   experiment_outs/Hypertune/<model_prefix>_BMCLab/0/models/train_BMCLab_23fold/fold<N>/latest_epoch.pth.tr
#
# These paths are exactly what run_shap_baselines.sh and run_shap_actor.sh
# expect when called with CONFIG=BMCLab.json.
#
# USAGE
# -----
#   # Train all models in parallel, one per GPU (default GPUs 4-7):
#   bash scripts/train_classifiers_bmclab.sh
#
#   # Use specific GPUs:
#   GPUS="0 1 2 3" bash scripts/train_classifiers_bmclab.sh
#
#   # Force sequential training on a single GPU:
#   PARALLEL=0 GPUS="4" bash scripts/train_classifiers_bmclab.sh
#
#   # Train a single model:
#   MODELS=motionbert bash scripts/train_classifiers_bmclab.sh
#
# OVERRIDABLE ENV VARS
# --------------------
#   MODELS         space-separated list of backbones to train
#                  [potr motionbert motionagformer mixste poseformerv2]
#   GPUS           space-separated CUDA device indices to use        [4 5 6 7]
#   PARALLEL       1 = launch one model per GPU concurrently         [1]
#   SKIP_EXISTING  1 = skip models whose folds are already trained   [1]
#   LOG_DIR        directory for per-model log files                 [logs/train_bmclab]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project virtualenv so `python` resolves correctly.
# shellcheck disable=SC1091
source carepd/bin/activate

MODELS="${MODELS:-potr motionbert motionagformer mixste poseformerv2}"
GPUS="${GPUS:-4 5 6 7}"
PARALLEL="${PARALLEL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LOG_DIR="${LOG_DIR:-logs/train_bmclab}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$LOG_DIR"

# Convert GPUS string to array
read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"

# ---------------------------------------------------------------------------
# Per-backbone metadata
# ---------------------------------------------------------------------------
# model_prefix: matches generate_config_*.py (POTR_ → uppercase; rest lowercase)
# config:       JSON filename under configs/<backbone>/
# best_params:  pre-tuned hyperparameter fallback (skips Optuna study requirement)
# ---------------------------------------------------------------------------
declare -A MODEL_PREFIX BACKBONE_CONFIG BEST_PARAMS

MODEL_PREFIX=(
    [potr]="POTR"
    [motionbert]="motionbert"
    [motionagformer]="motionagformer"
    [mixste]="mixste"
    [poseformerv2]="poseformerv2"
)
BACKBONE_CONFIG=(
    [potr]="BMCLab.json"
    [motionbert]="BMCLab.json"
    [motionagformer]="BMCLab.json"
    [mixste]="BMCLab.json"
    [poseformerv2]="BMCLab.json"
)
BEST_PARAMS=(
    [potr]="configs/best_configs_augmented/Hypertune/POTR_BMCLABS/0/best_params.json"
    [motionbert]="configs/best_configs_augmented/Hypertune/motionbert_BMCLABS_backright/0/best_params.json"
    [motionagformer]="configs/best_configs_augmented/Hypertune/motionagformer_BMCLABS_backright/0/best_params.json"
    [mixste]="configs/best_configs_augmented/Hypertune/mixste_BMCLABS_backright/0/best_params.json"
    [poseformerv2]="configs/best_configs_augmented/Hypertune/poseformerv2_BMCLABS_backright/0/best_params.json"
)

NUM_FOLDS=23   # BMCLab LOSO (23 subjects)
EXPERIMENT_NAME="Hypertune"
RUN_NUM=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ckpt_path() {
    local backbone="$1" fold="$2"
    local prefix="${MODEL_PREFIX[$backbone]}"
    echo "${REPO_ROOT}/experiment_outs/${EXPERIMENT_NAME}/${prefix}_BMCLab/${RUN_NUM}/models/train_BMCLab_${NUM_FOLDS}fold/fold${fold}/latest_epoch.pth.tr"
}

_all_folds_done() {
    local backbone="$1"
    for fold in $(seq 1 "$NUM_FOLDS"); do
        [[ -f "$(_ckpt_path "$backbone" "$fold")" ]] || return 1
    done
    return 0
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  BMCLab LOSO classifier training  (${NUM_FOLDS} folds)"
echo "  Models   : $MODELS"
echo "  GPUs     : $GPUS"
echo "  Parallel : $PARALLEL"
echo "  Logs     : $LOG_DIR"
echo "  Skip existing: $SKIP_EXISTING"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Build the list of models that actually need training
# ---------------------------------------------------------------------------
MODELS_TO_TRAIN=()
for backbone in $MODELS; do
    if [[ -z "${MODEL_PREFIX[$backbone]+_}" ]]; then
        echo "⚠️  Unknown backbone '$backbone' — skipping"
        continue
    fi
    if [[ "$SKIP_EXISTING" == "1" ]] && _all_folds_done "$backbone"; then
        echo "✅ $backbone — all ${NUM_FOLDS} folds already trained, skipping."
        continue
    fi
    bp="${BEST_PARAMS[$backbone]}"
    if [[ ! -f "$bp" ]]; then
        echo "⚠️  $backbone — best-params not found: $bp  (skipping)"
        continue
    fi
    MODELS_TO_TRAIN+=("$backbone")
done
echo ""

if [[ "${#MODELS_TO_TRAIN[@]}" -eq 0 ]]; then
    echo "Nothing to train — all models already have checkpoints."
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch training — parallel (one model per GPU) or sequential
# ---------------------------------------------------------------------------
PIDS=()
GPU_IDX=0

for backbone in "${MODELS_TO_TRAIN[@]}"; do
    gpu="${GPU_ARRAY[$((GPU_IDX % NUM_GPUS))]}"
    GPU_IDX=$((GPU_IDX + 1))

    prefix="${MODEL_PREFIX[$backbone]}"
    cfg="${BACKBONE_CONFIG[$backbone]}"
    bp="${BEST_PARAMS[$backbone]}"
    logfile="${LOG_DIR}/${TIMESTAMP}-${backbone}-BMCLab.log"

    echo "------------------------------------------------------------"
    echo "  Backbone : $backbone  →  ${prefix}_BMCLab/  on GPU $gpu"
    echo "  Config   : $cfg"
    echo "  HParams  : $bp"
    echo "  Log      : $logfile"
    echo "------------------------------------------------------------"

    if [[ "$PARALLEL" == "1" ]]; then
        echo "  ▶ Launching in background on GPU ${gpu} …"
        (CUDA_VISIBLE_DEVICES="$gpu" \
         python run.py \
             --backbone           "$backbone" \
             --config             "$cfg" \
             --hypertune          0 \
             --this_run_num       "$RUN_NUM" \
             --num_folds          -1 \
             --pretrained         0 \
             --tuned_model_config "$bp" \
             2>&1 | tee "$logfile") &
        PIDS+=($!)
    else
        echo "  ▶ Training on GPU ${gpu} (sequential) …"
        CUDA_VISIBLE_DEVICES="$gpu" \
        python run.py \
            --backbone           "$backbone" \
            --config             "$cfg" \
            --hypertune          0 \
            --this_run_num       "$RUN_NUM" \
            --num_folds          -1 \
            --pretrained         0 \
            --tuned_model_config "$bp" \
            2>&1 | tee "$logfile"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Wait for parallel jobs and report
# ---------------------------------------------------------------------------
if [[ "$PARALLEL" == "1" ]] && [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Waiting for ${#PIDS[@]} background training job(s) …"
    FAILED=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || { echo "  ⚠️  PID $pid exited with error"; FAILED=$((FAILED+1)); }
    done
    if [[ "$FAILED" -gt 0 ]]; then
        echo "⚠️  $FAILED job(s) failed — check log files in $LOG_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Final checkpoint report
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Checkpoint status:"
for backbone in "${MODELS_TO_TRAIN[@]}"; do
    missing=0
    for fold in $(seq 1 "$NUM_FOLDS"); do
        [[ -f "$(_ckpt_path "$backbone" "$fold")" ]] || missing=$((missing+1))
    done
    if [[ "$missing" -eq 0 ]]; then
        echo "  $backbone : ✅  all ${NUM_FOLDS} folds"
    else
        echo "  $backbone : ❌  ${missing}/${NUM_FOLDS} folds missing — check log"
    fi
done
echo ""
echo "  Run SHAP evaluations with:"
echo "    BACKBONE=<model> FOLD=<N> CONFIG=BMCLab.json \\"
echo "      bash scripts/run_shap_baselines.sh"
echo ""
echo "    BACKBONE=<model> FOLD=<N> CONFIG=BMCLab.json \\"
echo "      ACTOR_SHAP_CKPT=<path> bash scripts/run_shap_actor.sh"
echo "============================================================"
