#!/usr/bin/env bash
# train_vaeac_olsen.sh — train the three Olsen-faithful VAEAC checkpoints
# (k4 / k8 / spatial) using the Ivanov output head + prior-memory skip
# connections, then run the oracle-vs-VAEAC comparison on each.
#
# Runs are sequential on a single GPU because each training is ~3 min and
# DDP overhead would dominate at this model size.
#
# Usage:
#   GPU=4 bash scripts/train_vaeac_olsen.sh
#
# Skips re-training any variant whose `last.ckpt` already exists unless
# RETRAIN=1 is set.

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-4}"
RETRAIN="${RETRAIN:-0}"
CONFIG="configs/vaeac/synthetic_gaussian.json"
LOG_DIR="experiment_outs/flow_matching_synthetic/logs"
CKPT_ROOT="experiment_outs/vaeac_synthetic"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

train_one () {
    local tag="$1"
    local bench="$2"
    local ckpt_dir="${CKPT_ROOT}/${tag}"
    local log="${LOG_DIR}/03b_train_vaeac_${tag}.log"

    if [[ -f "${ckpt_dir}/last.ckpt" && "$RETRAIN" != "1" ]]; then
        echo "[olsen] SKIP ${tag} (checkpoint exists at ${ckpt_dir}/last.ckpt)"
        return 0
    fi

    echo "[olsen] TRAIN ${tag}  bench=${bench}  ckpt=${ckpt_dir}"
    CUDA_VISIBLE_DEVICES="$GPU" python train_vaeac.py \
        --config "$CONFIG" --no_wandb \
        --ckpt_dir_override "$ckpt_dir" \
        --bench_path "$bench" \
        --mask_mode benchmark \
        > "$log" 2>&1
    echo "[olsen] DONE  ${tag}  →  see $log"
}

train_one gaussian_k4_olsen \
    experiment_outs/actor_shap_synthetic/synthetic_gaussian_k4/synthetic_benchmark.pkl

train_one gaussian_k8_olsen \
    experiment_outs/actor_shap_synthetic/synthetic_gaussian_k8/synthetic_benchmark.pkl

train_one gaussian_spatial_olsen \
    experiment_outs/actor_shap_synthetic/synthetic_gaussian_spatial/synthetic_benchmark.pkl

echo "[olsen] All three Olsen VAEAC checkpoints trained."
ls -lh "${CKPT_ROOT}"/gaussian_{k4,k8,spatial}_olsen/last.ckpt 2>/dev/null || true
