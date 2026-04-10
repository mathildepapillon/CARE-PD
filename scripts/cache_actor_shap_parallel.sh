#!/usr/bin/env bash
# cache_actor_shap_parallel.sh
# Pre-generate ActorSHAP completions for all KernelSHAP coalitions across 4 GPUs.
# Run from the repo root:
#   bash scripts/cache_actor_shap_parallel.sh
#
# Output: results/shap_cache_fold1/seq_*.npz  (one file per test sequence)
# Each .npz contains completions for spatial KernelSHAP coalitions + all 16
# temporal coalitions, stored as float16 arrays.

set -euo pipefail

CKPT="experiment_outs/actor_shap/actor_shap_BMCLab_fold1_20260409_152427/actor_shap_last.ckpt"
BACKBONE="potr"
CONFIG="BMCLab.json"
NUM_FOLDS=23
FOLD=1
N_KERNEL_SAMPLES=200
N_COMPLETION_SAMPLES=20
OUTPUT_DIR="results/shap_cache_fold1"
GPUS=(4 5 6 7)
NUM_SHARDS=${#GPUS[@]}

mkdir -p "$OUTPUT_DIR"
echo "Caching ActorSHAP completions → $OUTPUT_DIR"
echo "  checkpoint:          $CKPT"
echo "  n_kernel_samples:    $N_KERNEL_SAMPLES"
echo "  n_completion_samples: $N_COMPLETION_SAMPLES"
echo "  shards:              $NUM_SHARDS  (GPUs: ${GPUS[*]})"
echo ""

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    LOG="/tmp/shap_cache_shard${i}.log"
    echo "Launching shard $i on cuda:$GPU  →  $LOG"
    python cache_actor_shap_completions.py \
        --actor_shap_ckpt "$CKPT" \
        --backbone "$BACKBONE" \
        --config "$CONFIG" \
        --num_folds $NUM_FOLDS \
        --fold $FOLD \
        --n_kernel_samples $N_KERNEL_SAMPLES \
        --n_completion_samples $N_COMPLETION_SAMPLES \
        --output_dir "$OUTPUT_DIR" \
        --num_shards $NUM_SHARDS \
        --shard_id $i \
        --device "cuda:$GPU" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All shards running. PIDs: ${PIDS[*]}"
echo "Tail logs with:"
for i in "${!GPUS[@]}"; do
    echo "  tail -f /tmp/shap_cache_shard${i}.log"
done
echo ""

# Wait for all shards and report outcome.
ALL_OK=1
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "Shard $i finished OK"
    else
        echo "Shard $i FAILED (exit $?)"
        ALL_OK=0
    fi
done

echo ""
if [ $ALL_OK -eq 1 ]; then
    N=$(ls "$OUTPUT_DIR"/seq_*.npz 2>/dev/null | wc -l)
    echo "Done. $N sequences cached in $OUTPUT_DIR"
else
    echo "One or more shards failed. Check logs above."
    exit 1
fi
