#!/usr/bin/env bash
# run_flow_matching_synthetic.sh — end-to-end synthetic Gaussian pipeline.
#
# Phases:
#   1. Build synthetic Gaussian data + classifier (K-windows or J-joints player
#      mode).  Writes benchmark/classifier/test tensors to $DATA_DIR.
#   2. (Optional) Build flow-matching training cache from $DATA_DIR.  Skipped
#      if $CACHE_DIR already has a ``cache.npz``.
#   3. (Optional) Train the flow matching model.  Skipped if $FLOW_CKPT_DIR
#      already holds a ``last.ckpt`` file.
#   3b. (Optional) Train the VAEAC baseline on the SAME cache.  Skipped if
#       $VAEAC_CKPT_DIR already holds a ``last.ckpt`` file, or if
#       $TRAIN_VAEAC != 1.
#   4. OTFlow-SHAP attributions on the Gaussian classifier.
#   5. EC1/EC2/EC3 benchmark sweep: zero / mean / marginal / gaussian_oracle /
#      flow_matching [ / vaeac ].  Produces $EC_OUT_DIR/ec_summary.json and a
#      printed table.
#
# Configure via env vars:
#
#   GPU           CUDA device index        (default 0)
#   TAG           sub-directory for this run (default synthetic_gaussian)
#   K             number of temporal windows (default 4; use 8 for "half-size windows")
#   PLAYERS       'temporal' or 'spatial'   (default temporal)
#   SIGNAL_JOINTS 4 space-separated indices for PLAYERS=spatial (default "0 1 2 3")
#   RETRAIN_FLOW  '1' to force re-training the flow even if last.ckpt exists (default 0)
#   N_TEST        number of test sequences to EC-sweep (default 100)
#   K_MC_TRUE     MC samples for oracle v(S) (default 1000)
#   N_COMP        completions per coalition for stochastic imputers (default 20)
#   N_KERNEL      paired KernelSHAP samples for spatial mode (default 250)
#   FLOW_CONFIG   path to flow config JSON   (default configs/flow_matching/synthetic_gaussian.json)
#   FLOW_CKPT_DIR where to find or train the flow checkpoint
#                 (default experiment_outs/flow_matching_synthetic/synthetic_gaussian)
#
# Example: K=8 temporal, reuse existing flow at gaussian_gpu_xl:
#
#   GPU=0 TAG=synthetic_gaussian_k8 K=8 PLAYERS=temporal \
#       FLOW_CKPT_DIR=experiment_outs/flow_matching_synthetic/gaussian_gpu_xl \
#       bash scripts/run_flow_matching_synthetic.sh
#
# Example: spatial J=17 players, reuse existing flow:
#
#   GPU=1 TAG=synthetic_gaussian_spatial PLAYERS=spatial \
#       FLOW_CKPT_DIR=experiment_outs/flow_matching_synthetic/gaussian_gpu_xl \
#       bash scripts/run_flow_matching_synthetic.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
TAG="${TAG:-synthetic_gaussian}"
K="${K:-4}"
PLAYERS="${PLAYERS:-temporal}"
SIGNAL_JOINTS="${SIGNAL_JOINTS:-0 1 2 3}"
RETRAIN_FLOW="${RETRAIN_FLOW:-0}"
N_TEST="${N_TEST:-100}"
K_MC_TRUE="${K_MC_TRUE:-1000}"
N_COMP="${N_COMP:-20}"
N_KERNEL="${N_KERNEL:-250}"
FLOW_NUM_STEPS="${FLOW_NUM_STEPS:-50}"
FLOW_SOLVER="${FLOW_SOLVER:-midpoint}"
FLOW_CONFIG="${FLOW_CONFIG:-configs/flow_matching/synthetic_gaussian.json}"
FLOW_CKPT_DIR="${FLOW_CKPT_DIR:-experiment_outs/flow_matching_synthetic/synthetic_gaussian}"

# VAEAC baseline (optional).  Set TRAIN_VAEAC=1 to train in Phase 3b (or point
# VAEAC_CKPT_DIR at a pre-trained run).  If VAEAC_CKPT_DIR exists (contains a
# last.ckpt), it is used in Phase 5 as an extra imputer row, regardless of
# TRAIN_VAEAC.
TRAIN_VAEAC="${TRAIN_VAEAC:-0}"
VAEAC_CONFIG="${VAEAC_CONFIG:-configs/vaeac/synthetic_gaussian.json}"
VAEAC_CKPT_DIR="${VAEAC_CKPT_DIR:-experiment_outs/vaeac_synthetic/synthetic_gaussian}"
VAEAC_TEMPERATURE="${VAEAC_TEMPERATURE:-1.0}"
RETRAIN_VAEAC="${RETRAIN_VAEAC:-0}"

DATA_DIR="experiment_outs/actor_shap_synthetic/${TAG}"
CACHE_DIR="cache/flow_matching/synthetic_gaussian"
SHAP_OUT_DIR="${FLOW_CKPT_DIR}/ig_${TAG}"
EC_OUT_DIR="experiment_outs/flow_matching_synthetic/ec_${TAG}"
LOG_DIR="experiment_outs/flow_matching_synthetic/logs"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[run] TAG=${TAG}  GPU=${GPU}  K=${K}  PLAYERS=${PLAYERS}"
echo "[run] DATA_DIR=${DATA_DIR}"
echo "[run] FLOW_CKPT_DIR=${FLOW_CKPT_DIR}"
echo "[run] VAEAC_CKPT_DIR=${VAEAC_CKPT_DIR}  (TRAIN_VAEAC=${TRAIN_VAEAC})"
echo "[run] EC_OUT_DIR=${EC_OUT_DIR}"

# ---- PHASE 1: build data + classifier --------------------------------------
BUILD_ARGS=(
    --out_dir "$DATA_DIR"
    --rho 0.5 --alpha 0.8
    --K "$K"
    --players "$PLAYERS"
    --n_train 2000 --n_val 500 --n_test 500
    --clf_epochs 80 --seed 0
)
if [[ "$PLAYERS" == "spatial" ]]; then
    # shellcheck disable=SC2206
    SJ=($SIGNAL_JOINTS)
    BUILD_ARGS+=(--signal_joints "${SJ[@]}")
fi
echo "[run] PHASE 1 — build data + classifier (player_mode=${PLAYERS}, K=${K})"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/build_synthetic_gaussian_data.py "${BUILD_ARGS[@]}" \
    > "$LOG_DIR/01_build_${TAG}.log" 2>&1

# ---- PHASE 2: flow cache (skip if already present) -------------------------
if [[ -f "${CACHE_DIR}/cache.npz" ]]; then
    echo "[run] PHASE 2 — cache already present at ${CACHE_DIR}/cache.npz, skipping."
else
    echo "[run] PHASE 2 — build flow-matching cache"
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/generate_velocity_synthetic.py \
        --ckpt_dir "$DATA_DIR" --out_dir "$CACHE_DIR" \
        > "$LOG_DIR/02_cache_${TAG}.log" 2>&1
fi

# ---- PHASE 3: flow training (skip if checkpoint exists) --------------------
if [[ -f "${FLOW_CKPT_DIR}/last.ckpt" && "$RETRAIN_FLOW" != "1" ]]; then
    echo "[run] PHASE 3 — flow checkpoint present at ${FLOW_CKPT_DIR}/last.ckpt, skipping."
else
    echo "[run] PHASE 3 — train flow matching model"
    CUDA_VISIBLE_DEVICES="$GPU" python train_flow_matching.py \
        --config "$FLOW_CONFIG" --no_wandb \
        > "$LOG_DIR/03_train_flow_${TAG}.log" 2>&1
fi

# ---- PHASE 3b: VAEAC training (optional; same cache as flow) ---------------
# Skipped unless TRAIN_VAEAC=1.  If a last.ckpt already exists and RETRAIN_VAEAC
# is 0 (default), training is also skipped.  Either way, if a checkpoint is
# present at the end, Phase 5 will include the `vaeac` imputer row.
if [[ "$TRAIN_VAEAC" == "1" ]]; then
    if [[ -f "${VAEAC_CKPT_DIR}/last.ckpt" && "$RETRAIN_VAEAC" != "1" ]]; then
        echo "[run] PHASE 3b — VAEAC checkpoint present at ${VAEAC_CKPT_DIR}/last.ckpt, skipping."
    else
        echo "[run] PHASE 3b — train VAEAC baseline (mask_mode=benchmark)"
        CUDA_VISIBLE_DEVICES="$GPU" python train_vaeac.py \
            --config "$VAEAC_CONFIG" --no_wandb \
            --ckpt_dir_override "$VAEAC_CKPT_DIR" \
            --bench_path "$DATA_DIR/synthetic_benchmark.pkl" \
            --mask_mode benchmark \
            > "$LOG_DIR/03b_train_vaeac_${TAG}.log" 2>&1
    fi
else
    echo "[run] PHASE 3b — TRAIN_VAEAC=0, skipping VAEAC training."
fi

# ---- PHASE 4: OTFlow-SHAP (integrated-gradient attributions) ---------------
# Non-fatal: failure here must not block the EC sweep (Phase 5).
SKIP_OTFLOW="${SKIP_OTFLOW:-0}"
if [[ "$SKIP_OTFLOW" == "1" ]]; then
    echo "[run] PHASE 4 — SKIPPED (SKIP_OTFLOW=1)"
else
    echo "[run] PHASE 4 — OTFlow-SHAP integrated gradients"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/compute_flow_shap_synthetic.py \
        --ckpt_dir      "$DATA_DIR" \
        --flow_ckpt_dir "$FLOW_CKPT_DIR" \
        --flow_config   "$FLOW_CONFIG" \
        --data_mode     synthetic_gaussian \
        --n_clips       "$N_TEST" --class_idx 0 \
        --output_dir    "$SHAP_OUT_DIR" \
        > "$LOG_DIR/04_flow_shap_${TAG}.log" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "[run] WARN: PHASE 4 failed (rc=$rc); see $LOG_DIR/04_flow_shap_${TAG}.log — continuing."
    fi
fi

# ---- PHASE 5: EC1/EC2/EC3 sweep across imputers ----------------------------
EC_ARGS=(
    --ckpt_dir              "$DATA_DIR"
    --flow_ckpt_dir         "$FLOW_CKPT_DIR"
    --flow_config           "$FLOW_CONFIG"
    --output_dir            "$EC_OUT_DIR"
    --n_test_sequences      "$N_TEST"
    --K_mc_true             "$K_MC_TRUE"
    --n_completion_samples  "$N_COMP"
    --n_kernel_samples      "$N_KERNEL"
    --flow_num_steps        "$FLOW_NUM_STEPS"
    --flow_solver           "$FLOW_SOLVER"
    --seed 0
)
if [[ -f "${VAEAC_CKPT_DIR}/last.ckpt" ]]; then
    echo "[run] PHASE 5 — EC sweep (zero / mean / marginal / oracle / flow / vaeac)"
    EC_ARGS+=(
        --vaeac_ckpt_dir     "$VAEAC_CKPT_DIR"
        --vaeac_config       "$VAEAC_CONFIG"
        --vaeac_temperature  "$VAEAC_TEMPERATURE"
    )
else
    echo "[run] PHASE 5 — EC sweep (zero / mean / marginal / oracle / flow) — "\
         "no VAEAC checkpoint at ${VAEAC_CKPT_DIR}/last.ckpt, skipping vaeac row"
fi
CUDA_VISIBLE_DEVICES="$GPU" python scripts/evaluate_shap_synthetic_gaussian.py \
    "${EC_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/05_ec_sweep_${TAG}.log"

echo "[run] DONE"
echo "  data:          $DATA_DIR"
echo "  cache:         $CACHE_DIR"
echo "  flow ckpt:     $FLOW_CKPT_DIR"
echo "  VAEAC ckpt:    $VAEAC_CKPT_DIR"
echo "  OTFlow psi:    $SHAP_OUT_DIR"
echo "  EC summary:    $EC_OUT_DIR/ec_summary.json"
