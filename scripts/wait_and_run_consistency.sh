#!/usr/bin/env bash
# Wait for fold-2 and fold-8 baseline runs to finish, then run the multi-group
# SHAP consistency analysis across all available folds.
#
# Usage (from repo root):
#   bash scripts/wait_and_run_consistency.sh [--poll 120]
#
# Polls every POLL_SECS seconds until both aggregate.json files exist.

set -euo pipefail
cd "$(dirname "$0")/.."

# ── configurable ──────────────────────────────────────────────────────────────
POLL_SECS=120

FOLD2_BASELINE_DIR="results/shap_baselines_potr_bmclab_fold2"
FOLD8_BASELINE_DIR="results/shap_baselines_potr_bmclab_fold8"
FOLD1_BASELINE_DIR="results/shap_baselines_potr_bmclab_fold1_full"
ACTOR_FOLD1_DIR="results/shap_actor_potr_bmclab_fold1_full"

PKL_1="assets/preprocessed_data/potr_processing/Hypertune/BMCLab_center_True_zscore/23fold/BMCLab_eval_1.pkl"
PKL_2="assets/preprocessed_data/potr_processing/Hypertune/BMCLab_center_True_zscore/23fold/BMCLab_eval_2.pkl"
PKL_8="assets/preprocessed_data/potr_processing/Hypertune/BMCLab_center_True_zscore/23fold/BMCLab_eval_8.pkl"

OUTPUT_DIR="results/shap_consistency_multigroup"
# ─────────────────────────────────────────────────────────────────────────────

source carepd/bin/activate

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Wait for both baseline runs ───────────────────────────────────────────────
log "Waiting for baseline runs to finish (polling every ${POLL_SECS}s)..."

while true; do
    F2_DONE=0; F8_DONE=0
    [[ -f "${FOLD2_BASELINE_DIR}/aggregate.json" ]] && F2_DONE=1
    [[ -f "${FOLD8_BASELINE_DIR}/aggregate.json" ]] && F8_DONE=1

    if [[ $F2_DONE -eq 1 && $F8_DONE -eq 1 ]]; then
        log "Both baseline runs complete."
        break
    fi

    STATUS_F2=$([ $F2_DONE -eq 1 ] && echo "DONE" || echo "waiting")
    STATUS_F8=$([ $F8_DONE -eq 1 ] && echo "DONE" || echo "waiting")
    log "  fold2=$STATUS_F2  fold8=$STATUS_F8"
    sleep "${POLL_SECS}"
done

# ── Build argument lists (include actor fold1 if present) ─────────────────────
ACTOR_ARGS=()
if [[ -d "${ACTOR_FOLD1_DIR}/shards" ]]; then
    ACTOR_ARGS+=(--actor_shap_dirs "${ACTOR_FOLD1_DIR}")
    log "Including ActorSHAP results from ${ACTOR_FOLD1_DIR}"
else
    log "WARNING: ActorSHAP fold1 dir not found; running baseline-only comparison."
fi

# ── Run consistency analysis ──────────────────────────────────────────────────
log "Running evaluate_shap_consistency.py …"

python3 evaluate_shap_consistency.py \
    "${ACTOR_ARGS[@]}" \
    --baseline_dirs \
        "${FOLD1_BASELINE_DIR}" \
        "${FOLD2_BASELINE_DIR}" \
        "${FOLD8_BASELINE_DIR}" \
    --eval_pkls \
        "${PKL_1}" \
        "${PKL_2}" \
        "${PKL_8}" \
    --output_dir "${OUTPUT_DIR}"

log "Done. Results written to ${OUTPUT_DIR}/"
