"""evaluate_shap_crosseval.py — Cross-evaluation of SHAP rankings under a fixed neutral protocol.

Motivation
==========
In the self-referential evaluation (evaluate_shap.py / evaluate_shap_baselines.py),
each method's SHAP values are both computed AND evaluated using the same imputation
function.  This makes cross-method comparison apples-to-oranges: ActorSHAP's PGI
measures on-manifold removal vs on-manifold prediction; Zero's PGI measures zeroed
removal vs zeroed prediction.  They do not answer the same question.

This script separates the two steps:

  1. SHAP rankings  — loaded from pre-computed per_sequence.jsonl files.
                       Four methods supported: actor, zero, mean, marginal.
  2. Faithfulness   — uses a SINGLE fixed evaluation protocol (Zero imputation)
                       for ALL methods.

The resulting PGI/PGU tables directly answer: "given a fixed, neutral measurement
ruler, does this method's joint ranking correctly identify what the classifier relies
on?"

Additional improvements over the original evaluation
-----------------------------------------------------
* ``PGI_norm@k = PGI@k / p_full`` (per sequence, then aggregated) removes the
  classifier-confidence floor effect caused by near-chance sequences.
* ``--min_p_full`` filter excludes near-chance sequences from the aggregate.
* Random baseline uses the same Zero evaluation, so rand_pgi and rand_pgu are
  directly comparable across methods (no per-method imputation difference).

Usage
=====
    python evaluate_shap_crosseval.py \\
        --backbone potr \\
        --config BMCLab.json \\
        --num_folds 23 \\
        --fold 1 \\
        --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \\
        --actor_results results/shap_actor_potr_bmclab_fold1_full \\
        --baseline_results results/shap_baselines_potr_bmclab_fold1_full \\
        --cache_dir results/shap_cache_fold1 \\
        --output_dir results/shap_crosseval_potr_bmclab_fold1 \\
        --device cuda:0

    # With confidence filter:
    python evaluate_shap_crosseval.py ... --min_p_full 0.35

    # Actor-only fold (no actor results yet, baselines only):
    python evaluate_shap_crosseval.py ... --actor_results none
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_shap_baselines import (
    _load_backbone_params,
    _raw_data_args,
    build_classifier_fn,
    build_train_pool,
    load_motion_encoder,
)
from model.actor.backbone_projection import compute_zscore_stats
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from data.dataloaders import collate_fn
from torch.utils.data import DataLoader
from model.actor.shap_masking import H36M_JOINT_NAMES
from model.actor.shap_metrics import (
    _class_prob,
    compute_pgi_pgu,
    compute_pgi_pgu_random,
    make_zero_impute_fn,
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_actor_shap_values(actor_results_dir: str) -> dict[int, dict]:
    """Load actor SHAP values from shards or per_sequence.jsonl, dedup by seq_idx."""
    by_idx: dict[int, dict] = {}

    shard_pattern = os.path.join(actor_results_dir, "shards", "per_sequence_shard*.jsonl")
    shard_files = sorted(glob.glob(shard_pattern))

    if shard_files:
        for path in shard_files:
            for rec in _load_jsonl(path):
                idx = rec["seq_idx"]
                if idx not in by_idx:
                    by_idx[idx] = rec
    else:
        main_jsonl = os.path.join(actor_results_dir, "per_sequence.jsonl")
        if os.path.exists(main_jsonl):
            for rec in _load_jsonl(main_jsonl):
                idx = rec["seq_idx"]
                if idx not in by_idx:
                    by_idx[idx] = rec

    return by_idx


def _load_baseline_shap_values(baseline_results_dir: str) -> dict[int, dict]:
    """Load baseline SHAP values from per_sequence.jsonl, dedup by seq_idx."""
    by_idx: dict[int, dict] = {}
    path = os.path.join(baseline_results_dir, "per_sequence.jsonl")
    for rec in _load_jsonl(path):
        idx = rec["seq_idx"]
        if idx not in by_idx:
            by_idx[idx] = rec
    return by_idx


def _load_cached_sequence(
    cache_dir: str,
    seq_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load x, y, mask, lengths from a cached .npz file."""
    path = os.path.join(cache_dir, f"seq_{seq_idx:05d}.npz")
    d = np.load(path, allow_pickle=True)
    x       = torch.tensor(d["x"].astype(np.float32)).unsqueeze(0).to(device)  # (1,J,F,T)
    y       = torch.tensor([int(d["y"])], dtype=torch.int64).to(device)
    mask    = torch.tensor(d["mask"]).unsqueeze(0).to(device)                  # (1,T)
    lengths = torch.tensor([int(d["lengths"])], dtype=torch.int64).to(device)
    return x, y, mask, lengths


def _build_sequence_store(
    backbone_params: dict,
    fold: int,
    device: torch.device,
    seq_indices: Optional[list[int]] = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load test sequences directly from the dataset (fallback when no cache).

    Returns a dict mapping seq_idx → (x, y, mask, lengths).
    Only loads the indices in seq_indices (or all if None).
    """
    data_args = _raw_data_args(backbone_params, fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    wanted = set(seq_indices) if seq_indices is not None else None
    store: dict = {}
    for idx, raw_batch in enumerate(loader):
        if wanted is not None and idx not in wanted:
            continue
        x_raw, labels, _, _, pad_mask = raw_batch
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params["num_classes"],
            device=device, y=labels,
        )
        store[idx] = (
            batch["x"],       # (1, J, F, T)
            batch["y"],       # (1,)
            batch["mask"],    # (1, T)
            batch["lengths"], # (1,)
        )
        if wanted is not None and len(store) == len(wanted):
            break

    return store


# ---------------------------------------------------------------------------
# Per-sequence cross-evaluation
# ---------------------------------------------------------------------------

def _extract_joint_shap(shap_values: dict) -> dict[str, float]:
    """Return only the 17 individual H36M joint entries from a SHAP values dict."""
    return {j: float(shap_values[j]) for j in H36M_JOINT_NAMES if j in shap_values}


def evaluate_sequence_crosseval(
    seq_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    shap_by_method: dict[str, dict],   # method -> {joint: value}
    classifier_fn,
    zero_impute_fn,
    p_full: float,
    k_list: tuple[int, ...],
) -> dict:
    """Evaluate all methods' SHAP rankings under Zero imputation.

    Args:
        seq_idx:         Sequence index (used as random seed for rand baseline).
        x:               (1, J, F, T) root-centred tensor.
        y:               (1,) class label.
        mask:            (1, T) valid-frame mask.
        lengths:         (1,) frame count.
        shap_by_method:  {method: {joint_name: shap_value}} for each method to eval.
        zero_impute_fn:  Imputation function wrapping the classifier (Zero protocol).
        p_full:          Baseline classifier probability (full sequence, true class).
        k_list:          k values for PGI/PGU.

    Returns:
        Dict with structure:
          {method: {
              "pgi_pgu": {k: {"pgi": float, "pgu": float}},
              "pgi_pgu_rand": {k: {"pgi": float, "pgu": float}},
              "pgi_norm": {k: float},   # PGI / p_full
              "pgu_norm": {k: float},   # PGU / p_full
              "pgi_minus_rand": {k: float},
              "pgu_minus_rand": {k: float},
              "pgi_norm_minus_rand_norm": {k: float},
          }}
          plus "rand_baseline" (shared across methods): {k: float}
    """
    # --- shared random baseline under zero imputation ---
    rand_result = compute_pgi_pgu_random(
        classifier_fn, x, y, mask, lengths, zero_impute_fn,
        k_list=k_list, seed=seq_idx,
    )

    out: dict = {}
    for method, shap_vals in shap_by_method.items():
        pgi_pgu = compute_pgi_pgu(
            classifier_fn, x, y, mask, lengths, shap_vals, zero_impute_fn,
            k_list=k_list,
        )

        method_result: dict = {
            "pgi_pgu":           {str(k): v for k, v in pgi_pgu.items()},
            "pgi_pgu_rand":      {str(k): v for k, v in rand_result.items()},
            "pgi_norm":          {},
            "pgu_norm":          {},
            "pgi_minus_rand":    {},
            "pgu_minus_rand":    {},
            "pgi_norm_minus_rand_norm": {},
        }

        for k, vals in pgi_pgu.items():
            rand_gap = rand_result[k]["pgi"]  # rand_pgi == rand_pgu by construction
            pgi = vals["pgi"]
            pgu = vals["pgu"]
            rand_norm = rand_gap / p_full if p_full > 0 else float("nan")

            method_result["pgi_norm"][str(k)]       = pgi / p_full if p_full > 0 else float("nan")
            method_result["pgu_norm"][str(k)]       = pgu / p_full if p_full > 0 else float("nan")
            method_result["pgi_minus_rand"][str(k)] = pgi - rand_gap
            method_result["pgu_minus_rand"][str(k)] = pgu - rand_gap
            method_result["pgi_norm_minus_rand_norm"][str(k)] = (
                (pgi / p_full - rand_norm) if p_full > 0 else float("nan")
            )

        out[method] = method_result

    return out


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _compute_aggregate(
    results: list[dict],
    k_list: list[int],
    methods: list[str],
    min_p_full: Optional[float] = None,
) -> dict:
    """Compute mean ± std for all metrics, optionally filtered by p_full."""
    filtered = results
    if min_p_full is not None:
        filtered = [r for r in results if r.get("p_full", 0.0) >= min_p_full]

    agg: dict = defaultdict(list)
    agg["_n_sequences"].append(len(filtered))

    for r in filtered:
        agg["p_full"].append(r["p_full"])
        ce = r.get("crosseval", {})
        for method in methods:
            if method not in ce:
                continue
            m = ce[method]
            for k in k_list:
                ks = str(k)
                prefix = f"{method}"
                if ks in m.get("pgi_pgu", {}):
                    agg[f"{prefix}/pgi@{k}"].append(m["pgi_pgu"][ks]["pgi"])
                    agg[f"{prefix}/pgu@{k}"].append(m["pgi_pgu"][ks]["pgu"])
                if ks in m.get("pgi_pgu_rand", {}):
                    agg[f"{prefix}/rand@{k}"].append(m["pgi_pgu_rand"][ks]["pgi"])
                if ks in m.get("pgi_norm", {}):
                    agg[f"{prefix}/pgi_norm@{k}"].append(m["pgi_norm"][ks])
                    agg[f"{prefix}/pgu_norm@{k}"].append(m["pgu_norm"][ks])
                if ks in m.get("pgi_minus_rand", {}):
                    agg[f"{prefix}/pgi_minus_rand@{k}"].append(m["pgi_minus_rand"][ks])
                    agg[f"{prefix}/pgu_minus_rand@{k}"].append(m["pgu_minus_rand"][ks])
                if ks in m.get("pgi_norm_minus_rand_norm", {}):
                    agg[f"{prefix}/pgi_norm_minus_rand_norm@{k}"].append(
                        m["pgi_norm_minus_rand_norm"][ks]
                    )

    summary = {"_n_sequences": len(filtered)}
    for key, vals in agg.items():
        if key == "_n_sequences":
            continue
        clean = [v for v in vals if v == v]  # drop NaN
        if clean:
            summary[key] = {"mean": float(np.mean(clean)), "std": float(np.std(clean))}

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-evaluation of SHAP rankings under Zero imputation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone",        required=True)
    parser.add_argument("--config",          required=True)
    parser.add_argument("--num_folds",       type=int, required=True)
    parser.add_argument("--fold",            type=int, required=True)
    parser.add_argument("--classifier_ckpt", required=True)
    parser.add_argument(
        "--actor_results",
        default=None,
        help="Directory containing actor SHAP results (shards/ or per_sequence.jsonl). "
             "Pass 'none' to skip actor method.",
    )
    parser.add_argument(
        "--baseline_results",
        required=True,
        help="Directory containing baseline per_sequence.jsonl.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Directory containing seq_NNNNN.npz cached tensors. "
             "If omitted or a file is missing, sequences are loaded from the raw dataset.",
    )
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--k_list", nargs="+", type=int, default=[1, 2, 3, 5],
    )
    parser.add_argument(
        "--min_p_full", type=float, default=None,
        help="Only include sequences with p_full >= this value in the aggregate. "
             "A second aggregate (all sequences) is always written.",
    )
    parser.add_argument(
        "--max_train_pool",         type=int, default=2000,
    )
    parser.add_argument(
        "--train_pool_batch_size",  type=int, default=64,
    )
    parser.add_argument(
        "--max_sequences", type=int, default=None,
        help="Cap on sequences for debugging (None = all).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load classifier + z-score stats
    # ------------------------------------------------------------------
    print("[1/4] Loading backbone params, classifier, and training pool …")
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    motion_encoder  = load_motion_encoder(args.classifier_ckpt, backbone_params, device)

    train_pool, joint_means = build_train_pool(
        backbone_params, args.fold, device,
        max_sequences=args.max_train_pool,
        batch_size=args.train_pool_batch_size,
    )
    zscore_mean, zscore_std = compute_zscore_stats(train_pool)
    zscore_mean = zscore_mean.to(device)
    zscore_std  = zscore_std.to(device)

    # ------------------------------------------------------------------
    # Load stored SHAP values
    # ------------------------------------------------------------------
    print("[2/4] Loading stored SHAP rankings …")

    actor_by_idx: dict[int, dict] = {}
    use_actor = args.actor_results and args.actor_results.lower() != "none"
    if use_actor:
        actor_by_idx = _load_actor_shap_values(args.actor_results)
        print(f"  Actor: {len(actor_by_idx)} unique sequences")

    baseline_by_idx = _load_baseline_shap_values(args.baseline_results)
    print(f"  Baselines: {len(baseline_by_idx)} unique sequences")

    # Determine common sequences
    if use_actor:
        common_idxs = sorted(set(actor_by_idx.keys()) & set(baseline_by_idx.keys()))
    else:
        common_idxs = sorted(baseline_by_idx.keys())

    if args.max_sequences is not None:
        common_idxs = common_idxs[: args.max_sequences]

    print(f"  Common sequences to evaluate: {len(common_idxs)}")

    # ------------------------------------------------------------------
    # Per-sequence cross-evaluation
    # ------------------------------------------------------------------
    print("[3/4] Running per-sequence cross-evaluation (Zero imputation for all) …")

    # Pre-load all sequences if cache is unavailable (dataset fallback).
    use_cache = args.cache_dir is not None and os.path.isdir(args.cache_dir)
    if not use_cache:
        print("  No cache dir — loading test sequences from raw dataset …")
        seq_store = _build_sequence_store(
            backbone_params, args.fold, device, seq_indices=common_idxs,
        )
        print(f"  Loaded {len(seq_store)} sequences from dataset.")
    else:
        seq_store = {}

    per_seq_results: list[dict] = []
    jsonl_path = os.path.join(args.output_dir, "per_sequence.jsonl")

    with open(jsonl_path, "w") as jsonl_out:
        for i, seq_idx in enumerate(common_idxs):
            print(f"  [{i+1}/{len(common_idxs)}] seq {seq_idx} …", end="\r")

            # Load tensors — prefer cache, fall back to dataset store.
            cache_path = (
                os.path.join(args.cache_dir, f"seq_{seq_idx:05d}.npz")
                if use_cache else None
            )
            if cache_path and os.path.exists(cache_path):
                x, y, mask, lengths = _load_cached_sequence(
                    args.cache_dir, seq_idx, device
                )
            elif seq_idx in seq_store:
                x, y, mask, lengths = seq_store[seq_idx]
            else:
                print(f"\n  [warn] seq {seq_idx}: no cache file and not in dataset store — skipping.")
                continue

            # Build per-sequence classifier with correct mask + z-score
            classifier_fn = build_classifier_fn(
                motion_encoder, mask, args.backbone,
                zscore_mean=zscore_mean, zscore_std=zscore_std,
            )
            zero_impute_fn = make_zero_impute_fn(classifier_fn)

            # p_full for this sequence
            p_full = float(_class_prob(classifier_fn, x, int(y[0].item())))

            # Collect SHAP rankings for all available methods
            shap_by_method: dict[str, dict] = {}

            if use_actor and seq_idx in actor_by_idx:
                actor_rec = actor_by_idx[seq_idx]
                raw_actor = actor_rec.get("shap_values", {}).get("actor", {})
                if raw_actor:
                    shap_by_method["actor"] = _extract_joint_shap(raw_actor)

            if seq_idx in baseline_by_idx:
                bl_rec = baseline_by_idx[seq_idx]
                for method in ("zero", "mean", "marginal"):
                    raw = bl_rec.get("shap_values", {}).get(method, {})
                    if raw:
                        shap_by_method[method] = _extract_joint_shap(raw)

            if not shap_by_method:
                continue

            # Run cross-evaluation
            crosseval = evaluate_sequence_crosseval(
                seq_idx=seq_idx,
                x=x, y=y, mask=mask, lengths=lengths,
                shap_by_method=shap_by_method,
                classifier_fn=classifier_fn,
                zero_impute_fn=zero_impute_fn,
                p_full=p_full,
                k_list=tuple(args.k_list),
            )

            result = {
                "seq_idx":    seq_idx,
                "true_class": int(y[0].item()),
                "p_full":     p_full,
                "crosseval":  crosseval,
            }

            per_seq_results.append(result)
            jsonl_out.write(json.dumps(result) + "\n")
            jsonl_out.flush()

    print(f"\n  Wrote {len(per_seq_results)} sequences to {jsonl_path}")

    # ------------------------------------------------------------------
    # Aggregates: all sequences + filtered by min_p_full
    # ------------------------------------------------------------------
    print("[4/4] Computing aggregate statistics …")

    available_methods = list(
        {m for r in per_seq_results for m in r.get("crosseval", {}).keys()}
    )
    available_methods = sorted(available_methods)

    agg_all = _compute_aggregate(
        per_seq_results, args.k_list, available_methods, min_p_full=None,
    )
    agg_all["_meta"] = {
        "backbone":        args.backbone,
        "config":          args.config,
        "num_folds":       args.num_folds,
        "fold":            args.fold,
        "evaluation":      "zero_imputation_crosseval",
        "n_sequences":     len(per_seq_results),
        "k_list":          args.k_list,
        "min_p_full_filter": None,
        "classifier_ckpt": args.classifier_ckpt,
    }

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg_all, f, indent=2)
    print(f"  Aggregate (all {len(per_seq_results)} seqs) → {agg_path}")

    if args.min_p_full is not None:
        agg_filtered = _compute_aggregate(
            per_seq_results, args.k_list, available_methods, min_p_full=args.min_p_full,
        )
        agg_filtered["_meta"] = {**agg_all["_meta"], "min_p_full_filter": args.min_p_full}
        filtered_path = os.path.join(
            args.output_dir,
            f"aggregate_pful{int(args.min_p_full * 100):02d}.json",
        )
        with open(filtered_path, "w") as f:
            json.dump(agg_filtered, f, indent=2)
        n_filtered = agg_filtered["_n_sequences"]
        print(f"  Aggregate (p_full≥{args.min_p_full}, n={n_filtered}) → {filtered_path}")


if __name__ == "__main__":
    main()
