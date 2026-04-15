#!/usr/bin/env python3
"""precompute_physics_shap.py — Offline physics-SHAP cache builder.

Runs the slow part of the physics baseline (per-coalition Cholesky
factorisation + sampling) once and writes per-sequence JSON files so that
evaluate_shap_baselines.py can skip it entirely via --physics_cache_dir.

Each output file  cache_dir/seq_{i:04d}.json  contains:
    {
        "seq_idx":    int,
        "subject_id": str,
        "true_class": int,
        "shap_values": {...},    # same dict as compute_spatial_shap_baseline
        "faithfulness": {...},   # same dict as compute_spatial_faithfulness_batched
    }

The script is resume-safe: at startup it scans ``cache_dir`` for existing
``seq_*.json`` files, prints how many will be skipped, and never reloads test
data for those indices (only missing sequences are loaded and computed).

Usage::

    # Full fold (overnight):
    python scripts/precompute_physics_shap.py \\
        --backbone potr --config BMCLab.json \\
        --num_folds 23 --fold 1 \\
        --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \\
        --physics_stats experiment_outs/motion_stats_fold1.pkl \\
        --cache_dir results/physics_cache_fold1 \\
        --physics_n_kernel_samples 500 \\
        --physics_n_samples 5 \\
        --device cpu

    # Quick smoke test (2 sequences):
    python scripts/precompute_physics_shap.py ... --max_sequences 2

    # Cap CPU usage (default 2 threads; avoids grabbing all cores for BLAS/torch):
    python scripts/precompute_physics_shap.py ... --num_threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time


def _configure_cpu_threads_before_imports() -> int:
    """Limit BLAS/OpenMP threads (must run before numpy/torch import).

    Reads ``--num_threads`` / ``--num_threads=N`` from argv, else env
    ``PRECOMPUTE_PHYSICS_NUM_THREADS``, else 2.
    """
    default = int(os.environ.get("PRECOMPUTE_PHYSICS_NUM_THREADS", "2"))
    n = default
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--num_threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
            break
        if a.startswith("--num_threads="):
            n = int(a.split("=", 1)[1])
            break
        i += 1
    n = max(1, n)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)
    return n


_PRECOMPUTE_NUM_THREADS = _configure_cpu_threads_before_imports()

import torch

torch.set_num_threads(_PRECOMPUTE_NUM_THREADS)
torch.set_num_interop_threads(1)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataloaders import collate_fn
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.physics_completer import load_physics_completer
from model.actor.shap_compute import compute_spatial_shap_baseline
from model.actor.shap_metrics import compute_spatial_faithfulness_batched
from model.actor.shap_metrics import _class_prob
from model.actor.shap_eval_shared import (
    _load_backbone_params,
    _raw_data_args,
    build_zscore_stats_for_potr,
    load_motion_encoder,
    build_classifier_fn,
)

_CACHE_SEQ_JSON_RE = re.compile(r"^seq_(\d{4})\.json$")


def _scan_cached_seq_indices(cache_dir: str) -> set[int]:
    """Return sequence indices that already have ``seq_{idx:04d}.json`` in *cache_dir*."""
    if not os.path.isdir(cache_dir):
        return set()
    found: set[int] = set()
    for name in os.listdir(cache_dir):
        m = _CACHE_SEQ_JSON_RE.match(name)
        if m:
            found.add(int(m.group(1)))
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Pre-compute physics SHAP results and cache to disk.",
    )
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--config",   required=True,
                    help="Config filename inside configs/<backbone>/, e.g. 'BMCLab.json'.")
    ap.add_argument("--num_folds", type=int, required=True)
    ap.add_argument("--fold",      type=int, required=True)
    ap.add_argument("--classifier_ckpt", required=True)
    ap.add_argument("--physics_stats",   required=True,
                    help="Path to motion_stats_fold*.pkl from compute_motion_stats.py.")
    ap.add_argument("--cache_dir", required=True,
                    help="Directory to write per-sequence JSON files.")
    ap.add_argument("--physics_n_kernel_samples", type=int, default=500)
    ap.add_argument("--physics_n_samples",        type=int, default=5)
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--max_sequences", type=int, default=None,
                    help="Cap on sequences (None = all; small values for smoke tests).")
    ap.add_argument("--start_seq", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="Physics sampling is CPU-bound; 'cpu' is fine. "
                         "Classifier forward passes benefit from 'cuda:0'.")
    ap.add_argument("--root_centered", action="store_true", default=False)
    ap.add_argument(
        "--num_threads",
        type=int,
        default=_PRECOMPUTE_NUM_THREADS,
        help="Max CPU threads for PyTorch and BLAS/OpenMP. Default 2, or "
             "PRECOMPUTE_PHYSICS_NUM_THREADS. Pass on the command line so it "
             "applies before heavy imports (same flag as read at startup).",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    k_list = tuple(args.k_list)

    os.makedirs(args.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    print("[1/4] Loading backbone params and classifier …", flush=True)
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    backbone_name   = backbone_params["backbone"]
    motion_encoder  = load_motion_encoder(args.classifier_ckpt, backbone_params, device)

    if backbone_name == "potr":
        print("  [potr] Computing z-score stats …", flush=True)
        zscore_mean, zscore_std = build_zscore_stats_for_potr(
            backbone_params, args.fold, device, root_centered=args.root_centered,
        )
    else:
        zscore_mean = zscore_std = None

    # ------------------------------------------------------------------
    print("[2/4] Loading physics completer …", flush=True)
    physics_completer = load_physics_completer(args.physics_stats, device)
    print(f"  n_kernel={args.physics_n_kernel_samples}  n_samples={args.physics_n_samples}",
          flush=True)

    # ------------------------------------------------------------------
    print("[3/4] Loading test data …", flush=True)
    data_args = _raw_data_args(backbone_params, args.fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    n_total = len(test_ds)
    print(f"  {n_total} test sequences", flush=True)

    cached_seqs = _scan_cached_seq_indices(args.cache_dir)
    if cached_seqs:
        ordered = sorted(cached_seqs)
        preview = ordered[:12]
        tail = ""
        if len(ordered) > 12:
            tail = f", … (+{len(ordered) - 12} more)"
        print(
            f"  Pre-scan: {len(cached_seqs)} sequences already in cache "
            f"(indices {preview}{tail}) — will skip those without loading data.",
            flush=True,
        )
    else:
        print("  Pre-scan: no seq_*.json in cache yet.", flush=True)

    # ------------------------------------------------------------------
    print("[4/4] Computing physics SHAP per sequence …", flush=True)

    n_done = n_skipped = 0
    for seq_idx in range(n_total):
        if seq_idx < args.start_seq:
            continue
        if args.max_sequences is not None and seq_idx >= args.start_seq + args.max_sequences:
            break

        out_path = os.path.join(args.cache_dir, f"seq_{seq_idx:04d}.json")
        if os.path.isfile(out_path):
            print(f"  seq {seq_idx}/{n_total-1}  already cached — skipping", flush=True)
            n_skipped += 1
            continue

        raw_batch = collate_fn([test_ds[seq_idx]])
        x_raw, labels, _, _, pad_mask = raw_batch
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params["num_classes"], device,
            y=labels, root_centered=args.root_centered,
        )
        x       = batch["x"]
        y       = batch["y"]
        mask    = batch["mask"]
        lengths = batch["lengths"]

        video_name = test_ds.video_names[seq_idx]
        subject_id = video_name.split("__")[0]
        true_class = int(y[0].item())

        physics_completer.set_subject(subject_id)

        classifier_fn = build_classifier_fn(
            motion_encoder, mask, backbone_name,
            zscore_mean=zscore_mean, zscore_std=zscore_std,
            x_orig=x,
        )

        t0 = time.perf_counter()

        # --- SHAP values ------------------------------------------------
        print(f"  seq {seq_idx}/{n_total-1}  subject={subject_id}  class={true_class}"
              f"  running KernelSHAP …", flush=True)
        shap_physics = compute_spatial_shap_baseline(
            "physics", classifier_fn, x, y, mask, lengths,
            physics_completer=physics_completer,
            n_kernel_samples=args.physics_n_kernel_samples,
            n_marginal_samples=args.physics_n_samples,
            seed=seq_idx,
        )

        # --- Faithfulness metrics ----------------------------------------
        p_full   = _class_prob(classifier_fn, x, true_class)
        p_ref    = shap_physics["_v_empty"]
        print(f"  seq {seq_idx}  running faithfulness metrics …", flush=True)
        metrics_physics = compute_spatial_faithfulness_batched(
            classifier_fn, x, y, mask, lengths, shap_physics, "physics",
            physics_completer=physics_completer,
            n_samples=args.physics_n_samples,
            k_list=k_list,
            p_full=p_full,
            p_ref=p_ref,
            seq_idx=seq_idx,
        )

        elapsed = time.perf_counter() - t0

        # --- Persist ----------------------------------------------------
        record = {
            "seq_idx":    seq_idx,
            "subject_id": subject_id,
            "true_class": true_class,
            "p_full":     p_full,
            "shap_values":  {k: float(v) for k, v in shap_physics.items()},
            "faithfulness": metrics_physics,
        }
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)

        n_done += 1
        print(f"  seq {seq_idx}  done  ({elapsed:.1f}s)  → {out_path}", flush=True)

    print(f"\nDone.  {n_done} computed, {n_skipped} skipped.")
    print(f"Cache dir: {args.cache_dir}")


if __name__ == "__main__":
    main()
