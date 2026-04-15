#!/usr/bin/env python3
"""run_physics_temporal_shap.py — Temporal SHAP with physics imputation (same setup as evaluate_shap_baselines).

Computes ``compute_temporal_shap_baseline(..., method='physics')`` and batched
temporal faithfulness (deletion/insertion AUC, Shapley completeness error) for
each test sequence index listed in ``--physics_seq_dir`` (same ``seq_*.json``
names as ``precompute_physics_shap.py`` / spatial cache).

Requires ``--physics_stats`` (motion stats pickle) so ``PhysicsInformedCompleter``
can run; spatial results can still come from a separate JSON cache when using
``evaluate_shap_baselines.py`` with both ``--physics_cache_dir`` and
``--physics_stats``.

Example::

    python scripts/run_physics_temporal_shap.py \\
        --backbone potr --config BMCLab.json --num_folds 23 --fold 1 \\
        --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \\
        --physics_stats experiment_outs/motion_stats_fold1.pkl \\
        --physics_seq_dir results/physics_cache_fold1 \\
        --output_jsonl results/temporal_physics_fold1.jsonl \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataloaders import collate_fn
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.physics_completer import load_physics_completer
from model.actor.shap_compute import compute_temporal_shap_baseline
from model.actor.shap_masking import build_temporal_windows, detect_stride_period
from model.actor.shap_metrics import compute_shapley_completeness
from model.actor.shap_eval_shared import (
    _load_backbone_params,
    _raw_data_args,
    build_train_pool,
    build_zscore_stats_for_potr,
    load_motion_encoder,
    build_classifier_fn,
    _temporal_deletion_insertion_auc_batched,
)

_SEQ_RE = re.compile(r"^seq_(\d{4})\.json$")


def _seq_indices_from_dir(d: str) -> list[int]:
    out: list[int] = []
    for name in sorted(os.listdir(d)):
        m = _SEQ_RE.match(name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Temporal physics SHAP for sequences listed in a physics cache directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backbone", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--num_folds", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--classifier_ckpt", required=True)
    p.add_argument(
        "--physics_stats",
        required=True,
        help="motion_stats_fold*.pkl for PhysicsInformedCompleter.",
    )
    p.add_argument(
        "--physics_seq_dir",
        required=True,
        help="Directory containing seq_*.json (indices determine which test sequences to run).",
    )
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_marginal_samples", type=int, default=5,
                   help="Physics completions averaged per temporal coalition.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_train_pool", type=int, default=2000)
    p.add_argument("--train_pool_batch_size", type=int, default=64)
    p.add_argument("--root_centered", action="store_true", default=False)
    args = p.parse_args()

    device = torch.device(args.device)
    want = set(_seq_indices_from_dir(args.physics_seq_dir))
    if not want:
        raise SystemExit(f"No seq_*.json files found in {args.physics_seq_dir!r}")

    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    backbone_name = backbone_params["backbone"]

    print("[1/3] Loading classifier …")
    motion_encoder = load_motion_encoder(args.classifier_ckpt, backbone_params, device)

    print("[2/3] Train pool + z-score + physics completer …")
    train_pool, joint_means = build_train_pool(
        backbone_params, args.fold, device,
        max_sequences=args.max_train_pool,
        batch_size=args.train_pool_batch_size,
        root_centered=args.root_centered,
    )
    if backbone_name == "potr":
        zscore_mean, zscore_std = build_zscore_stats_for_potr(
            backbone_params, args.fold, device,
            batch_size=args.train_pool_batch_size,
            root_centered=args.root_centered,
        )
    else:
        zscore_mean = zscore_std = None

    physics_completer = load_physics_completer(args.physics_stats, device)

    print("[3/3] Temporal physics SHAP per sequence …")
    data_args = _raw_data_args(backbone_params, args.fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    open(args.output_jsonl, "w").close()

    n_done = 0
    for seq_idx, raw_batch in enumerate(test_loader):
        if seq_idx not in want:
            continue

        x_raw, labels, _, _, pad_mask = raw_batch
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params["num_classes"], device,
            y=labels, root_centered=args.root_centered,
        )
        x = batch["x"]
        y = batch["y"]
        mask = batch["mask"]
        lengths = batch["lengths"]

        video_name = test_ds.video_names[seq_idx]
        subject_id = video_name.split("__")[0]
        physics_completer.set_subject(subject_id)

        classifier_fn = build_classifier_fn(
            motion_encoder, mask, backbone_name,
            zscore_mean=zscore_mean, zscore_std=zscore_std,
            x_orig=x,
        )

        x_np = x[0].permute(2, 0, 1).cpu().numpy()
        stride_period, fallback = detect_stride_period(x_np, fps=args.fps)
        window_assignments = build_temporal_windows(x.shape[-1], stride_period, K=4)

        t_shap = compute_temporal_shap_baseline(
            "physics", classifier_fn, x, y, mask, lengths,
            window_assignments=window_assignments,
            physics_completer=physics_completer,
            n_marginal_samples=args.n_marginal_samples,
            seed=seq_idx,
        )
        t_windows = [k for k in t_shap if not k.startswith("_")]
        comp_err = compute_shapley_completeness(
            t_shap, t_shap["_v_full"], t_shap["_v_empty"],
            joint_names=t_windows,
        )
        del_ins = _temporal_deletion_insertion_auc_batched(
            classifier_fn, x, y, mask, lengths, window_assignments, t_shap, "physics",
            physics_completer=physics_completer,
            seed=seq_idx,
            n_marginal_samples=args.n_marginal_samples,
        )

        rec = {
            "seq_idx": seq_idx,
            "subject_id": subject_id,
            "true_class": int(y[0].item()),
            "stride_fallback": bool(fallback),
            "temporal_shap_values": {"physics": {k: float(v) for k, v in t_shap.items()}},
            "temporal_faithfulness": {
                "physics": {
                    "deletion_auc": del_ins["deletion_auc"],
                    "insertion_auc": del_ins["insertion_auc"],
                    "completeness_error": float(comp_err),
                },
            },
        }
        with open(args.output_jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        n_done += 1
        print(
            f"  seq {seq_idx}  del_auc={del_ins['deletion_auc']:.3f}  "
            f"ins_auc={del_ins['insertion_auc']:.3f}  comp_err={comp_err:.4f}",
            flush=True,
        )

    print(f"Done. Wrote {n_done} lines to {args.output_jsonl}")


if __name__ == "__main__":
    main()
