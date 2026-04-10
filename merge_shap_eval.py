"""merge_shap_eval.py — Merge shard outputs from evaluate_shap.py and compute FID.

After all ``--num_shards`` jobs finish, run this once on a single device::

    python merge_shap_eval.py \\
        --output_dir results/shap_run \\
        --actor_shap_ckpt … \\
        --backbone potr \\
        --config BMCLab.json \\
        --num_folds 23 \\
        --fold 1 \\
        --classifier_ckpt …

``--classifier_ckpt`` is accepted for parity with ``evaluate_shap.py`` (recorded in
``aggregate.json`` meta) but is not required to compute FID.

Reads ``<output_dir>/shards/per_sequence_shard*_of_*.jsonl``, merges and sorts by
``seq_idx``, writes ``per_sequence.jsonl`` and ``aggregate.json``, then runs
``compute_test_fid`` over the full (capped) test set.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import torch

from evaluate_shap import (
    aggregate_results,
    collect_test_batches_for_fid,
    compute_test_fid,
    load_actor_shap,
    _load_backbone_params,
)


def _load_shard_metas(shards_dir: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(shards_dir, "shard_meta_shard*_of_*.json")))
    out: list[dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge evaluate_shap shard JSONLs and compute aggregate + FID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Same directory passed to evaluate_shap.py (must contain shards/).",
    )
    parser.add_argument(
        "--actor_shap_ckpt", required=True,
        help="ActorSHAP checkpoint (for FID).",
    )
    parser.add_argument("--actor_shap_config", default=None)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--classifier_ckpt", default="",
        help="Optional; stored in aggregate.json _meta only.",
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--num_folds", type=int, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--n_completion_samples", type=int, default=20,
        help="Must match evaluate_shap FID setting.",
    )
    parser.add_argument(
        "--max_test_sequences", type=int, default=None,
        help="Must match evaluate_shap cap (default: from shard meta if present).",
    )
    parser.add_argument(
        "--skip_fid", action="store_true",
        help="Merge JSONLs and aggregates only; skip FID (debug).",
    )
    args = parser.parse_args()

    shards_dir = os.path.join(args.output_dir, "shards")
    if not os.path.isdir(shards_dir):
        raise SystemExit(f"Missing shards directory: {shards_dir}")

    pattern = os.path.join(shards_dir, "per_sequence_shard*_of_*.jsonl")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise SystemExit(f"No shard jsonl files matching {pattern}")

    rows: list[dict[str, Any]] = []
    for path in shard_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    if not rows:
        raise SystemExit("No JSON lines found in shard files.")

    rows.sort(key=lambda r: int(r["seq_idx"]))
    seq_indices = [int(r["seq_idx"]) for r in rows]

    metas = _load_shard_metas(shards_dir)
    n_cap_meta: int | None = None
    if metas:
        raw_n = metas[0].get("n_test_effective")
        if raw_n is not None:
            n_cap_meta = int(raw_n)
        for m in metas[1:]:
            r2 = m.get("n_test_effective")
            if r2 is None or n_cap_meta is None:
                continue
            if int(r2) != n_cap_meta:
                print(
                    "[merge_shap_eval] WARNING: inconsistent n_test_effective across "
                    "shard_meta files.",
                )
                break

    max_test = args.max_test_sequences
    if max_test is None and n_cap_meta is not None:
        max_test = n_cap_meta

    if n_cap_meta is not None and len(rows) != n_cap_meta:
        print(
            f"[merge_shap_eval] WARNING: merged {len(rows)} sequences but shard meta "
            f"expects n_test_effective={n_cap_meta} (incomplete shard set or cap mismatch).",
        )

    dup = len(seq_indices) - len(set(seq_indices))
    if dup:
        raise SystemExit(f"Duplicate seq_idx entries in shard data ({dup} duplicates).")

    merged_jsonl = os.path.join(args.output_dir, "per_sequence.jsonl")
    with open(merged_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    agg = aggregate_results(rows)
    fid: float | None = None

    if not args.skip_fid:
        device = torch.device(args.device)
        print("[merge] Loading ActorSHAP for FID …")
        actor_shap = load_actor_shap(
            args.actor_shap_ckpt, device, config_path=args.actor_shap_config,
        )
        print("[merge] Collecting test batches …")
        backbone_params = _load_backbone_params(
            args.backbone, args.config, args.num_folds,
        )
        batches = collect_test_batches_for_fid(
            backbone_params, args.fold, device,
            max_test_sequences=max_test,
        )
        print("[merge] Computing FID …")
        fid = compute_test_fid(
            actor_shap, batches, device,
            n_completions=args.n_completion_samples,
        )
        print(f"  FID = {fid:.4f}")
        agg["fid"] = {"mean": fid, "std": 0.0}
    else:
        agg["fid"] = {"mean": float("nan"), "std": 0.0}

    m0 = metas[0] if metas else {}
    m0_args = m0.get("args") or {}
    agg["_meta"] = {
        "backbone": args.backbone,
        "config": args.config,
        "num_folds": args.num_folds,
        "fold": args.fold,
        "n_test_sequences": len(rows),
        "n_kernel_samples": m0_args.get("n_kernel_samples"),
        "n_completion_samples": args.n_completion_samples,
        "n_rank_runs": m0_args.get("n_rank_runs"),
        "k_list": m0_args.get("k_list"),
        "fps": m0_args.get("fps"),
        "max_test_sequences": max_test,
        "actor_shap_ckpt": args.actor_shap_ckpt,
        "actor_shap_config": args.actor_shap_config,
        "classifier_ckpt": args.classifier_ckpt or None,
        "merged_from_shards": True,
        "num_shards": m0.get("num_shards"),
    }

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\nWrote {merged_jsonl} ({len(rows)} sequences)")
    print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
