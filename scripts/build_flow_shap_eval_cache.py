"""
build_flow_shap_eval_cache.py — build a flow-SHAP eval cache on the *classifier's*
held-out subjects so that flow-SHAP faithfulness can be evaluated on the same
test fold as the SHAP baselines in ``evaluate_shap_baselines.py``.

The flow model was trained with a 6-fold subject split, so its val split
(held-out for the flow) does *not* correspond to the 23-fold LOSO split used
by the POTR classifier. That made earlier flow-SHAP runs leak two POTR-training
subjects (SUB15, SUB21) into the metrics, which invalidated comparisons.

This script:

1. Loads the flow's train cache (to reuse its z-score stats — the flow was
   trained with those stats, so we must keep them).
2. Loads the classifier's fold pickle (e.g. BMCLab 23-fold) and extracts the
   list of eval subjects (e.g. ``SUB01``).
3. Gathers pelvis-centered 80-frame clips at stride 80 for those subjects.
4. Applies the flow's z-score stats.
5. Writes a new ``cache.npz`` with the eval subject clips placed in the
   ``_val`` slot (train slots empty), so ``model.flow_shap.load_flow_cache``
   loads them unchanged with ``split="val"``.

Usage::

    python scripts/build_flow_shap_eval_cache.py \
        --source_cache cache/flow_matching/BMCLab_h36m_80_fold1 \
        --dataset BMCLab --data_type h36m \
        --num_folds 23 --fold 1 \
        --seq_len 80 --clip_stride 80 \
        --out_dir cache/flow_matching/BMCLab_h36m_80_classifier23fold_fold1_eval
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_velocity import (  # noqa: E402
    DATASET_NPZ,
    FOLD_PICKLE,
    LABEL_PICKLE,
    apply_zscore,
    gather_clips,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_cache", required=True,
                   help="Directory containing the flow-training cache.npz "
                        "(used only for its stats_mean / stats_std).")
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_NPZ.keys()))
    p.add_argument("--data_type", default="h36m")
    p.add_argument("--num_folds", type=int, required=True,
                   help="Classifier fold count (e.g. 23 for BMCLab LOSO).")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--clip_stride", type=int, default=80,
                   help="Stride for gathering eval clips (80 = non-overlap).")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    src_cache_path = Path(args.source_cache) / "cache.npz"
    if not src_cache_path.is_absolute():
        src_cache_path = PROJECT_ROOT / src_cache_path
    if not src_cache_path.exists():
        raise FileNotFoundError(f"Source cache not found: {src_cache_path}")
    print(f"[eval_cache] loading z-score stats from: {src_cache_path}", flush=True)
    src = np.load(str(src_cache_path), allow_pickle=True)
    stats_mean = src["stats_mean"].astype(np.float32)
    stats_std = src["stats_std"].astype(np.float32)
    assert stats_mean.shape == (17, 3), f"unexpected stats_mean shape: {stats_mean.shape}"

    fold_key = (args.dataset, args.num_folds)
    if fold_key not in FOLD_PICKLE:
        raise KeyError(
            f"No fold pickle registered for {fold_key}. "
            f"Available: {sorted(FOLD_PICKLE.keys())}"
        )
    fold_pkl_path = PROJECT_ROOT / FOLD_PICKLE[fold_key]
    with open(fold_pkl_path, "rb") as f:
        fold_splits = pickle.load(f)
    if args.fold not in fold_splits:
        raise KeyError(
            f"Fold {args.fold} not in {fold_pkl_path}. "
            f"Available: {sorted(fold_splits.keys())}"
        )
    eval_pids = set(fold_splits[args.fold]["eval"])
    print(f"[eval_cache] eval subjects for fold {args.fold}: "
          f"{sorted(eval_pids)}", flush=True)

    # labels for meta enrichment
    labels = None
    if args.dataset in LABEL_PICKLE:
        lbl_path = PROJECT_ROOT / LABEL_PICKLE[args.dataset]
        if lbl_path.exists():
            with open(lbl_path, "rb") as f:
                labels = pickle.load(f)

    npz_path = PROJECT_ROOT / DATASET_NPZ[args.dataset]
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {npz_path}")
    npz = np.load(str(npz_path), allow_pickle=False)

    print(f"[eval_cache] gathering eval clips (stride={args.clip_stride}, "
          f"seq_len={args.seq_len})", flush=True)
    eval_clips_raw, eval_masks, eval_meta = gather_clips(
        npz, eval_pids, args.seq_len, args.clip_stride, labels,
    )
    assert eval_clips_raw.shape[1:] == (args.seq_len, 17, 3), (
        f"eval clips shape {eval_clips_raw.shape} unexpected"
    )
    # pelvis after centering should be exactly zero
    pelvis_max = float(np.abs(eval_clips_raw[:, :, 0, :]).max())
    assert pelvis_max < 1e-4, f"pelvis not zeroed after centering ({pelvis_max})"

    x1_eval = apply_zscore(eval_clips_raw, stats_mean, stats_std)
    print(f"[eval_cache] eval clips: {x1_eval.shape[0]}  "
          f"(real-frame fraction: {float(eval_masks.mean()):.3f})", flush=True)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.npz"
    if cache_path.exists() and not args.force:
        raise FileExistsError(f"{cache_path} already exists. Use --force.")

    def _meta_to_arrays(meta: list[dict]) -> dict[str, np.ndarray]:
        return {
            "meta_pid": np.array([m["pid"] for m in meta], dtype=object),
            "meta_walk_id": np.array([m["walk_id"] for m in meta], dtype=object),
            "meta_seq_key": np.array([m["seq_key"] for m in meta], dtype=object),
            "meta_clip_idx": np.array([m["clip_idx"] for m in meta], dtype=np.int32),
            "meta_updrs_gait": np.array([m["updrs_gait"] for m in meta], dtype=np.int32),
            "meta_medication": np.array([m["medication"] for m in meta], dtype=object),
        }

    val_meta_arrays = {f"{k}_val": v for k, v in _meta_to_arrays(eval_meta).items()}

    empty_meta = {
        "meta_pid_train": np.array([], dtype=object),
        "meta_walk_id_train": np.array([], dtype=object),
        "meta_seq_key_train": np.array([], dtype=object),
        "meta_clip_idx_train": np.array([], dtype=np.int32),
        "meta_updrs_gait_train": np.array([], dtype=np.int32),
        "meta_medication_train": np.array([], dtype=object),
    }

    # Empty train arrays with compatible shape (N=0, T, J, C) so flow-matching
    # schema is preserved even though no training data is exported.
    payload = {
        "x1_train": np.zeros((0, args.seq_len, 17, 3), dtype=np.float32),
        "mask_train": np.zeros((0, args.seq_len), dtype=bool),
        "x1_val": x1_eval,
        "mask_val": eval_masks,
        "stats_mean": stats_mean,
        "stats_std": stats_std,
        "seq_len": np.int32(args.seq_len),
        "fold": np.int32(args.fold),
        "num_folds": np.int32(args.num_folds),
        "clip_stride_train": np.int32(args.clip_stride),
        "clip_stride_val": np.int32(args.clip_stride),
        "dataset": np.array(args.dataset, dtype=object),
        "data_type": np.array(args.data_type, dtype=object),
        **empty_meta,
        **val_meta_arrays,
    }
    np.savez_compressed(cache_path, **payload)
    print(f"[eval_cache] wrote {cache_path}", flush=True)

    summary = {
        "source_cache": str(src_cache_path),
        "dataset": args.dataset,
        "data_type": args.data_type,
        "num_folds": args.num_folds,
        "fold": args.fold,
        "seq_len": args.seq_len,
        "clip_stride": args.clip_stride,
        "eval_subjects": sorted(eval_pids),
        "num_eval_clips": int(x1_eval.shape[0]),
        "stats_reused_from_source": True,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval_cache] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
