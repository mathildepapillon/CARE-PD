"""
generate_velocity.py — cache normalized x_1 clips for flow-matching training.

For a CARE-PD dataset (default BMCLab h36m 3D) and a given fold, this script:

1.  Loads the NPZ of ``(T, 17, 3)`` world-coordinate sequences.
2.  Partitions sequences by participant ID into a ``train`` / ``val`` / ``eval``
    set using the project's per-dataset fold pickle (``eval`` is never read —
    it is reserved for downstream SHAP).
3.  Pelvis-centres every frame (``seq - seq[:, :1, :]``).
4.  Splits each sequence into 80-frame clips using
    ``--clip_stride_train`` (default 20, i.e. 75 percent overlap) for train
    and ``--clip_stride_val`` (default 80, non-overlap) for val.
5.  Computes z-score statistics ``(mean, std)`` of shape ``(17, 3)`` from
    the **train** clips only (over real, non-padded frames).
6.  Normalizes all clips with those train stats.
7.  Saves a compressed ``cache.npz`` + ``summary.json`` + ``sanity.npz``
    (deterministic ``(x_0, t, x_t, u_t)`` tuples produced via
    ``flow_matching.path.AffineProbPath(CondOTScheduler())``).

Running sanity checks (asserted inline, fail-fast):
* shape assertions
* clip count matches stride arithmetic
* pelvis joint ~ 0 after centering (before z-score)
* post-z-score per-coord train mean ~ 0, std ~ 1 on real frames
* AffineProbPath identities ``x_t = (1-t) x_0 + t x_1`` and ``dx_t = x_1 - x_0``
* no participant leakage between train / val / eval

Usage
-----
::

    python scripts/generate_velocity.py \
        --dataset BMCLab --data_type h36m --num_folds 6 --fold 1 \
        --seq_len 80 --clip_stride_train 20 --clip_stride_val 80 \
        --val_frac 0.1 --seed 42 \
        --out_dir cache/flow_matching/BMCLab_h36m_80_fold1
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Dataset / fold registries (mirror train_lstm_vae.py's conventions)
# ---------------------------------------------------------------------------

DATASET_NPZ = {
    "BMCLab":   "assets/datasets/h36m/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
    "T-SDU-PD": "assets/datasets/h36m/T-SDU-PD/h36m_3d_world_floorXZZplus_30f_or_longer_slopeCorrected.npz",
    "PD-GaM":   "assets/datasets/h36m/PD-GaM/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
    "3DGait":   "assets/datasets/h36m/3DGait/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
    "H36M":     "assets/datasets/h36m/H36M/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
}

FOLD_PICKLE = {
    ("BMCLab",   6):  "assets/datasets/folds/UPDRS_Datasets/BMCLab_6fold_participants.pkl",
    ("BMCLab",  23):  "assets/datasets/folds/UPDRS_Datasets/BMCLab_23fold_participants.pkl",
    ("T-SDU-PD", 14): "assets/datasets/folds/UPDRS_Datasets/T-SDU-PD_14fold_participants.pkl",
    ("3DGait",   6):  "assets/datasets/folds/UPDRS_Datasets/3DGait_6fold_participants.pkl",
    ("3DGait",  43):  "assets/datasets/folds/UPDRS_Datasets/3DGait_43fold_participants.pkl",
    ("PD-GaM",   6):  "assets/datasets/folds/UPDRS_Datasets/PD-GaM_6fold_participants.pkl",
    ("PD-GaM",  30):  "assets/datasets/folds/UPDRS_Datasets/PD-GaM_30fold_participants.pkl",
}

LABEL_PICKLE = {
    "BMCLab":   "assets/datasets/BMCLab.pkl",
    "T-SDU-PD": "assets/datasets/T-SDU-PD.pkl",
    "PD-GaM":   "assets/datasets/PD-GaM.pkl",
    "3DGait":   "assets/datasets/3DGait.pkl",
}

N_JOINTS = 17


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clips(seq: np.ndarray, clip_len: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``seq`` of shape ``(T, J, C)`` into ``(n, clip_len, J, C)`` windows
    using the given ``stride``.

    Any trailing remainder that does not yield a clip start is dropped.
    The last *started* window is zero-padded if it runs past ``T``.

    Returns ``(clips, masks)`` where ``masks`` is ``(n, clip_len)`` bool with
    ``True`` on real frames and ``False`` on zero-padded frames.
    """
    T, J, C = seq.shape
    if T < clip_len:
        pad = np.zeros((clip_len - T, J, C), dtype=seq.dtype)
        clip = np.concatenate([seq, pad], axis=0)
        mask = np.zeros((clip_len,), dtype=bool)
        mask[:T] = True
        return clip[None], mask[None]

    starts = list(range(0, T - clip_len + 1, stride))
    # Keep 1-based tail if the stride missed the final valid window, so we
    # don't drop the last real piece of motion entirely.
    if starts and starts[-1] + clip_len < T:
        starts.append(T - clip_len)

    clips = np.stack([seq[s:s + clip_len] for s in starts], axis=0)
    masks = np.ones((len(starts), clip_len), dtype=bool)
    return clips, masks


def pelvis_centre(seq: np.ndarray) -> np.ndarray:
    """Subtract joint 0 (pelvis/hip) at every frame."""
    return seq - seq[:, :1, :]


def parse_seq_key(key: str) -> tuple[str, str]:
    """NPZ keys look like ``"SUB01__SUB01_off_walk_1_down0"`` (CARE-PD
    convention). Returns ``(participant_id, walk_id)``."""
    head, _, tail = key.partition("__")
    # walk_id = ``tail`` stripped of downsample suffix (e.g. ``_down0``)
    walk_id = tail.rsplit("_down", 1)[0]
    return head, walk_id


def make_partition(
    keys: list[str],
    train_pids: set[str],
    eval_pids: set[str],
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[set[str], set[str], set[str]]:
    """Split training participants into train/val by participant (seeded)."""
    train_pids_sorted = sorted(train_pids)
    n_val = max(1, int(round(val_frac * len(train_pids_sorted))))
    idx = rng.permutation(len(train_pids_sorted))
    val_set = {train_pids_sorted[i] for i in idx[:n_val]}
    tr_set = {train_pids_sorted[i] for i in idx[n_val:]}
    return tr_set, val_set, set(eval_pids)


def gather_clips(
    npz: np.lib.npyio.NpzFile,
    pid_set: set[str],
    clip_len: int,
    stride: int,
    labels: dict | None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Pelvis-centre and slice every sequence whose participant is in
    ``pid_set``. Returns ``(clips, masks, meta)``."""
    clip_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    meta: list[dict] = []
    for key in npz.files:
        pid, walk_id = parse_seq_key(key)
        if pid not in pid_set:
            continue
        seq = npz[key].astype(np.float32)
        if seq.ndim != 3 or seq.shape[1] != N_JOINTS or seq.shape[2] != 3:
            raise ValueError(
                f"Unexpected shape for key {key!r}: {seq.shape}; "
                f"expected (T, {N_JOINTS}, 3)."
            )
        seq = pelvis_centre(seq)
        clips, masks = make_clips(seq, clip_len, stride)
        walk_lbl = (labels.get(pid, {}) or {}).get(walk_id, {}) if labels else {}
        updrs = walk_lbl.get("UPDRS_GAIT", -1) if isinstance(walk_lbl, dict) else -1
        med = walk_lbl.get("medication", "na") if isinstance(walk_lbl, dict) else "na"
        for i in range(clips.shape[0]):
            meta.append(
                {
                    "pid": pid,
                    "walk_id": walk_id,
                    "seq_key": key,
                    "clip_idx": i,
                    "updrs_gait": int(updrs) if isinstance(updrs, (int, np.integer)) else -1,
                    "medication": str(med),
                }
            )
        clip_chunks.append(clips)
        mask_chunks.append(masks)
    if not clip_chunks:
        raise RuntimeError(f"No clips produced for pid_set of size {len(pid_set)}.")
    return (
        np.concatenate(clip_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0),
        meta,
    )


def compute_zscore_stats(
    clips: np.ndarray, masks: np.ndarray, std_floor: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-(joint, coord) mean and std over real frames only.

    ``clips`` is ``(N, T, 17, 3)``; ``masks`` is ``(N, T)`` bool.

    Stats are accumulated in float64 to avoid numerical issues for joints
    whose pelvis-relative offset is nearly constant across the dataset
    (e.g. the hip joint in retargeted h36m skeletons, where the std in
    metres can be ~1 mm). A small ``std_floor`` (default 1 cm) is applied
    to prevent dividing by near-zero which would otherwise amplify float32
    rounding error into huge post-z-score values. Joints that barely move
    are not informative anyway, so saturating their normalization is safe.
    """
    real = clips[masks].astype(np.float64)  # (M, 17, 3)
    mean = real.mean(axis=0)
    std = real.std(axis=0)
    std = np.maximum(std, std_floor)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(
    clips: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    # Do the subtraction/division in float64 to minimise loss-of-significance
    # on near-constant joint coordinates; cast back to float32 at the end.
    return ((clips.astype(np.float64) - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Sanity tuple generation (fixed (x_0, t, x_t, u_t) for reproducibility)
# ---------------------------------------------------------------------------

def make_sanity_tuples(
    x1: np.ndarray, n_sanity: int, seed: int
) -> dict[str, np.ndarray]:
    """Run a deterministic AffineProbPath/CondOTScheduler sample on the first
    ``n_sanity`` clips of ``x1`` and return the tuple plus identity checks."""
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler

    n = min(n_sanity, x1.shape[0])
    g = torch.Generator().manual_seed(seed)
    x1_t = torch.from_numpy(x1[:n].copy())
    x0_t = torch.randn(x1_t.shape, generator=g, dtype=x1_t.dtype)
    t_t = torch.rand((n,), generator=g, dtype=x1_t.dtype)

    path = AffineProbPath(scheduler=CondOTScheduler())
    out = path.sample(t=t_t, x_0=x0_t, x_1=x1_t)
    t_broadcast = t_t.view(-1, *([1] * (x1_t.ndim - 1)))
    assert torch.allclose(out.x_t, (1 - t_broadcast) * x0_t + t_broadcast * x1_t, atol=1e-5), \
        "AffineProbPath.x_t does not match (1-t)*x_0 + t*x_1"
    assert torch.allclose(out.dx_t, x1_t - x0_t, atol=1e-5), \
        "AffineProbPath.dx_t does not match x_1 - x_0"

    return {
        "x0": x0_t.numpy(),
        "t": t_t.numpy(),
        "x_t": out.x_t.numpy(),
        "u_t": out.dx_t.numpy(),
        "x1": x1_t.numpy(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _safe_int_cast(x: Iterable[int]) -> list[int]:
    return [int(v) for v in x]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BMCLab", choices=sorted(DATASET_NPZ.keys()))
    p.add_argument("--data_type", default="h36m", choices=["h36m"])
    p.add_argument("--num_folds", type=int, default=6)
    p.add_argument("--fold", type=int, default=1,
                   help="Fold key present in the fold pickle (1-based in CARE-PD).")
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--clip_stride_train", type=int, default=20)
    p.add_argument("--clip_stride_val", type=int, default=80)
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of training participants to hold out for flow validation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_sanity_clips", type=int, default=16,
                   help="Number of clips to use when generating sanity (x0, t, x_t, u_t) tuples.")
    p.add_argument("--out_dir", default=None,
                   help="Output directory. Defaults to cache/flow_matching/<dataset>_<data_type>_<T>_fold<F>.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing cache.npz if present.")
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = (
            f"cache/flow_matching/"
            f"{args.dataset}_{args.data_type}_{args.seq_len}_fold{args.fold}"
        )
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.npz"
    summary_path = out_dir / "summary.json"
    sanity_path = out_dir / "sanity.npz"

    if cache_path.exists() and not args.force:
        raise FileExistsError(
            f"{cache_path} already exists. Use --force to overwrite."
        )

    rng = np.random.default_rng(args.seed)

    # -- load fold pickle -----------------------------------------------------
    fold_key = (args.dataset, args.num_folds)
    if fold_key not in FOLD_PICKLE:
        raise KeyError(
            f"No fold pickle registered for dataset={args.dataset!r}, "
            f"num_folds={args.num_folds}. Registered: {sorted(FOLD_PICKLE.keys())}"
        )
    fold_pkl_path = PROJECT_ROOT / FOLD_PICKLE[fold_key]
    with open(fold_pkl_path, "rb") as f:
        fold_splits = pickle.load(f)
    if args.fold not in fold_splits:
        raise KeyError(
            f"Fold {args.fold} not found in {fold_pkl_path}. "
            f"Available: {sorted(fold_splits.keys())}"
        )
    fold = fold_splits[args.fold]
    train_pids_all = set(fold["train"])
    eval_pids = set(fold["eval"])

    tr_pids, val_pids, eval_pids = make_partition(
        keys=[], train_pids=train_pids_all, eval_pids=eval_pids,
        val_frac=args.val_frac, rng=rng,
    )
    assert tr_pids.isdisjoint(val_pids), "train/val pids overlap"
    assert tr_pids.isdisjoint(eval_pids), "train/eval pids overlap"
    assert val_pids.isdisjoint(eval_pids), "val/eval pids overlap"

    # -- load data + labels ---------------------------------------------------
    npz_path = PROJECT_ROOT / DATASET_NPZ[args.dataset]
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {npz_path}")
    npz = np.load(str(npz_path), allow_pickle=False)

    labels = None
    if args.dataset in LABEL_PICKLE:
        lbl_path = PROJECT_ROOT / LABEL_PICKLE[args.dataset]
        if lbl_path.exists():
            with open(lbl_path, "rb") as f:
                labels = pickle.load(f)

    # -- build train / val clips ---------------------------------------------
    print(f"[generate_velocity] gathering train clips (stride={args.clip_stride_train})")
    train_clips_raw, train_masks, train_meta = gather_clips(
        npz, tr_pids, args.seq_len, args.clip_stride_train, labels,
    )
    print(f"[generate_velocity] gathering val clips (stride={args.clip_stride_val})")
    val_clips_raw, val_masks, val_meta = gather_clips(
        npz, val_pids, args.seq_len, args.clip_stride_val, labels,
    )

    # -- shape / pelvis sanity -----------------------------------------------
    assert train_clips_raw.shape[1:] == (args.seq_len, N_JOINTS, 3), (
        f"train clips shape {train_clips_raw.shape} unexpected"
    )
    assert val_clips_raw.shape[1:] == (args.seq_len, N_JOINTS, 3), (
        f"val clips shape {val_clips_raw.shape} unexpected"
    )
    assert train_masks.dtype == bool and val_masks.dtype == bool
    pelvis_max = np.abs(train_clips_raw[:, :, 0, :]).max()
    assert pelvis_max < 1e-4, f"pelvis joint not zeroed after centering (|max|={pelvis_max})"

    # -- clip-count arithmetic sanity ----------------------------------------
    expected_tr = 0
    for key in npz.files:
        pid, _ = parse_seq_key(key)
        if pid in tr_pids:
            Traw = npz[key].shape[0]
            if Traw < args.seq_len:
                expected_tr += 1
                continue
            starts = list(range(0, Traw - args.seq_len + 1, args.clip_stride_train))
            if starts and starts[-1] + args.seq_len < Traw:
                starts.append(Traw - args.seq_len)
            expected_tr += len(starts)
    assert expected_tr == train_clips_raw.shape[0], (
        f"train clip count mismatch: expected {expected_tr}, got {train_clips_raw.shape[0]}"
    )

    # -- z-score --------------------------------------------------------------
    mean, std = compute_zscore_stats(train_clips_raw, train_masks)
    x1_train = apply_zscore(train_clips_raw, mean, std)
    x1_val = apply_zscore(val_clips_raw, mean, std)

    # post-z-score sanity. Pelvis (joint 0) is all zeros pre-zscore; post-zscore
    # it stays 0 so we skip it. For other joints, mean should be ~0 exactly; std
    # should be ~1 except for joints whose raw std was below ``std_floor`` (1 cm)
    # — those saturate to raw_std/std_floor < 1 by design, which we allow.
    real_tr = x1_train[train_masks]
    tr_mean_per_coord = real_tr.astype(np.float64).mean(axis=0)
    tr_std_per_coord = real_tr.astype(np.float64).std(axis=0)
    non_pelvis_mean = tr_mean_per_coord[1:]
    non_pelvis_std = tr_std_per_coord[1:]
    assert np.abs(non_pelvis_mean).max() < 1e-2, (
        f"post-zscore non-pelvis per-coord mean too large: {np.abs(non_pelvis_mean).max()}"
    )
    assert non_pelvis_std.max() < 1.0 + 1e-2, (
        f"post-zscore non-pelvis per-coord std exceeds 1: max={non_pelvis_std.max()}"
    )
    assert non_pelvis_std.min() >= 0.0, (
        f"post-zscore std went negative?: min={non_pelvis_std.min()}"
    )

    # -- sanity tuples via AffineProbPath ------------------------------------
    sanity = make_sanity_tuples(x1_val, n_sanity=args.n_sanity_clips, seed=args.seed)

    # -- save -----------------------------------------------------------------
    def _meta_to_arrays(meta: list[dict]) -> dict[str, np.ndarray]:
        if not meta:
            return {
                "meta_pid": np.array([], dtype=object),
                "meta_walk_id": np.array([], dtype=object),
                "meta_seq_key": np.array([], dtype=object),
                "meta_clip_idx": np.array([], dtype=np.int32),
                "meta_updrs_gait": np.array([], dtype=np.int32),
                "meta_medication": np.array([], dtype=object),
            }
        return {
            "meta_pid": np.array([m["pid"] for m in meta], dtype=object),
            "meta_walk_id": np.array([m["walk_id"] for m in meta], dtype=object),
            "meta_seq_key": np.array([m["seq_key"] for m in meta], dtype=object),
            "meta_clip_idx": np.array([m["clip_idx"] for m in meta], dtype=np.int32),
            "meta_updrs_gait": np.array([m["updrs_gait"] for m in meta], dtype=np.int32),
            "meta_medication": np.array([m["medication"] for m in meta], dtype=object),
        }

    train_meta_arrays = {f"{k}_train": v for k, v in _meta_to_arrays(train_meta).items()}
    val_meta_arrays = {f"{k}_val": v for k, v in _meta_to_arrays(val_meta).items()}

    payload = {
        "x1_train": x1_train,
        "mask_train": train_masks,
        "x1_val": x1_val,
        "mask_val": val_masks,
        "stats_mean": mean,
        "stats_std": std,
        "seq_len": np.int32(args.seq_len),
        "fold": np.int32(args.fold),
        "num_folds": np.int32(args.num_folds),
        "clip_stride_train": np.int32(args.clip_stride_train),
        "clip_stride_val": np.int32(args.clip_stride_val),
        "dataset": np.array(args.dataset, dtype=object),
        "data_type": np.array(args.data_type, dtype=object),
        **train_meta_arrays,
        **val_meta_arrays,
    }
    np.savez_compressed(cache_path, **payload)
    np.savez_compressed(sanity_path, **sanity)

    summary = {
        "dataset": args.dataset,
        "data_type": args.data_type,
        "num_folds": args.num_folds,
        "fold": args.fold,
        "seq_len": args.seq_len,
        "clip_stride_train": args.clip_stride_train,
        "clip_stride_val": args.clip_stride_val,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_train_clips": int(x1_train.shape[0]),
        "n_val_clips": int(x1_val.shape[0]),
        "n_train_pids": len(tr_pids),
        "n_val_pids": len(val_pids),
        "n_eval_pids": len(eval_pids),
        "train_pids": sorted(tr_pids),
        "val_pids": sorted(val_pids),
        "eval_pids_reserved_for_shap": sorted(eval_pids),
        "stats_mean_shape": list(mean.shape),
        "stats_std_shape": list(std.shape),
        "post_zscore_train_nonpelvis_abs_mean_max": float(np.abs(non_pelvis_mean).max()),
        "post_zscore_train_nonpelvis_std_range": [
            float(non_pelvis_std.min()), float(non_pelvis_std.max())
        ],
        "std_floor_m": 1e-2,
        "n_saturated_joint_coords": int(((tr_std_per_coord > 1e-8) & (tr_std_per_coord < 1.0 - 1e-2)).sum()),
        "pelvis_abs_max_before_zscore": float(pelvis_max),
        "cache_path": str(cache_path),
        "sanity_path": str(sanity_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[generate_velocity] wrote cache: {cache_path}")
    print(f"[generate_velocity] wrote sanity: {sanity_path}")
    print(f"[generate_velocity] wrote summary: {summary_path}")
    print(json.dumps({k: summary[k] for k in [
        "n_train_clips", "n_val_clips", "n_train_pids", "n_val_pids",
        "n_eval_pids", "pelvis_abs_max_before_zscore",
        "post_zscore_train_nonpelvis_abs_mean_max",
        "post_zscore_train_nonpelvis_std_range",
    ]}, indent=2))


if __name__ == "__main__":
    main()
