"""
generate_velocity_synthetic.py — cache synthetic benchmark clips for flow-matching training.

Reads the outputs of ``train_actor_shap_synthetic.py`` and produces a
``cache.npz`` / ``sanity.npz`` pair compatible with ``train_flow_matching.py``.

Unlike ``scripts/generate_velocity.py`` (which pelvis-centres and z-scores
CARE-PD clips), this script leaves the synthetic data as-is:

* The Gaussian benchmark is already ``N(0, Sigma)`` with per-coord std ≈ 1, so
  z-scoring would be near-identity and would wipe the subtle scale structure
  the classifier cares about.
* There is no notion of a "pelvis" in synthetic data — joint 0 carries
  information just like every other joint. We do NOT subtract joint 0.

We therefore write ``stats_mean = 0`` and ``stats_std = 1`` so that flow
space == data space. ``compute_flow_shap_synthetic.py`` honours this.

Inputs
------
``--ckpt_dir`` (produced by ``train_actor_shap_synthetic.py``):

* ``x_train_jft.npy``    (N_tr, J, F, T) float32 — training sequences
* ``synthetic_test.pt``  dict with ``x`` (N_test, T, J, F) — reused as flow-val
* ``config.json``        (for J, F, T and ``data_mode``)

A fraction ``--val_frac`` of the training sequences is held out as the
flow-matching validation split, so the test set stays untouched for the
downstream SHAP evaluation.

Outputs (``<out_dir>/``)
------------------------

* ``cache.npz``   — ``x1_train``, ``mask_train``, ``x1_val``, ``mask_val``,
                    ``stats_mean``, ``stats_std``, plus metadata arrays.
* ``sanity.npz``  — deterministic ``(x0, t, x_t, u_t)`` tuples built with
                    ``flow_matching.path.AffineProbPath``.
* ``summary.json`` — shape counts and file paths.

Usage
-----

    python scripts/generate_velocity_synthetic.py \\
        --ckpt_dir experiment_outs/actor_shap_synthetic/actor_shap_synthetic_synthetic_gaussian \\
        --out_dir  cache/flow_matching/synthetic_gaussian
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the sanity-tuple helper from the real-dataset script.
from scripts.generate_velocity import make_sanity_tuples  # noqa: E402


def _load_train_val(
    ckpt_dir: Path, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Return ``(x1_train, mask_train, x1_val, mask_val, J, F, T)`` in
    ``(N, T, J, F)`` layout. A ``val_frac`` prefix of ``x_train_jft.npy`` is
    split off as the flow validation set (never touches ``synthetic_test.pt``).
    """
    jft_path = ckpt_dir / "x_train_jft.npy"
    if not jft_path.exists():
        raise FileNotFoundError(
            f"{jft_path} not found — re-run train_actor_shap_synthetic.py with "
            f"--checkpoint_dir {ckpt_dir.parent}."
        )
    x_train_jft = np.load(jft_path).astype(np.float32)  # (N, J, F, T)
    if x_train_jft.ndim != 4:
        raise ValueError(
            f"Expected (N, J, F, T) in {jft_path}, got shape {x_train_jft.shape}"
        )
    N, J, F, T = x_train_jft.shape
    if J != 17 or F != 3:
        raise ValueError(
            f"VelocityNet requires J=17 / F=3; synthetic data has J={J} F={F}."
        )

    x_train_btjf = np.transpose(x_train_jft, (0, 3, 1, 2))  # (N, T, J, F)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(round(val_frac * N)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    x1_train = np.ascontiguousarray(x_train_btjf[tr_idx])
    x1_val = np.ascontiguousarray(x_train_btjf[val_idx])
    mask_train = np.ones(x1_train.shape[:2], dtype=bool)
    mask_val = np.ones(x1_val.shape[:2], dtype=bool)
    return x1_train, mask_train, x1_val, mask_val, J, F, T


def _make_placeholder_meta(n: int, dataset: str) -> dict[str, np.ndarray]:
    """Minimal meta arrays so train_flow_matching.py's ``FlowClipDataset``
    is happy and downstream scripts can filter by 'split' without crashing.
    Most fields are not meaningful for synthetic data."""
    return {
        "meta_pid":         np.array([f"synth_{i}" for i in range(n)], dtype=object),
        "meta_walk_id":     np.array([f"walk_{i}"   for i in range(n)], dtype=object),
        "meta_seq_key":     np.array([f"synth_{i}__walk_{i}_down0" for i in range(n)], dtype=object),
        "meta_clip_idx":    np.zeros(n, dtype=np.int32),
        "meta_updrs_gait":  np.full(n, -1, dtype=np.int32),
        "meta_medication":  np.array(["na"] * n, dtype=object),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory produced by train_actor_shap_synthetic.py "
                        "(must contain x_train_jft.npy and synthetic_test.pt).")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for cache.npz / sanity.npz.")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of training clips held out as flow-matching val.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_sanity_clips", type=int, default=16)
    p.add_argument("--force", action="store_true",
                   help="Overwrite cache.npz / sanity.npz if they already exist.")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.npz"
    sanity_path = out_dir / "sanity.npz"
    summary_path = out_dir / "summary.json"

    if cache_path.exists() and not args.force:
        raise FileExistsError(
            f"{cache_path} already exists. Use --force to overwrite."
        )

    # --- Resolve dataset name from config.json (best-effort, informational) --
    cfg_path = ckpt_dir / "config.json"
    data_mode = "synthetic"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        data_mode = str(cfg.get("data_mode", "synthetic"))

    # --- Load training clips and split a val prefix --------------------------
    x1_train, mask_train, x1_val, mask_val, J, F, T = _load_train_val(
        ckpt_dir, val_frac=args.val_frac, seed=args.seed,
    )

    # Sanity: data should be roughly zero-mean, O(1) std (no z-scoring applied).
    real_tr = x1_train[mask_train].astype(np.float64)
    abs_mean_max = float(np.abs(real_tr.mean(axis=0)).max())
    std_max = float(real_tr.std(axis=0).max())
    if abs_mean_max > 5.0 or std_max > 20.0:
        print(f"[WARN] synthetic data has unusual scale: |mean|_max={abs_mean_max:.3f}, "
              f"std_max={std_max:.3f}. Flow matching expects ~zero mean / ~unit std.")

    # --- stats_mean = 0, stats_std = 1 (identity normalization) --------------
    stats_mean = np.zeros((J, F), dtype=np.float32)
    stats_std  = np.ones((J, F), dtype=np.float32)

    # --- Sanity tuples via AffineProbPath ------------------------------------
    sanity = make_sanity_tuples(x1_val, n_sanity=args.n_sanity_clips, seed=args.seed)

    # --- Save ----------------------------------------------------------------
    train_meta = _make_placeholder_meta(x1_train.shape[0], data_mode)
    val_meta   = _make_placeholder_meta(x1_val.shape[0],   data_mode)
    train_meta_arrays = {f"{k}_train": v for k, v in train_meta.items()}
    val_meta_arrays   = {f"{k}_val":   v for k, v in val_meta.items()}

    payload = {
        "x1_train":          x1_train.astype(np.float32),
        "mask_train":        mask_train.astype(bool),
        "x1_val":            x1_val.astype(np.float32),
        "mask_val":          mask_val.astype(bool),
        "stats_mean":        stats_mean,
        "stats_std":         stats_std,
        "seq_len":           np.int32(T),
        "fold":              np.int32(0),
        "num_folds":         np.int32(1),
        "clip_stride_train": np.int32(T),
        "clip_stride_val":   np.int32(T),
        "dataset":           np.array(data_mode, dtype=object),
        "data_type":         np.array("synthetic", dtype=object),
        **train_meta_arrays,
        **val_meta_arrays,
    }
    np.savez_compressed(cache_path, **payload)
    np.savez_compressed(sanity_path, **sanity)

    summary = {
        "source_ckpt_dir":   str(ckpt_dir),
        "data_mode":         data_mode,
        "n_train_clips":     int(x1_train.shape[0]),
        "n_val_clips":       int(x1_val.shape[0]),
        "seq_len":           int(T),
        "n_joints":          int(J),
        "n_feats":           int(F),
        "val_frac":          float(args.val_frac),
        "seed":              int(args.seed),
        "train_abs_mean_max": abs_mean_max,
        "train_std_max":      std_max,
        "cache_path":        str(cache_path),
        "sanity_path":       str(sanity_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[generate_velocity_synthetic] wrote {cache_path}")
    print(f"[generate_velocity_synthetic] wrote {sanity_path}")
    print(f"[generate_velocity_synthetic] wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
