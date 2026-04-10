"""cache_actor_shap_completions.py — Pre-generate ActorSHAP completions for fast evaluation.

WHY THIS EXISTS
--------------
evaluate_shap.py is slow because for every test sequence it calls
ActorSHAP.sample_completions once per KernelSHAP coalition (2 × n_kernel_samples
coalitions × n_completion_samples stochastic draws = the dominant compute cost).

The coalitions are DETERMINISTIC given seed=seq_idx (the exact seed used in
compute_spatial_shap and compute_temporal_shap).  We can therefore pre-generate
and save all completions once, then during evaluation serve them from disk — no
ActorSHAP GPU calls needed for the KernelSHAP phase.

Faithfulness metrics (PGI/PGU, deletion/insertion AUC) evaluate a small number
of additional coalitions (~50 per sequence) that depend on the SHAP values.
Those are still evaluated on-line with the real model, but at ~50 calls instead
of ~400 they are much cheaper.  evaluate_shap.py handles this automatically via
the CachedActorSHAP wrapper (see that file's --actor_shap_cache_dir flag).

STORAGE
-------
Per-sequence NPZ (~float16):
  n_kernel_samples=200: 400 coalitions × n_samp × (17×3×81) × 2 bytes ≈ 12 MB/seq
  For 50 test sequences: ~600 MB  (manageable)

To reduce size, use --n_completion_samples 3 for caching (evaluate_shap.py
can be told to use the cached n_samp even if --n_completion_samples differs).

USAGE — single GPU
------------------
    python cache_actor_shap_completions.py \\
        --actor_shap_ckpt experiment_outs/actor_shap/<run>/actor_shap_last.ckpt \\
        --backbone potr --config BMCLab.json \\
        --num_folds 23 --fold 1 \\
        --n_kernel_samples 200 \\
        --n_completion_samples 5 \\
        --output_dir results/shap_cache/actor_potr_bmclab_fold1 \\
        --device cuda:4

USAGE — parallel across 4 GPUs
-------------------------------
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$((4+i)) python cache_actor_shap_completions.py \\
            --actor_shap_ckpt ... \\
            --backbone potr --config BMCLab.json \\
            --num_folds 23 --fold 1 \\
            --n_kernel_samples 200 --n_completion_samples 5 \\
            --output_dir results/shap_cache/actor_potr_bmclab_fold1 \\
            --num_shards 4 --shard_id $i &
    done
    wait

OUTPUT FORMAT — one file per test sequence
------------------------------------------
    {output_dir}/seq_{seq_idx:05d}.npz
    Keys:
      coalition_masks   bool    (N_coal, 17)           True = observed joint
      completions       float16 (N_coal, n_samp, 17, 3, T)
      x                 float16 (17, 3, T)              original sequence
      y                 int32   scalar                  class label
      mask              bool    (T,)                    valid-frame mask
      lengths           int32   scalar                  frame count
      seq_idx           int32   scalar
      n_kernel_samples  int32   scalar                  for consistency check
      n_completion_samples int32 scalar
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from data.dataloaders import collate_fn
from evaluate_shap import (
    _load_backbone_params,
    _raw_data_args,
    load_actor_shap,
)
from model.actor.actor_shap import ActorSHAP
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.shap_compute import _sample_kernel_coalitions, build_spatial_shap_mask
from model.actor.shap_masking import build_temporal_windows


# ---------------------------------------------------------------------------
# Coalition generation — mirrors evaluate_shap.py exactly
# ---------------------------------------------------------------------------

def _spatial_coalitions_for_seq(seq_idx: int, n_kernel_samples: int) -> np.ndarray:
    """Return the (2*N, 17) int coalition array used by compute_spatial_shap.

    Uses the same rng seed as evaluate_shap.py (seed=seq_idx).
    """
    rng = np.random.default_rng(seq_idx)
    coalitions, _ = _sample_kernel_coalitions(17, n_kernel_samples, rng)
    return coalitions  # (2*N, 17) int, 1=observed


def _temporal_coalition_masks(T: int, device: torch.device,
                               stride_period: int | None = None) -> list[torch.Tensor]:
    """Return all 16 exact-enumeration temporal coalitions (K=4 windows).

    These are the same coalitions as compute_temporal_shap (exact enumeration).
    stride_period defaults to T//4 (one stride = one quarter of the clip).
    """
    import itertools
    from model.actor.shap_masking import build_temporal_shap_mask
    sp = stride_period if stride_period is not None else max(1, T // 4)
    windows = build_temporal_windows(T, stride_period=sp)
    K = len(windows)
    masks = []
    for bits in itertools.product([0, 1], repeat=K):
        observed_idx = [k for k, b in enumerate(bits) if b == 1]
        tm = build_temporal_shap_mask(observed_idx, windows, T, device)
        masks.append(tm.unsqueeze(0))  # (1, T)
    return masks  # list of (1, T) bool tensors


# ---------------------------------------------------------------------------
# Per-sequence caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_sequence(
    seq_idx: int,
    x: torch.Tensor,    # (1, J, F, T)
    y: torch.Tensor,    # (1,)
    mask: torch.Tensor, # (1, T)
    lengths: torch.Tensor,
    model: ActorSHAP,
    n_kernel_samples: int,
    n_completion_samples: int,
    output_dir: str,
) -> str:
    """Generate and save all KernelSHAP completions for one test sequence.

    Spatial: 2*n_kernel_samples coalitions (deterministic, seed=seq_idx).
    Temporal: all 16 exact-enumeration window coalitions.
    Extra: all-zeros (all joints masked) and all-ones (all joints observed).

    Returns the path to the saved NPZ file.
    """
    out_path = os.path.join(output_dir, f"seq_{seq_idx:05d}.npz")
    if os.path.isfile(out_path):
        return out_path  # already done

    device = x.device
    B, J, F, T = x.shape

    # ---- Spatial coalitions ------------------------------------------------
    spatial_cols = _spatial_coalitions_for_seq(seq_idx, n_kernel_samples)  # (N, 17) int
    # Deduplicate (complement pairs can produce identical rows).
    unique_spatial = np.unique(spatial_cols, axis=0)

    # Also add the all-zeros coalition (needed for completeness/p_ref).
    all_zeros = np.zeros((1, J), dtype=int)
    unique_spatial = np.unique(np.vstack([unique_spatial, all_zeros]), axis=0)

    all_coalition_masks: list[np.ndarray] = []
    all_completions: list[np.ndarray] = []

    for col in unique_spatial:
        observed = np.where(col == 1)[0].tolist()
        cm = build_spatial_shap_mask(observed, device, n_joints=J).unsqueeze(0)  # (1, J)
        comps = model.sample_completions(
            x, y, mask, lengths, cm,
            n_samples=n_completion_samples, paste_observed=True,
        )
        # comps: list of n_completion_samples (1, J, F, T) tensors
        comps_np = np.stack(
            [c[0].cpu().to(torch.float16).numpy() for c in comps], axis=0
        )  # (n_samp, J, F, T)
        all_coalition_masks.append(col.astype(bool))
        all_completions.append(comps_np)

    coalition_masks_arr = np.stack(all_coalition_masks, axis=0)   # (N_coal, J) bool
    completions_arr     = np.stack(all_completions, axis=0)        # (N_coal, n_samp, J, F, T) float16

    # ---- Temporal coalitions -----------------------------------------------
    temporal_masks  = _temporal_coalition_masks(T, device)
    temporal_cols: list[np.ndarray] = []
    temporal_comps: list[np.ndarray] = []

    for tm in temporal_masks:
        comps = model.sample_completions(
            x, y, mask, lengths, tm,
            n_samples=n_completion_samples, paste_observed=True,
        )
        comps_np = np.stack(
            [c[0].cpu().to(torch.float16).numpy() for c in comps], axis=0
        )  # (n_samp, J, F, T)
        temporal_cols.append(tm[0].cpu().numpy())       # (T,) bool
        temporal_comps.append(comps_np)

    temporal_masks_arr  = np.stack(temporal_cols,  axis=0)    # (16, T) bool
    temporal_comps_arr  = np.stack(temporal_comps, axis=0)    # (16, n_samp, J, F, T)

    # ---- Save --------------------------------------------------------------
    np.savez_compressed(
        out_path,
        # Spatial
        coalition_masks        = coalition_masks_arr,
        completions            = completions_arr,
        # Temporal
        temporal_coalition_masks = temporal_masks_arr,
        temporal_completions     = temporal_comps_arr,
        # Sequence data (for standalone use without the original dataset)
        x                      = x[0].cpu().to(torch.float16).numpy(),
        y                      = int(y[0].item()),
        mask                   = mask[0].cpu().numpy(),
        lengths                = int(lengths[0].item()),
        # Metadata
        seq_idx                = seq_idx,
        n_kernel_samples       = n_kernel_samples,
        n_completion_samples   = n_completion_samples,
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-generate ActorSHAP completions for fast evaluate_shap.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--actor_shap_ckpt", required=True,
                   help="Path to trained ActorSHAP Lightning checkpoint.")
    p.add_argument("--actor_shap_config", default=None,
                   help="Optional config.json override (architecture hyper-params).")
    p.add_argument("--backbone", required=True,
                   help="Backbone name, e.g. 'potr'.")
    p.add_argument("--config", required=True,
                   help="Config filename inside configs/<backbone>/, e.g. 'BMCLab.json'.")
    p.add_argument("--num_folds", type=int, default=23)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n_kernel_samples", type=int, default=200,
                   help="Must match --n_kernel_samples used in evaluate_shap.py.")
    p.add_argument("--n_completion_samples", type=int, default=5,
                   help="Completions per coalition. 3-5 is enough for caching "
                        "(evaluate_shap.py will use all cached samples, so setting this "
                        "lower than evaluate_shap's --n_completion_samples just means "
                        "the cached value_fn averages over fewer samples — still valid).")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write per-sequence NPZ files.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_sequences", type=int, default=None,
                   help="Cap total test sequences (for quick tests).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of parallel jobs (for splitting across GPUs).")
    p.add_argument("--shard_id", type=int, default=0,
                   help="This job's shard index (0-indexed).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Loading ActorSHAP from {args.actor_shap_ckpt} ...")
    model = load_actor_shap(args.actor_shap_ckpt, device,
                            config_path=args.actor_shap_config)
    model.eval()

    print("[2/4] Loading backbone params and dataset ...")
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    backbone_params["dataset"] = backbone_params.get("dataset", "BMCLab")
    backbone_params["num_folds"] = args.num_folds

    data_args = _raw_data_args(backbone_params, args.fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)

    n_full = len(test_ds)
    n_cap  = n_full if args.max_sequences is None else min(n_full, args.max_sequences)

    # Shard: contiguous block assignment (matches evaluate_shap.py sharding).
    shard_start = (args.shard_id * n_cap) // args.num_shards
    shard_end   = ((args.shard_id + 1) * n_cap) // args.num_shards
    n_shard     = shard_end - shard_start

    print(f"[3/4] Shard {args.shard_id}/{args.num_shards}: "
          f"sequences {shard_start}–{shard_end-1} ({n_shard} total)")

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    print(f"[4/4] Generating completions "
          f"(n_kernel_samples={args.n_kernel_samples}, "
          f"n_completion_samples={args.n_completion_samples}) ...")
    saved = 0
    skipped = 0
    for seq_idx, raw_batch in enumerate(tqdm(loader, total=shard_end, desc="sequences")):
        if seq_idx < shard_start:
            continue
        if seq_idx >= shard_end:
            break

        x_raw, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            x_raw.float(), pad_mask,
            num_classes=3, device=device, y=labels,
        )

        out_path = os.path.join(args.output_dir, f"seq_{seq_idx:05d}.npz")
        if os.path.isfile(out_path):
            skipped += 1
            continue

        cache_sequence(
            seq_idx=seq_idx,
            x=b["x"], y=b["y"], mask=b["mask"], lengths=b["lengths"],
            model=model,
            n_kernel_samples=args.n_kernel_samples,
            n_completion_samples=args.n_completion_samples,
            output_dir=args.output_dir,
        )
        saved += 1

    print(f"\nDone. Saved {saved} new files, skipped {skipped} existing.")
    print(f"Cache directory: {args.output_dir}")


if __name__ == "__main__":
    main()
