#!/usr/bin/env python3
"""
Quantify whether Actor CVAE reconstructions are **articulated** vs **rigid + translation**.

If the skeleton only translates (or rigidly moves) with a fixed joint configuration,
then after subtracting the root (pelvis) each frame, joint positions are **constant**
over time → temporal variance of root-relative positions ≈ 0.

Ground-truth gait should show clearly larger values.

Usage::

    python scripts/metrics_actor_cvae_articulation.py \\
        --checkpoint experiment_outs/actor_cvae/<run>/actor_cvae_best.ckpt \\
        --max_batches 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from data.dataloaders import collate_fn  # noqa: E402
from model.actor.cvae_data import (  # noqa: E402
    actor_batch_from_carepd,
    get_carepd_datasets,
)
from visualize_actor_cvae import (  # noqa: E402
    H36M17_EDGES,
    build_model_from_cfg,
    load_actor_weights,
)


def _masked_temporal_var_mean(
    rel_bttj: torch.Tensor,
    mask_bt: torch.Tensor,
) -> torch.Tensor:
    """
    rel: (B, T, J, 3) — already root-relative or any signal over time.
    mask: (B, T) bool, True = valid.

    Returns scalar: mean over batch elements of mean temporal variance over joints & xyz.
    """
    B, T, J, _ = rel_bttj.shape
    vals = []
    for b in range(B):
        idx = mask_bt[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        r = rel_bttj[b, idx]  # (Tv, J, 3)
        v = r.var(dim=0, unbiased=False)  # (J, 3)
        vals.append(v.mean())
    if not vals:
        return rel_bttj.new_zeros(())
    return torch.stack(vals).mean()


def root_relative_temporal_variance(
    x_bttf: torch.Tensor,
    mask_bt: torch.Tensor,
) -> torch.Tensor:
    """Mean temporal variance of (joint - root) for joints 1..J-1."""
    root = x_bttf[:, :, 0:1, :]
    rel = x_bttf[:, :, 1:, :] - root
    return _masked_temporal_var_mean(rel, mask_bt)


def bone_length_temporal_variance(
    x_bttf: torch.Tensor,
    mask_bt: torch.Tensor,
    edges: set[tuple[int, int]],
) -> torch.Tensor:
    """
    Mean over bones and batch of temporal variance of edge length ||x_i - x_j||.
    Pure translation of a fixed pose → bone lengths constant → variance ≈ 0.
    """
    B, T, J, _ = x_bttf.shape
    edge_list = list(edges)
    vals_batch = []
    for b in range(B):
        idx = mask_bt[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        per_edge = []
        for i, j in edge_list:
            seg = x_bttf[b, idx, i, :] - x_bttf[b, idx, j, :]  # (Tv, 3)
            L = seg.norm(dim=-1)  # (Tv,)
            per_edge.append(L.var(unbiased=False))
        vals_batch.append(torch.stack(per_edge).mean())
    if not vals_batch:
        return x_bttf.new_zeros(())
    return torch.stack(vals_batch).mean()


def bttf_from_bjft(x_bjft: torch.Tensor) -> torch.Tensor:
    """(B, J, F, T) -> (B, T, J, F)."""
    return x_bjft.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    acc = {
        "gt_rr_var": [],
        "recon_rr_var": [],
        "gt_bone_var": [],
        "recon_bone_var": [],
        "ratio_rr": [],
    }
    n_seq = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, _lab, _vidx, _meta, pad_mask = batch
        pad_mask = pad_mask.bool()
        x = x.to(device)
        pad_mask = pad_mask.to(device)
        b_dict = actor_batch_from_carepd(x, pad_mask, model.num_classes, device)
        out = model(b_dict)
        hat = bttf_from_bjft(out["output"])

        for b in range(x.shape[0]):
            m = pad_mask[b : b + 1]
            if m.sum() < 2:
                continue
            gt = x[b : b + 1]
            rec = hat[b : b + 1]
            g_rr = root_relative_temporal_variance(gt, m)
            r_rr = root_relative_temporal_variance(rec, m)
            g_b = bone_length_temporal_variance(gt, m, H36M17_EDGES)
            r_b = bone_length_temporal_variance(rec, m, H36M17_EDGES)
            acc["gt_rr_var"].append(float(g_rr))
            acc["recon_rr_var"].append(float(r_rr))
            acc["gt_bone_var"].append(float(g_b))
            acc["recon_bone_var"].append(float(r_b))
            acc["ratio_rr"].append(float(r_rr / (g_rr + 1e-12)))
            n_seq += 1

    def _mean(key: str) -> float:
        return float(sum(acc[key]) / max(len(acc[key]), 1))

    return {
        "n_sequences": n_seq,
        "mean_gt_root_rel_temporal_var": _mean("gt_rr_var"),
        "mean_recon_root_rel_temporal_var": _mean("recon_rr_var"),
        "mean_gt_bone_length_temporal_var": _mean("gt_bone_var"),
        "mean_recon_bone_length_temporal_var": _mean("recon_bone_var"),
        "mean_ratio_recon_over_gt_rr_var": _mean("ratio_rr"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = os.path.abspath(args.checkpoint)
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if not os.path.isfile(cfg_path):
        sys.exit(f"Missing config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    model = build_model_from_cfg(cfg, device)
    load_actor_weights(model, ckpt_path, device)

    import types

    ns = types.SimpleNamespace(**cfg)
    bs = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 16)
    _, val_ds = get_carepd_datasets(ns)
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(4, bs),
        pin_memory=(device.type == "cuda"),
    )

    stats = evaluate_loader(model, loader, device, args.max_batches)

    print("Actor CVAE articulation metrics (validation sequences)")
    print("=" * 60)
    print(
        "Root-relative temporal variance: mean over joints 1..J-1 of "
        "Var_t[(x_j - x_root)]. Near 0 ⇒ fixed pose + global motion only."
    )
    print(f"  GT    mean: {stats['mean_gt_root_rel_temporal_var']:.6e}")
    print(f"  Recon mean: {stats['mean_recon_root_rel_temporal_var']:.6e}")
    print()
    print(
        "Bone-length temporal variance: mean over edges of Var_t[||x_i-x_j||]. "
        "Near 0 ⇒ constant bone lengths over time (rigid shape)."
    )
    print(f"  GT    mean: {stats['mean_gt_bone_length_temporal_var']:.6e}")
    print(f"  Recon mean: {stats['mean_recon_bone_length_temporal_var']:.6e}")
    print()
    print(f"Sequences aggregated: {stats['n_sequences']}")
    print(
        f"Mean (recon_rr_var / gt_rr_var): {stats['mean_ratio_recon_over_gt_rr_var']:.6e} "
        "(≪1 suggests recon is much more rigid than GT)"
    )


if __name__ == "__main__":
    main()
