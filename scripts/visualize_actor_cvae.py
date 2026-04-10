#!/usr/bin/env python3
"""
visualize_actor_cvae.py — Render ActorCVAE reconstructions as animated GIFs.

Supports both data modes:

``carepd``
    Stick figures drawn from the raw 17-joint H36M XYZ output.

``6dsmpl`` (ACTOR-parity)
    Stick figures drawn from the SMPL FK output (24 joints, ``output_xyz``).
    This means the GIF reflects true spatial articulation, not raw 6D rotation
    values.  The SMPL body model must be available at the path recorded in the
    training config (``smpl_path``).

Usage::

    python scripts/visualize_actor_cvae.py \\
        --checkpoint experiment_outs/actor_cvae/<run>/actor_cvae_best.ckpt \\
        --out_dir artifacts/actor_recon_gifs

    # Use the training split instead of validation:
    python scripts/visualize_actor_cvae.py \\
        --checkpoint ... --split train --out_dir artifacts/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _SCRIPTS)

import viz_utils  # noqa: E402

from data.dataloaders import collate_fn  # noqa: E402
from model.actor.cvae_data import (  # noqa: E402
    actor_batch_from_carepd,
    actor_batch_from_6dsmpl,
    get_carepd_datasets,
    get_6dsmpl_datasets,
)
from model.actor.cvae import ActorCVAE  # noqa: E402
from model.actor.transformer_arch import (  # noqa: E402
    Decoder_TRANSFORMER,
    Encoder_TRANSFORMER,
)


# ---------------------------------------------------------------------------
# Model construction from saved config
# ---------------------------------------------------------------------------

def build_model_from_cfg(
    cfg: dict,
    device: torch.device,
) -> ActorCVAE:
    """Reconstruct an ActorCVAE from a training config dict.

    Handles both the legacy ``xyz`` mode and the ACTOR-parity ``rot6d``
    (6dsmpl) mode.  For ``rot6d`` mode a ``Rotation2xyz`` instance is
    attached automatically so that the model can compute ``x_xyz`` /
    ``output_xyz`` for the ``rcxyz`` loss and GIF rendering.
    """
    lambdas = {}
    for key in ("rc", "rr", "kl", "rcxyz", "vel"):
        v = cfg.get(f"lambda_{key}", 0.0)
        if v and float(v) > 0:
            lambdas[key] = float(v)

    pose_rep   = cfg.get("pose_rep", "xyz")
    data_mode  = cfg.get("data_mode", "carepd")

    common = dict(
        modeltype="cvae",
        njoints=cfg["njoints"],
        nfeats=cfg["nfeats"],
        num_frames=0,
        num_classes=cfg["num_classes"],
        translation=True,
        pose_rep=pose_rep,
        glob=True,
        glob_rot=[3.141592653589793, 0, 0],
        latent_dim=cfg["latent_dim"],
        ff_size=cfg["ff_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        ablation=None,
        activation="gelu",
    )
    enc = Encoder_TRANSFORMER(**common)
    dec = Decoder_TRANSFORMER(**common)

    rotation2xyz = None
    if data_mode == "6dsmpl" or pose_rep == "rot6d":
        from model.actor.rotation2xyz import Rotation2xyz
        r2xyz_kwargs = {}
        if cfg.get("smpl_path"):
            r2xyz_kwargs["smpl_path"] = cfg["smpl_path"]
        rotation2xyz = Rotation2xyz(device, **r2xyz_kwargs)

    return ActorCVAE(
        enc, dec,
        lambdas=lambdas,
        latent_dim=cfg["latent_dim"],
        device=device,
        pose_rep=pose_rep,
        num_classes=cfg["num_classes"],
        rotation2xyz=rotation2xyz,
    ).to(device)


def load_actor_weights(model: ActorCVAE, ckpt_path: str, device: torch.device) -> None:
    """Load weights from a PyTorch Lightning checkpoint into the raw ActorCVAE."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict issue missing={missing} unexpected={unexpected}")


# ---------------------------------------------------------------------------
# GIF generation
# ---------------------------------------------------------------------------

def save_actor_recon_gifs(
    model: ActorCVAE,
    x_batch: torch.Tensor,
    pad_mask: torch.Tensor,
    device: torch.device,
    out_dir: str,
    batch_idx: int,
    n_examples: int,
    fps: int,
    *,
    output_filenames: Optional[list[str]] = None,
    verbose: bool = True,
    data_mode: str = "carepd",
    labels: Optional[torch.Tensor] = None,
) -> list[str]:
    """Run model inference and save one GIF per sequence.

    For ``data_mode='6dsmpl'`` the stick figures are drawn from the FK-derived
    XYZ positions (``batch['output_xyz']``), not the raw 6D rotation outputs.
    The GT stick figure is drawn from ``batch['x_xyz']``.  This ensures that
    the animation reflects true spatial articulation.

    Args:
        model:            ActorCVAE in eval mode (caller should ensure this).
        x_batch:          Raw batch tensor from the dataloader.
                          Shape ``(B, T, J, F)`` where ``J/F`` depend on mode.
        pad_mask:         ``(B, T)`` bool mask from the dataloader.
        device:           Target device.
        out_dir:          Directory to write GIF files.
        batch_idx:        Index of this batch (used in default filenames).
        n_examples:       How many sequences from the batch to render.
        fps:              GIF frame rate.
        output_filenames: Optional list of explicit filenames (length ≥ n_examples).
        verbose:          Print a line per saved file.
        data_mode:        One of ``'carepd'``, ``'6dsmpl'``.
        labels:           Optional ``(B,)`` label tensor (used for ``y`` in 6dsmpl mode).

    Returns:
        List of absolute paths to the saved GIF files.
    """
    model.eval()
    pad_mask = pad_mask.bool()

    # Build ACTOR batch dict from the raw dataloader tensor
    if data_mode == "6dsmpl":
        b_dict = actor_batch_from_6dsmpl(x_batch.to(device), pad_mask.to(device), device)
        if labels is not None:
            b_dict["y"] = labels.long().to(device)
    else:
        b_dict = actor_batch_from_carepd(
            x_batch.to(device), pad_mask.to(device),
            model.num_classes, device,
        )

    with torch.no_grad():
        out = model(b_dict)

    # For rot6d, render the FK XYZ positions so the GIF shows 3D articulation.
    # For XYZ, x_xyz == x and output_xyz == output (identity FK).
    gt_bjft  = out.get("x_xyz",      out["x"])       # (B, J, 3_or_F, T)
    out_bjft = out.get("output_xyz", out["output"])   # (B, J, 3_or_F, T)
    mask_bt  = out["mask"]                            # (B, T)

    # Permute to (B, T, J, 3) for frame-by-frame rendering
    gt_btj3  = gt_bjft.permute(0, 3, 1, 2).cpu().numpy()
    out_btj3 = out_bjft.permute(0, 3, 1, 2).cpu().numpy()
    m_np     = mask_bt.cpu().numpy()

    n_joints = gt_btj3.shape[2]
    edges = viz_utils.edges_for_njoints(n_joints)

    n_examples = min(n_examples, gt_btj3.shape[0])
    if output_filenames is not None and len(output_filenames) < n_examples:
        raise ValueError(
            f"output_filenames ({len(output_filenames)}) < n_examples ({n_examples})"
        )

    saved = []

    for b in range(n_examples):
        real_len = int(m_np[b].sum())
        if real_len < 2:
            continue
        gt = gt_btj3[b, :real_len]
        hr = out_btj3[b, :real_len]

        if output_filenames is not None:
            out_path = os.path.join(out_dir, output_filenames[b])
        else:
            out_path = os.path.join(out_dir, f"actor_sample_{batch_idx:02d}_{b:02d}.gif")
        viz_utils.save_motion_comparison_gif(
            gt,
            hr,
            edges,
            out_path,
            fps,
            title_prefix=f"batch {batch_idx:02d} seq {b}",
            verbose=verbose,
        )
        saved.append(os.path.abspath(out_path))

    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize ActorCVAE reconstructions as animated GIFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to actor_cvae_best.ckpt (config.json must be alongside).")
    p.add_argument("--out_dir", default="artifacts/actor_recon_gifs")
    p.add_argument("--n_examples", type=int, default=5,
                   help="Sequences per batch to render.")
    p.add_argument("--n_batches", type=int, default=2,
                   help="Number of batches to process.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size from config.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--split", type=str, default="val", choices=("train", "val"),
        help="Which data split to render.",
    )
    p.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle the dataset before sampling (useful to get diverse val examples).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if not os.path.isfile(cfg_path):
        sys.exit(f"config.json not found next to checkpoint: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"Loaded config from {cfg_path}")

    model = build_model_from_cfg(cfg, device)
    load_actor_weights(model, ckpt_path, device)
    model.eval()
    print(f"ActorCVAE loaded — pose_rep={model.pose_rep}  data_mode={cfg.get('data_mode','carepd')}")

    import types
    ns = types.SimpleNamespace(**cfg)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 16)
    data_mode  = cfg.get("data_mode", "carepd")

    if data_mode == "6dsmpl":
        train_ds, val_ds = get_6dsmpl_datasets(ns)
    else:
        train_ds, val_ds = get_carepd_datasets(ns)

    ds = train_ds if args.split == "train" else val_ds
    shuffle = (args.split == "train") or args.shuffle
    print(f"Using {args.split} split ({len(ds)} sequences, shuffle={shuffle}).")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=min(4, batch_size),
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    all_paths = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.n_batches:
            break
        x, lab, _vidx, _meta, pad_mask = batch
        print(f"Batch {batch_idx} shape={tuple(x.shape)}")
        paths = save_actor_recon_gifs(
            model, x, pad_mask, device, args.out_dir,
            batch_idx, args.n_examples, args.fps,
            data_mode=data_mode, labels=lab,
        )
        all_paths.extend(paths)

    print(f"Done. {len(all_paths)} GIF(s) → {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
