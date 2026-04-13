#!/usr/bin/env python3
"""Diagnostic: compare ActorSHAP reconstruction across phases.

Phase 1 — z ~ q_ϕ, no masking (pure CVAE reconstruction).
           If the decoder is broken, this looks wrong.
Phase 3 — z ~ r_ψ, masked input (inference path used in GIF callback).
           If only this looks wrong, the problem is r_ψ z-gap, not the decoder.

Usage:
    python scripts/diag_actor_shap_phases.py \\
        --checkpoint experiment_outs/actor_shap/<run>/actor_shap_epochepoch=0099.ckpt \\
        --out_dir artifacts/diag_phases
"""
from __future__ import annotations

import argparse, json, os, sys, types
import torch

_REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _SCRIPTS)

import viz_utils
from data.dataloaders import collate_fn
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.motion_utils import unroot_to_global
from model.actor.shap_masking import H36M_GROUPS


def load_model(ckpt_path: str):
    from model.actor.actor_shap import ActorSHAP, MaskedActorEncoder, CoalitionFullEncoder
    from model.actor.transformer_arch import Decoder_TRANSFORMER

    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    import math
    common = dict(
        modeltype   = "cvae",
        njoints     = 17,
        nfeats      = 3,
        num_frames  = 0,
        num_classes = cfg["num_classes"],
        translation = True,
        pose_rep    = "xyz",
        glob        = True,
        glob_rot    = [math.pi, 0, 0],
        latent_dim  = cfg["latent_dim"],
        ff_size     = cfg.get("ff_size", 1024),
        num_layers  = cfg.get("num_layers", 8),
        num_heads   = cfg.get("num_heads", 4),
        dropout     = cfg.get("dropout", 0.1),
        ablation    = None,
        activation  = "gelu",
    )

    full_enc    = CoalitionFullEncoder(**common)
    masked_enc  = MaskedActorEncoder(**common)
    decoder     = Decoder_TRANSFORMER(**common)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorSHAP(
        encoder=full_enc, masked_encoder=masked_enc, decoder=decoder,
        latent_dim=cfg["latent_dim"], device=device,
        pose_rep=cfg.get("pose_rep","xyz"), num_classes=cfg["num_classes"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir",    default="artifacts/diag_phases")
    p.add_argument("--n_seqs",     type=int, default=3)
    p.add_argument("--split",      default="val")
    p.add_argument("--fps",        type=int, default=12)
    args = p.parse_args()

    model, cfg, device = load_model(args.checkpoint)
    os.makedirs(args.out_dir, exist_ok=True)

    ns = types.SimpleNamespace(
        dataset        = cfg.get("dataset", "BMCLab"),
        data_dir       = cfg.get("data_dir", "data/"),
        num_folds      = cfg.get("num_folds", 23),
        fold           = cfg.get("fold", 1),
        batch_size     = 8,
        num_workers    = 0,
        experiment_name= "diag",
        split          = args.split,
    )

    train_ds, val_ds = get_carepd_datasets(ns)
    ds = val_ds if args.split == "val" else train_ds
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.n_seqs,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    x_raw, lab, _vidx, _meta, pad_mask = next(iter(loader))
    n = min(args.n_seqs, x_raw.shape[0])

    b = actor_batch_from_carepd(x_raw[:n].float().to(device), pad_mask[:n].to(device),
                                 model.num_classes, device)
    b["y"] = lab[:n].long().to(device)

    # Pick one spatial coalition for the masked test (e.g. right_leg)
    group = "right_leg"
    joint_ids = H36M_GROUPS[group]
    cm_j = torch.ones(17, dtype=torch.bool, device=device)
    cm_j[joint_ids] = False
    coalition_mask = cm_j.unsqueeze(0)  # (1, J)

    with torch.no_grad():
        for bi in range(n):
            x1       = b["x"][bi:bi+1]
            y1       = b["y"][bi:bi+1]
            mask1    = b["mask"][bi:bi+1]
            lengths1 = b["lengths"][bi:bi+1]
            real_len = int(mask1[0].sum().item())

            gt_global = unroot_to_global(
                x1[:, :, :, :real_len].permute(0, 3, 1, 2)
            )[0].cpu().numpy()  # (T, J, 3)

            # ── Phase 1: z ~ q_ϕ, no masking (training path reconstruction) ──
            b1 = {"x": x1, "y": y1, "mask": mask1, "lengths": lengths1}
            b1 = model.forward(b1, phase=1)
            p1_global = unroot_to_global(
                b1["output"][:, :, :, :real_len].permute(0, 3, 1, 2)
            )[0].cpu().numpy()  # (T, J, 3)

            # ── Phase 3: z ~ r_ψ, masked input (inference path) ──
            b3 = {"x": x1, "y": y1, "mask": mask1, "lengths": lengths1,
                  "coalition_mask": coalition_mask.expand(1, -1)}
            b3 = model.forward(b3, phase=3)
            p3_global = unroot_to_global(
                b3["output"][:, :, :, :real_len].permute(0, 3, 1, 2)
            )[0].cpu().numpy()  # (T, J, 3)

            # Save GT, phase-1 reconstruction, phase-3 reconstruction as GIFs
            prefix = os.path.join(args.out_dir, f"seq{bi:02d}")

            edges = viz_utils.edges_for_njoints(17)

            viz_utils.save_motion_comparison_gif(
                gt_global, p1_global, edges,
                f"{prefix}_phase1_full_reconstruction.gif",
                fps=args.fps,
                legend_gt_label="GT",
                legend_pred_label="phase1 (z~q_φ, no mask)",
            )

            viz_utils.save_motion_comparison_gif(
                gt_global, p3_global, edges,
                f"{prefix}_phase3_masked_{group}.gif",
                fps=args.fps,
                legend_gt_label="GT",
                legend_pred_label=f"phase3 (z~r_ψ, {group} masked)",
            )

            # Quantitative: MPJPE for phase-1 vs GT (global coords)
            diff_p1 = ((gt_global - p1_global) ** 2).sum(-1) ** 0.5
            diff_p3 = ((gt_global - p3_global) ** 2).sum(-1) ** 0.5
            print(f"seq{bi} phase1 MPJPE={diff_p1.mean():.4f}  phase3 MPJPE={diff_p3.mean():.4f}")

            # Check r_ψ sigma (low sigma = near-deterministic = z doesn't vary)
            if "logvar_masked" in b3:
                sigma_psi = b3["logvar_masked"].mul(0.5).exp().mean().item()
                print(f"seq{bi} r_ψ sigma_mean={sigma_psi:.4f}")


if __name__ == "__main__":
    main()
