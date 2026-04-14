#!/usr/bin/env python3
"""visualize_physics_completions.py

Generate comparison GIFs showing PhysicsInformedCompleter diversity.

Each GIF has two panels side-by-side:
  Left  — GT skeleton (blue) with held-out joints highlighted in red.
  Right — K physics completions overlaid (orange palette).

Usage::

    python scripts/visualize_physics_completions.py \\
        --stats experiment_outs/motion_stats_fold1.pkl \\
        --out_dir experiment_outs/physics_gifs \\
        --fold 1 --n_seq 3 --n_samples 5
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image as PILImage

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataloaders import collate_fn
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.motion_utils import unroot_to_global
from model.actor.physics_completer import load_physics_completer
from model.actor.shap_masking import H36M_GROUPS, H36M_JOINT_NAMES
from scripts.viz_utils import H36M17_EDGES, draw_skeleton, to_display_xyz
from torch.utils.data import DataLoader

# Colour palette for K completions
_COMPLETION_COLORS = [
    "#e63946",   # vivid red
    "#f4a261",   # orange
    "#2a9d8f",   # teal
    "#e9c46a",   # gold
    "#a8dadc",   # light teal
    "#457b9d",   # mid blue
    "#e76f51",   # coral
]


# ---------------------------------------------------------------------------
# Multi-completion GIF
# ---------------------------------------------------------------------------

def _to_world(x_actor: torch.Tensor) -> np.ndarray:
    """Convert (1, 17, 3, T) global-pelvis → (T, 17, 3) world-space numpy."""
    x_btjf = x_actor.permute(0, 3, 1, 2)     # (1, T, 17, 3)
    world = unroot_to_global(x_btjf)          # (1, T, 17, 3) full world
    return world[0].cpu().numpy()              # (T, 17, 3)


def save_multi_completion_gif(
    gt_world: np.ndarray,             # (T, 17, 3)
    completions_world: list[np.ndarray],  # K × (T, 17, 3)
    held_joints: list[int],
    out_path: str,
    fps: int = 15,
    coalition_name: str = "",
    frame_stride: int = 3,
) -> str:
    T = gt_world.shape[0]
    K = len(completions_world)
    edges = H36M17_EDGES
    held_set = set(held_joints)

    gt_d = to_display_xyz(gt_world)                         # (T, 17, 3)
    comps_d = [to_display_xyz(c) for c in completions_world]  # K × (T, 17, 3)

    all_pos = np.concatenate([gt_d] + comps_d, axis=0)     # (T*(K+1), 17, 3)
    flat = all_pos.reshape(-1, 3)
    margin = 0.10 * (np.ptp(flat, axis=0).max() + 1e-6)
    lo = flat.min(0) - margin
    hi = flat.max(0) + margin

    frame_indices = list(range(0, T, frame_stride))

    frames = []
    for fi in frame_indices:
        fig, (ax_gt, ax_comp) = plt.subplots(
            1, 2, figsize=(9, 5),
            subplot_kw={"projection": "3d"},
        )

        # ---- Left panel: GT ------------------------------------------------
        ax_gt.set_title("Ground truth\n(red = held-out joints)", fontsize=8)
        # Draw all edges; held edges in a different colour
        obs_joints = [j for j in range(17) if j not in held_set]
        for a, b in edges:
            is_held = (a in held_set) or (b in held_set)
            ax_gt.plot(
                [gt_d[fi, a, 0], gt_d[fi, b, 0]],
                [gt_d[fi, a, 1], gt_d[fi, b, 1]],
                [gt_d[fi, a, 2], gt_d[fi, b, 2]],
                color="#cc0000" if is_held else "steelblue",
                linewidth=2.0, alpha=0.9,
            )
        # Joint dots
        for j in range(17):
            c = "#cc0000" if j in held_set else "steelblue"
            sz = 28 if j in held_set else 12
            ax_gt.scatter(*gt_d[fi, j], c=c, s=sz, alpha=0.95, zorder=5)

        # ---- Right panel: completions overlaid --------------------------------
        ax_comp.set_title(
            f"{coalition_name}  —  {K} physics samples\n(grey=GT, colour=completions)",
            fontsize=8,
        )
        # Draw GT faintly in grey
        draw_skeleton(ax_comp, gt_d[fi], edges, color="grey", alpha=0.30)

        # Draw K completions in different colours
        for ki, comp_d in enumerate(comps_d):
            col = _COMPLETION_COLORS[ki % len(_COMPLETION_COLORS)]
            alpha = 0.55 if ki > 0 else 0.85
            draw_skeleton(ax_comp, comp_d[fi], edges, color=col, alpha=alpha)

            # Highlight held-out joints with large dots
            for j in held_joints:
                ax_comp.scatter(
                    *comp_d[fi, j], c=col, s=40, alpha=alpha, zorder=5,
                    edgecolors="black", linewidths=0.5,
                )

        # Apply consistent axis limits to both panels
        for ax in (ax_gt, ax_comp):
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.view_init(elev=15, azim=-75)
            ax.set_xlabel("x", fontsize=6)
            ax.set_ylabel("z", fontsize=6)
            ax.set_zlabel("y", fontsize=6)
            ax.tick_params(labelsize=6)

        fig.suptitle(
            f"frame {fi+1}/{T}  |  {coalition_name}",
            fontsize=9, y=1.01,
        )
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())
        buf.close()

    n_frames = len(frame_indices)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        loop=0, duration=max(1, int(1000 / fps)),
    )
    print(f"  saved {out_path}  ({n_frames} frames @ {fps} fps, stride={frame_stride})")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stats", default="experiment_outs/motion_stats_fold1.pkl")
    ap.add_argument("--out_dir", default="experiment_outs/physics_gifs")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--num_folds", type=int, default=23)
    ap.add_argument("--dataset", default="BMCLab")
    ap.add_argument("--n_seq", type=int, default=3,
                    help="Number of test sequences to visualize.")
    ap.add_argument("--n_samples", type=int, default=2,
                    help="Number of physics completions per coalition.")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="Which coalition groups to show (default: all 7 anatomical groups).")
    ap.add_argument("--frame_stride", type=int, default=2,
                    help="Render every N-th frame (speeds up GIF generation).")
    return ap.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)

    completer = load_physics_completer(args.stats, device)

    class _DataArgs:
        dataset = args.dataset
        num_folds = args.num_folds
        fold = args.fold
        batch_size = 1
        source_seq_len = 81
        experiment_name = "VaeacMotion"
        carepd_pose_npz = None
        carepd_labels_pkl = None

    _, test_ds = get_carepd_datasets(_DataArgs())

    groups_to_show = args.groups or list(H36M_GROUPS.keys())

    os.makedirs(args.out_dir, exist_ok=True)

    for seq_idx in range(min(args.n_seq, len(test_ds))):
        sample = test_ds[seq_idx]
        x_raw = torch.from_numpy(sample["encoder_inputs"]).float().unsqueeze(0)  # (1,T,17,3)
        label = torch.tensor([sample["label"]])
        pad_mask = torch.from_numpy(sample["pad_mask"]).bool().unsqueeze(0)
        video_name: str = test_ds.video_names[seq_idx]
        subject_id = video_name.split("__")[0]

        b = actor_batch_from_carepd(x_raw, pad_mask, 3, device, y=label)
        x, y_t, mask_t, lengths = b["x"], b["y"], b["mask"], b["lengths"]

        completer.set_subject(subject_id)
        gt_world = _to_world(x)   # (T, 17, 3)
        T = int(lengths[0].item())

        print(f"\n[seq {seq_idx+1}/{args.n_seq}]  {video_name}  subject={subject_id}  "
              f"UPDRS={label.item()}  T={T}")

        for group_name in groups_to_show:
            joint_ids = H36M_GROUPS[group_name]
            cm = torch.ones(1, 17, dtype=torch.bool, device=device)
            cm[0, joint_ids] = False

            completions = completer.sample_completions(
                x, y_t, mask_t, lengths, cm,
                n_samples=args.n_samples, paste_observed=True,
            )

            # Convert completions to world space
            comps_world = [_to_world(c.cpu())[:T] for c in completions]
            gt_T = gt_world[:T]

            fname = f"seq{seq_idx+1:02d}_{subject_id}_{group_name}.gif"
            out_path = os.path.join(args.out_dir, fname)
            save_multi_completion_gif(
                gt_T, comps_world, joint_ids,
                out_path, fps=args.fps,
                coalition_name=f"{group_name} ({len(joint_ids)} joints)",
                frame_stride=args.frame_stride,
            )


if __name__ == "__main__":
    main()
