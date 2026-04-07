"""
inspect_6dsmpl_batch.py — Visual sanity-check for 6D_SMPL data + FK pipeline.

What it does
------------
1. Loads one batch from the 6D_SMPL data loader (default: BMCLab fold 1).
2. Runs the SMPL forward-kinematics (Rotation2xyz) to convert 6D rotations
   → root-centred XYZ joint positions.
3. Saves a PNG showing a grid of **4 evenly-spaced frames** from the first
   sequence in the batch.  Each panel is a front-view stick figure with every
   joint numbered so you can verify joint ordering at a glance.
4. Prints a short text report: shapes, value ranges, root-joint zero-check,
   and a bone-length consistency check.

Usage
-----
    python scripts/inspect_6dsmpl_batch.py                        # BMCLab fold 1
    python scripts/inspect_6dsmpl_batch.py --dataset T-SDU-PD --fold 2
    python scripts/inspect_6dsmpl_batch.py --out artifacts/6d_inspect.png
    python scripts/inspect_6dsmpl_batch.py --no_vis          # text report only

Joint numbering (SMPL 24-joint layout)
---------------------------------------
 0 = Pelvis (root)
 1 = L_Hip        2 = R_Hip        3 = Spine1
 4 = L_Knee       5 = R_Knee       6 = Spine2
 7 = L_Ankle      8 = R_Ankle      9 = Spine3
10 = L_Foot      11 = R_Foot      12 = Neck
13 = L_Collar    14 = R_Collar    15 = Head
16 = L_Shoulder  17 = R_Shoulder
18 = L_Elbow     19 = R_Elbow
20 = L_Wrist     21 = R_Wrist
22 = L_Hand      23 = R_Hand
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataloaders import collate_fn
from model.actor.cvae_data import get_6dsmpl_datasets, actor_batch_from_6dsmpl
from model.actor.rotation2xyz import Rotation2xyz


# ---------------------------------------------------------------------------
# SMPL kinematic skeleton: (parent, child) pairs for drawing limb lines.
# ---------------------------------------------------------------------------
SMPL_BONES = [
    # spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # left arm
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    # right arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visual + text sanity check for 6D_SMPL → FK pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="BMCLab",
                   choices=["BMCLab", "T-SDU-PD", "PD-GaM", "3DGait"])
    p.add_argument("--fold", type=int, default=1, help="Cross-validation fold index.")
    p.add_argument("--num_folds", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_idx", type=int, default=0,
                   help="Which sequence in the batch to visualise.")
    p.add_argument("--n_frames", type=int, default=4,
                   help="Number of evenly-spaced frames to show per sequence.")
    p.add_argument("--out", default="artifacts/6dsmpl_inspect.png",
                   help="Output PNG path.")
    p.add_argument("--no_vis", action="store_true",
                   help="Skip PNG output (text report only).")
    p.add_argument("--smpl_path", default=None,
                   help="Override path to SMPL_NEUTRAL.pkl.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stick-figure drawing helpers
# ---------------------------------------------------------------------------


def _draw_stick_figure_3d(ax, joints: np.ndarray, *, color: str = "#4488cc") -> None:
    """Draw a 3D stick figure on a matplotlib 3D Axes.

    Args:
        ax:     Matplotlib 3D axes.
        joints: ``(24, 3)`` joint positions in *display* coordinates
                ``(x_disp, z_world, y_world)`` = right-hand, depth, up.
        color:  Line / scatter colour.
    """
    for (pa, ch) in SMPL_BONES:
        ax.plot(
            [joints[pa, 0], joints[ch, 0]],
            [joints[pa, 1], joints[ch, 1]],
            [joints[pa, 2], joints[ch, 2]],
            color=color, linewidth=1.5, zorder=1,
        )
    # Highlight root in red, rest green
    for j in range(24):
        c = "#e84040" if j == 0 else "#33aa55"
        ax.scatter(joints[j, 0], joints[j, 1], joints[j, 2], s=20, color=c, zorder=2)
        ax.text(joints[j, 0], joints[j, 1], joints[j, 2], str(j),
                fontsize=4, color="#222222", zorder=3)


def save_skeleton_grid(
    xyz_seq: np.ndarray,   # (T, 24, 3)  world coords: x=left, y=up, z=forward
    n_frames: int,
    out_path: str,
    seq_label: str = "seq 0",
) -> None:
    """Save a grid of ``n_frames`` **3D perspective** stick figures.

    Each panel shows a 3D view with elevation=20°, azimuth=-70° so you see
    the person from slightly above and to the side — the same view used in
    the GIF renderer.  This avoids the misleading 2D projection artefact where
    arm swing along the walking axis makes the arms appear above the head.

    Note on axes: SMPL world space has y=up, x=left-right (person's left = +x
    when facing +z), z=forward.  We re-map to display space for the Axes:
        display-x  = world-z  (walking/forward direction, horizontal)
        display-y  = world-x  (left-right, horizontal)
        display-z  = world-y  (up-down, vertical)

    This puts the vertical axis (display-z) as the height axis in matplotlib's
    3D plot, which gives the most natural appearance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = xyz_seq.shape[0]
    indices = np.linspace(0, T - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(4 * n_frames, 4))
    fig.suptitle(
        f"6D_SMPL FK output — {seq_label}\n"
        "(red=pelvis/root, numbers=SMPL joint idx; 3D perspective view)",
        fontsize=8,
    )

    # Compute shared axis limits across all displayed frames
    shown = xyz_seq[indices]  # (n_frames, 24, 3)
    flat = shown.reshape(-1, 3)
    span = flat.ptp(axis=0).max() / 2 + 0.05   # half-range with margin
    centre = (flat.max(axis=0) + flat.min(axis=0)) / 2

    for col, fidx in enumerate(indices):
        ax = fig.add_subplot(1, n_frames, col + 1, projection="3d")
        joints_world = xyz_seq[fidx]  # (24, 3): x=left, y=up, z=fwd

        # Re-map to display coordinates: disp = (z, x, y) so vertical is on z-axis
        joints_disp = joints_world[:, [2, 0, 1]]  # (24, 3): fwd, left, up

        _draw_stick_figure_3d(ax, joints_disp, color="#4488cc")

        # Symmetric axis limits for each display axis
        c_disp = centre[[2, 0, 1]]   # centre in display coords
        ax.set_xlim(c_disp[0] - span, c_disp[0] + span)
        ax.set_ylim(c_disp[1] - span, c_disp[1] + span)
        ax.set_zlim(c_disp[2] - span, c_disp[2] + span)
        ax.set_xlabel("fwd (z)", fontsize=5)
        ax.set_ylabel("left (x)", fontsize=5)
        ax.set_zlabel("up (y)", fontsize=5)
        ax.tick_params(labelsize=4)
        ax.view_init(elev=20, azim=-70)
        ax.set_title(f"frame {fidx}", fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect] Saved 3D stick-figure grid → {out_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(
    x_raw: torch.Tensor,       # (B, T, 25, 6)
    x_actor: torch.Tensor,     # (B, 24, 6, T)
    xyz: torch.Tensor,         # (B, 24, 3, T)
    mask: torch.Tensor,        # (B, T)
    seq_idx: int,
) -> None:
    B, J, F, T = x_actor.shape
    print("=" * 60)
    print("6D_SMPL batch inspection report")
    print("=" * 60)
    print(f"Raw loader output  : {tuple(x_raw.shape)}  (B, T, 25, 6)")
    print(f"After drop+permute : {tuple(x_actor.shape)}  (B, 24, 6, T)")
    print(f"FK xyz output      : {tuple(xyz.shape)}  (B, 24, 3, T)")
    print()

    # Range checks
    a1 = x_actor[:, :, :3, :]
    a2 = x_actor[:, :, 3:, :]
    n1 = a1.norm(dim=2).mean().item()
    n2 = a2.norm(dim=2).mean().item()
    dot = (a1 * a2).sum(dim=2).abs().mean().item()
    print(f"6D col-0 mean norm : {n1:.6f}  (should be 1.0)")
    print(f"6D col-1 mean norm : {n2:.6f}  (should be 1.0)")
    print(f"6D col·col mean    : {dot:.2e}  (should be ≈ 0)")
    print()

    # Root joint should be zero (root-centred output)
    root_max = xyz[:, 0, :, :].abs().max().item()
    print(f"Root joint max |xyz|: {root_max:.2e}  (should be 0)")
    print()

    # Bone-length consistency across frames (should be ~constant)
    seq = xyz[seq_idx]           # (24, 3, T)
    bone_lengths = []
    for pa, ch in SMPL_BONES:
        diff = seq[ch] - seq[pa]     # (3, T)
        bl = diff.norm(dim=0)        # (T,) — length per frame
        # Only valid frames
        valid = mask[seq_idx].bool()
        if valid.sum() > 1:
            bl_valid = bl[valid]
            bone_lengths.append((bl_valid.std() / (bl_valid.mean() + 1e-8)).item())
    cv_mean = float(np.mean(bone_lengths))
    print(f"Bone-length CV (seq {seq_idx}): {cv_mean:.4f}  (should be < 0.01 for rigid body)")
    print()

    # Joint motion: average displacement between consecutive frames
    seq_t = seq.permute(2, 0, 1)   # (T, 24, 3)
    valid_idx = mask[seq_idx].nonzero(as_tuple=True)[0]
    if len(valid_idx) > 1:
        diffs = seq_t[valid_idx[1:]] - seq_t[valid_idx[:-1]]   # (T-1, 24, 3)
        mean_vel = diffs.norm(dim=-1).mean().item()
        print(f"Mean inter-frame joint displacement: {mean_vel*100:.3f} cm/frame")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.experiment_name = "inspect_6dsmpl"

    print(f"[inspect] Loading {args.dataset} fold {args.fold} …")
    _, val_ds = get_6dsmpl_datasets(args)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    x_raw, _, _, _, pad = next(iter(loader))
    # x_raw: (B, T, 25, 6)

    device = torch.device("cpu")
    batch = actor_batch_from_6dsmpl(x_raw, pad, device=device)
    x_actor = batch["x"]   # (B, 24, 6, T)
    mask    = batch["mask"]

    smpl_kwargs = {}
    if args.smpl_path:
        smpl_kwargs["smpl_path"] = args.smpl_path
    r2xyz = Rotation2xyz(device, **smpl_kwargs)
    xyz = r2xyz(x_actor, mask)   # (B, 24, 3, T)

    print_report(x_raw, x_actor, xyz, mask, args.seq_idx)

    if not args.no_vis:
        seq = xyz[args.seq_idx]          # (24, 3, T)
        seq_np = seq.permute(2, 0, 1).cpu().numpy()  # (T, 24, 3)
        label = f"{args.dataset} fold {args.fold} seq {args.seq_idx}"
        save_skeleton_grid(seq_np, args.n_frames, args.out, label)


if __name__ == "__main__":
    main()
