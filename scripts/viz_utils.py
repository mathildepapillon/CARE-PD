#!/usr/bin/env python3
"""Shared skeleton drawing and GIF export for Actor motion visualization scripts.

Used by ``visualize_actor_cvae.py`` and ``visualize_actor_shap.py``.  Sets the
matplotlib Agg backend on import so headless runs work.

Run scripts from the repo root, e.g. ``python scripts/visualize_actor_cvae.py``.
"""

from __future__ import annotations

import io
import os
from typing import Iterable, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Skeleton edge definitions
# ---------------------------------------------------------------------------

H36M17_EDGES: set[Tuple[int, int]] = {
    (0, 4), (0, 1), (4, 5), (5, 6), (1, 2), (2, 3),
    (0, 7), (7, 8), (8, 14), (14, 15), (15, 16),
    (8, 11), (11, 12), (12, 13), (8, 9), (9, 10),
}

SMPL24_EDGES: list[Tuple[int, int]] = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def edges_for_njoints(n_joints: int) -> Union[set[Tuple[int, int]], list[Tuple[int, int]]]:
    """Return skeleton edges for H36M-17 or SMPL-24 layouts."""
    if n_joints == 24:
        return SMPL24_EDGES
    return H36M17_EDGES


# ---------------------------------------------------------------------------
# Geometry / drawing
# ---------------------------------------------------------------------------

def to_display_xyz(pos: np.ndarray) -> np.ndarray:
    """Map dataset axes to matplotlib Z-up: (x, y, z) → (x, z, y).

    Legacy helper kept for any callers that use it directly.
    ``save_motion_comparison_gif`` no longer calls this — it draws 2-D
    front/side panels instead to avoid perspective-projection artefacts.
    """
    return pos[..., [0, 2, 1]]


def axis_limits(pos: np.ndarray) -> dict[str, Tuple[float, float]]:
    """pos: (N, J, 3) after display remap."""
    flat = pos.reshape(-1, 3)
    margin = 0.08 * (np.ptp(flat, axis=0).max() + 1e-6)
    lo = flat.min(axis=0) - margin
    hi = flat.max(axis=0) + margin
    return {"x": (lo[0], hi[0]), "y": (lo[1], hi[1]), "z": (lo[2], hi[2])}


def draw_skeleton(
    ax,
    joints: np.ndarray,
    edges: Iterable[Tuple[int, int]],
    *,
    color: str,
    alpha: float = 0.9,
) -> None:
    """Draw a single-frame stick figure on a 3-D axes.

    Args:
        ax:     Matplotlib 3D axes.
        joints: ``(J, 3)`` positions in display coordinates.
        edges:  Iterable of ``(i, j)`` joint index pairs.
        color:  Line and scatter colour.
        alpha:  Transparency.
    """
    for a, b in edges:
        ax.plot(
            [joints[a, 0], joints[b, 0]],
            [joints[a, 1], joints[b, 1]],
            [joints[a, 2], joints[b, 2]],
            color=color, linewidth=2.0, alpha=alpha,
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=12, alpha=alpha)


def draw_skeleton_2d(
    ax,
    joints: np.ndarray,
    edges: Iterable[Tuple[int, int]],
    col_h: int,
    col_v: int,
    *,
    color: str,
    alpha: float = 0.9,
) -> None:
    """Draw a single-frame stick figure on a 2-D axes.

    Args:
        ax:     Matplotlib 2D axes.
        joints: ``(J, 3)`` positions in **dataset** XYZ coordinates.
        edges:  Iterable of ``(i, j)`` joint index pairs.
        col_h:  Column index for horizontal axis (0=X, 1=Y, 2=Z).
        col_v:  Column index for vertical axis.
        color:  Line and scatter colour.
        alpha:  Transparency.
    """
    for a, b in edges:
        ax.plot(
            [joints[a, col_h], joints[b, col_h]],
            [joints[a, col_v], joints[b, col_v]],
            color=color, linewidth=2.0, alpha=alpha,
        )
    ax.scatter(joints[:, col_h], joints[:, col_v], c=color, s=12, alpha=alpha, zorder=5)


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------

def save_motion_comparison_gif(
    gt_tj3: np.ndarray,
    pred_tj3: np.ndarray,
    edges: Union[set[Tuple[int, int]], list[Tuple[int, int]]],
    out_path: str,
    fps: int,
    *,
    title_prefix: str = "",
    gt_color: str = "steelblue",
    pred_color: str = "darkorange",
    gt_alpha: float = 0.9,
    pred_alpha: float = 0.85,
    legend_gt_label: str = "GT",
    legend_pred_label: str = "recon",
    verbose: bool = True,
) -> str:
    """Save an animated GIF comparing two motion sequences (T, J, 3) in dataset XYZ space.

    Renders two side-by-side 2-D panels per frame:
      - Left:  Front view (X horizontal, Y vertical — "facing the camera").
      - Right: Side view  (Z horizontal, Y vertical — sagittal plane).

    Both panels use the dataset convention where Y is the vertical (up) axis,
    avoiding the 3-D perspective distortions that make upright figures look
    horizontal in the old single-3D-axes layout.

    Args:
        gt_tj3:       Ground truth, shape ``(T, J, 3)``.
        pred_tj3:     Prediction / completion, same shape as ``gt_tj3``.
        edges:        Bone list for ``draw_skeleton_2d``.
        out_path:     Output ``.gif`` path (parent directory must exist or be creatable).
        fps:          Frames per second.
        title_prefix: Shown before ``frame i/T`` on each frame.

    Returns:
        Absolute path to the written file.
    """
    assert gt_tj3.shape == pred_tj3.shape, (gt_tj3.shape, pred_tj3.shape)
    t = gt_tj3.shape[0]
    if t < 2:
        raise ValueError("Need at least 2 frames for a GIF.")

    # Compute unified axis limits (in raw dataset XYZ) so both panels and
    # both sequences share the same scale across all frames.
    both = np.concatenate([gt_tj3, pred_tj3], axis=0).reshape(-1, 3)
    margin = 0.08 * (np.ptp(both, axis=0).max() + 1e-6)
    lo = both.min(axis=0) - margin   # (3,)
    hi = both.max(axis=0) + margin   # (3,)
    xlim = (lo[0], hi[0])   # lateral
    ylim = (lo[1], hi[1])   # vertical (Y=up)
    zlim = (lo[2], hi[2])   # forward

    legend_elems = [
        Line2D([0], [0], color=gt_color, linewidth=2, label=legend_gt_label),
        Line2D([0], [0], color=pred_color, linewidth=2, label=legend_pred_label),
    ]

    frames: list = []
    for fi in range(t):
        fig, (ax_front, ax_side) = plt.subplots(1, 2, figsize=(7, 5))

        for ax, (col_h, col_v, h_label, v_label, h_lim, v_lim, title_view) in [
            (ax_front, (0, 1, "X (lateral)", "Y (up)", xlim, ylim, "Front (X-Y)")),
            (ax_side,  (2, 1, "Z (forward)", "Y (up)", zlim, ylim, "Side  (Z-Y)")),
        ]:
            draw_skeleton_2d(ax, gt_tj3[fi],   edges, col_h, col_v,
                             color=gt_color,   alpha=gt_alpha)
            draw_skeleton_2d(ax, pred_tj3[fi], edges, col_h, col_v,
                             color=pred_color, alpha=pred_alpha)
            ax.set_xlim(*h_lim)
            ax.set_ylim(*v_lim)
            ax.set_xlabel(h_label, fontsize=8)
            ax.set_ylabel(v_label, fontsize=8)
            ax.set_aspect("equal")
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(0, color="gray", lw=0.5, ls="--")
            ax.set_title(title_view, fontsize=8)

        ax_front.legend(handles=legend_elems, fontsize=7,
                        loc="upper right", framealpha=0.6)
        title = f"{title_prefix}  frame {fi + 1}/{t}" if title_prefix else f"frame {fi + 1}/{t}"
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())
        buf.close()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_abs = os.path.abspath(out_path)
    frames[0].save(
        out_abs,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=max(1, int(1000 / fps)),
    )
    if verbose:
        print(f"  saved {out_abs} ({t} frames @ {fps} fps)")
    return out_abs
