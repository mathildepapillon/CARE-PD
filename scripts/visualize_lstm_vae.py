"""
scripts/visualize_lstm_vae.py — Visualise LSTM VAE reconstructions as GIFs.

Saves one GIF per clip: GT (blue) and reconstruction (orange) drawn on top of
each other, in a front view (X horizontal, Y vertical) and a side view
(Z horizontal, Y vertical) — both in the same figure so the overlay is clear.

Only validation sequences (sequences the model never saw during training) are
used.

Usage
-----
python scripts/visualize_lstm_vae.py \\
    --checkpoint experiment_outs/lstm_vae/BMCLab_fold1/checkpoints/last.ckpt \\
    --dataset BMCLab \\
    --fold 1 \\
    --n_gifs 8 \\
    --fps 15 \\
    --out_dir artifacts/lstm_vae_recons
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_lstm_vae import (
    LstmVAELit,
    SequenceDataset,
    PROJECT_ROOT,
    DATASET_NPZ,
    FOLD_PICKLE,
)
from scripts.viz_utils import H36M17_EDGES

N_JOINTS = 17


# ---------------------------------------------------------------------------
# GIF: overlay GT + recon on the same axes
# ---------------------------------------------------------------------------

def _draw_skeleton_2d(ax, joints, edges, col_h, col_v, *, color, alpha=0.9):
    """Draw a stick figure on 2-D axes. joints: (J, 3) in dataset XYZ."""
    for a, b in edges:
        ax.plot(
            [joints[a, col_h], joints[b, col_h]],
            [joints[a, col_v], joints[b, col_v]],
            color=color, linewidth=2.0, alpha=alpha,
        )
    ax.scatter(joints[:, col_h], joints[:, col_v],
               c=color, s=14, alpha=alpha, zorder=5)


def save_overlay_gif(
    gt_tj3: np.ndarray,
    recon_tj3: np.ndarray,
    edges,
    out_path: str,
    fps: int = 15,
    title_prefix: str = "",
    gt_color: str = "steelblue",
    recon_color: str = "darkorange",
) -> str:
    """Save a GIF with GT and reconstruction overlaid on the same axes.

    Two panels per frame: front view (X-Y) and side view (Z-Y). Both GT and
    reconstruction are drawn on each panel so they are directly comparable.

    Parameters
    ----------
    gt_tj3, recon_tj3 : ``(T, J, 3)`` in dataset XYZ (Y up).
    """
    assert gt_tj3.shape == recon_tj3.shape
    T = gt_tj3.shape[0]

    # Shared axis limits across all frames and both sequences
    both = np.concatenate([gt_tj3, recon_tj3], axis=0).reshape(-1, 3)
    margin = 0.1 * (np.ptp(both, axis=0).max() + 1e-6)
    lo, hi = both.min(0) - margin, both.max(0) + margin
    xlim, ylim, zlim = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])

    legend_elems = [
        Line2D([0], [0], color=gt_color,    linewidth=2, label="GT"),
        Line2D([0], [0], color=recon_color, linewidth=2, label="recon"),
    ]

    frames = []
    for fi in range(T):
        fig, (ax_front, ax_side) = plt.subplots(1, 2, figsize=(8, 4.5))

        for ax, col_h, col_v, h_lim, v_lim, view_title in [
            (ax_front, 0, 1, xlim, ylim, "Front  (X – Y)"),
            (ax_side,  2, 1, zlim, ylim, "Side   (Z – Y)"),
        ]:
            _draw_skeleton_2d(ax, gt_tj3[fi],    edges, col_h, col_v,
                              color=gt_color,    alpha=0.85)
            _draw_skeleton_2d(ax, recon_tj3[fi], edges, col_h, col_v,
                              color=recon_color, alpha=0.75)
            ax.set_xlim(*h_lim)
            ax.set_ylim(*v_lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="lightgray", lw=0.5, ls="--")
            ax.axvline(0, color="lightgray", lw=0.5, ls="--")
            ax.set_title(view_title, fontsize=8)
            ax.tick_params(labelsize=7)

        ax_front.legend(handles=legend_elems, fontsize=7,
                        loc="upper right", framealpha=0.6)
        title = f"{title_prefix}   frame {fi + 1}/{T}" if title_prefix else f"frame {fi + 1}/{T}"
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())
        buf.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=max(1, int(1000 / fps)),
    )
    print(f"  saved {Path(out_path).resolve()}  ({T} frames @ {fps} fps)")
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_dataset(
    dataset: str,
    num_folds: int,
    fold: int,
    seq_len: int,
) -> SequenceDataset:
    import pickle
    fold_key = (dataset, num_folds)
    if fold_key not in FOLD_PICKLE:
        raise ValueError(f"No fold pickle for dataset='{dataset}', num_folds={num_folds}.")
    fold_splits = pickle.load(open(PROJECT_ROOT / FOLD_PICKLE[fold_key], "rb"))
    eval_pids = set(fold_splits[fold]["eval"])
    npz_path = PROJECT_ROOT / DATASET_NPZ[dataset]
    return SequenceDataset(npz_path, eval_pids, seq_len=seq_len)


def reconstruct(lit: LstmVAELit, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """x: (T, 51) → recon: (T, 17, 3)."""
    lit.eval()
    with torch.no_grad():
        recon, _, _ = lit.model(x.unsqueeze(0).to(device))
    return recon.squeeze(0).cpu().numpy().reshape(-1, N_JOINTS, 3)


def impute(
    lit: LstmVAELit,
    x: torch.Tensor,
    coalition_mask: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Sample a completion from the masked encoder's mixture.

    x:  (T, 51)
    coalition_mask:  (n_joints,) bool for spatial or (T,) bool for temporal.

    Returns: (T, 17, 3) imputed pose sequence.
    """
    lit.eval()
    with torch.no_grad():
        x_b = x.unsqueeze(0).to(device)
        cm = coalition_mask.unsqueeze(0).to(device)
        out = lit.model.sample_masked(x_b, cm)
    return out.squeeze(0).cpu().numpy().reshape(-1, N_JOINTS, 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save GT-vs-reconstruction overlay GIFs for validation sequences."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .ckpt produced by train_lstm_vae.py.")
    p.add_argument("--dataset",    type=str, default="BMCLab",
                   choices=list(DATASET_NPZ.keys()))
    p.add_argument("--fold",       type=int, default=1)
    p.add_argument("--num_folds",  type=int, default=6)
    p.add_argument("--seq_len",    type=int, default=80,
                   help="Must match the seq_len used during training.")
    p.add_argument("--n_gifs",     type=int, default=8)
    p.add_argument("--fps",        type=int, default=15)
    p.add_argument("--out_dir",    type=str, default="artifacts/lstm_vae_recons")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Imputation mode
    p.add_argument("--mode", type=str, default="recon",
                   choices=["recon", "impute_spatial", "impute_temporal"],
                   help="recon: full-encoder reconstruction; "
                        "impute_spatial: mask specific joints; "
                        "impute_temporal: mask specific temporal windows.")
    p.add_argument("--mask_joints", type=int, nargs="+", default=None,
                   help="Joint indices to MASK (hold out) for spatial imputation. "
                        "Remaining joints are observed.")
    p.add_argument("--mask_windows", type=int, nargs="+", default=None,
                   help="Temporal window indices (0-3) to MASK for temporal imputation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    print(f"Loading checkpoint: {args.checkpoint}")
    lit = LstmVAELit.load_from_checkpoint(args.checkpoint, map_location=device).to(device)

    print(f"Loading {args.dataset} fold={args.fold} validation sequences ...")
    ds = load_eval_dataset(args.dataset, args.num_folds, args.fold, args.seq_len)
    print(f"  {len(ds)} validation clips available.")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_gifs, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False).tolist()

    # Build coalition mask for imputation modes
    coalition_mask: torch.Tensor | None = None
    mode_label = "recon"
    if args.mode == "impute_spatial":
        if lit.model.masked_encoder is None:
            raise RuntimeError("Checkpoint has no masked encoder (n_mix=0 at train time).")
        mask_joints = args.mask_joints or [14, 15, 16]
        mask = torch.ones(N_JOINTS, dtype=torch.bool)
        mask[mask_joints] = False
        coalition_mask = mask
        mode_label = f"impute_mask_joints{'_'.join(map(str, mask_joints))}"
        print(f"  Spatial imputation — masking joints {mask_joints}")
    elif args.mode == "impute_temporal":
        if lit.model.masked_encoder is None:
            raise RuntimeError("Checkpoint has no masked encoder (n_mix=0 at train time).")
        mask_windows = args.mask_windows or [2, 3]
        T = args.seq_len
        quarter = T // 4
        mask = torch.ones(T, dtype=torch.bool)
        for w in mask_windows:
            start = w * quarter
            end = (w + 1) * quarter if w < 3 else T
            mask[start:end] = False
        coalition_mask = mask
        mode_label = f"impute_mask_win{'_'.join(map(str, mask_windows))}"
        print(f"  Temporal imputation — masking windows {mask_windows}")

    for rank, idx in enumerate(indices):
        x = ds[idx]                                                # (T, 51)
        gt = x.numpy().reshape(args.seq_len, N_JOINTS, 3)         # (T, 17, 3)

        if coalition_mask is not None:
            result = impute(lit, x, coalition_mask, device)
            recon_label = "imputed"
            recon_color = "crimson"
        else:
            result = reconstruct(lit, x, device)
            recon_label = "recon"
            recon_color = "darkorange"

        out_path = out_dir / f"{args.dataset}_fold{args.fold}_{mode_label}_{rank:03d}.gif"
        save_overlay_gif(
            gt_tj3=gt,
            recon_tj3=result,
            edges=H36M17_EDGES,
            out_path=str(out_path),
            fps=args.fps,
            title_prefix=f"{args.dataset} fold{args.fold} val clip#{idx}",
            recon_color=recon_color,
        )

    print(f"\nDone — {n} GIFs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
