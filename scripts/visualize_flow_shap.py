"""
visualize_flow_shap.py — plots showing how OTFlow-SHAP attributions
accumulate across joints and time.

Reads a ``psi.npz`` written by :mod:`scripts.compute_flow_shap` and produces
the following figures in ``--output_dir`` (defaults to ``<psi_dir>/figs/``):

1. ``summary_aggregates.png`` — per-joint total |psi| bar chart +
   per-frame total |psi| line (averaged across clips and coords).
2. ``joint_time_heatmap.png``  — mean |psi|(joint, frame), averaged over
   clips and coords.
3. ``anatomical_groups.png``   — per-(anatomical group) |psi| over time.
4. ``per_class_heatmaps.png``  — joint × time mean |psi| heatmaps faceted
   by predicted class (top) plus the signed contribution curves per class.
5. ``example_clips.png``       — joint × time heatmaps for a few
   representative clips (highest |Δf| per predicted class).
6. ``cumulative_over_time.png``— cumulative |psi| along the clip (running
   sum over frames), aggregated + per class.

Usage::

    python scripts/visualize_flow_shap.py \\
        --psi experiment_outs/flow_shap/bmclab_potr_fold1/psi.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.actor.shap_masking import H36M_GROUPS, H36M_JOINT_NAMES


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_psi(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    psi          = data["psi"]                 # (N, T, J, C) float32
    mask         = data["mask"]                # (N, T)       bool
    class_idx    = data["class_idx"]           # (N,)
    f_xstar      = data["f_xstar"]
    f_x0         = data["f_x0"]
    completeness = data["completeness_rel"]
    return {
        "psi":          psi,
        "mask":         mask,
        "class_idx":    class_idx,
        "f_xstar":      f_xstar,
        "f_x0":         f_x0,
        "completeness": completeness,
        "pelvis_world": data["pelvis_world"],
    }


def _apply_mask(psi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-out padded frames so aggregations respect real-frame count."""
    return psi * mask[:, :, None, None].astype(psi.dtype)


def _frame_counts_per_clip(mask: np.ndarray) -> np.ndarray:
    """Return ``(N,)`` number of real frames per clip."""
    return mask.sum(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Plot 1 — summary aggregates
# ---------------------------------------------------------------------------

def plot_summary_aggregates(
    psi: np.ndarray, mask: np.ndarray, out_path: Path,
) -> None:
    """Per-joint bar chart + per-frame curve of |psi| aggregated across clips."""
    N, T, J, C = psi.shape
    abs_psi = np.abs(psi) * mask[:, :, None, None].astype(psi.dtype)

    # Per-joint: sum over (N, T, C), normalise to fraction.
    per_joint = abs_psi.sum(axis=(0, 1, 3))                   # (J,)
    per_joint_frac = per_joint / per_joint.sum()

    # Per-frame: mean over (N, J, C). Weight frame index by masked mean across clips.
    per_frame_num = abs_psi.sum(axis=(0, 2, 3))               # (T,)
    per_frame_den = mask.sum(axis=0).clip(min=1.0) * J * C    # (T,)
    per_frame = per_frame_num / per_frame_den                  # mean |psi| per frame/joint/coord

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={"height_ratios": [1.2, 1]})
    colors = _joint_palette(J)
    bars = ax0.bar(range(J), per_joint_frac, color=colors, edgecolor="black", linewidth=0.5)
    ax0.set_xticks(range(J))
    ax0.set_xticklabels(H36M_JOINT_NAMES, rotation=45, ha="right")
    ax0.set_ylabel("Fraction of total |ψ|")
    ax0.set_title("Per-joint attribution magnitude (aggregated across clips)")
    ax0.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, per_joint_frac):
        ax0.text(bar.get_x() + bar.get_width() / 2, v, f"{v*100:.1f}%",
                 ha="center", va="bottom", fontsize=7)

    ax1.plot(np.arange(T), per_frame, color="tab:blue", linewidth=1.5)
    ax1.fill_between(np.arange(T), per_frame, color="tab:blue", alpha=0.2)
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Mean |ψ| per (joint, coord)")
    ax1.set_title("Per-frame attribution magnitude (averaged across clips)")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — joint × time heatmap
# ---------------------------------------------------------------------------

def plot_joint_time_heatmap(
    psi: np.ndarray, mask: np.ndarray, out_path: Path,
    title: str = "Mean |ψ|(joint, frame) — all clips",
) -> None:
    """2-D heatmap of mean |psi| at each (joint, frame)."""
    N, T, J, C = psi.shape
    abs_psi = np.abs(psi) * mask[:, :, None, None].astype(psi.dtype)
    num = abs_psi.sum(axis=(0, 3))                             # (T, J)
    den = (mask.sum(axis=0)[:, None] * C).clip(min=1.0)        # (T, 1)
    heat = (num / den).T                                        # (J, T)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(J))
    ax.set_yticklabels(H36M_JOINT_NAMES)
    ax.set_xlabel("Frame index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |ψ| per coord")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — anatomical groups over time
# ---------------------------------------------------------------------------

def plot_anatomical_groups(
    psi: np.ndarray, mask: np.ndarray, out_path: Path,
) -> None:
    N, T, J, C = psi.shape
    abs_psi = np.abs(psi) * mask[:, :, None, None].astype(psi.dtype)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.get_cmap("tab10").colors
    for ci, (group_name, joints) in enumerate(H36M_GROUPS.items()):
        num = abs_psi[:, :, joints, :].sum(axis=(0, 2, 3))        # (T,)
        den = mask.sum(axis=0).clip(min=1.0) * len(joints) * C   # (T,)
        vals = num / den
        ax.plot(range(T), vals, label=group_name, color=colors[ci % len(colors)], linewidth=1.8)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean |ψ| per (joint, coord) in group")
    ax.set_title("Anatomical-group attribution over time")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4 — per-class heatmaps + signed curves
# ---------------------------------------------------------------------------

def plot_per_class(
    psi: np.ndarray, mask: np.ndarray, class_idx: np.ndarray,
    num_classes: int, out_path: Path,
) -> None:
    N, T, J, C = psi.shape
    classes_present = sorted({int(c) for c in class_idx})

    # Layout: top row heatmaps (|psi|), bottom row signed per-joint curves
    n_cls = len(classes_present)
    fig, axs = plt.subplots(2, n_cls, figsize=(6 * n_cls, 8), squeeze=False)

    # Consistent colour scale for heatmaps across classes.
    per_class_heat: Dict[int, np.ndarray] = {}
    for c in classes_present:
        sel = (class_idx == c)
        if sel.sum() == 0:
            continue
        abs_psi = np.abs(psi[sel]) * mask[sel][:, :, None, None].astype(psi.dtype)
        num = abs_psi.sum(axis=(0, 3))                            # (T, J)
        den = (mask[sel].sum(axis=0)[:, None] * C).clip(min=1.0)
        per_class_heat[c] = (num / den).T                          # (J, T)

    vmax = max((h.max() for h in per_class_heat.values()), default=1.0)

    for col, c in enumerate(classes_present):
        ax_top = axs[0, col]
        sel = (class_idx == c)
        heat = per_class_heat[c]
        im = ax_top.imshow(heat, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=vmax)
        ax_top.set_yticks(range(J))
        ax_top.set_yticklabels(H36M_JOINT_NAMES, fontsize=8)
        ax_top.set_xlabel("Frame index")
        ax_top.set_title(f"Class {c} — mean |ψ|  (n={int(sel.sum())})")
        fig.colorbar(im, ax=ax_top, fraction=0.04, pad=0.02)

        # Bottom: signed per-joint attribution (sum over frames + coords).
        ax_bot = axs[1, col]
        signed = psi[sel] * mask[sel][:, :, None, None].astype(psi.dtype)
        signed = signed.sum(axis=(1, 3))                           # (n_sel, J)
        mean_signed = signed.mean(axis=0)
        ci_low = np.percentile(signed, 25, axis=0)
        ci_high = np.percentile(signed, 75, axis=0)

        x = np.arange(J)
        ax_bot.bar(x, mean_signed,
                   color=["tab:red" if v < 0 else "tab:blue" for v in mean_signed],
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        ax_bot.errorbar(x, mean_signed, yerr=[mean_signed - ci_low, ci_high - mean_signed],
                        fmt="none", ecolor="black", alpha=0.4, capsize=2)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(H36M_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
        ax_bot.axhline(0, color="black", linewidth=0.5)
        ax_bot.set_ylabel("Mean signed Σ ψ per clip")
        ax_bot.set_title(f"Class {c} — signed per-joint contribution (IQR bars)")
        ax_bot.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-class OTFlow-SHAP attributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5 — example clips
# ---------------------------------------------------------------------------

def plot_example_clips(
    psi: np.ndarray, mask: np.ndarray, class_idx: np.ndarray,
    delta_f: np.ndarray, out_path: Path,
    n_per_class: int = 2,
) -> None:
    """Signed joint × time heatmaps for a few high-|Δf| clips per class."""
    classes = sorted({int(c) for c in class_idx})
    rows: List[Tuple[int, int, float]] = []  # (clip_idx, class, delta_f)
    for c in classes:
        idxs = np.where(class_idx == c)[0]
        if idxs.size == 0:
            continue
        # pick clips with the largest |delta_f| — they are the most confident.
        order = np.argsort(-np.abs(delta_f[idxs]))
        for j in order[:n_per_class]:
            rows.append((int(idxs[j]), int(c), float(delta_f[idxs[j]])))

    n = len(rows)
    if n == 0:
        return
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.2 * nrows),
                            squeeze=False)

    # Symmetric colour scale across example clips.
    vmax = max(
        np.abs(psi[ci].sum(axis=-1) * mask[ci][:, None].astype(psi.dtype)).max()
        for ci, _, _ in rows
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for idx, (ci, cls, df) in enumerate(rows):
        ax = axs[idx // ncols, idx % ncols]
        heat = (psi[ci].sum(axis=-1) *
                mask[ci][:, None].astype(psi.dtype)).T            # (J, T)
        im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", norm=norm, origin="lower")
        ax.set_yticks(range(heat.shape[0]))
        ax.set_yticklabels(H36M_JOINT_NAMES, fontsize=7)
        ax.set_xlabel("Frame index")
        ax.set_title(f"clip {ci} — class {cls}, Δf={df:+.2f}")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    for k in range(n, nrows * ncols):
        axs[k // ncols, k % ncols].axis("off")

    fig.suptitle("Signed Σ_coord ψ  —  representative clips per predicted class",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6 — cumulative attribution over time
# ---------------------------------------------------------------------------

def plot_cumulative_over_time(
    psi: np.ndarray, mask: np.ndarray, class_idx: np.ndarray,
    out_path: Path,
) -> None:
    """Running-sum of |psi| (over joints+coords) along the clip, normalised
    to the per-clip total so all curves end at 1.0."""
    N, T, J, C = psi.shape
    abs_psi_frame = (np.abs(psi).sum(axis=(2, 3))                # (N, T)
                     * mask.astype(psi.dtype))
    total_per_clip = abs_psi_frame.sum(axis=1).clip(min=1e-8)    # (N,)
    cum = np.cumsum(abs_psi_frame, axis=1) / total_per_clip[:, None]   # (N, T)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Overall mean + 25-75 IQR band
    mean_all = cum.mean(axis=0)
    lo = np.percentile(cum, 25, axis=0)
    hi = np.percentile(cum, 75, axis=0)
    ax.plot(np.arange(T), mean_all, label="all clips (mean)", color="black", linewidth=2.0)
    ax.fill_between(np.arange(T), lo, hi, color="black", alpha=0.15, label="IQR")

    colors = plt.get_cmap("tab10").colors
    for i, c in enumerate(sorted({int(x) for x in class_idx})):
        sel = (class_idx == c)
        if sel.sum() == 0:
            continue
        ax.plot(np.arange(T), cum[sel].mean(axis=0),
                label=f"class {c} (mean, n={int(sel.sum())})",
                color=colors[i % len(colors)], linewidth=1.6, linestyle="--")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Cumulative fraction of Σ|ψ|")
    ax.set_title("Cumulative attribution along the clip (normalised per clip)")
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle=":")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aesthetic helpers
# ---------------------------------------------------------------------------

def _joint_palette(J: int = 17) -> List[str]:
    """Colour joints by anatomical group for the per-joint bar chart."""
    color_map = {
        "root":      "#5c5c5c",
        "right_leg": "#1f77b4",
        "left_leg":  "#17becf",
        "spine":     "#2ca02c",
        "head":      "#9467bd",
        "left_arm":  "#ff7f0e",
        "right_arm": "#d62728",
    }
    colors = ["grey"] * J
    for grp, joints in H36M_GROUPS.items():
        for j in joints:
            colors[j] = color_map.get(grp, "grey")
    return colors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psi", required=True, help="Path to psi.npz.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write figures. Defaults to <psi_dir>/figs/.")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--examples_per_class", type=int, default=2)
    args = p.parse_args()

    psi_path = Path(args.psi).resolve()
    if not psi_path.exists():
        raise FileNotFoundError(psi_path)
    out_dir = Path(args.output_dir) if args.output_dir else psi_path.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] loading {psi_path}")
    data = _load_psi(psi_path)
    psi          = data["psi"]
    mask         = data["mask"]
    class_idx    = data["class_idx"]
    delta_f      = data["f_xstar"] - data["f_x0"]

    print(f"[viz] psi shape = {psi.shape}, "
          f"{int(class_idx.size)} clips, "
          f"classes = {sorted(set(int(c) for c in class_idx))}")

    plot_summary_aggregates(psi, mask, out_dir / "summary_aggregates.png")
    print(f"[viz] wrote {out_dir / 'summary_aggregates.png'}")
    plot_joint_time_heatmap(psi, mask, out_dir / "joint_time_heatmap.png")
    print(f"[viz] wrote {out_dir / 'joint_time_heatmap.png'}")
    plot_anatomical_groups(psi, mask, out_dir / "anatomical_groups.png")
    print(f"[viz] wrote {out_dir / 'anatomical_groups.png'}")
    plot_per_class(psi, mask, class_idx, args.num_classes,
                   out_dir / "per_class_heatmaps.png")
    print(f"[viz] wrote {out_dir / 'per_class_heatmaps.png'}")
    plot_example_clips(psi, mask, class_idx, delta_f,
                       out_dir / "example_clips.png",
                       n_per_class=args.examples_per_class)
    print(f"[viz] wrote {out_dir / 'example_clips.png'}")
    plot_cumulative_over_time(psi, mask, class_idx,
                              out_dir / "cumulative_over_time.png")
    print(f"[viz] wrote {out_dir / 'cumulative_over_time.png'}")

    # Quick quantitative summary to JSON.
    abs_psi = np.abs(psi) * mask[:, :, None, None].astype(psi.dtype)
    per_joint = abs_psi.sum(axis=(0, 1, 3))
    per_joint_frac = per_joint / per_joint.sum()
    top_joints = np.argsort(-per_joint_frac)[:5]
    numeric_summary = {
        "num_clips": int(psi.shape[0]),
        "sequence_length": int(psi.shape[1]),
        "class_counts": {
            str(c): int((class_idx == c).sum())
            for c in sorted({int(x) for x in class_idx})
        },
        "top5_joints": [
            {"idx": int(j), "name": H36M_JOINT_NAMES[j],
             "fraction_of_total_abs_psi": float(per_joint_frac[j])}
            for j in top_joints
        ],
        "median_frame_of_50pct_cumulative": _median_t50(psi, mask),
    }
    summary_path = out_dir / "viz_summary.json"
    with open(summary_path, "w") as f:
        json.dump(numeric_summary, f, indent=2)
    print(f"[viz] wrote {summary_path}")
    print(json.dumps(numeric_summary, indent=2))


def _median_t50(psi: np.ndarray, mask: np.ndarray) -> float:
    """Median over clips of the earliest frame whose cumulative |psi|
    reaches 50 % of the clip's total."""
    abs_psi_frame = (np.abs(psi).sum(axis=(2, 3))
                     * mask.astype(psi.dtype))                   # (N, T)
    total = abs_psi_frame.sum(axis=1).clip(min=1e-8)
    cum = np.cumsum(abs_psi_frame, axis=1) / total[:, None]
    t50 = np.argmax(cum >= 0.5, axis=1)
    return float(np.median(t50))


if __name__ == "__main__":
    main()
