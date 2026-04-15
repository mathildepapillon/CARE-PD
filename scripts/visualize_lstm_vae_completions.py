"""
scripts/visualize_lstm_vae_completions.py — GIF visualisations and diversity
metrics for LSTM VAE reconstructions and masked completions.

Automatically selects the best masked-encoder checkpoint (lowest
``val/mpjpe_masked_obs``) from the run directory, then for every sampled
validation clip produces:

  1. A reconstruction GIF (full encoder → decoder).
  2. Spatial completion GIFs — each anatomical group masked in turn.
  3. Temporal completion GIFs — each quarter window masked individually, plus
     first-half (windows 0+1) and second-half (windows 2+3) combinations.
  4. Diversity GIFs — K completions overlaid in different colours so you can
     visually assess the spread of the masked encoder's posterior.

Diversity metrics
-----------------
For every (clip × mask) the script samples K completions and computes:

  APD (Average Pairwise Distance):
    Mean pairwise MPJPE across all K*(K-1)/2 pairs, measured **only on the
    masked region** (masked joints for spatial, masked frames for temporal).
    Near-zero = mode collapse; healthy spread = several cm.

  FID in latent space  (opt-in with ``--fid``):
    Fréchet distance between the distribution of z vectors drawn from the full
    encoder (deterministic mu) and from the masked encoder (sampled from the
    mixture posterior) across ``--fid_n_sequences`` validation clips.  One FID
    is reported per mask type.  Lower = masked encoder posterior is well-
    aligned with the full encoder's posterior.

Results are printed as a table and saved to ``<out_dir>/diversity_stats.json``.

Usage
-----
python scripts/visualize_lstm_vae_completions.py \\
    --dataset BMCLab --fold 1 \\
    --n_clips 4 --fps 15 \\
    --n_diversity_samples 20 \\
    --out_dir artifacts/lstm_vae_completions

Add ``--fid --fid_n_sequences 200`` to also compute latent-space FID.
"""

from __future__ import annotations

import argparse
import io
import json
import re
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
from model.actor.shap_masking import H36M_GROUPS, H36M_JOINT_NAMES

N_JOINTS = 17
FEAT_PER_JOINT = 3

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_masked_ckpt(checkpoints_dir: Path) -> Path:
    """Return the checkpoint with the lowest ``masked_obs`` metric.

    Checkpoints are named ``epoch<N>-masked_obs<F>.ckpt``.  The sentinel
    value 999.0000 written at the start of training (before any real masked
    validation has run) is excluded.
    """
    pattern = re.compile(r"masked_obs(-?\d+\.\d+)\.ckpt$")
    candidates: list[tuple[float, Path]] = []
    for p in sorted(checkpoints_dir.glob("*masked_obs*.ckpt")):
        m = pattern.search(p.name)
        if m is None:
            continue
        val = float(m.group(1))
        if val >= 999.0:  # sentinel — not yet trained
            continue
        if val <= 0.0:  # uninitialized / pre-warmup epoch (real MPJPE > 0)
            continue
        candidates.append((val, p))

    if not candidates:
        raise FileNotFoundError(
            f"No valid masked_obs checkpoints found in {checkpoints_dir}.\n"
            "Train with --n_mix > 0 to generate a masked encoder."
        )

    candidates.sort(key=lambda t: t[0])
    best_val, best_path = candidates[0]
    print(f"  Best masked-encoder ckpt: {best_path.name}  (masked_obs={best_val:.4f})")
    return best_path


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_skeleton_2d(
    ax,
    joints: np.ndarray,   # (J, 3) in dataset XYZ
    edges,
    col_h: int,
    col_v: int,
    *,
    color: str,
    alpha: float = 0.9,
    joint_subset: list[int] | None = None,
    linewidth: float = 2.0,
    markersize: float = 14,
) -> None:
    """Draw a stick figure on 2-D axes.

    If ``joint_subset`` is given, only edges where *both* endpoints belong to
    the subset are drawn (and only subset joints are scattered).
    """
    subset_set = set(joint_subset) if joint_subset is not None else None

    for a, b in edges:
        if subset_set is not None and not (a in subset_set and b in subset_set):
            continue
        ax.plot(
            [joints[a, col_h], joints[b, col_h]],
            [joints[a, col_v], joints[b, col_v]],
            color=color, linewidth=linewidth, alpha=alpha,
        )

    js = list(joint_subset) if joint_subset is not None else list(range(len(joints)))
    ax.scatter(
        joints[js, col_h], joints[js, col_v],
        c=color, s=markersize, alpha=alpha, zorder=5,
    )


# ---------------------------------------------------------------------------
# Reconstruction GIF (full encoder)
# ---------------------------------------------------------------------------

def save_overlay_gif(
    gt_tj3: np.ndarray,
    recon_tj3: np.ndarray,
    edges,
    out_path: str,
    fps: int = 15,
    title_prefix: str = "",
    gt_color: str = "steelblue",
    recon_color: str = "darkorange",
) -> None:
    """GT and reconstruction overlaid — front (X–Y) and side (Z–Y) panels.

    Parameters
    ----------
    gt_tj3, recon_tj3 : ``(T, J, 3)``
    """
    assert gt_tj3.shape == recon_tj3.shape
    T = gt_tj3.shape[0]

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
            _draw_skeleton_2d(ax, gt_tj3[fi],    edges, col_h, col_v, color=gt_color,    alpha=0.85)
            _draw_skeleton_2d(ax, recon_tj3[fi], edges, col_h, col_v, color=recon_color, alpha=0.75)
            ax.set_xlim(*h_lim); ax.set_ylim(*v_lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="lightgray", lw=0.5, ls="--")
            ax.axvline(0, color="lightgray", lw=0.5, ls="--")
            ax.set_title(view_title, fontsize=8)
            ax.tick_params(labelsize=7)
        ax_front.legend(handles=legend_elems, fontsize=7, loc="upper right", framealpha=0.6)
        title = f"{title_prefix}   frame {fi + 1}/{T}" if title_prefix else f"frame {fi + 1}/{T}"
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        frames.append(_fig_to_pil(fig))

    _save_gif(frames, out_path, fps)


# ---------------------------------------------------------------------------
# Completion GIF helpers
# ---------------------------------------------------------------------------

def save_spatial_completion_gif(
    gt_tj3: np.ndarray,
    imputed_tj3: np.ndarray,
    masked_joint_indices: list[int],
    edges,
    out_path: str,
    fps: int = 15,
    title_prefix: str = "",
    mask_label: str = "",
) -> None:
    """GIF comparing GT and spatial completion.

    GT is drawn full in steelblue.  The imputed sequence is drawn in two
    layers:
    - Observed joints (not masked): omitted (they are identical to GT after
      paste_observed, so no visual difference would be shown).
    - Masked joints: drawn in crimson from the imputed sequence, showing
      what the model predicted for the held-out group.

    Parameters
    ----------
    gt_tj3, imputed_tj3 : ``(T, J, 3)``
    masked_joint_indices : joint indices that were held out (masked = imputed
        by the model).
    """
    T = gt_tj3.shape[0]
    both = np.concatenate([gt_tj3, imputed_tj3], axis=0).reshape(-1, 3)
    margin = 0.1 * (np.ptp(both, axis=0).max() + 1e-6)
    lo, hi = both.min(0) - margin, both.max(0) + margin
    xlim, ylim, zlim = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])

    legend_elems = [
        Line2D([0], [0], color="steelblue", linewidth=2, label="GT"),
        Line2D([0], [0], color="crimson",   linewidth=2, label=f"model  [{mask_label}]"),
    ]

    frames = []
    for fi in range(T):
        fig, (ax_front, ax_side) = plt.subplots(1, 2, figsize=(8, 4.5))
        for ax, col_h, col_v, h_lim, v_lim, view_title in [
            (ax_front, 0, 1, xlim, ylim, "Front  (X – Y)"),
            (ax_side,  2, 1, zlim, ylim, "Side   (Z – Y)"),
        ]:
            # Full GT skeleton in blue
            _draw_skeleton_2d(ax, gt_tj3[fi], edges, col_h, col_v,
                              color="steelblue", alpha=0.85)
            # Model prediction for masked joints only, in crimson
            _draw_skeleton_2d(ax, imputed_tj3[fi], edges, col_h, col_v,
                              color="crimson", alpha=0.85,
                              joint_subset=masked_joint_indices)
            ax.set_xlim(*h_lim); ax.set_ylim(*v_lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="lightgray", lw=0.5, ls="--")
            ax.axvline(0, color="lightgray", lw=0.5, ls="--")
            ax.set_title(view_title, fontsize=8)
            ax.tick_params(labelsize=7)
        ax_front.legend(handles=legend_elems, fontsize=7, loc="upper right", framealpha=0.6)
        title = f"{title_prefix}   frame {fi + 1}/{T}" if title_prefix else f"frame {fi + 1}/{T}"
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        frames.append(_fig_to_pil(fig))

    _save_gif(frames, out_path, fps)


def save_temporal_completion_gif(
    gt_tj3: np.ndarray,
    imputed_tj3: np.ndarray,
    obs_mask: np.ndarray,    # (T,) bool — True = observed frame
    edges,
    out_path: str,
    fps: int = 15,
    title_prefix: str = "",
    mask_label: str = "",
) -> None:
    """GIF comparing GT and temporal completion.

    On **observed** frames the GT skeleton is drawn in steelblue (the imputed
    sequence is identical here due to paste_observed, so we skip it).
    On **masked** frames the GT is drawn in light gray and the model's
    prediction is drawn in crimson, making it easy to see the difference.

    Parameters
    ----------
    gt_tj3, imputed_tj3 : ``(T, J, 3)``
    obs_mask : ``(T,)`` bool, True = frame was observed (not masked).
    """
    T = gt_tj3.shape[0]
    both = np.concatenate([gt_tj3, imputed_tj3], axis=0).reshape(-1, 3)
    margin = 0.1 * (np.ptp(both, axis=0).max() + 1e-6)
    lo, hi = both.min(0) - margin, both.max(0) + margin
    xlim, ylim, zlim = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])

    legend_elems = [
        Line2D([0], [0], color="steelblue",  linewidth=2, label="GT (observed)"),
        Line2D([0], [0], color="lightgray",  linewidth=2, label="GT (masked region)"),
        Line2D([0], [0], color="crimson",    linewidth=2, label=f"model  [{mask_label}]"),
    ]

    frames = []
    for fi in range(T):
        observed = bool(obs_mask[fi])
        fig, (ax_front, ax_side) = plt.subplots(1, 2, figsize=(8, 4.5))
        for ax, col_h, col_v, h_lim, v_lim, view_title in [
            (ax_front, 0, 1, xlim, ylim, "Front  (X – Y)"),
            (ax_side,  2, 1, zlim, ylim, "Side   (Z – Y)"),
        ]:
            if observed:
                # Observed frame: GT only (imputed == GT here)
                _draw_skeleton_2d(ax, gt_tj3[fi], edges, col_h, col_v,
                                  color="steelblue", alpha=0.85)
            else:
                # Masked frame: faint GT + model prediction
                _draw_skeleton_2d(ax, gt_tj3[fi], edges, col_h, col_v,
                                  color="lightgray", alpha=0.6)
                _draw_skeleton_2d(ax, imputed_tj3[fi], edges, col_h, col_v,
                                  color="crimson", alpha=0.85)
            ax.set_xlim(*h_lim); ax.set_ylim(*v_lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="lightgray", lw=0.5, ls="--")
            ax.axvline(0, color="lightgray", lw=0.5, ls="--")
            ax.set_title(view_title, fontsize=8)
            ax.tick_params(labelsize=7)
        ax_front.legend(handles=legend_elems, fontsize=7, loc="upper right", framealpha=0.6)
        obs_str = "OBS" if observed else "MASKED"
        title = (
            f"{title_prefix}   frame {fi + 1}/{T}  [{obs_str}]"
            if title_prefix else f"frame {fi + 1}/{T}  [{obs_str}]"
        )
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        frames.append(_fig_to_pil(fig))

    _save_gif(frames, out_path, fps)


# ---------------------------------------------------------------------------
# PIL / GIF utilities
# ---------------------------------------------------------------------------

def _fig_to_pil(fig) -> PILImage.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = PILImage.open(buf).copy()
    buf.close()
    return img


def _save_gif(frames: list[PILImage.Image], out_path: str, fps: int) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=max(1, int(1000 / fps)),
    )
    print(f"  saved {Path(out_path).resolve()}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct(lit: LstmVAELit, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """Full encoder path.  x: (T, 51) → (T, 17, 3)."""
    lit.eval()
    recon, _, _ = lit.model(x.unsqueeze(0).to(device))
    return recon.squeeze(0).cpu().numpy().reshape(-1, N_JOINTS, FEAT_PER_JOINT)


@torch.no_grad()
def complete(
    lit: LstmVAELit,
    x: torch.Tensor,
    coalition_mask: torch.Tensor,
    device: torch.device,
    paste_observed: bool = True,
) -> np.ndarray:
    """Masked-encoder completion path.

    Mirrors the exact logic of ``LstmVaeShapWrapper.sample_completions``
    (single sample variant):
        root-centre → forward_masked → sample z → decode → restore pelvis
        → paste observed parts back.

    Since ``SequenceDataset`` already root-centres (joint 0 == 0 at all
    frames), the "restore pelvis" step is a no-op here.

    Parameters
    ----------
    x               : (T, 51) — root-centred input from SequenceDataset.
    coalition_mask  : (J,) bool for spatial or (T,) bool for temporal.
    paste_observed  : if True, copy observed joints/frames from x onto the
                      decoded output (same as SHAP evaluation).

    Returns
    -------
    (T, 17, 3) completed pose sequence.
    """
    lit.eval()
    x_b = x.unsqueeze(0).to(device)   # (1, T, 51)
    cm  = coalition_mask.unsqueeze(0).to(device)

    out = lit.model.sample_masked(x_b, cm)  # (1, T, 51)

    if paste_observed:
        T_len = x_b.shape[1]
        D = x_b.shape[2]
        if coalition_mask.shape[-1] == N_JOINTS:
            # Spatial paste: restore observed joints from GT
            gt_4d  = x_b.reshape(1, T_len, N_JOINTS, FEAT_PER_JOINT)
            out_4d = out.reshape(1, T_len, N_JOINTS, FEAT_PER_JOINT)
            obs = cm[:, None, :, None].expand_as(gt_4d)
            out_4d = torch.where(obs, gt_4d, out_4d)
            out = out_4d.reshape(1, T_len, D)
        else:
            # Temporal paste: restore observed frames from GT
            obs = cm[:, :, None].expand_as(x_b)
            out = torch.where(obs, x_b, out)

    return out.squeeze(0).cpu().numpy().reshape(-1, N_JOINTS, FEAT_PER_JOINT)


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def build_spatial_masks() -> list[tuple[str, list[int]]]:
    """Return (label, masked_joint_indices) for each anatomical group."""
    return [(name, joints) for name, joints in H36M_GROUPS.items()]


def build_temporal_masks(seq_len: int) -> list[tuple[str, np.ndarray, list[int]]]:
    """Return (label, obs_mask_bool, masked_window_indices) for each temporal perturbation.

    Splits the clip into 4 equal quarters and returns masks for:
    - each quarter individually (win0 … win3)
    - first half  (win0 + win1)
    - second half (win2 + win3)

    Returns
    -------
    list of (label, obs_mask, window_indices) where
        obs_mask : (T,) bool numpy array — True = observed frame
    """
    T = seq_len
    quarter = T // 4
    boundaries = [0, quarter, 2 * quarter, 3 * quarter, T]

    combos: list[tuple[str, list[int]]] = [
        ("win0",    [0]),
        ("win1",    [1]),
        ("win2",    [2]),
        ("win3",    [3]),
        ("win0+1",  [0, 1]),
        ("win2+3",  [2, 3]),
    ]

    result = []
    for label, win_idxs in combos:
        obs_mask = np.ones(T, dtype=bool)
        for w in win_idxs:
            obs_mask[boundaries[w]:boundaries[w + 1]] = False
        result.append((label, obs_mask, win_idxs))
    return result


# ---------------------------------------------------------------------------
# Multi-sample completion (batched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_k_completions(
    lit: LstmVAELit,
    x: torch.Tensor,               # (T, 51)
    coalition_mask: torch.Tensor,  # (J,) bool  or  (T,) bool
    device: torch.device,
    K: int = 20,
    paste_observed: bool = True,
) -> np.ndarray:
    """Sample K completions in one batched GPU call.

    Uses a single ``forward_masked`` pass to get the mixture posterior, then
    draws K independent reparameterised samples and decodes them all at once.
    This is equivalent to calling ``complete()`` K times but ~K× faster.

    Returns
    -------
    (K, T, 17, 3)
    """
    lit.eval()
    x_b  = x.unsqueeze(0).to(device)            # (1, T, 51)
    cm   = coalition_mask.unsqueeze(0).to(device)
    T_len = x_b.shape[1]

    # Encode once, sample K times from the mixture posterior
    log_pi, mu, logvar = lit.model.forward_masked(x_b, cm)  # each: (1, n_mix, *)
    log_pi_k = log_pi.expand(K, -1)             # (K, n_mix)
    mu_k     = mu.expand(K, -1, -1)             # (K, n_mix, D)
    logvar_k = logvar.expand(K, -1, -1)
    z_all    = lit.model.masked_encoder.sample(log_pi_k, mu_k, logvar_k)  # (K, D)

    # Decode all K latents in one LSTM forward pass
    recon_all = lit.model.decode(z_all, seq_len=T_len)  # (K, T, 51)

    if paste_observed:
        x_exp = x_b.expand(K, -1, -1)  # (K, T, 51)
        if coalition_mask.shape[-1] == N_JOINTS:
            # Spatial: paste observed joints back from GT
            gt_4d   = x_exp.reshape(K, T_len, N_JOINTS, FEAT_PER_JOINT)
            out_4d  = recon_all.reshape(K, T_len, N_JOINTS, FEAT_PER_JOINT)
            obs     = cm.expand(K, -1)[:, None, :, None].expand_as(gt_4d)
            recon_all = torch.where(obs, gt_4d, out_4d).reshape(K, T_len, N_JOINTS * FEAT_PER_JOINT)
        else:
            # Temporal: paste observed frames back from GT
            obs = cm.expand(K, -1)[:, :, None].expand_as(x_exp)
            recon_all = torch.where(obs, x_exp, recon_all)

    return recon_all.cpu().numpy().reshape(K, T_len, N_JOINTS, FEAT_PER_JOINT)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def compute_apd_spatial(
    samples: np.ndarray,             # (K, T, J, 3)
    masked_joint_indices: list[int],
) -> float:
    """Average Pairwise Distance on masked joints only.

    Computes mean MPJPE over all K*(K-1)/2 pairs of completions, where MPJPE
    is averaged over frames AND only the masked joints.

    Returns the APD in the same units as the joint positions (metres).
    """
    pts = samples[:, :, masked_joint_indices, :]   # (K, T, n_mj, 3)
    K = pts.shape[0]
    # (K, K, T, n_mj) pairwise L2
    diff  = pts[:, None] - pts[None, :]            # (K, K, T, n_mj, 3)
    dists = np.linalg.norm(diff, axis=-1).mean(axis=(-2, -1))  # (K, K)  avg over T, n_mj
    mask  = np.triu(np.ones((K, K), dtype=bool), k=1)
    return float(dists[mask].mean()) if mask.any() else 0.0


def compute_apd_temporal(
    samples: np.ndarray,             # (K, T, J, 3)
    obs_mask: np.ndarray,            # (T,) bool — True = observed, False = masked
) -> float:
    """Average Pairwise Distance on masked frames only.

    Returns the APD in the same units as the joint positions (metres).
    """
    masked_frames = ~obs_mask                       # True where frames were masked
    pts  = samples[:, masked_frames, :, :]          # (K, n_mf, J, 3)
    K    = pts.shape[0]
    diff = pts[:, None] - pts[None, :]              # (K, K, n_mf, J, 3)
    dists = np.linalg.norm(diff, axis=-1).mean(axis=(-2, -1))  # (K, K)
    mask  = np.triu(np.ones((K, K), dtype=bool), k=1)
    return float(dists[mask].mean()) if mask.any() else 0.0


@torch.no_grad()
def compute_fid_latent(
    lit: LstmVAELit,
    dataset: "SequenceDataset",
    coalition_mask: torch.Tensor,  # (J,) or (T,)
    device: torch.device,
    n_sequences: int = 200,
    n_samples_per_seq: int = 5,
    eps: float = 1e-4,
    chunk_size: int = 64,
) -> float:
    """Fréchet distance between full-encoder and masked-encoder z distributions.

    Encodes ``n_sequences`` validation clips with both encoders:
      - Full encoder → deterministic ``mu``  shape (N, latent_dim)
      - Masked encoder → ``n_samples_per_seq`` sampled z per clip
        shape (N × K, latent_dim)

    All LSTM forward passes are batched in ``chunk_size`` chunks so the cost is
    dominated by a small number of GPU kernel launches rather than N sequential
    single-sample calls.  Runs well on both GPU (fast) and CPU (acceptable for
    moderate N).

    Fits independent Gaussians to each point cloud and returns the Fréchet
    distance (lower = masked encoder posterior aligns with full encoder).
    ``eps * I`` is added to each covariance matrix to regularise the matrix
    square root when N is close to latent_dim.
    """
    from scipy.linalg import sqrtm

    lit.eval()
    n = min(n_sequences, len(dataset))
    K = n_samples_per_seq

    # Load all N sequences onto device in one transfer
    xs = torch.stack([dataset[i] for i in range(n)], dim=0).to(device)  # (N, T, 51)
    cm_full = coalition_mask.to(device)

    z_full_chunks:   list[np.ndarray] = []
    z_masked_chunks: list[np.ndarray] = []

    for start in range(0, n, chunk_size):
        end    = min(start + chunk_size, n)
        x_c    = xs[start:end]                                     # (B, T, 51)
        B      = x_c.shape[0]
        cm_c   = cm_full.unsqueeze(0).expand(B, -1)               # (B, J_or_T)

        # --- full encoder (deterministic mu) ---
        mu_f, _ = lit.model.encode(x_c)                           # (B, D)
        z_full_chunks.append(mu_f.cpu().numpy())

        # --- masked encoder: K samples per sequence ---
        log_pi, mu_m, logvar_m = lit.model.forward_masked(x_c, cm_c)
        # repeat_interleave so each sequence gets K independent samples
        log_pi_k = log_pi.repeat_interleave(K, dim=0)             # (B*K, n_mix)
        mu_k     = mu_m.repeat_interleave(K, dim=0)               # (B*K, n_mix, D)
        logvar_k = logvar_m.repeat_interleave(K, dim=0)
        z_m      = lit.model.masked_encoder.sample(log_pi_k, mu_k, logvar_k)  # (B*K, D)
        z_masked_chunks.append(z_m.cpu().numpy())

    z_full   = np.concatenate(z_full_chunks,   axis=0)            # (N, D)
    z_masked = np.concatenate(z_masked_chunks, axis=0)            # (N*K, D)

    D    = z_full.shape[1]
    mu1  = z_full.mean(0);   sigma1 = np.cov(z_full,   rowvar=False) + eps * np.eye(D)
    mu2  = z_masked.mean(0); sigma2 = np.cov(z_masked, rowvar=False) + eps * np.eye(D)

    diff    = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # numerical artefact from near-singular matrices

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ---------------------------------------------------------------------------
# Diversity GIF (K completions overlaid)
# ---------------------------------------------------------------------------

# Colour palette for up to 8 completions
_DIVERSITY_PALETTE = [
    "darkorange", "crimson", "mediumorchid", "forestgreen",
    "deeppink",   "saddlebrown", "steelblue",  "goldenrod",
]


def save_diversity_gif(
    gt_tj3: np.ndarray,           # (T, J, 3)
    samples_ktj3: np.ndarray,     # (K, T, J, 3)
    masked_joint_indices: list[int] | None,  # None → temporal, use obs_mask
    obs_mask: np.ndarray | None,  # (T,) bool — None → spatial, use joint indices
    edges,
    out_path: str,
    fps: int = 15,
    title_prefix: str = "",
    mask_label: str = "",
) -> None:
    """GIF showing GT alongside K diverse completions.

    GT is drawn in steelblue (full skeleton).  Each of the K completions is
    drawn in a distinct colour, but **only for the masked region** (masked
    joints for spatial, masked frames for temporal) so the spread is visible
    without cluttering observed parts.

    Parameters
    ----------
    masked_joint_indices : list of ints for spatial mode (None for temporal).
    obs_mask             : (T,) bool for temporal mode (None for spatial).
    """
    assert (masked_joint_indices is None) != (obs_mask is None), (
        "Exactly one of masked_joint_indices or obs_mask must be provided."
    )
    K, T, _, _ = samples_ktj3.shape

    all_data = np.concatenate([gt_tj3[None], samples_ktj3], axis=0)
    both = all_data.reshape(-1, 3)
    margin = 0.1 * (np.ptp(both, axis=0).max() + 1e-6)
    lo, hi = both.min(0) - margin, both.max(0) + margin
    xlim = (lo[0], hi[0]); ylim = (lo[1], hi[1]); zlim = (lo[2], hi[2])

    k_colors = _DIVERSITY_PALETTE[:K]

    legend_elems = [Line2D([0], [0], color="steelblue", linewidth=2, label="GT")]
    for ki, col in enumerate(k_colors):
        legend_elems.append(Line2D([0], [0], color=col, linewidth=1.5, alpha=0.75,
                                   label=f"sample {ki + 1}"))

    frames = []
    for fi in range(T):
        observed_frame = True if obs_mask is None else bool(obs_mask[fi])
        fig, (ax_front, ax_side) = plt.subplots(1, 2, figsize=(8, 4.5))

        for ax, col_h, col_v, h_lim, v_lim, view_title in [
            (ax_front, 0, 1, xlim, ylim, "Front  (X – Y)"),
            (ax_side,  2, 1, zlim, ylim, "Side   (Z – Y)"),
        ]:
            # GT: always draw full skeleton
            gt_color = "steelblue" if observed_frame else "lightgray"
            _draw_skeleton_2d(ax, gt_tj3[fi], edges, col_h, col_v,
                              color=gt_color, alpha=0.85)

            # Completions: draw only masked region
            for ki, col in enumerate(k_colors):
                sample = samples_ktj3[ki, fi]  # (J, 3)
                if masked_joint_indices is not None:
                    # Spatial: draw masked joints only
                    _draw_skeleton_2d(ax, sample, edges, col_h, col_v,
                                      color=col, alpha=0.65, linewidth=1.5,
                                      joint_subset=masked_joint_indices)
                else:
                    # Temporal: draw full skeleton only on masked frames
                    if not observed_frame:
                        _draw_skeleton_2d(ax, sample, edges, col_h, col_v,
                                          color=col, alpha=0.65, linewidth=1.5)

            ax.set_xlim(*h_lim); ax.set_ylim(*v_lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="lightgray", lw=0.5, ls="--")
            ax.axvline(0, color="lightgray", lw=0.5, ls="--")
            ax.set_title(view_title, fontsize=8)
            ax.tick_params(labelsize=7)

        ax_front.legend(handles=legend_elems, fontsize=6, loc="upper right",
                        framealpha=0.6, ncol=2)
        obs_str = "" if obs_mask is None else ("  [OBS]" if observed_frame else "  [MASKED]")
        title = (f"{title_prefix}   frame {fi + 1}/{T}{obs_str}"
                 if title_prefix else f"frame {fi + 1}/{T}{obs_str}")
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        frames.append(_fig_to_pil(fig))

    _save_gif(frames, out_path, fps)


# ---------------------------------------------------------------------------
# Dataset loading
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
        raise ValueError(
            f"No fold pickle registered for dataset='{dataset}', "
            f"num_folds={num_folds}.  Available: {list(FOLD_PICKLE.keys())}"
        )
    fold_splits = pickle.load(open(PROJECT_ROOT / FOLD_PICKLE[fold_key], "rb"))
    eval_pids   = set(fold_splits[fold]["eval"])
    npz_path    = PROJECT_ROOT / DATASET_NPZ[dataset]
    return SequenceDataset(npz_path, eval_pids, seq_len=seq_len)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate reconstruction and masked-completion GIFs for LSTM VAE. "
            "Automatically selects the best masked-encoder checkpoint."
        )
    )
    p.add_argument("--dataset",   type=str, default="BMCLab",
                   choices=list(DATASET_NPZ.keys()))
    p.add_argument("--fold",      type=int, default=1)
    p.add_argument("--num_folds", type=int, default=6)
    p.add_argument("--seq_len",   type=int, default=80,
                   help="Must match the seq_len used during training.")
    p.add_argument("--run_dir",   type=str, default=None,
                   help="Override the run directory (default: "
                        "experiment_outs/lstm_vae/{dataset}_fold{fold}).")
    p.add_argument("--n_clips",   type=int, default=4,
                   help="Number of validation clips to visualise.")
    p.add_argument("--fps",       type=int, default=15)
    p.add_argument("--out_dir",   type=str, default="artifacts/lstm_vae_completions")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_spatial",   action="store_true",
                   help="Skip spatial completion GIFs.")
    p.add_argument("--no_temporal",  action="store_true",
                   help="Skip temporal completion GIFs.")
    p.add_argument("--no_recon",     action="store_true",
                   help="Skip full-encoder reconstruction GIFs.")

    # Diversity / FID
    p.add_argument("--n_diversity_samples", type=int, default=20,
                   help="Number of completions to sample per (clip, mask) for "
                        "APD computation and diversity GIFs. Set to 0 to skip.")
    p.add_argument("--n_diversity_gif_samples", type=int, default=5,
                   help="How many of the diversity samples to show in the "
                        "diversity GIF (≤ n_diversity_samples).")
    p.add_argument("--no_diversity_gifs", action="store_true",
                   help="Compute APD but skip saving diversity GIFs.")
    p.add_argument("--metrics_only", action="store_true",
                   help="Skip ALL GIF generation; only compute APD diversity "
                        "stats (and FID if --fid). Useful for a fast diagnostic "
                        "pass before committing to a full render run.")
    p.add_argument("--fid", action="store_true",
                   help="Compute latent-space FID per mask type across the "
                        "validation set (slow — set --fid_n_sequences to limit).")
    p.add_argument("--fid_n_sequences", type=int, default=200,
                   help="Number of validation sequences to use for FID "
                        "computation (default 200).")
    p.add_argument("--fid_n_samples_per_seq", type=int, default=5,
                   help="Masked-encoder samples per sequence for FID (default 5).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    tag     = f"{args.dataset}_fold{args.fold}"

    do_diversity  = args.n_diversity_samples > 1
    n_div         = args.n_diversity_samples
    n_div_gif     = min(args.n_diversity_gif_samples, n_div)
    metrics_only  = args.metrics_only
    skip_div_gifs = args.no_diversity_gifs or metrics_only

    # ------------------------------------------------------------------ ckpt
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = PROJECT_ROOT / "experiment_outs" / "lstm_vae" / tag
    ckpts_dir = run_dir / "checkpoints"
    ckpt_path = find_best_masked_ckpt(ckpts_dir)

    print(f"Loading checkpoint …")
    lit = LstmVAELit.load_from_checkpoint(str(ckpt_path), map_location=device).to(device)
    lit.eval()

    if lit.model.masked_encoder is None:
        raise RuntimeError(
            "The loaded checkpoint has no masked encoder (n_mix=0). "
            "This usually means the checkpoint was saved before masked training began."
        )

    # ------------------------------------------------------------------ data
    print(f"Loading {args.dataset} fold={args.fold} validation sequences …")
    ds = load_eval_dataset(args.dataset, args.num_folds, args.fold, args.seq_len)
    print(f"  {len(ds)} validation clips available.")

    rng = np.random.default_rng(args.seed)
    n   = min(args.n_clips, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False).tolist()

    # ------------------------------------------------------------------ masks
    spatial_masks  = build_spatial_masks()
    temporal_masks = build_temporal_masks(args.seq_len)

    if metrics_only:
        gif_count = 0
        print(f"\n--metrics_only: skipping all GIF generation, computing APD"
              f"{' + FID' if args.fid else ''} only …\n")
    else:
        gif_count = n * (
            (0 if args.no_recon    else 1) +
            (0 if args.no_spatial  else len(spatial_masks)) +
            (0 if args.no_temporal else len(temporal_masks))
        )
        if do_diversity and not skip_div_gifs:
            gif_count += n * (
                (0 if args.no_spatial  else len(spatial_masks)) +
                (0 if args.no_temporal else len(temporal_masks))
            )
        print(f"\nGenerating {gif_count} GIFs for {n} clips …\n")

    # acc[mask_key] = list of APD floats (one per clip)
    apd_acc: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ loop
    for rank, idx in enumerate(indices):
        x   = ds[idx]                                                  # (T, 51)
        gt  = x.numpy().reshape(args.seq_len, N_JOINTS, FEAT_PER_JOINT)  # (T, 17, 3)
        prefix = f"{tag} val clip#{idx}"

        # -- 1. Reconstruction
        if not args.no_recon and not metrics_only:
            recon = reconstruct(lit, x, device)
            save_overlay_gif(
                gt_tj3=gt,
                recon_tj3=recon,
                edges=H36M17_EDGES,
                out_path=str(out_dir / f"{tag}_recon_{rank:03d}.gif"),
                fps=args.fps,
                title_prefix=f"{prefix} | recon",
            )

        # -- 2. Spatial completions
        if not args.no_spatial:
            for group_name, masked_joints in spatial_masks:
                coalition = torch.ones(N_JOINTS, dtype=torch.bool)
                coalition[masked_joints] = False

                joint_labels = [H36M_JOINT_NAMES[j] for j in masked_joints]
                mask_label   = f"{group_name} ({', '.join(joint_labels)})"
                mask_key     = f"spatial/{group_name}"

                # Single-sample completion GIF
                if not metrics_only:
                    imputed = complete(lit, x, coalition, device, paste_observed=True)
                    save_spatial_completion_gif(
                        gt_tj3=gt,
                        imputed_tj3=imputed,
                        masked_joint_indices=masked_joints,
                        edges=H36M17_EDGES,
                        out_path=str(out_dir / f"{tag}_spatial_{group_name}_{rank:03d}.gif"),
                        fps=args.fps,
                        title_prefix=f"{prefix} | spatial mask: {group_name}",
                        mask_label=mask_label,
                    )

                # Diversity — K completions
                if do_diversity:
                    samples = sample_k_completions(
                        lit, x, coalition, device, K=n_div, paste_observed=True,
                    )  # (K, T, J, 3)

                    apd = compute_apd_spatial(samples, masked_joints)
                    apd_acc.setdefault(mask_key, []).append(apd)

                    if not skip_div_gifs:
                        save_diversity_gif(
                            gt_tj3=gt,
                            samples_ktj3=samples[:n_div_gif],
                            masked_joint_indices=masked_joints,
                            obs_mask=None,
                            edges=H36M17_EDGES,
                            out_path=str(out_dir / f"{tag}_spatial_{group_name}_diversity_{rank:03d}.gif"),
                            fps=args.fps,
                            title_prefix=(
                                f"{prefix} | spatial diversity: {group_name} "
                                f"APD={apd * 1000:.1f} mm"
                            ),
                            mask_label=mask_label,
                        )

        # -- 3. Temporal completions
        if not args.no_temporal:
            for win_label, obs_mask_np, _win_idxs in temporal_masks:
                obs_mask_t = torch.from_numpy(obs_mask_np)
                n_masked   = int((~obs_mask_np).sum())
                mask_label = f"{win_label} ({n_masked} frames masked)"
                mask_key   = f"temporal/{win_label}"

                # Single-sample completion GIF
                if not metrics_only:
                    imputed = complete(lit, x, obs_mask_t, device, paste_observed=True)
                    save_temporal_completion_gif(
                        gt_tj3=gt,
                        imputed_tj3=imputed,
                        obs_mask=obs_mask_np,
                        edges=H36M17_EDGES,
                        out_path=str(out_dir / f"{tag}_temporal_{win_label}_{rank:03d}.gif"),
                        fps=args.fps,
                        title_prefix=f"{prefix} | temporal mask: {win_label}",
                        mask_label=mask_label,
                    )

                # Diversity — K completions
                if do_diversity:
                    samples = sample_k_completions(
                        lit, x, obs_mask_t, device, K=n_div, paste_observed=True,
                    )  # (K, T, J, 3)

                    apd = compute_apd_temporal(samples, obs_mask_np)
                    apd_acc.setdefault(mask_key, []).append(apd)

                    if not skip_div_gifs:
                        save_diversity_gif(
                            gt_tj3=gt,
                            samples_ktj3=samples[:n_div_gif],
                            masked_joint_indices=None,
                            obs_mask=obs_mask_np,
                            edges=H36M17_EDGES,
                            out_path=str(out_dir / f"{tag}_temporal_{win_label}_diversity_{rank:03d}.gif"),
                            fps=args.fps,
                            title_prefix=(
                                f"{prefix} | temporal diversity: {win_label} "
                                f"APD={apd * 1000:.1f} mm"
                            ),
                            mask_label=mask_label,
                        )

    # ------------------------------------------------------------------ APD summary
    diversity_results: dict = {}

    if do_diversity and apd_acc:
        print("\n" + "=" * 62)
        print(f"  Diversity summary  (K={n_div} samples, n_clips={n})")
        print("=" * 62)
        print(f"  {'Mask type':<30}  {'APD mean':>10}  {'APD std':>9}")
        print("-" * 62)
        for key in sorted(apd_acc.keys()):
            vals = np.array(apd_acc[key])
            mean_mm = vals.mean() * 1000
            std_mm  = vals.std()  * 1000
            print(f"  {key:<30}  {mean_mm:>9.2f}mm  {std_mm:>8.2f}mm")
            diversity_results[key] = {
                "apd_mean_m": float(vals.mean()),
                "apd_std_m":  float(vals.std()),
                "apd_per_clip_m": vals.tolist(),
                "n_samples": n_div,
                "n_clips":   len(vals),
            }
        print("=" * 62)
        print("  APD = average pairwise MPJPE on the masked region.")
        print("  Near-zero → mode collapse; healthy diversity → several mm.\n")

    # ------------------------------------------------------------------ FID
    if args.fid:
        print(f"\nComputing latent-space FID ({args.fid_n_sequences} seqs, "
              f"{args.fid_n_samples_per_seq} samples/seq per mask) …")
        fid_results: dict[str, float] = {}

        all_masks: list[tuple[str, torch.Tensor]] = []
        if not args.no_spatial:
            for group_name, masked_joints in spatial_masks:
                cm = torch.ones(N_JOINTS, dtype=torch.bool)
                cm[masked_joints] = False
                all_masks.append((f"spatial/{group_name}", cm))
        if not args.no_temporal:
            for win_label, obs_mask_np, _ in temporal_masks:
                cm = torch.from_numpy(obs_mask_np)
                all_masks.append((f"temporal/{win_label}", cm))

        print(f"  {'Mask type':<30}  {'FID (latent)':>14}")
        print("-" * 50)
        for mask_key, cm in all_masks:
            fid_val = compute_fid_latent(
                lit, ds, cm, device,
                n_sequences=args.fid_n_sequences,
                n_samples_per_seq=args.fid_n_samples_per_seq,
            )
            print(f"  {mask_key:<30}  {fid_val:>14.3f}")
            fid_results[mask_key] = fid_val
            diversity_results.setdefault(mask_key, {})["fid_latent"] = fid_val

        print("-" * 50)
        print("  FID: lower = masked encoder posterior aligns with full encoder.")

    # ------------------------------------------------------------------ save JSON
    if diversity_results:
        json_path = out_dir / "diversity_stats.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(diversity_results, indent=2))
        print(f"\n  Diversity stats saved → {json_path.resolve()}")

    print(f"\nDone — GIFs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
