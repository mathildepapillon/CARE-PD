"""diagnostics.py — numerical quality indicators for OTFlow-SHAP outputs.

These are intentionally small, stateless helpers that operate on the tensors
returned by :func:`attribution.compute_flow_shap` plus a few simple signals
from the flow trajectory. The driver script batches them up and writes the
aggregated statistics to ``summary.json`` per fold.

All functions return CPU float tensors so they are safe to log via wandb /
JSON without extra coercion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from model.flow_matching import VelocityNet


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def completeness_residual(
    psi: Tensor,                  # (B, T, J, C)
    f_xstar: Tensor,              # (B,)
    f_x0: Tensor,                 # (B,)
    mask: Optional[Tensor] = None,  # (B, T) bool
    eps: float = 1e-8,
) -> Dict[str, Tensor]:
    """Per-sample completeness residual and its relative form.

    Returns:
        ``{"abs": (B,), "rel": (B,), "delta_f": (B,)}`` where

        - ``abs = |sum(psi) - (f_xstar - f_x0)|``
        - ``rel = abs / max(|f_xstar - f_x0|, eps)``
        - ``delta_f = f_xstar - f_x0``  (sign preserved).
    """
    if mask is not None:
        m = mask.to(dtype=psi.dtype)[..., None, None]       # (B, T, 1, 1)
        psi_sum = (psi * m).sum(dim=(1, 2, 3))
    else:
        psi_sum = psi.sum(dim=(1, 2, 3))
    delta_f = f_xstar - f_x0
    abs_res = (psi_sum - delta_f).abs()
    rel_res = abs_res / delta_f.abs().clamp(min=eps)
    return {"abs": abs_res.detach().cpu(),
            "rel": rel_res.detach().cpu(),
            "delta_f": delta_f.detach().cpu()}


# ---------------------------------------------------------------------------
# Flow Consistency Error (Eq. 10, OTFlow-SHAP paper)
# ---------------------------------------------------------------------------

def flow_consistency_error(
    velocity_net: VelocityNet,
    trajectory: Sequence[Tensor],     # K+1 tensors (B, T, J, C) for t=0..1
    t_grid: Tensor,                   # (K+1,)
    mask: Tensor,                     # (B, T) bool
) -> Dict[str, Tensor]:
    """Average squared mismatch between ODE steps and the learned velocity.

    ``FCE = mean_k || (x_{k+1} - x_k)/dt - v_theta(x_k, t_k) ||^2``

    A large FCE indicates the integrator is not following ``v_theta``
    faithfully — typically either the step size is too coarse or ``x_k`` has
    drifted off-manifold.

    Args:
        velocity_net: trained flow.
        trajectory: output of ``compute_flow_shap(..., return_trajectory=True)``
            — length ``K+1``, indexed forward in time.
        t_grid: ``(K+1,)`` support times.
        mask: ``(B, T)`` bool pad mask.

    Returns:
        ``{"per_sample": (B,), "overall": (1,)}`` — mean FCE per clip and
        over all clips.
    """
    velocity_net.eval()
    K = len(trajectory) - 1
    if K < 1:
        raise ValueError("trajectory must contain at least 2 points")
    dt = (t_grid[1:] - t_grid[:-1])                   # (K,)
    B = trajectory[0].shape[0]
    device = trajectory[0].device
    m = mask.to(device=device, dtype=trajectory[0].dtype)[..., None, None]   # (B, T, 1, 1)

    acc = torch.zeros(B, device=device, dtype=trajectory[0].dtype)
    with torch.no_grad():
        for k in range(K):
            x_k = trajectory[k]
            x_kp1 = trajectory[k + 1]
            t_k = t_grid[k].expand(B)
            v_k = velocity_net(x_k, t_k, mask=mask.bool())
            residual = (x_kp1 - x_k) / dt[k] - v_k                               # (B, T, J, C)
            r2 = (residual ** 2 * m).sum(dim=(1, 2, 3))
            denom = m.sum(dim=(1, 2, 3)).clamp(min=1.0)
            acc = acc + (r2 / denom)
    per_sample = acc / float(K)
    return {"per_sample": per_sample.detach().cpu(),
            "overall": per_sample.mean().unsqueeze(0).detach().cpu()}


# ---------------------------------------------------------------------------
# Pelvis leak
# ---------------------------------------------------------------------------

def pelvis_leak(
    psi_preserved: Tensor,      # (B, T, J, C)  attributions BEFORE zeroing
    mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Fraction of total |psi| that falls on the pelvis (joint 0).

    Under the pelvis-quotient construction (constant pelvis trajectory along
    the flow path) the pelvis attribution should be identically zero. Any
    non-trivial fraction indicates a leak through the projection step
    (``unroot_to_global + project_for_backbone``) — e.g. PoseFormerV2's
    perspective projection mixes pelvis position into all 2D joint coords.
    """
    if mask is not None:
        m = mask.to(dtype=psi_preserved.dtype)[..., None, None]
        psi_m = psi_preserved * m
    else:
        psi_m = psi_preserved
    pelvis_abs = psi_m[..., 0, :].abs().sum(dim=(1, 2))        # (B,)
    total_abs  = psi_m.abs().sum(dim=(1, 2, 3))                # (B,)
    frac = pelvis_abs / total_abs.clamp(min=1e-12)
    return {"pelvis_abs": pelvis_abs.detach().cpu(),
            "total_abs":  total_abs.detach().cpu(),
            "fraction":   frac.detach().cpu()}


# ---------------------------------------------------------------------------
# Endpoint-bucket integrand magnitudes (diagnoses endpoint-loss sensitivity)
# ---------------------------------------------------------------------------

def integrand_magnitude_by_bucket(
    t_grid: Tensor,                   # (K+1,)
    g_per_step: Sequence[Tensor],     # list of K+1 tensors (B, T, J, C)
    v_per_step: Sequence[Tensor],
    buckets: Sequence[tuple] = (
        (0.0, 0.1), (0.1, 0.3), (0.3, 0.5),
        (0.5, 0.7), (0.7, 0.9), (0.9, 1.0),
    ),
) -> List[Dict[str, float]]:
    """Median ``|g|``, ``|v|`` and ``|g * v|`` inside each ``t`` bucket.

    The caller has to provide the per-step gradient and velocity tensors (not
    cached inside :func:`compute_flow_shap` by default). This helper is
    intended for ad-hoc deep dives — the main driver logs coarser summaries.
    """
    t_vals = t_grid.detach().cpu().tolist()
    per_bucket: List[Dict[str, float]] = []
    for lo, hi in buckets:
        idxs = [i for i, t in enumerate(t_vals) if lo <= t < hi or (hi == 1.0 and t == 1.0)]
        if not idxs:
            per_bucket.append({"lo": lo, "hi": hi, "n_points": 0,
                               "median_abs_g": float("nan"),
                               "median_abs_v": float("nan"),
                               "median_abs_gv": float("nan")})
            continue
        gs = torch.stack([g_per_step[i].abs().median() for i in idxs])
        vs = torch.stack([v_per_step[i].abs().median() for i in idxs])
        gv = torch.stack([(g_per_step[i] * v_per_step[i]).abs().median()
                          for i in idxs])
        per_bucket.append({
            "lo": lo, "hi": hi, "n_points": len(idxs),
            "median_abs_g":  float(gs.mean().item()),
            "median_abs_v":  float(vs.mean().item()),
            "median_abs_gv": float(gv.mean().item()),
        })
    return per_bucket
