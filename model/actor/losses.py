"""ACTOR-style losses (rc, rr, kl, rcxyz, vel) — masked MSE / KL.

KL matches Mathux/ACTOR ``src/models/tools/losses.py``: analytic ELBO KL term with
**reduction=sum** over ``mu`` / ``logvar`` (not mean). ``lambda_kl`` must be
tuned together with batch size and ``latent_dim`` (same as upstream).

**XYZ vs rot6d:** ACTOR's default CVAE uses ``rc`` (on rotation params) plus
``rcxyz`` (on FK xyz). For ``pose_rep=\"xyz\"``, ``rc`` and ``rcxyz`` are the
same — we add **``rr``** (root-relative MSE) as the extra reconstruction term so
articulation is supervised, not only absolute coordinates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_rc_loss(_model, batch: dict) -> torch.Tensor:
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]
    perm = x.permute(0, 3, 1, 2)
    gtmasked = perm[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    if batch.get("_rc_l1", False):
        return F.l1_loss(outmasked, gtmasked, reduction="mean")
    return F.mse_loss(outmasked, gtmasked, reduction="mean")


def compute_rr_loss(_model, batch: dict) -> torch.Tensor:
    """Masked MSE on root-relative positions (joints relative to joint 0)."""
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]
    rel_x = x - x[:, 0:1, :, :]
    rel_o = output - output[:, 0:1, :, :]
    gtmasked = rel_x.permute(0, 3, 1, 2)[mask]
    outmasked = rel_o.permute(0, 3, 1, 2)[mask]
    return F.mse_loss(outmasked, gtmasked, reduction="mean")


def compute_rcxyz_loss(_model, batch: dict) -> torch.Tensor:
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    return F.mse_loss(outmasked, gtmasked, reduction="mean")


def compute_vel_loss(_model, batch: dict) -> torch.Tensor:
    x = batch["x"]
    output = batch["output"]
    gtvel = x[..., 1:] - x[..., :-1]
    outvel = output[..., 1:] - output[..., :-1]
    mask = batch["mask"][..., 1:]

    jw = batch.get("vel_joint_weights")
    if jw is not None:
        # jw: (J,) → (1, J, 1, 1) broadcast with (B, J, F, T-1)
        w = jw.view(1, -1, 1, 1)
        gtvel = gtvel * w
        outvel = outvel * w

    gtmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outmasked = outvel.permute(0, 3, 1, 2)[mask]
    return F.mse_loss(outmasked, gtmasked, reduction="mean")


def compute_velxyz_loss(_model, batch: dict) -> torch.Tensor:
    """Velocity MSE on FK-derived XYZ positions (root-centred metres).

    For rot6d mode ``x_xyz`` / ``output_xyz`` come from the SMPL FK and are
    already root-centred.  This loss directly penalises a static output: if
    the model predicts the same XYZ for every frame the ankle/knee velocity
    is zero while the GT velocity is non-zero, so the loss is large.

    Falls back to the raw representation velocity if ``x_xyz`` is absent
    (legacy xyz mode where ``x_xyz == x``).
    """
    x = batch.get("x_xyz", batch["x"])
    output = batch.get("output_xyz", batch["output"])
    gtvel  = x[..., 1:] - x[..., :-1]
    outvel = output[..., 1:] - output[..., :-1]
    mask   = batch["mask"][..., 1:]

    jw = batch.get("velxyz_joint_weights")
    if jw is not None:
        w = jw.view(1, -1, 1, 1)
        gtvel = gtvel * w
        outvel = outvel * w

    gtmasked  = gtvel.permute(0, 3, 1, 2)[mask]
    outmasked = outvel.permute(0, 3, 1, 2)[mask]
    return F.mse_loss(outmasked, gtmasked, reduction="mean")


def compute_kl_loss(_model, batch: dict) -> torch.Tensor:
    """Same as ACTOR: sum over batch × latent dims (see upstream compute_kl_loss)."""
    mu, logvar = batch["mu"], batch["logvar"]
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def compute_kl_free_bits_loss(_model, batch: dict) -> torch.Tensor:
    """Mean-reduced KL with a per-dimension free-bits floor.

    Each latent dimension is clamped to use at least ``free_nats`` of
    information, preventing posterior collapse while still regularising.
    ``free_nats`` is read from ``batch["_free_nats"]`` (default 0.1).
    """
    mu, logvar = batch["mu"], batch["logvar"]
    free_nats = batch.get("_free_nats", 0.1)
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(kl_per_dim, min=free_nats).mean()


def _masked_temporal_std(
    tensor: torch.Tensor,
    mask_f: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute per-(B, J, F) temporal std, excluding padded frames.

    Note: ``var`` is clamped to ``1e-8`` before sqrt. When the reconstruction
    collapses to a perfectly static sequence (var == 0), plain ``sqrt`` has a
    singular derivative and produces NaN gradients — exactly when the hinge
    loss most needs a clean push away from collapse.
    """
    t = tensor * mask_f
    mean = t.sum(dim=-1) / lengths.unsqueeze(-1)          # (B, J, F)
    diff2 = ((t - mean.unsqueeze(-1)) * mask_f).pow(2)
    var = diff2.sum(dim=-1) / (lengths.unsqueeze(-1) - 1)
    return var.clamp_min(1e-8).sqrt()


def compute_tstd_loss(_model, batch: dict) -> torch.Tensor:
    """Two-sided temporal std matching (MSE).  Penalises both under- and
    over-motion equally.  Prefer ``tstd_hinge`` for anti-collapse training.
    """
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]
    mask_f = mask[:, None, None, :].expand_as(x).float()
    lengths = mask.sum(dim=1, keepdim=True).clamp(min=2)
    std_out = _masked_temporal_std(output, mask_f, lengths)
    std_gt  = _masked_temporal_std(x,      mask_f, lengths)
    return F.mse_loss(std_out, std_gt)


def compute_tstd_hinge_loss(_model, batch: dict) -> torch.Tensor:
    """One-sided temporal std hinge loss.

    Only fires when ``recon_std < gt_std`` — i.e. the reconstruction is too
    static.  The loss is the mean L1 distance between the shortfall and zero:

        loss = mean(max(0, gt_std - recon_std))

    This creates a strong, asymmetric gradient that pushes the model to match
    or exceed GT temporal variability, while never penalising it for being
    more dynamic than the GT.  Combined with a large weight (≥500) this
    overcomes the mean-pose attractor without being softened by the MSE
    squaring of already-small residuals.
    """
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]
    mask_f = mask[:, None, None, :].expand_as(x).float()
    lengths = mask.sum(dim=1, keepdim=True).clamp(min=2)
    std_out = _masked_temporal_std(output, mask_f, lengths)  # (B, J, F)
    std_gt  = _masked_temporal_std(x,      mask_f, lengths)  # (B, J, F)
    shortfall = (std_gt - std_out).clamp(min=0.0)            # zero when recon >= gt
    return shortfall.mean()


_MATCHING = {
    "rc": compute_rc_loss,
    "rr": compute_rr_loss,
    "kl": compute_kl_loss,
    "kl_fb": compute_kl_free_bits_loss,
    "rcxyz": compute_rcxyz_loss,
    "vel": compute_vel_loss,
    "velxyz": compute_velxyz_loss,
    "tstd": compute_tstd_loss,
    "tstd_hinge": compute_tstd_hinge_loss,
}


def get_loss_function(name: str):
    return _MATCHING[name]
