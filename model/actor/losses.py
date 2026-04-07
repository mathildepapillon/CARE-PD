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
    gtmasked  = gtvel.permute(0, 3, 1, 2)[mask]
    outmasked = outvel.permute(0, 3, 1, 2)[mask]
    return F.mse_loss(outmasked, gtmasked, reduction="mean")


def compute_kl_loss(_model, batch: dict) -> torch.Tensor:
    """Same as ACTOR: sum over batch × latent dims (see upstream compute_kl_loss)."""
    mu, logvar = batch["mu"], batch["logvar"]
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


_MATCHING = {
    "rc": compute_rc_loss,
    "rr": compute_rr_loss,
    "kl": compute_kl_loss,
    "rcxyz": compute_rcxyz_loss,
    "vel": compute_vel_loss,
    "velxyz": compute_velxyz_loss,
}


def get_loss_function(name: str):
    return _MATCHING[name]
