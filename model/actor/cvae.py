"""
cvae.py — ACTOR-style Transformer CVAE for motion sequences.

Architecture overview
---------------------
``ActorCVAE`` is a thin wrapper that orchestrates the following pipeline
(matching Mathux/ACTOR ``src/models/cvae.py``):

1. **Encoder** ``q_φ(z | x, y)``
   - Takes the motion sequence ``x: (B, njoints, nfeats, T)`` and class
     label ``y: (B,)`` via positional + class token.
   - Returns ``mu, logvar`` (posterior parameters).

2. **Reparameterisation** ``z = mu + ε·exp(0.5·logvar)``

3. **Decoder** ``p_θ(x̂ | z, y)``
   - Conditions on ``z`` (latent action token) and ``y``.
   - Returns ``output: (B, njoints, nfeats, T)``.

4. **FK** (``rot6d`` only)
   - If ``pose_rep == "rot6d"`` and a ``rotation2xyz`` callable is provided,
     both ``x`` and ``output`` are passed through SMPL FK to produce
     ``x_xyz`` and ``output_xyz``, which are needed by the ``rcxyz`` loss.
   - For ``pose_rep == "xyz"`` the FK step is skipped and
     ``x_xyz = x``, ``output_xyz = output`` (copies, no computation).

Loss terms (all present in ``_MATCHING`` in losses.py)
------------------------------------------------------
``rc``    — MSE on raw representation (rotations or XYZ).
``rcxyz`` — MSE on FK-derived XYZ positions (requires ``rotation2xyz`` for rot6d).
``kl``    — KL divergence: ``-0.5 * sum(1 + logvar - mu² - exp(logvar))``.
``rr``    — Root-relative MSE (only useful for pose_rep=="xyz").
``vel``   — Velocity (frame-delta) MSE.

ACTOR default config: ``rc=1, rcxyz=1, kl=1e-5``.

Batch layout (ACTOR convention)
--------------------------------
    x        : (B, njoints, nfeats, nframes)
    y        : (B,) long, class index ∈ [0, num_classes)
    mask     : (B, nframes) bool, True = valid frame
    lengths  : (B,) long
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from model.actor.losses import get_loss_function
from model.actor.h36m_rotation2xyz import h36m_vel_joint_weights


class ActorCVAE(nn.Module):
    """ACTOR-compatible conditional VAE for motion sequences.

    Args:
        encoder:      Transformer encoder (``Encoder_TRANSFORMER``).
        decoder:      Transformer decoder (``Decoder_TRANSFORMER``).
        lambdas:      Dict mapping loss name → weight.  Zero-weight terms are
                      dropped.  Example: ``{"rc": 1.0, "rcxyz": 1.0, "kl": 1e-5}``.
        latent_dim:   Dimensionality of the latent space.
        device:       Torch device (stored for convenience).
        pose_rep:     ``"xyz"`` or ``"rot6d"``.  Determines FK behaviour.
        num_classes:  Number of action classes (1 = unconditional).
        rotation2xyz: Optional callable ``(x, mask) → x_xyz`` that runs SMPL FK.
                      Required when ``"rcxyz"`` is in ``lambdas`` and
                      ``pose_rep == "rot6d"``.  Ignored for ``pose_rep == "xyz"``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lambdas: dict[str, float],
        latent_dim: int,
        device: torch.device,
        pose_rep: str = "xyz",
        num_classes: int = 1,
        rotation2xyz: Optional[Callable] = None,
        use_frame_tokens: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambdas = {k: float(v) for k, v in lambdas.items() if float(v) != 0.0}
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.num_classes = num_classes
        self.device = device
        self.rotation2xyz = rotation2xyz
        # When False, frame_tokens produced by the encoder are NOT forwarded to
        # the decoder.  The decoder must then reconstruct all T frames from z
        # alone (z-only mode).  This forces z to encode full temporal dynamics,
        # which is required for diverse ActorSHAP completions.
        self.use_frame_tokens = use_frame_tokens

        # Validate: rcxyz loss requires FK for rot6d data.
        if "rcxyz" in self.lambdas and pose_rep == "rot6d" and rotation2xyz is None:
            raise ValueError(
                "lambda_rcxyz > 0 with pose_rep='rot6d' requires a rotation2xyz "
                "callable.  Pass rotation2xyz=Rotation2xyz(device) to ActorCVAE."
            )

        # Per-joint velocity weights: concentrate vel gradient on dynamic joints.
        if pose_rep == "rot6d" and encoder.njoints == 32:
            self.register_buffer(
                "vel_joint_weights", h36m_vel_joint_weights(32),
            )
        else:
            self.vel_joint_weights = None

        self.losses = list(self.lambdas.keys()) + ["mixed"]

    # ------------------------------------------------------------------
    # Reparameterisation
    # ------------------------------------------------------------------

    def reparameterize(self, batch: dict, seed: int | None = None) -> torch.Tensor:
        """Sample latent z ~ N(mu, exp(logvar)) via the reparameterisation trick.

        When ``batch["_ae_deterministic"]`` is True, returns ``mu`` directly
        (autoencoder warmup — no KL noise).

        ``batch["_noise_scale"]`` (float, 0-1) can be set to linearly
        interpolate between deterministic (0) and full stochastic (1).
        """
        mu, logvar = batch["mu"], batch["logvar"]
        if batch.get("_ae_deterministic", False):
            return mu
        std = torch.exp(0.5 * logvar)
        if seed is None:
            eps = torch.randn_like(std)
        else:
            gen = torch.Generator(device=mu.device)
            gen.manual_seed(seed)
            eps = torch.randn(std.shape, device=mu.device, generator=gen)
        noise_scale = batch.get("_noise_scale", 1.0)
        return mu + noise_scale * eps * std

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """Run encoder → reparameterise → decoder → (optional) FK.

        Modifies ``batch`` in-place with the keys added by each step and
        returns it.  After this call the batch contains:
        - ``mu``, ``logvar`` — from encoder
        - ``z``              — sampled latent
        - ``output``         — decoded sequence ``(B, njoints, nfeats, T)``
        - ``x_xyz``          — XYZ positions of input (FK or copy)
        - ``output_xyz``     — XYZ positions of output (FK or copy)
        """
        # ---- FK on input (needs to happen before encoder for rcxyz) -----
        if self.pose_rep == "xyz":
            # XYZ data: FK is identity.
            batch["x_xyz"] = batch["x"]
        elif self.rotation2xyz is not None:
            batch["x_xyz"] = self.rotation2xyz(batch["x"], batch["mask"])

        # ---- Encoder + reparameterisation + decoder ----------------------
        batch.update(self.encoder(batch))
        if not self.use_frame_tokens:
            batch.pop("frame_tokens", None)
        batch["z"] = self.reparameterize(batch)
        batch.update(self.decoder(batch))

        # ---- FK on decoder output ----------------------------------------
        if self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        elif self.rotation2xyz is not None:
            batch["output_xyz"] = self.rotation2xyz(batch["output"], batch["mask"])

        return batch

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the weighted sum of all active losses.

        Returns:
            mixed: scalar loss tensor for ``loss.backward()``.
            losses: dict of individual loss values (detached floats) including
                    ``"mixed"`` for logging.
        """
        if self.vel_joint_weights is not None:
            batch["vel_joint_weights"] = self.vel_joint_weights
        x0 = batch["x"]
        mixed = torch.zeros((), device=x0.device, dtype=x0.dtype)
        losses: dict[str, float] = {}
        for ltype, lam in self.lambdas.items():
            fn = get_loss_function(ltype)
            v = fn(self, batch)
            mixed = mixed + float(lam) * v
            losses[ltype] = float(v.detach())
        losses["mixed"] = float(mixed.detach())
        return mixed, losses
