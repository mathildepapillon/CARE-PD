"""
vaeac_actor_motion.py — VAEAC on the ACTOR Transformer backbone (global z).

Architecture  (identical to ACTOR for the training path)
---------------------------------------------------------
q_φ(z | x, y)      Full encoder.  Encoder_TRANSFORMER from ACTOR.
                    Produces mu, logvar ∈ ℝ^{B × D} via muQuery / sigmaQuery.

r_ψ(z | x_S, S)    Masked encoder.  Same Encoder_TRANSFORMER, but unobserved
                    positions are replaced by learnable mask tokens before the
                    Transformer runs.  Produces its own mu, logvar.

p_θ(x̂ | z, y)      Decoder_TRANSFORMER (unchanged from ACTOR).  Cross-attention
                    memory is z + actionBiases[y] (single global token, same as
                    the original ACTOR paper).  With use_frame_tokens=True,
                    per-frame encoder outputs are additionally prepended; however
                    the default (False) matches the original ACTOR architecture.

VAEAC addition
--------------
Training:  q_φ encodes the full sequence → z → decoder.
           r_ψ encodes the masked sequence (for KL computation only).
           KL(q_φ ‖ r_ψ) aligns r_ψ to q_φ.  q_φ is DETACHED from KL.
           KL(q_φ ‖ N(0,1)) regularises q_φ (identical to ACTOR's KL term).
Inference: r_ψ encodes the masked sequence → z → decoder.
           q_φ is never called.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.actor.transformer_arch import Decoder_TRANSFORMER, Encoder_TRANSFORMER
from model.actor.h36m_rotation2xyz import h36m_vel_joint_weights


# ---------------------------------------------------------------------------
# Full encoder q_φ(z | x, y)
# ---------------------------------------------------------------------------

class VaeacActorFullEncoder(nn.Module):
    """Full encoder (proposal) q_φ — thin wrapper around Encoder_TRANSFORMER."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = Encoder_TRANSFORMER(**encoder_kwargs)

    def forward(self, batch: dict) -> dict:
        out = self._encoder(batch)
        return {
            "mu_full": out["mu"],            # (B, D)
            "logvar_full": out["logvar"],     # (B, D)
            "frame_tokens": out["frame_tokens"],  # (T, B, D) deterministic
        }


# ---------------------------------------------------------------------------
# Masked encoder r_ψ(z | x_S, S)
# ---------------------------------------------------------------------------

class VaeacActorMaskedEncoder(nn.Module):
    """Masked encoder (prior) r_ψ — Encoder_TRANSFORMER with input masking.

    Spatial  mask (coalition_mask shape B×J):
        Replaces unobserved joints in raw feature space with learnable
        mask_token_spatial[j] before skelEmbedding.

    Temporal mask (coalition_mask shape B×T):
        Replaces unobserved frame embeddings (after skelEmbedding) with a
        learnable mask_token_temporal, then continues with the standard
        Encoder_TRANSFORMER pipeline (muQuery/sigmaQuery prepend, PE, etc.).
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = Encoder_TRANSFORMER(**encoder_kwargs)
        njoints = encoder_kwargs["njoints"]
        nfeats = encoder_kwargs["nfeats"]
        latent_dim = encoder_kwargs.get("latent_dim", 256)
        self.mask_token_spatial = nn.Parameter(torch.zeros(njoints, nfeats))
        self.mask_token_temporal = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, batch: dict) -> dict:
        cm = batch["coalition_mask"]
        x = batch["x"]
        is_spatial = cm.shape[-1] == x.shape[1]
        if is_spatial:
            return self._encode_spatial(batch)
        return self._encode_temporal(batch)

    def _encode_spatial(self, batch: dict) -> dict:
        """Replace unobserved joints → call Encoder_TRANSFORMER normally."""
        x = batch["x"]
        cm = batch["coalition_mask"]  # (B, J) True=observed
        B, J, F, T = x.shape
        # mask_token_spatial: (J, F) → (1, J, F, 1) for broadcast with (B, J, F, T)
        mask_vals = self.mask_token_spatial.unsqueeze(0).unsqueeze(-1).expand_as(x)
        obs = cm.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # (B,J,1,1) → (B,J,F,T)
        x = torch.where(obs, x, mask_vals)
        out = self._encoder({**batch, "x": x})
        return {
            "mu_masked": out["mu"],
            "logvar_masked": out["logvar"],
            "frame_tokens_masked": out["frame_tokens"],
        }

    def _encode_temporal(self, batch: dict) -> dict:
        """Replace unobserved frame embeddings after skelEmbedding.

        We decompose the Encoder_TRANSFORMER forward pass so we can inject
        mask tokens between skelEmbedding and the muQuery/sigmaQuery prepend.
        """
        x, y, mask, cm = batch["x"], batch["y"], batch["mask"], batch["coalition_mask"]
        enc = self._encoder
        bs, njoints, nfeats, nframes = x.shape

        x_flat = x.permute(3, 0, 1, 2).reshape(nframes, bs, njoints * nfeats)
        h = enc.skelEmbedding(x_flat)  # (T, B, D)

        obs_tf = cm.permute(1, 0).float()      # (T, B) 1.0=observed
        unobs_tf = (~cm).permute(1, 0).float()  # (T, B) 1.0=unobserved
        h = h * obs_tf.unsqueeze(-1) + unobs_tf.unsqueeze(-1) * self.mask_token_temporal

        xseq = torch.cat(
            (enc.muQuery[y][None], enc.sigmaQuery[y][None], h), dim=0,
        )
        xseq = enc.sequence_pos_encoder(xseq)
        mu_sigma_mask = torch.ones((bs, 2), dtype=torch.bool, device=h.device)
        maskseq = torch.cat((mu_sigma_mask, mask), dim=1)

        final = enc.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1]
        frame_tokens = final[2:]

        return {
            "mu_masked": mu,                    # (B, D)
            "logvar_masked": logvar,            # (B, D)
            "frame_tokens_masked": frame_tokens,  # (T, B, D)
        }


# ---------------------------------------------------------------------------
# VaeacActorMotion — the full VAEAC model
# ---------------------------------------------------------------------------

class VaeacActorMotion(nn.Module):
    """VAEAC on the ACTOR backbone with global z.

    The training path replicates ACTOR exactly:
        Encoder_TRANSFORMER → reparameterize(mu, logvar) → Decoder_TRANSFORMER
        with z as the sole cross-attention memory token (use_frame_tokens=False,
        matching the original ACTOR paper).

    The only addition is the masked encoder r_ψ, trained by forward KL to
    match q_φ.  At inference, r_ψ replaces q_φ.

    Gradient routing (Ivanov 2019 §3):
        q_φ: reconstruction + velocity + KL(q_φ ‖ N(0,1)).
        r_ψ: KL(q_φ ‖ r_ψ) + prior regularisation.
        p_θ: reconstruction + velocity.
    """

    def __init__(
        self,
        full_encoder: VaeacActorFullEncoder,
        masked_encoder: VaeacActorMaskedEncoder,
        decoder: Decoder_TRANSFORMER,
        latent_dim: int,
        njoints: int,
        nfeats: int,
        device: torch.device,
        pose_rep: str = "xyz",
        num_classes: int = 3,
        use_frame_tokens: bool = False,
        rotation2xyz: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.full_encoder = full_encoder
        self.masked_encoder = masked_encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.pose_rep = pose_rep
        self.num_classes = num_classes
        self.device = device
        self.use_frame_tokens = use_frame_tokens
        self.rotation2xyz = rotation2xyz

        if pose_rep == "rot6d" and njoints == 32:
            self.register_buffer(
                "vel_joint_weights", h36m_vel_joint_weights(32),
            )
        else:
            self.vel_joint_weights = None

        self.losses = ["rc", "vel", "kl", "kl_prior", "reg",
                       "rcxyz", "velxyz", "mixed"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return torch.randn_like(std) * std + mu

    def _coalition_to_xmask(
        self, coalition_mask: torch.Tensor, x_shape: torch.Size,
    ) -> torch.Tensor:
        B, J, F, T = x_shape
        if coalition_mask.shape[-1] == J:
            return coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
        return coalition_mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict, phase: str = "train") -> dict:
        """VAEAC forward pass.

        Training:  q_φ → z → decoder.  r_ψ runs for KL.
        Inference: r_ψ → z → decoder.  q_φ not called.
        """
        y = batch["y"]
        if y.max() >= self.num_classes or y.min() < 0:
            raise ValueError(
                f"Label y has values in [{int(y.min())}, {int(y.max())}] "
                f"but num_classes={self.num_classes}. "
                f"Encoder muQuery/sigmaQuery and decoder actionBiases have "
                f"only {self.num_classes} rows — labels must be in "
                f"[0, {self.num_classes - 1}]."
            )
        if self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        elif self.rotation2xyz is not None:
            batch["x_xyz"] = self.rotation2xyz(batch["x"], batch["mask"])

        coalition_mask = batch.get("coalition_mask")

        if phase == "infer":
            masked_out = self.masked_encoder(batch)
            mu = masked_out["mu_masked"]
            logvar = masked_out["logvar_masked"]
            batch["mu_masked"] = mu
            batch["logvar_masked"] = logvar
            batch["z"] = self.reparameterize(mu, logvar)
            if self.use_frame_tokens:
                batch["frame_tokens"] = masked_out["frame_tokens_masked"]

        else:
            full_out = self.full_encoder(batch)
            batch["mu_full"] = full_out["mu_full"]
            batch["logvar_full"] = full_out["logvar_full"]
            batch["z"] = self.reparameterize(
                full_out["mu_full"], full_out["logvar_full"],
            )
            if self.use_frame_tokens:
                batch["frame_tokens"] = full_out["frame_tokens"]

            if coalition_mask is not None:
                masked_out = self.masked_encoder(batch)
                batch["mu_masked"] = masked_out["mu_masked"]
                batch["logvar_masked"] = masked_out["logvar_masked"]

        batch.update(self.decoder(batch))

        if self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        elif self.rotation2xyz is not None:
            batch["output_xyz"] = self.rotation2xyz(batch["output"], batch["mask"])

        return batch

    # ------------------------------------------------------------------
    # VAEAC ELBO
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: dict,
        lambda_kl: float = 1.0,
        lambda_reg: float = 1e-6,
        lambda_vel: float = 5.0,
        lambda_kl_prior: float = 1e-5,
        lambda_rcxyz: float = 0.0,
        lambda_velxyz: float = 0.0,
        prior_sigma_mu: float = 1e4,
        prior_sigma_sigma: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """ACTOR-style reconstruction + forward KL(q_φ ‖ r_ψ).

        Losses:
            rc       — MSE on raw representation (rotations or XYZ).
            vel      — velocity MSE on raw representation.
            rcxyz    — MSE on FK-derived XYZ positions (rot6d only).
            velxyz   — velocity MSE on FK-derived XYZ (rot6d only).
            kl       — forward KL on the global (B, D) latent.  q_φ detached.
            kl_prior — KL(q_φ ‖ N(0,1)), sum reduction, matching ACTOR.
            reg      — Ivanov prior regulariser on r_ψ.
        """
        x = batch["x"]
        output = batch["output"]
        mask = batch["mask"]
        B, J, Ft, T = x.shape

        real_frames = mask.unsqueeze(1).unsqueeze(1).expand(B, J, Ft, T)
        n_real = real_frames.float().sum().clamp(min=1.0)

        # ---- Full-sequence reconstruction (raw representation) ----
        rc = ((output - x).pow(2) * real_frames.float()).sum() / n_real

        # ---- Full-sequence velocity (raw representation) ----
        vel = x.new_zeros(())
        if T > 1:
            gt_vel = x[..., 1:] - x[..., :-1]
            out_vel = output[..., 1:] - output[..., :-1]
            vel_real = real_frames[..., 1:] & real_frames[..., :-1]
            n_vel = vel_real.float().sum().clamp(min=1.0)
            diff2 = (out_vel - gt_vel).pow(2)
            if self.vel_joint_weights is not None:
                diff2 = diff2 * self.vel_joint_weights.view(1, J, 1, 1)
            vel = (diff2 * vel_real.float()).sum() / n_vel

        # ---- FK XYZ losses (rot6d only) ----
        rcxyz = x.new_zeros(())
        velxyz = x.new_zeros(())
        if lambda_rcxyz > 0 or lambda_velxyz > 0:
            x_xyz = batch.get("x_xyz")
            out_xyz = batch.get("output_xyz")
            if x_xyz is not None and out_xyz is not None:
                rcxyz_perm = x_xyz.permute(0, 3, 1, 2)  # (B, T, J', 3)
                out_perm = out_xyz.permute(0, 3, 1, 2)
                rcxyz = F.mse_loss(out_perm[mask], rcxyz_perm[mask], reduction="mean")

                if T > 1 and lambda_velxyz > 0:
                    gt_v = x_xyz[..., 1:] - x_xyz[..., :-1]
                    out_v = out_xyz[..., 1:] - out_xyz[..., :-1]
                    mask_v = mask[..., 1:]
                    gt_v_p = gt_v.permute(0, 3, 1, 2)
                    out_v_p = out_v.permute(0, 3, 1, 2)
                    velxyz = F.mse_loss(out_v_p[mask_v], gt_v_p[mask_v], reduction="mean")

        # ---- Forward KL: KL(q_φ ‖ r_ψ) — global latent (B, D) ----
        kl = x.new_zeros(())
        if "mu_masked" in batch and "logvar_masked" in batch:
            mu_phi = batch["mu_full"].detach()       # (B, D)
            lv_phi = batch["logvar_full"].detach()    # (B, D)
            mu_psi = batch["mu_masked"]               # (B, D)
            lv_psi = batch["logvar_masked"]           # (B, D)
            var_phi = lv_phi.exp()
            var_psi = lv_psi.exp()
            kl_cell = 0.5 * (
                lv_psi - lv_phi
                + (var_phi + (mu_phi - mu_psi).pow(2)) / var_psi.clamp(min=1e-8)
                - 1.0
            )  # (B, D)
            kl = kl_cell.sum(dim=-1).mean()  # sum over D, mean over B

        # ---- KL prior on q_φ: KL(q_φ ‖ N(0,1)) — matches ACTOR ----
        kl_prior = x.new_zeros(())
        if "mu_full" in batch and "logvar_full" in batch:
            mu_phi = batch["mu_full"]
            lv_phi = batch["logvar_full"]
            kl_prior = -0.5 * torch.sum(1 + lv_phi - mu_phi.pow(2) - lv_phi.exp())

        # ---- Prior regularisation on r_ψ (Ivanov 2019 Eq.8) ----
        reg = x.new_zeros(())
        if "mu_masked" in batch:
            mu_psi = batch["mu_masked"]
            lv_psi = batch["logvar_masked"]
            reg_mu = mu_psi.pow(2).sum(dim=-1).mean() / (2.0 * prior_sigma_mu ** 2)
            sigma_psi = (0.5 * lv_psi).exp()
            reg_sigma = (lv_psi * 0.5 - sigma_psi).sum(dim=-1).mean() * prior_sigma_sigma
            reg = reg_mu - reg_sigma

        loss = (rc + lambda_vel * vel + lambda_kl * kl
                + lambda_kl_prior * kl_prior + lambda_reg * reg
                + lambda_rcxyz * rcxyz + lambda_velxyz * velxyz)
        return loss, {
            "rc":       float(rc.detach()),
            "vel":      float(vel.detach()),
            "kl":       float(kl.detach()),
            "kl_prior": float(kl_prior.detach()),
            "reg":      float(reg.detach()),
            "rcxyz":    float(rcxyz.detach()),
            "velxyz":   float(velxyz.detach()),
            "mixed":    float(loss.detach()),
        }

    # ------------------------------------------------------------------
    # Inference: sample completions for KernelSHAP
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_completions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        coalition_mask: torch.Tensor,
        n_samples: int = 20,
        paste_observed: bool = True,
    ) -> list[torch.Tensor]:
        """Draw n_samples manifold-constrained completions via r_ψ + p_θ.

        At inference only r_ψ is called (matching VAEAC deployment).
        Each sample re-draws z from r_ψ's distribution.
        """
        b = {
            "x": x, "y": y, "mask": mask,
            "lengths": lengths, "coalition_mask": coalition_mask,
        }
        masked_out = self.masked_encoder(b)
        mu = masked_out["mu_masked"]
        logvar = masked_out["logvar_masked"]
        std = (0.5 * logvar).exp()

        obs_xmask = self._coalition_to_xmask(coalition_mask, x.shape)

        completions: list[torch.Tensor] = []
        for _ in range(n_samples):
            z = torch.randn_like(std) * std + mu
            dec_b = dict(b)
            dec_b["z"] = z
            if self.use_frame_tokens:
                dec_b["frame_tokens"] = masked_out["frame_tokens_masked"]
            dec_b.update(self.decoder(dec_b))
            x_hat = dec_b["output"]
            if paste_observed:
                x_hat = x_hat.clone()
                x_hat[obs_xmask] = x[obs_xmask]
            completions.append(x_hat)
        return completions


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dev = torch.device("cpu")
    B, J, F, T = 2, 17, 3, 16

    common = dict(
        modeltype="cvae",
        njoints=J, nfeats=F,
        num_frames=0, num_classes=3,
        translation=True, pose_rep="xyz",
        glob=True, glob_rot=[3.14, 0, 0],
        latent_dim=16, ff_size=64,
        num_layers=2, num_heads=2,
        dropout=0.0, ablation=None, activation="gelu",
    )

    x = torch.randn(B, J, F, T)
    y = torch.zeros(B, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.bool)
    lengths = torch.full((B,), T, dtype=torch.long)
    cm_sp = torch.ones(B, J, dtype=torch.bool)
    cm_sp[:, 3:6] = False
    cm_t = torch.ones(B, T, dtype=torch.bool)
    cm_t[:, :T // 4] = False

    def _build_model(use_ft=False):
        full_enc = VaeacActorFullEncoder(**common)
        masked_enc = VaeacActorMaskedEncoder(**common)
        dec = Decoder_TRANSFORMER(**common)
        return VaeacActorMotion(full_enc, masked_enc, dec,
                                latent_dim=16, njoints=J, nfeats=F, device=dev,
                                use_frame_tokens=use_ft)

    def _smoke(model, tag):
        model.train()
        batch = {"x": x, "y": y, "mask": mask, "lengths": lengths,
                 "coalition_mask": cm_sp}
        out = model(dict(batch), phase="train")
        assert out["output"].shape == (B, J, F, T), f"{tag}: bad output shape"
        assert "mu_full" in out and "mu_masked" in out, f"{tag}: missing encoder keys"
        assert out["mu_full"].shape == (B, 16), f"{tag}: mu_full wrong shape"
        assert out["mu_masked"].shape == (B, 16), f"{tag}: mu_masked wrong shape"

        loss, ld = model.compute_loss(out, lambda_kl=1.0)
        assert loss.isfinite(), f"{tag}: loss not finite"
        assert ld["kl"] >= 0, f"{tag}: KL negative"

        # temporal coalition mask
        batch_t = {"x": x, "y": y, "mask": mask, "lengths": lengths,
                   "coalition_mask": cm_t}
        out_t = model(dict(batch_t), phase="train")
        assert model.compute_loss(out_t, lambda_kl=1.0)[0].isfinite(), \
            f"{tag}: temporal loss not finite"

        # no coalition mask (pure ACTOR mode)
        batch_none = {"x": x, "y": y, "mask": mask, "lengths": lengths}
        out_none = model(dict(batch_none), phase="train")
        assert "mu_full" in out_none and "mu_masked" not in out_none, \
            f"{tag}: no-mask mode should skip masked encoder"
        loss_none, _ = model.compute_loss(out_none, lambda_kl=1.0)
        assert loss_none.isfinite(), f"{tag}: no-mask loss not finite"

        # inference phase
        model.eval()
        out_i = model(dict(batch), phase="infer")
        assert "mu_masked" in out_i and "mu_full" not in out_i, \
            f"{tag}: infer keys wrong"

        # sample_completions: spatial mask
        comps = model.sample_completions(
            x[:1], y[:1], mask[:1], lengths[:1], cm_sp[:1], n_samples=3,
        )
        assert len(comps) == 3 and comps[0].shape == (1, J, F, T), \
            f"{tag}: completions shape wrong"
        obs_joints = cm_sp[0]
        for c in comps:
            for j in range(J):
                if obs_joints[j]:
                    assert torch.allclose(c[0, j], x[0, j]), \
                        f"{tag}: observed joint {j} not pasted back"

        # sample_completions: temporal mask
        comps_t = model.sample_completions(
            x[:1], y[:1], mask[:1], lengths[:1], cm_t[:1], n_samples=2,
        )
        assert len(comps_t) == 2 and comps_t[0].shape == (1, J, F, T), \
            f"{tag}: temporal completions shape wrong"

        print(f"  {tag}: OK")

    print("--- VaeacActorMotion smoke tests ---")
    _smoke(_build_model(use_ft=False), "z-only (ACTOR default)")
    _smoke(_build_model(use_ft=True), "with frame_tokens")
    print("All smoke tests passed.")
