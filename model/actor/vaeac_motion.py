"""
vaeac_motion.py — VAEAC for manifold-constrained motion imputation (SHAP).

Implements the Variational Autoencoder with Arbitrary Conditioning (Ivanov
et al. ICLR 2019; Olsen et al. JMLR 2022) adapted for temporal motion data
on the ACTOR Transformer backbone.

Three networks
--------------
q_ϕ(z|x,S)     Full encoder.  Sees the COMPLETE sequence x plus coalition
                mask S (unobserved joints get an additive target marker).
                Used ONLY during training to provide the KL target.

r_ψ(z|x_S,S)   Masked encoder (prior network).  Sees only observed joints
                (mask tokens for unobserved).  Used at inference.

p_θ(x̂|z,x_S,S) Decoder conditioned on z AND observed features x_S.
                This is the key difference from previous implementations:
                instead of z-only or encoder-frame-tokens, the decoder cross-
                attends to the raw observed-feature embeddings.  These are
                identical during training and inference, eliminating the
                train/infer mismatch that caused mean-pose collapse.

Why the decoder conditions on x_S
----------------------------------
Original VAEAC (Ivanov 2019, §4.2; code: skip connections from prior→decoder)
explicitly models p_θ(x_{S̄}|z, x_{1-b}, b) — the decoder sees z AND x_S.
Original ACTOR's decoder sees only z (single memory token), forcing z to
encode the entire 81-frame sequence — an impossible bottleneck for imputation.

Our fix: pass observed-joint embeddings as additional decoder memory, giving
the decoder direct per-frame access to x_S.  z now only needs to encode the
dynamics of the UNOBSERVED joints, which is a much easier task.

VAEAC ELBO (Ivanov 2019, Eq.6; Olsen 2022, Eq.6)
--------------------------------------------------
L = E_{z~q_ϕ} [log p_θ(x_{S̄}|z, x_S, S)]
    − KL(q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S))
    − μ²_ψ / (2σ²_μ)              [normal prior on μ_ψ]
    + (1/σ_σ)(log σ_ψ − σ_ψ)     [gamma prior on σ_ψ]

Gradient routing:
  ∂L/∂θ (decoder):        reconstruction + velocity.
  ∂L/∂ϕ (full encoder):   reconstruction via reparameterization of z.
                           Detached from KL so KL doesn't collapse q_ϕ.
  ∂L/∂ψ (masked encoder): KL + prior regularization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.actor.transformer_arch import Decoder_TRANSFORMER, Encoder_TRANSFORMER


# ---------------------------------------------------------------------------
# Full encoder q_ϕ(z | x, S)
# ---------------------------------------------------------------------------

class VaeacFullEncoder(nn.Module):
    """Full encoder (proposal) q_ϕ(z|x,S).

    Sees the COMPLETE sequence.  Coalition-awareness: an additive learnable
    ``target_marker`` is injected into the input features of unobserved
    (held-out) joints.  This makes μ_ϕ depend on which joints are in S,
    providing coalition-specific z targets for r_ψ.

    Zero-init so q_ϕ == plain CVAE encoder at training start.
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = Encoder_TRANSFORMER(**encoder_kwargs)
        njoints = encoder_kwargs["njoints"]
        nfeats = encoder_kwargs["nfeats"]
        self.target_marker = nn.Parameter(torch.zeros(njoints, nfeats))

    def forward(self, batch: dict) -> dict:
        x = batch["x"].clone()
        cm = batch.get("coalition_mask")

        if cm is not None and cm.shape[-1] == x.shape[1]:
            unobs = ~cm
            B, J, F, T = x.shape
            for j in range(J):
                x[unobs[:, j], j, :, :] += self.target_marker[j, :].unsqueeze(-1)

        out = self._encoder({**batch, "x": x})
        return {"mu_full": out["mu"], "logvar_full": out["logvar"]}


# ---------------------------------------------------------------------------
# Masked encoder r_ψ(z | x_S, S)
# ---------------------------------------------------------------------------

class VaeacMaskedEncoder(nn.Module):
    """Masked encoder (prior) r_ψ(z|x_S,S).

    Unobserved joints REPLACED by learnable mask tokens so the Transformer
    sees only a "this joint is missing" signal.  Self-attention propagates
    observed-joint information to all frame positions.
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = Encoder_TRANSFORMER(**encoder_kwargs)
        njoints = encoder_kwargs["njoints"]
        nfeats = encoder_kwargs["nfeats"]
        latent_dim = encoder_kwargs["latent_dim"]
        self.mask_token_spatial = nn.Parameter(torch.zeros(njoints, nfeats))
        self.mask_token_temporal = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, batch: dict) -> dict:
        cm = batch["coalition_mask"]
        is_spatial = cm.shape[-1] == batch["x"].shape[1]
        if is_spatial:
            return self._encode_spatial(batch)
        return self._encode_temporal(batch)

    def _encode_spatial(self, batch: dict) -> dict:
        x = batch["x"].clone()
        cm = batch["coalition_mask"]
        unobs = ~cm
        B, J, F, T = x.shape
        for j in range(J):
            x[unobs[:, j], j, :, :] = self.mask_token_spatial[j, :].unsqueeze(-1)
        out = self._encoder({**batch, "x": x})
        return {"mu_masked": out["mu"], "logvar_masked": out["logvar"]}

    def _encode_temporal(self, batch: dict) -> dict:
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        cm = batch["coalition_mask"]
        B, J, F, T = x.shape
        enc = self._encoder

        x_t = x.permute(3, 0, 1, 2).reshape(T, B, J * F)
        x_emb = enc.skelEmbedding(x_t)
        unobs_tf = ~cm.permute(1, 0)
        obs_tf = cm.permute(1, 0)
        x_emb = (
            x_emb * obs_tf.unsqueeze(-1).float()
            + unobs_tf.unsqueeze(-1).float() * self.mask_token_temporal
        )

        xseq = torch.cat(
            (enc.muQuery[y][None], enc.sigmaQuery[y][None], x_emb), dim=0,
        )
        xseq = enc.sequence_pos_encoder(xseq)
        muandsigmaMask = torch.ones((B, 2), dtype=torch.bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), dim=1)
        final = enc.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        return {"mu_masked": final[0], "logvar_masked": final[1]}


# ---------------------------------------------------------------------------
# VaeacMotion — the full model
# ---------------------------------------------------------------------------

class VaeacMotion(nn.Module):
    """VAEAC for manifold-constrained motion imputation.

    The decoder cross-attends to [z, observed_emb] where observed_emb is the
    raw observed-feature embedding (NOT encoder representations).  This means
    the decoder's conditioning is identical during training and inference.

    Args:
        full_encoder:    VaeacFullEncoder instance (q_ϕ).
        masked_encoder:  VaeacMaskedEncoder instance (r_ψ).
        decoder:         Decoder_TRANSFORMER instance (p_θ).
        latent_dim:      Latent dimensionality.
        njoints:         Number of joints.
        nfeats:          Features per joint.
        device:          Torch device.
        pose_rep:        ``"xyz"`` or ``"rot6d"``.
        num_classes:     Number of class labels.
    """

    def __init__(
        self,
        full_encoder: VaeacFullEncoder,
        masked_encoder: VaeacMaskedEncoder,
        decoder: Decoder_TRANSFORMER,
        latent_dim: int,
        njoints: int,
        nfeats: int,
        device: torch.device,
        pose_rep: str = "xyz",
        num_classes: int = 3,
        use_obs_indicator: bool = True,
        obs_emb_drop: float = 0.0,
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
        self.use_obs_indicator = use_obs_indicator
        self.obs_emb_drop = obs_emb_drop

        # Project raw observed features (+ optional binary coalition indicator)
        # to decoder latent space.
        #
        # use_obs_indicator=True  (default, new checkpoints):
        #   Input = [x_obs_flat | indicator], shape (J*F + J,).
        #   Appending the indicator removes the ambiguity between
        #   "joint coordinates happen to be zero" and "this joint is masked".
        #   Follows Ivanov et al. (2019) §4.2: "concatenate [x◦(1−b), b]".
        #
        # use_obs_indicator=False (legacy checkpoints trained before the
        #   indicator was added):
        #   Input = x_obs_flat only, shape (J*F,).
        obs_proj_in = njoints * nfeats + (njoints if use_obs_indicator else 0)
        self.observed_projection = nn.Linear(obs_proj_in, latent_dim)

        self.losses = ["rc", "vel", "kl", "reg", "mixed"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observed_input(
        self, x: torch.Tensor, coalition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Zero out unobserved features in x (VAEAC-style observed input).

        For spatial masks (B, J): unobserved joints → zero.
        For temporal masks (B, T): unobserved frames → zero.
        """
        x_obs = x.clone()
        B, J, F, T = x.shape
        if coalition_mask.shape[-1] == J:
            unobs = ~coalition_mask  # (B, J)
            for j in range(J):
                x_obs[unobs[:, j], j, :, :] = 0.0
        else:
            unobs = ~coalition_mask  # (B, T)
            for t in range(T):
                x_obs[unobs[:, t], :, :, t] = 0.0
        return x_obs

    def _make_observed_memory(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        coalition_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project observed features + binary indicator to decoder latent space.

        Builds the (T, B, J*F + J) input:
          - x_obs_flat  (J*F): raw joint features, zeroed at unobserved positions.
          - indicator   (J):   1.0 = observed, 0.0 = held out.

        For spatial masks (B, J):  indicator[t, b, j] = coalition_mask[b, j]
                                   — same per-joint signal at every frame.
        For temporal masks (B, T): indicator[t, b, j] = coalition_mask[b, t]
                                   — all J joints share a single 0/1 flag that
                                   says whether frame t is in the coalition.
        When coalition_mask is None (no masking): indicator is all ones.

        Returns:
            obs_emb:  (T, B, D) — per-frame observed-feature embeddings.
            obs_mask: (B, T) bool — True where frame is real (padding mask).
        """
        B, J, F, T = x_obs.shape
        x_flat = x_obs.permute(3, 0, 1, 2).reshape(T, B, J * F)  # (T, B, J*F)

        if not self.use_obs_indicator:
            obs_emb = self.observed_projection(x_flat)   # (T, B, D)  — legacy path
            return obs_emb, mask

        if coalition_mask is None:
            indicator = torch.ones(T, B, J, device=x_obs.device)
        elif coalition_mask.shape[-1] == J:
            # Spatial (B, J) → (T, B, J): same joint indicators at every frame.
            indicator = coalition_mask.float().unsqueeze(0).expand(T, B, J)
        else:
            # Temporal (B, T) → (T, B, J): per-frame scalar broadcast to all joints.
            indicator = (
                coalition_mask.float()           # (B, T)
                .permute(1, 0)                   # (T, B)
                .unsqueeze(-1)                   # (T, B, 1)
                .expand(T, B, J)                 # (T, B, J)
            )

        x_with_ind = torch.cat([x_flat, indicator], dim=-1)  # (T, B, J*F + J)
        obs_emb = self.observed_projection(x_with_ind)        # (T, B, D)
        return obs_emb, mask

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return torch.randn_like(std) * std + mu

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict, phase: str = "train") -> dict:
        """VAEAC forward pass.

        phase="train":
            q_ϕ → z ~ q_ϕ → decoder(z, x_S).  Also runs r_ψ for KL.
        phase="infer":
            r_ψ → z ~ r_ψ → decoder(z, x_S).  q_ϕ not called.
        """
        if self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]

        x = batch["x"]
        coalition_mask = batch.get("coalition_mask")
        B, J, F, T = x.shape

        # Observed features: x with zeros at masked positions.
        if coalition_mask is not None:
            x_obs = self._make_observed_input(x, coalition_mask)
        else:
            x_obs = x
        obs_emb, obs_mask = self._make_observed_memory(x_obs, batch["mask"], coalition_mask)

        if phase == "infer":
            masked_out = self.masked_encoder(batch)
            mu = masked_out["mu_masked"]
            logvar = masked_out["logvar_masked"]
            batch["mu_masked"] = mu
            batch["logvar_masked"] = logvar
        else:
            # q_ϕ: coalition-aware full encoder
            full_out = self.full_encoder(batch)
            batch["mu_full"] = full_out["mu_full"]
            batch["logvar_full"] = full_out["logvar_full"]
            mu = full_out["mu_full"]
            logvar = full_out["logvar_full"]

            # r_ψ: masked encoder (for KL)
            if coalition_mask is not None:
                masked_out = self.masked_encoder(batch)
                batch["mu_masked"] = masked_out["mu_masked"]
                batch["logvar_masked"] = masked_out["logvar_masked"]

        # Sample z and decode.
        z = self.reparameterize(mu, logvar)
        batch["z"] = z
        batch["mu"] = mu
        batch["logvar"] = logvar

        # Provide observed-feature embeddings as "frame_tokens" so the
        # existing Decoder_TRANSFORMER picks them up as additional memory.
        # During training, randomly drop entire frame-token time-steps to
        # weaken the decoder's per-frame shortcut through obs_emb, forcing
        # it to rely on z for information about unobserved joints.
        if self.training and self.obs_emb_drop > 0.0:
            T_obs = obs_emb.shape[0]
            B_obs = obs_emb.shape[1]
            keep = torch.rand(T_obs, B_obs, 1, device=obs_emb.device) >= self.obs_emb_drop
            obs_emb = obs_emb * keep.float()
        batch["frame_tokens"] = obs_emb

        batch.update(self.decoder(batch))

        if self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]

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
        lambda_rc_obs: float = 0.1,
        prior_sigma_mu: float = 1e4,
        prior_sigma_sigma: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """VAEAC ELBO (Ivanov 2019 Eq.6, Olsen 2022 Eq.6).

        L = E_{z~q_ϕ} [log p_θ(x_{S̄}|z, x_S, S)]
            − λ_kl · KL(q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S))
            − λ_reg · prior_reg(r_ψ)
            + λ_vel · velocity_loss(held-out)
            + λ_rc_obs · MSE(observed joints)

        Gradient routing:
          q_ϕ is DETACHED in KL → q_ϕ trained only by reconstruction.
          r_ψ trained only by KL + prior reg.
          Decoder trained by reconstruction + velocity.
        """
        x = batch["x"]
        output = batch["output"]
        mask = batch["mask"]
        coalition_mask = batch["coalition_mask"]
        B, J, F, T = x.shape

        # Build per-element masks.
        if coalition_mask.shape[-1] == J:
            held_out = ~coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
        else:
            held_out = ~coalition_mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)
        real_frames = mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)

        # ---- Reconstruction on HELD-OUT features (VAEAC likelihood) ----
        rc_mask = held_out & real_frames
        n_held = rc_mask.float().sum().clamp(min=1.0)
        rc = ((output - x).pow(2) * rc_mask.float()).sum() / n_held

        # ---- Optional reconstruction on observed features ----
        obs_mask = (~held_out) & real_frames
        n_obs = obs_mask.float().sum().clamp(min=1.0)
        rc_obs = ((output - x).pow(2) * obs_mask.float()).sum() / n_obs

        # ---- Velocity on held-out features ----
        vel = x.new_zeros(())
        if T > 1:
            gt_vel = x[..., 1:] - x[..., :-1]
            out_vel = output[..., 1:] - output[..., :-1]
            vel_mask = rc_mask[..., 1:] & rc_mask[..., :-1]
            n_vel = vel_mask.float().sum().clamp(min=1.0)
            vel = ((out_vel - gt_vel).pow(2) * vel_mask.float()).sum() / n_vel

        # ---- Forward KL: KL(q_ϕ ‖ r_ψ) ----
        # q_ϕ DETACHED: only r_ψ receives KL gradient (Ivanov 2019).
        kl = x.new_zeros(())
        if "mu_masked" in batch and "logvar_masked" in batch:
            mu_phi = batch["mu_full"].detach()
            lv_phi = batch["logvar_full"].detach()
            mu_psi = batch["mu_masked"]
            lv_psi = batch["logvar_masked"]
            var_phi = lv_phi.exp()
            var_psi = lv_psi.exp()
            kl = 0.5 * (
                lv_psi - lv_phi
                + (var_phi + (mu_phi - mu_psi).pow(2)) / var_psi.clamp(min=1e-8)
                - 1.0
            ).sum(dim=-1).mean()

        # ---- Prior regularization on r_ψ (Ivanov 2019 Eq.8) ----
        # prior_sigma_mu  controls normal prior on μ_ψ (larger = looser).
        # prior_sigma_sigma controls gamma prior on σ_ψ (larger = more
        #   freedom for σ_ψ to deviate from 1.0; Ivanov default 1e-4 is
        #   very tight; 1e-2 allows coalition-dependent variance).
        reg = x.new_zeros(())
        if "mu_masked" in batch:
            mu_psi = batch["mu_masked"]
            lv_psi = batch["logvar_masked"]
            reg_mu = mu_psi.pow(2).sum(dim=-1).mean() / (2.0 * prior_sigma_mu ** 2)
            sigma_psi = (0.5 * lv_psi).exp()
            reg_sigma = (lv_psi * 0.5 - sigma_psi).sum(dim=-1).mean() * prior_sigma_sigma
            reg = reg_mu - reg_sigma

        loss = rc + lambda_rc_obs * rc_obs + lambda_vel * vel + lambda_kl * kl + lambda_reg * reg
        return loss, {
            "rc": float(rc.detach()),
            "rc_obs": float(rc_obs.detach()),
            "vel": float(vel.detach()),
            "kl": float(kl.detach()),
            "reg": float(reg.detach()),
            "mixed": float(loss.detach()),
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

        At inference, only r_ψ (masked encoder) and p_θ (decoder conditioned
        on x_S) are used — q_ϕ is never called.  This matches VAEAC deployment
        (Olsen 2022, §3.2).

        Args:
            x:              (1, J, F, T) input sequence.
            y:              (1,) class label.
            mask:           (1, T) real-frame mask.
            lengths:        (1,) frame count.
            coalition_mask: (1, J) spatial or (1, T) temporal; True=observed.
            n_samples:      Number of stochastic completions.
            paste_observed: If True, overwrite observed positions with GT.

        Returns:
            List of n_samples tensors, each (1, J, F, T).
        """
        b = {
            "x": x, "y": y, "mask": mask,
            "lengths": lengths, "coalition_mask": coalition_mask,
        }

        # r_ψ encoding (once — reuse mu/logvar for all samples).
        masked_out = self.masked_encoder(b)
        mu = masked_out["mu_masked"]
        logvar = masked_out["logvar_masked"]
        std = (0.5 * logvar).exp()

        # Observed-feature memory (same for all samples).
        x_obs = self._make_observed_input(x, coalition_mask)
        obs_emb, _ = self._make_observed_memory(x_obs, mask, coalition_mask)

        obs_xmask = self._coalition_to_xmask(coalition_mask, x.shape)

        completions: list[torch.Tensor] = []
        for _ in range(n_samples):
            z = torch.randn_like(std) * std + mu
            dec_batch = {
                **b, "z": z, "frame_tokens": obs_emb,
            }
            x_hat = self.decoder(dec_batch)["output"]
            if paste_observed:
                x_hat[obs_xmask] = x[obs_xmask]
            completions.append(x_hat)
        return completions

    def _coalition_to_xmask(
        self, coalition_mask: torch.Tensor, x_shape: torch.Size,
    ) -> torch.Tensor:
        B, J, F, T = x_shape
        if coalition_mask.shape[-1] == J:
            return coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
        return coalition_mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_from_cvae_checkpoint(self, ckpt_path: str) -> None:
        """Load encoder and decoder weights from a trained ActorCVAE checkpoint.

        Initialises both q_ϕ._encoder and r_ψ._encoder from the CVAE encoder
        weights.  Decoder loaded directly.  target_marker and mask tokens stay
        at zero so epoch-0 is identical to the base CVAE.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        enc_sd = {
            k.removeprefix("model.encoder."): v
            for k, v in sd.items()
            if k.startswith("model.encoder.")
        }
        dec_sd = {
            k.removeprefix("model.decoder."): v
            for k, v in sd.items()
            if k.startswith("model.decoder.")
        }
        if not enc_sd:
            enc_sd = {
                k.removeprefix("encoder."): v
                for k, v in sd.items()
                if k.startswith("encoder.")
            }
            dec_sd = {
                k.removeprefix("decoder."): v
                for k, v in sd.items()
                if k.startswith("decoder.")
            }

        self.full_encoder._encoder.load_state_dict(enc_sd)
        self.masked_encoder._encoder.load_state_dict(enc_sd)
        self.decoder.load_state_dict(dec_sd)


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
        num_layers=1, num_heads=2,
        dropout=0.0, ablation=None, activation="gelu",
    )

    full_enc = VaeacFullEncoder(**common)
    masked_enc = VaeacMaskedEncoder(**common)
    dec = Decoder_TRANSFORMER(**common)

    model = VaeacMotion(
        full_enc, masked_enc, dec,
        latent_dim=16, njoints=J, nfeats=F, device=dev,
    )

    x = torch.randn(B, J, F, T)
    y = torch.zeros(B, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.bool)
    lengths = torch.full((B,), T, dtype=torch.long)

    # ---- Training forward (spatial mask) ----
    cm_sp = torch.ones(B, J, dtype=torch.bool)
    cm_sp[:, 3:6] = False
    batch = {"x": x, "y": y, "mask": mask, "lengths": lengths,
             "coalition_mask": cm_sp}
    out = model(dict(batch), phase="train")
    assert out["output"].shape == (B, J, F, T)
    assert "mu_full" in out and "mu_masked" in out
    loss, ld = model.compute_loss(out, lambda_kl=1.0)
    assert loss.isfinite(), f"Loss not finite: {loss}"
    assert ld["kl"] >= 0, f"KL must be non-negative: {ld['kl']}"

    # ---- Training forward (temporal mask) ----
    cm_t = torch.ones(B, T, dtype=torch.bool)
    cm_t[:, :T // 4] = False
    batch_t = {"x": x, "y": y, "mask": mask, "lengths": lengths,
               "coalition_mask": cm_t}
    out_t = model(dict(batch_t), phase="train")
    loss_t, _ = model.compute_loss(out_t, lambda_kl=1.0)
    assert loss_t.isfinite()

    # ---- Inference (r_ψ only) ----
    out_i = model(dict(batch), phase="infer")
    assert "mu_masked" in out_i
    assert "mu_full" not in out_i

    # ---- sample_completions ----
    comps = model.sample_completions(
        x[:1], y[:1], mask[:1], lengths[:1], cm_sp[:1], n_samples=3,
    )
    assert len(comps) == 3 and comps[0].shape == (1, J, F, T)

    # Verify observed joints are pasted back.
    obs_joints = cm_sp[0]  # (J,)
    for c in comps:
        for j in range(J):
            if obs_joints[j]:
                assert torch.allclose(c[0, j], x[0, j]), \
                    f"Observed joint {j} not pasted back"

    print("Smoke test OK")
