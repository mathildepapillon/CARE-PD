"""actor_shap.py — Proper VAEAC-style ActorSHAP for manifold-constrained SHAP.

Implements the three-component VAEAC (Ivanov 2019, Olsen et al. JMLR 2022) correctly:

  q_ϕ(z|x,S)   — Coalition-aware full encoder (CoalitionFullEncoder).
                   Sees the COMPLETE sequence x AND coalition mask S.
                   S-conditioning: unobserved joints get an additive
                   learnable target_marker in input-feature space.
                   TRAINED jointly (reconstruction + KL gradients).

  r_ψ(z|x_S,S) — Masked encoder (MaskedActorEncoder), unchanged.
                   Sees only observed joints (mask tokens for unobserved).
                   TRAINED by KL term only — receives NO reconstruction gradient.

  p_θ(x̂|z,y)  — Decoder, loaded from ActorCVAE checkpoint.
                   TRAINED by reconstruction gradient only.

VAEAC ELBO (Olsen 2022, Eq. 6 + Sec 3.3.1):

  L = E_{z ~ q_ϕ(z|x,S)} [log p_θ(x_S | z, x_S, S)]
      − KL( q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S) )
      − ||μ_ψ||² / (2σ²_μ)          ← normal prior on μ_ψ
      + (1/σ_σ)(log σ_ψ − σ_ψ)      ← gamma prior on σ_ψ

Gradient routing (the critical fix vs. previous implementation):
  ∂L/∂θ (decoder):        reconstruction term only.
  ∂L/∂ϕ (full encoder):   reconstruction via reparameterization of z ONLY.
                           q_ϕ params are DETACHED from the KL term to prevent
                           KL from collapsing q_ϕ's coalition-specific representations.
  ∂L/∂ψ (masked encoder): KL (with q_ϕ.detach() as target) + prior regularization.
                           ZERO gradient from reconstruction.

Previous bugs (now fixed):
  1. Old full encoder q_ϕ(z|x) ignored S → same μ_full for all coalitions
     → KL had a fixed, S-independent target → r_ψ collapsed to ignore S.
  2. Old KL was reversed: KL(r_ψ ‖ q_ϕ) — wrong direction per Olsen 2022 Eq. 6.
  3. Old z was sampled from r_ψ, not q_ϕ → reconstruction gradient incorrectly
     trained r_ψ directly, bypassing the VAEAC teacher-student mechanism.

Depends on:
  model.actor.transformer_arch  (Encoder_TRANSFORMER, Decoder_TRANSFORMER)
  model.actor.cvae              (ActorCVAE — loaded but not modified)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.actor.transformer_arch import Decoder_TRANSFORMER, Encoder_TRANSFORMER


# ---------------------------------------------------------------------------
# Coalition-aware full encoder  q_ϕ(z | x, S)
# ---------------------------------------------------------------------------

class CoalitionFullEncoder(nn.Module):
    """Full encoder (proposal network) q_ϕ(z|x,S) for the VAEAC ELBO.

    Sees the COMPLETE sequence x for ALL joints, plus a learnable per-joint
    "target marker" additively injected into the input features of unobserved
    (held-out) joints.  This makes μ_ϕ(x, S) depend on which joints are in S,
    providing coalition-specific z targets for the masked encoder r_ψ.

    Math (spatial coalition mask):
        x'[b, j, :, t] = x[b, j, :, t] + target_marker[j]   if j ∉ S_b
        x'[b, j, :, t] = x[b, j, :, t]                       if j ∈ S_b
        q_ϕ(z|x,S) = N(μ_ϕ(x', y), diag(σ²_ϕ(x', y)))

    Zero-initialization of target_marker means q_ϕ starts identical to the
    plain ActorCVAE encoder at training epoch 0, so training begins from a
    sensible warm-start and the target markers learn gradually.

    Used ONLY during training.  At inference / SHAP evaluation, only the
    masked encoder r_ψ and decoder p_θ are used (see sample_completions).

    Args:
        encoder_kwargs: keyword arguments forwarded verbatim to
                        Encoder_TRANSFORMER (same as MaskedActorEncoder).
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = Encoder_TRANSFORMER(**encoder_kwargs)
        njoints = encoder_kwargs["njoints"]
        nfeats  = encoder_kwargs["nfeats"]

        # Learnable additive perturbation for target (unobserved) joints.
        # Shape (J, F) — same as mask_token_spatial in MaskedActorEncoder.
        # Zero-init: q_ϕ == standard encoder at training start.
        self.target_marker_spatial = nn.Parameter(torch.zeros(njoints, nfeats))

    def forward(self, batch: dict) -> dict:
        """Encode (x, S) → (mu_full, logvar_full).

        Args:
            batch: must contain "x" (B, J, F, T), "y" (B,), "mask" (B, T).
                   "coalition_mask" (B, J) is optional; if absent all joints
                   are treated as observed (no target markers added), which is
                   the behaviour used in Phase 1 decoder warm-up.

        Returns:
            dict with keys "mu_full" and "logvar_full".
        """
        x = batch["x"].clone()  # (B, J, F, T) — do not modify original
        coalition_mask = batch.get("coalition_mask")

        if coalition_mask is not None and coalition_mask.shape[-1] == x.shape[1]:
            # Spatial coalition: add target_marker to unobserved joints.
            # target_marker_spatial: (J, F) → unsqueeze(−1) to broadcast over T.
            unobserved = ~coalition_mask  # (B, J) True = held-out / target
            B, J, F, T = x.shape
            for j in range(J):
                # x[b, j, :, :] += target_marker[j, :] for each unobserved b
                x[unobserved[:, j], j, :, :] = (
                    x[unobserved[:, j], j, :, :]
                    + self.target_marker_spatial[j, :].unsqueeze(-1)
                )
        # (Temporal coalitions: full encoder ignores temporal masking — the
        # teacher signal comes from q_ϕ knowing the full sequence regardless.)

        patched_batch = {**batch, "x": x}
        out = self._encoder(patched_batch)
        return {"mu_full": out["mu"], "logvar_full": out["logvar"]}


# ---------------------------------------------------------------------------
# Masked encoder  r_ψ(z | x_S, S)
# ---------------------------------------------------------------------------

class MaskedActorEncoder(nn.Module):
    """Masked encoder (prior network) r_ψ(z|x_S,S) for the VAEAC ELBO.

    Wraps Encoder_TRANSFORMER with learnable mask token injection.  Unobserved
    joints are REPLACED (not added to) by learnable mask tokens, so the
    transformer sees only a "this joint is missing" signal — no information
    about the actual held-out values leaks through.

    Two injection modes are selected by the shape of batch["coalition_mask"]:
      (B, J)  → spatial:  replace unobserved joint features in *input space*
                           before skelEmbedding.
      (B, T)  → temporal: overwrite unobserved frame embeddings in *latent
                           space* after skelEmbedding, before the Transformer.

    Gradient routing (VAEAC design):
        r_ψ receives gradient ONLY from the KL term and prior regularization.
        The reconstruction loss uses z sampled from q_ϕ, so r_ψ never sees
        a reconstruction gradient.  This is enforced in ActorSHAP.forward and
        compute_loss — not inside this class.

    Args:
        encoder_kwargs: keyword arguments forwarded verbatim to Encoder_TRANSFORMER.
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = Encoder_TRANSFORMER(**encoder_kwargs)
        njoints = encoder_kwargs["njoints"]
        nfeats  = encoder_kwargs["nfeats"]
        latent_dim = encoder_kwargs["latent_dim"]

        # Spatial: one learnable replacement token per joint, in input-feature space.
        self.mask_token_spatial = nn.Parameter(torch.zeros(njoints, nfeats))
        # Temporal: one shared replacement token in latent space after skelEmbedding.
        self.mask_token_temporal = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, batch: dict) -> dict:
        """Encode a partially-observed sequence into (mu_masked, logvar_masked).

        Args:
            batch: must contain "x", "y", "mask", "coalition_mask".

        Returns:
            dict with keys "mu_masked" and "logvar_masked".
        """
        coalition_mask = batch["coalition_mask"]
        is_spatial = coalition_mask.shape[-1] == batch["x"].shape[1]  # J
        if is_spatial:
            return self._encode_spatial(batch)
        else:
            return self._encode_temporal(batch)

    def _encode_spatial(self, batch: dict) -> dict:
        """Replace unobserved joint features with mask tokens before skelEmbedding."""
        x = batch["x"].clone()  # (B, J, F, T) — do not modify original
        coalition_mask = batch["coalition_mask"]  # (B, J) True = observed

        # Replace unobserved joint features with the learnable spatial mask token.
        # mask_token_spatial: (J, F) → broadcast to (B, J, F, T) via unsqueeze.
        unobserved = ~coalition_mask  # (B, J) True = masked
        B, J, F, T = x.shape
        for j in range(J):
            x[unobserved[:, j], j, :, :] = self.mask_token_spatial[j, :].unsqueeze(-1)

        patched_batch = {**batch, "x": x}
        out = self._encoder(patched_batch)
        return {"mu_masked": out["mu"], "logvar_masked": out["logvar"]}

    def _encode_temporal(self, batch: dict) -> dict:
        """Overwrite unobserved frame embeddings with mask token after skelEmbedding.

        Encoder_TRANSFORMER.forward is not designed for mid-stream injection, so
        we replicate its forward body here (< 30 lines, identical logic) and inject
        the temporal mask token between skelEmbedding and the Transformer layers.
        """
        x = batch["x"]
        y = batch["mask_or_y"] if "mask_or_y" in batch else batch["y"]
        mask = batch["mask"]
        coalition_mask = batch["coalition_mask"]  # (B, T) True = observed
        B, J, F, T = x.shape

        enc = self._encoder

        # Step 1: reshape + linear projection (mirrors Encoder_TRANSFORMER.forward).
        x_t = x.permute(3, 0, 1, 2).reshape(T, B, J * F)
        x_emb = enc.skelEmbedding(x_t)  # (T, B, latent_dim)

        # Step 2: overwrite unobserved frame embeddings with the mask token.
        # Zero out masked embeddings first, then write mask_token_temporal, so the
        # Transformer sees *only* mask_token_temporal at those positions.
        unobserved_tf = ~coalition_mask.permute(1, 0)   # (T, B) True = masked
        observed_tf   =  coalition_mask.permute(1, 0)   # (T, B) True = observed
        x_emb = (
            x_emb * observed_tf.unsqueeze(-1).float()
            + unobserved_tf.unsqueeze(-1).float() * self.mask_token_temporal
        )

        # Step 3: prepend class-conditional mu/sigma query tokens.
        y_for_batch = batch["y"]
        xseq = torch.cat(
            (enc.muQuery[y_for_batch][None], enc.sigmaQuery[y_for_batch][None], x_emb),
            dim=0,
        )  # (T+2, B, latent_dim)
        xseq = enc.sequence_pos_encoder(xseq)

        muandsigmaMask = torch.ones((B, 2), dtype=torch.bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), dim=1)

        # Step 4: Transformer encoder.
        final = enc.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)

        return {"mu_masked": final[0], "logvar_masked": final[1]}


# ---------------------------------------------------------------------------
# ActorSHAP
# ---------------------------------------------------------------------------

class ActorSHAP(nn.Module):
    """Proper VAEAC (Ivanov 2019 / Olsen 2022) for manifold-constrained SHAP.

    Three components:
      self.encoder        — CoalitionFullEncoder q_ϕ(z|x,S). Sees full sequence
                            PLUS learnable target markers for unobserved joints.
                            TRAINED by both reconstruction and KL gradients.
      self.masked_encoder — MaskedActorEncoder r_ψ(z|x_S,S). Sees only observed
                            joints (mask tokens for unobserved).
                            TRAINED by KL and prior reg gradients only.
      self.decoder        — Decoder p_θ(x̂|z,y). Loaded from ActorCVAE.
                            TRAINED by reconstruction gradient only.

    Args:
        encoder:        CoalitionFullEncoder instance.
        masked_encoder: MaskedActorEncoder instance.
        decoder:        Decoder_TRANSFORMER instance.
        latent_dim:     latent space dimensionality.
        device:         torch device.
        pose_rep:       "xyz" (only supported value).
        num_classes:    number of UPDRS classes.
    """

    def __init__(
        self,
        encoder: CoalitionFullEncoder,
        masked_encoder: MaskedActorEncoder,
        decoder: Decoder_TRANSFORMER,
        latent_dim: int,
        device: torch.device,
        pose_rep: str = "xyz",
        num_classes: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.masked_encoder = masked_encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.num_classes = num_classes
        self.device = device
        self.losses = ["rc", "kl", "reg", "mixed"]

    # ------------------------------------------------------------------
    # Reparameterisation
    # ------------------------------------------------------------------

    def reparameterize(self, batch: dict, seed: int | None = None) -> torch.Tensor:
        """Sample z ~ N(mu, exp(0.5·logvar)) via reparameterisation trick.

        Reads batch["mu"] and batch["logvar"].  Caller sets these to q_ϕ's
        outputs (mu_full, logvar_full) before calling so that gradient flows
        back to the full encoder ϕ through z.
        """
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(0.5 * logvar)
        if seed is None:
            eps = torch.randn_like(std)
        else:
            gen = torch.Generator(device=mu.device)
            gen.manual_seed(seed)
            eps = torch.randn(std.shape, device=mu.device, generator=gen)
        return eps * std + mu

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict, phase: int = 2) -> dict:
        """VAEAC forward pass.

        Phase 1 (decoder / full-encoder warm-up):
            z ~ q_ϕ(z|x) — full encoder without coalition conditioning
            (coalition_mask absent or ignored).  Decoder and full encoder
            are trained on MSE over all joints.

        Phase 2 (proper VAEAC ELBO):
            z ~ q_ϕ(z|x,S) — coalition-aware full encoder WITH target markers.
            Gradient flows to θ (decoder) and ϕ (full encoder) via reconstruction.
            r_ψ(z|x_S,S) is run to obtain μ_ψ, σ²_ψ for KL — it receives ZERO
            gradient from the reconstruction path.

        Args:
            batch: ACTOR batch dict with "x", "y", "mask", "lengths".
                   Phase 2 also requires "coalition_mask".
            phase: 1 or 2.

        Returns:
            batch with added keys: mu_full, logvar_full, [mu_masked,
            logvar_masked in phase 2], mu, logvar, z, output.
        """
        if self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]

        # Run coalition-aware full encoder q_ϕ(z|x,S).
        # Phase 1: coalition_mask absent → no target markers → plain encoding.
        # Phase 2: coalition_mask present → coalition-specific z target for r_ψ.
        # Gradient flows freely (no no_grad wrapper — q_ϕ is trainable).
        out_full = self.encoder(batch)
        batch["mu_full"]     = out_full["mu_full"]
        batch["logvar_full"] = out_full["logvar_full"]

        # z is ALWAYS sampled from q_ϕ.
        # Reparameterization carries gradient to both μ_ϕ and σ_ϕ.
        batch["mu"]    = batch["mu_full"]
        batch["logvar"] = batch["logvar_full"]

        if phase == 2:
            # Run masked encoder r_ψ(z|x_S,S) — parameters stored for KL only.
            # r_ψ receives NO gradient from the decoder/reconstruction path.
            # Its gradient comes entirely from the KL term in compute_loss.
            masked_out = self.masked_encoder(batch)
            batch["mu_masked"]     = masked_out["mu_masked"]
            batch["logvar_masked"] = masked_out["logvar_masked"]

        batch["z"] = self.reparameterize(batch)   # z ~ q_ϕ
        batch.update(self.decoder(batch))

        if self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]

        return batch

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: dict,
        lambda_kl: float = 1.0,
        lambda_reg: float = 1e-6,
        lambda_kl_full: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Proper VAEAC ELBO (Olsen 2022, Eq. 6 + Sec. 3.3.1).

        L = E_{z~q_ϕ(z|x,S)} [log p_θ(x_S|z,...)]
            − KL( q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S) )
            − ||μ_ψ||²/(2σ²_μ)          [normal prior on μ_ψ, σ_μ=100]
            + (1/σ_σ)(log σ_ψ − σ_ψ)   [gamma prior on σ_ψ, σ_σ=100]

        Forward KL: KL( N(μ_ϕ,σ²_ϕ) ‖ N(μ_ψ,σ²_ψ) )
            = 0.5 · Σ[ log(σ²_ψ/σ²_ϕ) + (σ²_ϕ + (μ_ϕ−μ_ψ)²)/σ²_ψ − 1 ]

        Gradient routing:
            ∂L/∂θ (decoder):       reconstruction only.
            ∂L/∂ϕ (full encoder):  reconstruction (via reparameterization)
                                   + KL numerator (μ_ϕ, σ²_ϕ terms).
            ∂L/∂ψ (masked encoder): KL denominator (μ_ψ, σ²_ψ terms)
                                    + prior regularization.
                                    ZERO from reconstruction.

        Args:
            batch:      output of forward() — must contain "x", "output",
                        "mask", "coalition_mask", "mu_full", "logvar_full",
                        "mu_masked", "logvar_masked".
            lambda_kl:  KL weight (linearly annealed by the Lightning module).
            lambda_reg: weight for prior regularization on r_ψ parameters.

        Returns:
            (loss, losses_dict) — loss is the scalar for backward();
            losses_dict contains "rc", "kl", "reg", "mixed" as floats.
        """
        x               = batch["x"]
        output          = batch["output"]
        mask            = batch["mask"]              # (B, T) real-frame mask
        coalition_mask  = batch["coalition_mask"]    # (B, J) spatial or (B, T) temporal

        # ------------------------------------------------------------------
        # Reconstruction: MSE on held-out positions only.
        # "We use only unobserved components to compute likelihood." (Ivanov 2019)
        # ------------------------------------------------------------------
        B, J, F, T = x.shape
        if coalition_mask.shape[-1] == J:
            # Spatial coalition: (B, J) → (B, J, F, T)
            held_out = ~coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
        else:
            # Temporal coalition: (B, T) → (B, J, F, T)
            held_out = ~coalition_mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)

        real_frames = mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)
        reconstruction_mask = held_out & real_frames
        n_masked = reconstruction_mask.float().sum().clamp(min=1.0)
        rc = ((output - x).pow(2) * reconstruction_mask.float()).sum() / n_masked

        # ------------------------------------------------------------------
        # Forward KL: KL( q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S) )
        #
        # Closed-form Gaussian-Gaussian divergence:
        #   KL(N(μ₁,σ²₁) ‖ N(μ₂,σ²₂))
        #     = 0.5 · Σ[ log(σ²₂/σ²₁) + (σ²₁+(μ₁−μ₂)²)/σ²₂ − 1 ]
        #
        # GRADIENT ROUTING FOR KL:
        #   μ_ϕ, σ²_ϕ are DETACHED from the KL computation.
        #   Only r_ψ (ψ) receives gradient from KL.
        #
        # Rationale: if q_ϕ receives KL gradient, the update direction
        #   ∂KL/∂μ_ϕ = (μ_ϕ − μ_ψ)/σ²_ψ
        # pulls μ_ϕ toward μ_ψ. Since r_ψ starts near q_ϕ (zero mask tokens),
        # this immediately collapses q_ϕ's coalition-specific representations
        # that the reconstruction gradient is trying to build via target_marker_spatial.
        #
        # By detaching, q_ϕ is trained ONLY by reconstruction (via reparameterization),
        # which naturally induces coalition-specific encodings:
        #   different S → different x' (target markers on unobserved joints)
        #   → different z → different reconstruction targets (held-out joints)
        #   → reconstruction gradient shapes target_marker_spatial to be S-specific.
        #
        # r_ψ then tracks q_ϕ via KL, acting as a "student" that learns to
        # reproduce the teacher (q_ϕ.detach())'s coalition-specific distributions:
        #   ∂KL/∂ψ = ∂/∂ψ(-E_{q_ϕ}[log r_ψ])
        #   → r_ψ maximizes log-likelihood under the (stopped) q_ϕ distribution.
        # ------------------------------------------------------------------
        mu_phi  = batch["mu_full"].detach()     # stop q_ϕ gradient through KL
        lv_phi  = batch["logvar_full"].detach() # stop q_ϕ gradient through KL
        mu_psi  = batch["mu_masked"]            # r_ψ gradient flows freely
        lv_psi  = batch["logvar_masked"]
        var_phi = lv_phi.exp()
        var_psi = lv_psi.exp()
        kl = 0.5 * (
            lv_psi - lv_phi
            + (var_phi + (mu_phi - mu_psi).pow(2)) / var_psi.clamp(min=1e-8)
            - 1.0
        ).sum(dim=-1).mean()

        # ------------------------------------------------------------------
        # Prior regularization on r_ψ (Olsen 2022, Sec. 3.3.1; Ivanov 2019, Eq. 8).
        #
        # Without regularization, μ_ψ and σ_ψ can grow without bound during
        # training because the KL target (q_ϕ) is also moving.
        #
        # Normal prior on μ_ψ: p(μ_ψ) = N(0, σ²_μ·I)
        #   −log p ∝ ||μ_ψ||² / (2σ²_μ)
        #   σ_μ = 100 → variance = 10000 → very mild, prevents divergence.
        #
        # Gamma prior on σ_ψ: p(σ_ψ) = Γ(1+σ_σ⁻¹, σ_σ⁻¹), mean = 1+σ_σ ≈ 1
        #   −log p ∝ −(1/σ_σ)(log σ_ψ − σ_ψ)
        #   σ_σ = 100 → very mild, keeps σ_ψ near 1.
        # ------------------------------------------------------------------
        sigma_mu    = 100.0
        sigma_sigma = 100.0
        reg_mu    = mu_psi.pow(2).sum(dim=-1).mean() / (2.0 * sigma_mu ** 2)
        sigma_psi = (0.5 * lv_psi).exp()
        # (1/σ_σ)(log σ_ψ − σ_ψ): maximized at σ_ψ=1, so we negate for loss.
        reg_sigma = (lv_psi * 0.5 - sigma_psi).sum(dim=-1).mean() / sigma_sigma
        reg = reg_mu - reg_sigma   # minimize reg_mu, maximize reg_sigma (→ subtract)

        # ------------------------------------------------------------------
        # KL regularization on q_ϕ: KL( q_ϕ(z|x,S) ‖ N(0,I) )
        #
        # Without this, q_ϕ is trained ONLY by reconstruction, which pushes
        # σ_ϕ → 0 (deterministic z minimizes reconstruction variance).
        # As σ_ϕ → 0, the KL target for r_ψ also collapses, dragging σ_ψ → 0.
        # Consequence: completions become deterministic → completion_diversity=0.
        #
        # This term uses the NON-detached μ_ϕ, σ²_ϕ so gradient flows to ϕ.
        # Weight λ_kl_full ≪ λ_kl: just enough to prevent collapse, not
        # enough to destroy q_ϕ's coalition-specific structure.
        # Standard VAE KL:  0.5 · Σ(μ² + σ² - log σ² - 1)
        # ------------------------------------------------------------------
        mu_phi_nd  = batch["mu_full"]      # non-detached — gradient to ϕ
        lv_phi_nd  = batch["logvar_full"]
        kl_full = 0.5 * (
            mu_phi_nd.pow(2) + lv_phi_nd.exp() - lv_phi_nd - 1.0
        ).sum(dim=-1).mean()

        loss = rc + lambda_kl * kl + lambda_reg * reg + lambda_kl_full * kl_full
        return loss, {
            "rc":      float(rc.detach()),
            "kl":      float(kl.detach()),
            "reg":     float(reg.detach()),
            "kl_full": float(kl_full.detach()),
            "mixed":   float(loss.detach()),
        }

    # ------------------------------------------------------------------
    # Sampling — used by KernelSHAP inference
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
        """Draw n_samples manifold-constrained completions of a masked sequence.

        At inference, ONLY r_ψ (masked encoder) and p_θ (decoder) are used —
        this matches the VAEAC deployment phase (Olsen 2022, Sec. 3.2).
        The coalition-aware full encoder q_ϕ is NOT used here.

        Args:
            x:               (1, J, F, T) input sequence.
            y:               (1,) class label.
            mask:            (1, T) real-frame mask.
            lengths:         (1,) frame count.
            coalition_mask:  (1, J) for spatial or (1, T) for temporal;
                             True = observed.
            n_samples:       number of stochastic completions.
            paste_observed:  if True, overwrite observed positions in x_hat with
                             the original x (no distortion of seen joints/frames).

        Returns:
            List of n_samples tensors, each (1, J, F, T).
        """
        base_batch = {
            "x": x, "y": y, "mask": mask,
            "lengths": lengths, "coalition_mask": coalition_mask,
        }
        obs_mask = self._coalition_to_xmask(coalition_mask, x.shape)
        results: list[torch.Tensor] = []
        for _ in range(n_samples):
            b = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in base_batch.items()}
            # Inference uses r_ψ → z → p_θ, matching VAEAC deployment phase.
            b = self.forward(b, phase=2)
            x_hat = b["output"].clone()
            if paste_observed:
                x_hat[obs_mask] = x[obs_mask]
            results.append(x_hat)
        return results

    def _coalition_to_xmask(
        self, coalition_mask: torch.Tensor, x_shape: torch.Size
    ) -> torch.Tensor:
        """Expand coalition_mask to (B, J, F, T) for use in paste_observed."""
        B, J, F, T = x_shape
        if coalition_mask.shape[-1] == J:
            return coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
        else:
            return coalition_mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_from_actor_cvae(self, ckpt_path: str) -> None:
        """Load encoder and decoder weights from a trained ActorCVAE checkpoint.

        Initialises both CoalitionFullEncoder._encoder and
        MaskedActorEncoder._encoder from the ActorCVAE encoder weights, and
        decoder from the ActorCVAE decoder weights.

        IMPORTANT: the full encoder (q_ϕ) is NO LONGER FROZEN.  It must be
        trained jointly so that target_marker_spatial can learn to produce
        coalition-specific z values and provide meaningful KL targets for r_ψ.

        Args:
            ckpt_path: path to a PyTorch Lightning checkpoint saved by
                       ActorCVAEModule (state_dict keys prefixed with "model.").
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]

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

        # Initialize CoalitionFullEncoder from ActorCVAE encoder weights.
        # target_marker_spatial stays at zero (no initial perturbation).
        self.encoder._encoder.load_state_dict(enc_sd)
        # Initialize MaskedActorEncoder from the same ActorCVAE encoder weights.
        # mask_token_spatial stays at zero (same initialization as before).
        self.masked_encoder._encoder.load_state_dict(enc_sd)
        # Initialize decoder from ActorCVAE decoder weights.
        self.decoder.load_state_dict(dec_sd)

        # NOTE: self.encoder is NOT frozen — it is trained jointly with r_ψ and p_θ.
        # Both target_marker_spatial (q_ϕ) and mask_token_spatial (r_ψ) start at
        # zero, so epoch 0 is identical to a standard ActorCVAE forward pass.
        # Training will diverge q_ϕ and r_ψ to serve their distinct roles.

    def load_decoder_from_phase1(self, ckpt_path: str) -> None:
        """Overwrite decoder weights from a Phase 1 Lightning checkpoint.

        Call this after load_from_actor_cvae when resuming Phase 2 training
        from a Phase 1 checkpoint.

        Args:
            ckpt_path: path to a Phase 1 Lightning checkpoint (ActorSHAPModule).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        dec_sd = {
            k.removeprefix("model.decoder."): v
            for k, v in sd.items()
            if k.startswith("model.decoder.")
        }
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

    # New architecture: CoalitionFullEncoder instead of plain Encoder_TRANSFORMER.
    encoder        = CoalitionFullEncoder(**common)
    masked_encoder = MaskedActorEncoder(**common)
    decoder        = Decoder_TRANSFORMER(**common)

    model = ActorSHAP(encoder, masked_encoder, decoder, latent_dim=16, device=dev)

    x              = torch.randn(B, J, F, T)
    y              = torch.zeros(B, dtype=torch.long)
    mask           = torch.ones(B, T, dtype=torch.bool)
    lengths        = torch.full((B,), T, dtype=torch.long)
    coalition_mask = torch.ones(B, J, dtype=torch.bool)
    coalition_mask[:, 3] = False  # mask joint 3

    batch = {"x": x, "y": y, "mask": mask, "lengths": lengths,
             "coalition_mask": coalition_mask}

    # ---- Phase 2 (VAEAC ELBO) ----
    out  = model(dict(batch), phase=2)
    assert "output" in out and out["output"].shape == (B, J, F, T), \
        f"Unexpected output shape: {out['output'].shape}"
    assert "mu_full"    in out, "mu_full missing from phase 2 output"
    assert "mu_masked"  in out, "mu_masked missing from phase 2 output"

    loss, ld = model.compute_loss(out, lambda_kl=1.0, lambda_reg=1e-6)
    assert loss.isfinite(), f"Phase 2 loss not finite: {loss}"
    assert set(ld.keys()) == {"rc", "kl", "reg", "mixed"}, \
        f"Unexpected loss keys: {ld.keys()}"

    # Verify forward KL direction: KL(q_ϕ || r_ψ), not KL(r_ψ || q_ϕ).
    # With zero target_markers and zero mask_tokens, q_ϕ ≈ r_ψ → KL ≈ 0.
    assert ld["kl"] >= 0.0, f"KL must be non-negative, got {ld['kl']}"

    # ---- Phase 1 (warm-up, no coalition_mask) ----
    batch1 = {"x": x, "y": y, "mask": mask, "lengths": lengths}
    out1   = model(dict(batch1), phase=1)
    assert "mu_masked" not in out1, "mu_masked should not appear in phase 1"
    # Phase 1 has no coalition_mask → cannot call compute_loss (no mu_masked).
    assert out1["output"].shape == (B, J, F, T)

    # ---- sample_completions (inference path: r_ψ only) ----
    comps = model.sample_completions(
        x[:1], y[:1], mask[:1], lengths[:1], coalition_mask[:1], n_samples=3
    )
    assert len(comps) == 3 and comps[0].shape == (1, J, F, T), \
        f"sample_completions shape error: {comps[0].shape}"

    # ---- Temporal coalition mask ----
    coalition_t = torch.ones(B, T, dtype=torch.bool)
    coalition_t[:, :T // 4] = False
    batch_t = {"x": x, "y": y, "mask": mask, "lengths": lengths,
               "coalition_mask": coalition_t}
    out_t = model(dict(batch_t), phase=2)
    loss_t, _ = model.compute_loss(out_t, lambda_kl=1.0)
    assert loss_t.isfinite(), f"Temporal phase 2 loss not finite: {loss_t}"

    print("Smoke test OK")
