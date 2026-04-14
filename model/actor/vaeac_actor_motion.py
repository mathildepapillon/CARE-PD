"""
vaeac_actor_motion.py — Sequence-latent VAEAC on the ACTOR Transformer backbone.

Architecture
------------
q_φ(z | x)          Full encoder.  Sees the complete sequence x.
                     TransformerEncoder outputs one (D,) hidden state per frame;
                     mu_head / logvar_head project each to per-frame mu / logvar.
                     z_φ ∈ ℝ^{T × B × D} — a full sequence latent.

r_ψ(z | x_S, S)     Masked encoder.  Sees x with unobserved positions replaced
                     by learnable mask tokens.  Produces matching per-frame
                     (T, B, D) mu / logvar.  Used at inference.

p_θ(x̂ | z)          Decoder_TRANSFORMER (unchanged from ACTOR).  Receives z_seq
                     as frame_tokens — cross-attention memory is (T+1, B, D):
                       token 0   : zeros + actionBiases[y]  → class-conditioning
                       tokens 1…T: z_seq[0…T-1]             → per-frame latent

Key differences from VaeacMotion
---------------------------------
- Latent is (T, B, D) per-frame sequence, not a single (B, D) vector.
- No observed_projection / obs_emb shortcut — z_seq is the only decoder memory.
- KL: sum over D, mean over T and B (normalises by T).
- SeqEncoder uses class_embedding added to all T tokens instead of muQuery/
  sigmaQuery prepended tokens.  mu_head / logvar_head act frame-by-frame on
  the T TransformerEncoder output tokens.

Why this avoids mean-pose collapse
------------------------------------
With z ∈ ℝ^{T × D} the decoder cross-attends to T specific tokens; frame t
primarily attends to z[t].  r_ψ must encode observed-joint information per
frame into z_ψ (rather than routing it through a direct linear projection
shortcut), so z_ψ diversity translates directly to completion diversity.

KL normalisation
-----------------
Forward KL = Σ_{t,d} KL(q_φ[t,d] ‖ r_ψ[t,d]).
We divide by T via `.sum(dim=-1).mean()` on (T, B, D):
    sum over D → (T, B), then mean over (T, B) → normalised by T·B.
Use the same λ_kl as VaeacMotion; the /T keeps it comparable in magnitude.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.actor.transformer_arch import Decoder_TRANSFORMER
from model.motionclip.transformer import PositionalEncoding


# ---------------------------------------------------------------------------
# Shared encoder backbone
# ---------------------------------------------------------------------------

class SeqEncoder(nn.Module):
    """Per-frame Transformer encoder backbone shared by q_φ and r_ψ.

    Each instance (full / masked) is separately initialised and trained.
    Unlike ACTOR's Encoder_TRANSFORMER, there are no muQuery / sigmaQuery
    tokens.  Instead all T frame tokens are processed and per-frame
    mu_head / logvar_head projections produce the sequence latent.
    """

    def __init__(
        self,
        njoints: int,
        nfeats: int,
        num_classes: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,  # absorb unused args from common dict (modeltype, etc.)
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.latent_dim = latent_dim

        self.skelEmbedding = nn.Linear(njoints * nfeats, latent_dim)
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    # ------------------------------------------------------------------

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw pose features to latent space.

        Args:
            x: (B, J, F, T) raw joint features.

        Returns:
            h: (T, B, D) — skelEmbedding applied, class/pos not yet added.
        """
        B, J, F, T = x.shape
        x_flat = x.permute(3, 0, 1, 2).reshape(T, B, J * F)  # (T, B, J*F)
        return self.skelEmbedding(x_flat)                      # (T, B, D)

    def encode_from_embed(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add class / positional embeddings, run transformer, produce mu/logvar.

        Args:
            h:    (T, B, D) token sequence after skelEmbedding (may have mask
                  tokens substituted at relevant positions).
            y:    (B,) integer class labels.
            mask: (B, T) bool, True = real frame, False = padding.

        Returns:
            mu:     (T, B, D) per-frame mean.
            logvar: (T, B, D) per-frame log-variance.
        """
        # Class conditioning: add class_embedding[y] to every frame token.
        h = h + self.class_embedding(y).unsqueeze(0)  # (T, B, D)
        h = self.sequence_pos_encoder(h)
        # PyTorch convention: src_key_padding_mask True = ignore.
        h = self.seqTransEncoder(h, src_key_padding_mask=~mask)  # (T, B, D)
        return self.mu_head(h), self.logvar_head(h)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward: (B, J, F, T) → (T, B, D) mu, (T, B, D) logvar."""
        h = self.embed(x)
        return self.encode_from_embed(h, y, mask)


# ---------------------------------------------------------------------------
# Full encoder q_φ(z | x)
# ---------------------------------------------------------------------------

class VaeacActorFullEncoder(nn.Module):
    """Full encoder (proposal) q_φ(z | x).

    Sees the COMPLETE sequence with no masking.
    Returns per-frame mu_full / logvar_full of shape (T, B, D).
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = SeqEncoder(**encoder_kwargs)

    def forward(self, batch: dict) -> dict:
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        mu, logvar = self._encoder(x, y, mask)
        return {"mu_full": mu, "logvar_full": logvar}


# ---------------------------------------------------------------------------
# Masked encoder r_ψ(z | x_S, S)
# ---------------------------------------------------------------------------

class VaeacActorMaskedEncoder(nn.Module):
    """Masked encoder (prior) r_ψ(z | x_S, S).

    Unobserved positions are REPLACED by learnable mask tokens before the
    encoder backbone runs, so the Transformer sees a "this is missing" signal.

    Two kinds of masking:
        Spatial  (coalition_mask ∈ {True/False}^{B×J}):
            mask_token_spatial[j] replaces all frames of unobserved joint j
            in raw feature space before skelEmbedding.

        Temporal (coalition_mask ∈ {True/False}^{B×T}):
            mask_token_temporal replaces unobserved frame embeddings after
            skelEmbedding, so the token shape in latent space is D-dimensional.
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self._encoder = SeqEncoder(**encoder_kwargs)
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
        x = batch["x"].clone()
        y, mask, cm = batch["y"], batch["mask"], batch["coalition_mask"]
        unobs = ~cm                    # (B, J)
        B, J, F, T = x.shape
        for j in range(J):
            x[unobs[:, j], j, :, :] = self.mask_token_spatial[j].unsqueeze(-1)
        mu, logvar = self._encoder(x, y, mask)
        return {"mu_masked": mu, "logvar_masked": logvar}

    def _encode_temporal(self, batch: dict) -> dict:
        x, y, mask, cm = batch["x"], batch["y"], batch["mask"], batch["coalition_mask"]
        h = self._encoder.embed(x)              # (T, B, D)
        # cm: (B, T), True=observed → permute to (T, B)
        obs_tf  =  cm.permute(1, 0).float()     # (T, B) 1.0=observed
        unobs_tf = (~cm).permute(1, 0).float()  # (T, B) 1.0=unobserved
        h = (h * obs_tf.unsqueeze(-1)
             + unobs_tf.unsqueeze(-1) * self.mask_token_temporal)
        mu, logvar = self._encoder.encode_from_embed(h, y, mask)
        return {"mu_masked": mu, "logvar_masked": logvar}


# ---------------------------------------------------------------------------
# VaeacActorMotion — the full VAEAC model
# ---------------------------------------------------------------------------

class VaeacActorMotion(nn.Module):
    """Sequence-latent VAEAC for manifold-constrained motion imputation.

    The latent z is a full (T, B, D) sequence rather than a single (B, D) vector.
    The decoder receives this sequence as frame_tokens in cross-attention memory,
    alongside a class-conditioning token (actionBiases[y]).  No observed_projection
    / obs_emb shortcut is present — z_seq is the sole decoder memory.

    Gradient routing (Ivanov 2019 §3):
        q_φ: reconstruction  (via reparameterised z_φ decoded during training).
             DETACHED from KL so KL does not collapse q_φ.
        r_ψ: KL + prior regularisation.
        p_θ: reconstruction + velocity.

    Args:
        full_encoder:   VaeacActorFullEncoder (q_φ).
        masked_encoder: VaeacActorMaskedEncoder (r_ψ).
        decoder:        Decoder_TRANSFORMER (p_θ, unchanged from ACTOR).
        latent_dim:     Latent dimensionality D.
        njoints:        Number of skeleton joints J.
        nfeats:         Features per joint F.
        device:         Torch device.
        pose_rep:       ``"xyz"`` or ``"rot6d"``.
        num_classes:    Number of action/diagnosis class labels.
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
        self.losses = ["rc", "vel", "kl", "reg", "mixed"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return torch.randn_like(std) * std + mu

    def _decode(self, batch: dict, z_seq: torch.Tensor) -> dict:
        """Decode using per-frame latent sequence z_seq.

        Passes z_seq as frame_tokens so that decoder memory is:
            [actionBiases[y]  |  z_seq[0]  …  z_seq[T-1]]
               (1, B, D)            (T, B, D)
        = (T+1, B, D) total.

        The first token provides global class conditioning (via actionBiases[y]).
        The remaining T tokens provide per-frame latent content.
        We set batch["z"] = zeros(B, D) so that the decoder's
        ``z = z + actionBiases[y]`` produces exactly actionBiases[y].
        """
        B = z_seq.shape[1]
        device = z_seq.device
        batch = dict(batch)
        batch["z"] = torch.zeros(B, self.latent_dim, device=device)
        batch["frame_tokens"] = z_seq
        batch.update(self.decoder(batch))
        return batch

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

        phase="train":
            q_φ → z_φ ~ q_φ → decoder.  r_ψ also runs (for KL in compute_loss).
        phase="infer":
            r_ψ → z_ψ ~ r_ψ → decoder.  q_φ not called.
        """
        if self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]

        coalition_mask = batch.get("coalition_mask")

        if phase == "infer":
            masked_out = self.masked_encoder(batch)
            mu = masked_out["mu_masked"]
            logvar = masked_out["logvar_masked"]
            batch["mu_masked"] = mu
            batch["logvar_masked"] = logvar
            z_seq = self.reparameterize(mu, logvar)

        else:
            # q_φ: full encoder provides z during training.
            full_out = self.full_encoder(batch)
            batch["mu_full"] = full_out["mu_full"]
            batch["logvar_full"] = full_out["logvar_full"]
            mu = full_out["mu_full"]
            logvar = full_out["logvar_full"]

            # r_ψ: run for KL loss (always when a coalition_mask is present).
            if coalition_mask is not None:
                masked_out = self.masked_encoder(batch)
                batch["mu_masked"] = masked_out["mu_masked"]
                batch["logvar_masked"] = masked_out["logvar_masked"]

            z_seq = self.reparameterize(mu, logvar)

        batch = self._decode(batch, z_seq)

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
        prior_sigma_mu: float = 1e4,
        prior_sigma_sigma: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """ACTOR-style full-sequence reconstruction + KL(q_φ ‖ r_ψ) regularisation.

        Objective
        ---------
        L = MSE(p_θ(z_φ), x)          full-sequence reconstruction
            + λ_vel · vel_loss(x)      velocity smoothness over full sequence
            − λ_kl  · KL(q_φ ‖ r_ψ)  aligns r_ψ to q_φ per frame
            − λ_reg · prior_reg(r_ψ)  Ivanov prior stabiliser on r_ψ

        The reconstruction and velocity are over ALL real frames (identical to
        ACTOR's training objective).  This gives q_φ and p_θ strong, dense
        gradients across the entire sequence.

        The KL term trains r_ψ to produce the same per-frame distribution as
        q_φ would if it saw the full sequence.  q_φ is DETACHED from the KL so
        it is never collapsed — it trains only via reconstruction.

        At inference, r_ψ encodes the masked input and its per-frame latents
        z_ψ are decoded by the same p_θ that was trained to reconstruct from
        z_φ.  Because KL(q_φ ‖ r_ψ) → 0, z_ψ lies in the same space that p_θ
        expects, giving coherent completions.

        Gradient routing:
            q_φ: reconstruction + velocity only (DETACHED from KL).
            r_ψ: KL + prior reg only.
            p_θ: reconstruction + velocity.

        KL normalisation
        ----------------
        mu/logvar shapes: (T, B, D).  sum over D, mean over T·B — divides by T
        relative to a single-vector KL so λ_kl has the same effective scale.
        """
        x = batch["x"]
        output = batch["output"]
        mask = batch["mask"]
        B, J, F, T = x.shape

        real_frames = mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)
        n_real = real_frames.float().sum().clamp(min=1.0)

        # ---- Full-sequence reconstruction ----
        rc = ((output - x).pow(2) * real_frames.float()).sum() / n_real

        # ---- Full-sequence velocity ----
        vel = x.new_zeros(())
        if T > 1:
            gt_vel  = x[..., 1:]      - x[..., :-1]
            out_vel = output[..., 1:] - output[..., :-1]
            vel_real = real_frames[..., 1:] & real_frames[..., :-1]
            n_vel = vel_real.float().sum().clamp(min=1.0)
            vel = ((out_vel - gt_vel).pow(2) * vel_real.float()).sum() / n_vel

        # ---- Forward KL: KL(q_φ ‖ r_ψ) — per-frame, normalised by T ----
        # q_φ DETACHED: KL gradient flows only into r_ψ.
        kl = x.new_zeros(())
        if "mu_masked" in batch and "logvar_masked" in batch:
            mu_phi = batch["mu_full"].detach()      # (T, B, D)
            lv_phi = batch["logvar_full"].detach()  # (T, B, D)
            mu_psi = batch["mu_masked"]             # (T, B, D)
            lv_psi = batch["logvar_masked"]         # (T, B, D)
            var_phi = lv_phi.exp()
            var_psi = lv_psi.exp()
            kl_cell = 0.5 * (
                lv_psi - lv_phi
                + (var_phi + (mu_phi - mu_psi).pow(2)) / var_psi.clamp(min=1e-8)
                - 1.0
            )  # (T, B, D)
            kl = kl_cell.sum(dim=-1).mean()  # sum over D, mean over T and B

        # ---- Prior regularisation on r_ψ (Ivanov 2019 Eq.8) ----
        reg = x.new_zeros(())
        if "mu_masked" in batch:
            mu_psi = batch["mu_masked"]
            lv_psi = batch["logvar_masked"]
            reg_mu = mu_psi.pow(2).sum(dim=-1).mean() / (2.0 * prior_sigma_mu ** 2)
            sigma_psi = (0.5 * lv_psi).exp()
            reg_sigma = (lv_psi * 0.5 - sigma_psi).sum(dim=-1).mean() * prior_sigma_sigma
            reg = reg_mu - reg_sigma

        loss = rc + lambda_vel * vel + lambda_kl * kl + lambda_reg * reg
        return loss, {
            "rc":    float(rc.detach()),
            "vel":   float(vel.detach()),
            "kl":    float(kl.detach()),
            "reg":   float(reg.detach()),
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

        q_φ is never called at inference — only r_ψ and p_θ are used.
        This matches VAEAC deployment (Olsen 2022, §3.2).

        Args:
            x:              (1, J, F, T) input sequence.
            y:              (1,) class label.
            mask:           (1, T) real-frame mask.
            lengths:        (1,) frame count.
            coalition_mask: (1, J) spatial or (1, T) temporal; True = observed.
            n_samples:      Number of stochastic completions.
            paste_observed: If True, overwrite observed positions with GT.

        Returns:
            List of n_samples tensors, each (1, J, F, T).
        """
        b = {
            "x": x, "y": y, "mask": mask,
            "lengths": lengths, "coalition_mask": coalition_mask,
        }
        masked_out = self.masked_encoder(b)
        mu = masked_out["mu_masked"]        # (T, 1, D)
        logvar = masked_out["logvar_masked"] # (T, 1, D)
        std = (0.5 * logvar).exp()

        obs_xmask = self._coalition_to_xmask(coalition_mask, x.shape)

        completions: list[torch.Tensor] = []
        for _ in range(n_samples):
            z_seq = torch.randn_like(std) * std + mu   # (T, 1, D)
            dec_b = self._decode(dict(b), z_seq)
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

    def _build_model():
        full_enc = VaeacActorFullEncoder(**common)
        masked_enc = VaeacActorMaskedEncoder(**common)
        dec = Decoder_TRANSFORMER(**common)
        return VaeacActorMotion(full_enc, masked_enc, dec,
                                latent_dim=16, njoints=J, nfeats=F, device=dev)

    def _smoke(model, tag):
        model.train()
        batch = {"x": x, "y": y, "mask": mask, "lengths": lengths,
                 "coalition_mask": cm_sp}
        out = model(dict(batch), phase="train")
        assert out["output"].shape == (B, J, F, T), f"{tag}: bad output shape"
        assert "mu_full" in out and "mu_masked" in out, f"{tag}: missing encoder keys"
        assert out["mu_full"].shape == (T, B, 16), f"{tag}: mu_full wrong shape"
        assert out["mu_masked"].shape == (T, B, 16), f"{tag}: mu_masked wrong shape"

        loss, ld = model.compute_loss(out, lambda_kl=1.0)
        assert loss.isfinite(), f"{tag}: loss not finite"
        assert ld["kl"] >= 0, f"{tag}: KL negative"

        # temporal coalition mask
        batch_t = {"x": x, "y": y, "mask": mask, "lengths": lengths,
                   "coalition_mask": cm_t}
        out_t = model(dict(batch_t), phase="train")
        assert model.compute_loss(out_t, lambda_kl=1.0)[0].isfinite(), \
            f"{tag}: temporal loss not finite"

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
    _smoke(_build_model(), "full model")
    print("All smoke tests passed.")
