"""vaeac.py — Transformer-backbone VAEAC (Variational Autoencoder with
Arbitrary Conditioning) sized to match :class:`model.flow_matching.VelocityNet`.

Design
------
VAEAC (Ivanov et al. 2019; Olsen et al. JMLR 2022) models

    p(x_hidden | x_observed, mask)

by learning three networks:

* ``full_encoder``   q(z | x, mask)      — training only (sees everything).
* ``prior_encoder``  p(z | x_obs, mask)  — train + inference (sees only the
                                           observed coordinates + the mask).
* ``decoder``        p(x | z, x_obs, mask)

where ``z`` is a per-frame latent of shape ``(B, T, d_latent)``.  The training
loss is the ELBO

    L = E_q [ -log p(x_hid | z, x_obs, mask) ]  +  KL( q(z|x,mask) || p(z|x_obs,mask) )

Ivanov-style additions (``decoder_head="gaussian_ivanov"``, ``use_prior_memory=True``)
-------------------------------------------------------------------------------------
* **Output head**: faithful re-implementation of Ivanov/Olsen's per-feature
  heteroscedastic Gaussian.  For each continuous coordinate the decoder emits
  ``(μ, σ_param)``; the scale is ``σ = softplus(σ_param).clamp_min(min_sigma)``.
  See :class:`model.vaeac.heads.GaussianIvanovHead`.
* **Memory / skip connections**: the prior encoder's intermediate transformer
  outputs are concatenated into the decoder's token stream at matching layer
  depths, then projected back to ``d_model`` via a small merge linear — the
  transformer analogue of Ivanov's ``MemoryLayer(add=False)``.
* **Prior regularisation**: tiny penalty on the prior's ``(μ_p, σ_p)`` to keep
  them from drifting, matching ``VAEAC.prior_regularization`` in Ivanov's repo
  with ``σ_μ=1e4, σ_σ=1e-4``.

The legacy ``gaussian_scalar`` head is kept for ablation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model.flow_matching.velocity_net import FramePositionalEncoding
from model.vaeac.heads import make_head


# ---------------------------------------------------------------------------
# Shared transformer trunk (supports per-layer outputs & prior→decoder memory)
# ---------------------------------------------------------------------------

class _TransformerTrunk(nn.Module):
    """Per-frame transformer encoder used by all three VAEAC subnets.

    Always builds a :class:`torch.nn.ModuleList` of ``num_layers``
    :class:`nn.TransformerEncoderLayer`\ s so per-layer activations can be
    inspected or fused with prior-encoder ``memory`` at each depth.

    ``forward`` can either:

    * return just the final ``(B, T, d_model)`` tensor (default);
    * return the final tensor AND a list of intermediate outputs
      (one per layer), for use as Ivanov-style "memory" by the decoder;
    * consume a list of ``memory`` tensors and concatenate them with the
      decoder's token stream *before* each layer, followed by a learned
      ``Linear(2·d_model → d_model)`` merge.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        activation: str = "gelu",
        use_memory: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_memory = bool(use_memory)

        self.frame_pos = FramePositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=False,
            )
            for _ in range(num_layers)
        ])
        if self.use_memory:
            # One merge per layer: cat([tok, mem], dim=-1) → Linear(2d, d).
            # Initialised so the layer is approximately the identity on
            # ``tok`` (i.e. memory starts off silent) to stabilise early
            # training.  We zero the ``mem`` half of the weight and init the
            # ``tok`` half to identity-like (trunc_normal, small std).
            self.memory_merges = nn.ModuleList([
                nn.Linear(2 * d_model, d_model) for _ in range(num_layers)
            ])
            for lin in self.memory_merges:
                with torch.no_grad():
                    # Set first d_model cols ≈ identity, last d_model cols = 0.
                    nn.init.trunc_normal_(lin.weight, std=0.02)
                    lin.weight[:, d_model:].zero_()
                    if lin.bias is not None:
                        lin.bias.zero_()
        else:
            self.memory_merges = None

    def forward(
        self,
        tok: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        *,
        memory: Optional[List[torch.Tensor]] = None,
        return_layer_outputs: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        T = tok.shape[1]
        tok = tok + self.frame_pos(T).to(tok.dtype)
        src_kp = None if pad_mask is None else ~pad_mask

        if memory is not None:
            if not self.use_memory:
                raise RuntimeError(
                    "_TransformerTrunk got `memory` but was built with "
                    "`use_memory=False`."
                )
            if len(memory) != self.num_layers:
                raise ValueError(
                    f"memory list length {len(memory)} != num_layers "
                    f"{self.num_layers}"
                )

        layer_outputs: list[torch.Tensor] = []
        for l, layer in enumerate(self.layers):
            if memory is not None:
                mem_l = memory[l]
                if mem_l.shape != tok.shape:
                    raise ValueError(
                        f"memory[{l}] shape {tuple(mem_l.shape)} != "
                        f"tok shape {tuple(tok.shape)}"
                    )
                fused = torch.cat([tok, mem_l], dim=-1)
                tok = self.memory_merges[l](fused)
            tok = layer(tok, src_key_padding_mask=src_kp)
            if return_layer_outputs:
                layer_outputs.append(tok)

        if return_layer_outputs:
            return tok, layer_outputs
        return tok


def _init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Gaussian-head helpers
# ---------------------------------------------------------------------------

def _gaussian_kl(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    mu_p: torch.Tensor, logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) elementwise."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# Mask sampling
# ---------------------------------------------------------------------------

def build_window_to_frame(
    K: int, T: int, window_assignments,
    device: torch.device,
) -> torch.Tensor:
    """Precompute ``(K, T)`` boolean matrix mapping windows → frame indices.

    ``window_assignments[k]`` is the list/array of frame indices belonging
    to window ``k``.  This matches :class:`GaussianMotionBenchmark` exactly.
    Returned as a ``float32`` tensor so the broadcast can use a matmul.
    """
    w2f = torch.zeros(K, T, dtype=torch.float32, device=device)
    for k in range(K):
        idx = torch.as_tensor(list(window_assignments[k]), dtype=torch.long, device=device)
        w2f[k, idx] = 1.0
    return w2f


def sample_window_mask_uniform(
    B: int, T: int, J: int, C: int,
    window_to_frame: torch.Tensor,    # (K, T) float
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``B`` coalitions uniformly over ``{0,1}^K`` windows.

    Each batch item independently draws a bit vector ``z ∈ {0,1}^K``; frames
    in any observed window are marked observed for *all* joints/coords.  This
    exactly matches the coalition support used by the temporal EC evaluator
    (which enumerates all ``2^K`` window coalitions).

    Returns ``obs ∈ {True, False}^(B, T, J, C)`` with ``True`` = observed.
    """
    K = window_to_frame.shape[0]
    # Uniform over the hypercube — includes empty/full coalitions (rare).
    bits = torch.randint(
        0, 2, (B, K), device=device, generator=generator, dtype=torch.int64,
    ).to(window_to_frame.dtype)
    # (B, K) @ (K, T) → (B, T) obs-per-frame indicators in {0,1}.
    obs_T = (bits @ window_to_frame) > 0.5                           # (B, T)
    obs = obs_T.view(B, T, 1, 1).expand(B, T, J, C).contiguous()
    return obs


def sample_joint_mask_uniform(
    B: int, T: int, J: int, C: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``B`` coalitions uniformly over ``{0,1}^J`` joints.

    Each batch item draws a bit vector ``z ∈ {0,1}^J``; observed joints are
    observed across *all* frames and coords (mirrors the spatial-player
    coalition structure the EC evaluator uses).
    """
    bits = torch.randint(
        0, 2, (B, J), device=device, generator=generator, dtype=torch.bool,
    )
    obs = bits.view(B, 1, J, 1).expand(B, T, J, C).contiguous()
    return obs


def sample_training_mask(
    B: int,
    T: int,
    J: int,
    C: int,
    device: torch.device,
    *,
    p_temporal: float = 0.40,
    p_spatial:  float = 0.40,
    p_element:  float = 0.20,
    generator:  Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample per-batch-element observation masks ``obs ∈ (B, T, J, C)``.

    For each batch item:

    * with prob ``p_temporal`` → keep a uniformly-random fraction of frames
      (same frames observed across all joints/coords);
    * with prob ``p_spatial``  → keep a uniformly-random fraction of joints
      (same joints observed across all frames/coords);
    * with prob ``p_element``  → per-(t,j,c) Bernoulli with a random p.

    The three probabilities must sum to 1.  The fraction of observed entries
    is drawn from ``U(0, 1)`` independently per batch item.
    """
    if abs((p_temporal + p_spatial + p_element) - 1.0) > 1e-6:
        raise ValueError("Mask-type probabilities must sum to 1.")
    g = generator
    kind = torch.rand(B, device=device, generator=g)
    frac = torch.rand(B, device=device, generator=g)            # U(0,1) per item

    obs = torch.empty(B, T, J, C, device=device, dtype=torch.bool)
    for b in range(B):
        f = float(frac[b].item())
        k = float(kind[b].item())
        if k < p_temporal:
            t_keep = torch.rand(T, device=device, generator=g) < f
            m = t_keep.view(1, T, 1, 1).expand(1, T, J, C)
        elif k < p_temporal + p_spatial:
            j_keep = torch.rand(J, device=device, generator=g) < f
            m = j_keep.view(1, 1, J, 1).expand(1, T, J, C)
        else:
            m = torch.rand(1, T, J, C, device=device, generator=g) < f
        obs[b] = m[0]
    return obs


# ---------------------------------------------------------------------------
# VAEAC subnets
# ---------------------------------------------------------------------------

class _EncoderHead(nn.Module):
    """Transformer + (mu, logvar) head producing a per-frame latent.

    Can optionally expose per-layer transformer outputs as ``memory`` for the
    decoder (Ivanov's :class:`MemoryLayer` equivalent).
    """

    def __init__(
        self,
        input_feat_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        d_latent: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_feat_dim, d_model)
        self.trunk = _TransformerTrunk(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout, max_len=max_len,
            use_memory=False,
        )
        self.mu_head = nn.Linear(d_model, d_latent)
        self.logvar_head = nn.Linear(d_model, d_latent)
        # Clamp logvar to keep KL well-behaved.
        self.apply(_init_linear)

    def forward(
        self,
        tok_feat: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        *,
        return_memory: bool = False,
    ) -> tuple:
        tok = self.in_proj(tok_feat)
        if return_memory:
            h, memory = self.trunk(
                tok, pad_mask, return_layer_outputs=True,
            )                                                   # h, list[T]
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(min=-8.0, max=4.0)
            return mu, logvar, memory
        else:
            h = self.trunk(tok, pad_mask)                        # type: ignore[assignment]
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(min=-8.0, max=4.0)
            return mu, logvar


class _Decoder(nn.Module):
    """Transformer decoder for ``p(x | z, x_obs, mask)`` — outputs raw
    distribution parameters; the final reshape/split into ``(μ, σ)`` or a
    scalar mean is handled by the head.
    """

    def __init__(
        self,
        input_feat_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        use_memory: bool = False,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_feat_dim, d_model)
        self.trunk = _TransformerTrunk(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout, max_len=max_len,
            use_memory=use_memory,
        )
        self.out_proj = nn.Linear(d_model, output_dim)
        self.apply(_init_linear)

    def forward(
        self,
        tok_feat: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        *,
        memory: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        tok = self.in_proj(tok_feat)
        h = self.trunk(tok, pad_mask, memory=memory)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# VAEAC
# ---------------------------------------------------------------------------

class VAEAC(nn.Module):
    """Per-frame VAEAC with transformer subnets.

    Parameters
    ----------
    n_joints, n_coords
        Skeleton geometry (default 17, 3).
    d_model, nhead, num_layers, ff_dim, dropout
        Shared transformer hyperparameters.  Defaults give ~1.8 M params per
        subnet at ``d_model=192``, ``num_layers=4``, ``ff_dim=768``.
    d_latent
        Per-frame latent dim.
    max_len
        Max sequence length supported by the positional encoding buffer.
    decoder_head
        ``"gaussian_scalar"`` (single learnable scalar σ, legacy) or
        ``"gaussian_ivanov"`` (per-feature heteroscedastic σ via softplus,
        faithful to Ivanov/Olsen).
    ivanov_min_sigma
        Lower clamp on σ for the Ivanov head; ignored by the scalar head.
    use_prior_memory
        If ``True``, route prior-encoder intermediate activations into the
        decoder at each transformer layer (Ivanov's ``MemoryLayer``).
    prior_reg_sigma_mu, prior_reg_sigma_sigma
        Coefficients for the latent-prior regulariser
        ``-μ²/(2 σ_μ²) + σ_σ·(log σ − σ)`` applied on the prior network
        outputs.  Defaults match Ivanov's ``sigma_mu=1e4, sigma_sigma=1e-4``.
    """

    def __init__(
        self,
        n_joints: int = 17,
        n_coords: int = 3,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        ff_dim: int = 768,
        dropout: float = 0.1,
        d_latent: int = 64,
        max_len: int = 512,
        *,
        decoder_head: str = "gaussian_scalar",
        ivanov_min_sigma: float = 1e-2,
        use_prior_memory: bool = False,
        prior_reg_sigma_mu:    float = 1e4,
        prior_reg_sigma_sigma: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.n_coords = n_coords
        self.input_dim = n_joints * n_coords                    # 51
        self.d_latent = d_latent
        self.decoder_head_name = decoder_head
        self.use_prior_memory = bool(use_prior_memory)
        self.prior_reg_sigma_mu = float(prior_reg_sigma_mu)
        self.prior_reg_sigma_sigma = float(prior_reg_sigma_sigma)

        # Token feature layouts per subnet:
        #   full encoder:  [x_flat, mask_flat]                         (2 JC)
        #   prior encoder: [(x * mask)_flat, mask_flat]                (2 JC)
        #   decoder:       [z, (x * mask)_flat, mask_flat]             (d_latent + 2 JC)
        feat_enc = 2 * self.input_dim
        feat_dec = d_latent + 2 * self.input_dim

        self.full_encoder = _EncoderHead(
            input_feat_dim=feat_enc,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout, d_latent=d_latent,
            max_len=max_len,
        )
        self.prior_encoder = _EncoderHead(
            input_feat_dim=feat_enc,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout, d_latent=d_latent,
            max_len=max_len,
        )

        # Output head first — we need its ``out_dim_per_feat`` to size the
        # decoder's final projection.
        self.head = make_head(decoder_head, min_sigma=ivanov_min_sigma)
        dec_out_dim = self.input_dim * self.head.out_dim_per_feat

        self.decoder = _Decoder(
            input_feat_dim=feat_dec,
            output_dim=dec_out_dim,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout,
            max_len=max_len,
            use_memory=self.use_prior_memory,
        )

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _encode_tok_full(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """[x_flat, mask_flat]."""
        B, T, J, C = x.shape
        x_flat = x.reshape(B, T, J * C)
        m_flat = obs.reshape(B, T, J * C).to(x.dtype)
        return torch.cat([x_flat, m_flat], dim=-1)

    def _encode_tok_prior(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """[(x*mask)_flat, mask_flat] — hidden entries zeroed."""
        B, T, J, C = x.shape
        obs_f = obs.to(x.dtype)
        x_masked = (x * obs_f).reshape(B, T, J * C)
        m_flat = obs_f.reshape(B, T, J * C)
        return torch.cat([x_masked, m_flat], dim=-1)

    def _decode_tok(
        self, z: torch.Tensor, x: torch.Tensor, obs: torch.Tensor,
    ) -> torch.Tensor:
        """[z, (x*mask)_flat, mask_flat]."""
        B, T, J, C = x.shape
        obs_f = obs.to(x.dtype)
        x_masked = (x * obs_f).reshape(B, T, J * C)
        m_flat = obs_f.reshape(B, T, J * C)
        return torch.cat([z, x_masked, m_flat], dim=-1)

    def _reshape_head_out(
        self, dec_flat: torch.Tensor, B: int, T: int, J: int, C: int,
    ) -> torch.Tensor:
        """Reshape decoder output ``(B, T, J*C*out_dim_per_feat)`` into a
        ``(B, T, J, C*out_dim_per_feat)`` layout that the head can parse.

        For ``out_dim_per_feat == 1`` (scalar head) this gives ``(B, T, J, C)``
        identical to the raw mean.  For ``out_dim_per_feat == 2`` (Ivanov
        head) this gives ``(B, T, J, 2C)`` where the head splits off means
        and scale params internally.
        """
        P = self.head.out_dim_per_feat
        return dec_flat.reshape(B, T, J, C * P)

    # ------------------------------------------------------------------
    # Prior regularisation (Ivanov VAEAC.prior_regularization)
    # ------------------------------------------------------------------

    def prior_regularization(
        self,
        mu_p: torch.Tensor,            # (B, T, d_latent)
        logvar_p: torch.Tensor,        # (B, T, d_latent)
        pad_mask: Optional[torch.Tensor],   # (B, T) bool
    ) -> torch.Tensor:
        """Return the *per-frame, per-latent-dim* regulariser value.

        Matches Ivanov's ``VAEAC.prior_regularization`` exactly:

            mu_reg    = -(μ²) / (2 · σ_μ²)
            sigma_reg = (log σ − σ) · σ_σ

        Summed over the latent dim, averaged over real frames.  Because the
        caller adds this as a loss term (positive = penalty), we return
        ``-(mu_reg + sigma_reg)``.
        """
        sigma_p = (0.5 * logvar_p).exp()
        mu_term = -(mu_p ** 2) / (2.0 * self.prior_reg_sigma_mu ** 2)
        sigma_term = (sigma_p.log() - sigma_p) * self.prior_reg_sigma_sigma
        reg_per = mu_term + sigma_term                        # (B, T, d_latent)
        if pad_mask is not None:
            pm = pad_mask.to(reg_per.dtype).unsqueeze(-1)     # (B, T, 1)
            denom = (pm.sum() * reg_per.shape[-1]).clamp(min=1.0)
            reg_mean = (reg_per * pm).sum() / denom
        else:
            reg_mean = reg_per.mean()
        # Return as a *penalty* (positive = bad), so negate.
        return -reg_mean

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def elbo(
        self,
        x: torch.Tensor,                      # (B, T, J, C)
        obs: torch.Tensor,                    # (B, T, J, C) bool, True=observed
        pad_mask: Optional[torch.Tensor] = None,   # (B, T) bool, True=real
        kl_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"Expected x (B, T, J, C); got {tuple(x.shape)}")
        if obs.shape != x.shape:
            raise ValueError(
                f"obs mask must match x shape {tuple(x.shape)}; got {tuple(obs.shape)}"
            )
        B, T, J, C = x.shape

        # ---- Encoders ----
        mu_q, logvar_q = self.full_encoder(
            self._encode_tok_full(x, obs), pad_mask,
        )
        if self.use_prior_memory:
            mu_p, logvar_p, prior_mem = self.prior_encoder(
                self._encode_tok_prior(x, obs), pad_mask, return_memory=True,
            )
        else:
            mu_p, logvar_p = self.prior_encoder(
                self._encode_tok_prior(x, obs), pad_mask,
            )
            prior_mem = None

        # ---- Latent sample from q ----
        z = _reparameterize(mu_q, logvar_q)                   # (B, T, d_latent)

        # ---- Decode ----
        dec_flat = self.decoder(
            self._decode_tok(z, x, obs), pad_mask, memory=prior_mem,
        )                                                      # (B, T, JC·P)
        head_out = self._reshape_head_out(dec_flat, B, T, J, C)

        # ---- Reconstruction NLL on HIDDEN entries only ----
        hid = ~obs
        if pad_mask is not None:
            hid = hid & pad_mask.unsqueeze(-1).unsqueeze(-1)
        recon_nll, head_stats = self.head.nll(x, head_out, hid)

        # ---- KL per frame ----
        kl_per = _gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)  # (B, T, d_latent)
        if pad_mask is not None:
            pm_f = pad_mask.to(kl_per.dtype).unsqueeze(-1)     # (B, T, 1)
            kl_per = kl_per * pm_f
            kl = kl_per.sum() / (pm_f.sum() * kl_per.shape[-1]).clamp(min=1.0)
        else:
            kl = kl_per.mean()

        # ---- Prior regularisation (Ivanov) ----
        prior_reg = self.prior_regularization(mu_p, logvar_p, pad_mask)

        loss = recon_nll + kl_weight * kl + prior_reg
        # NOTE: we return the live (grad-tracking) tensors so the caller can
        # recompose with a custom kl-weight / free-bits schedule.  Detached
        # copies are provided for logging.
        out: dict[str, torch.Tensor] = {
            "loss":                loss,
            "recon_nll":           recon_nll,
            "kl":                  kl,
            "prior_reg":           prior_reg,
            "loss_detached":       loss.detach(),
            "recon_nll_detached":  recon_nll.detach(),
            "kl_detached":         kl.detach(),
            "prior_reg_detached":  prior_reg.detach(),
        }
        out.update({k: v for k, v in head_stats.items()})
        return loss, out

    # ------------------------------------------------------------------
    # Inference — sample completions from the prior encoder + decoder
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_completions(
        self,
        x: torch.Tensor,                      # (B, T, J, C) in flow-space units
        obs: torch.Tensor,                    # (B, T, J, C) bool
        pad_mask: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return ``(n_samples * B, T, J, C)`` conditional completions.

        Observed entries are copied from ``x`` verbatim; hidden entries are
        filled by sampling from the head (e.g. ``μ + σ·ε`` for the Ivanov
        head, or ``μ + σ_scalar·ε`` for the scalar head).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x (B, T, J, C); got {tuple(x.shape)}")
        B, T, J, C = x.shape

        x_rep = x.repeat_interleave(n_samples, dim=0)          # (B*N, T, J, C)
        obs_rep = obs.repeat_interleave(n_samples, dim=0)
        pad_rep = (
            pad_mask.repeat_interleave(n_samples, dim=0)
            if pad_mask is not None else None
        )

        if self.use_prior_memory:
            mu_p, logvar_p, prior_mem = self.prior_encoder(
                self._encode_tok_prior(x_rep, obs_rep), pad_rep,
                return_memory=True,
            )
        else:
            mu_p, logvar_p = self.prior_encoder(
                self._encode_tok_prior(x_rep, obs_rep), pad_rep,
            )
            prior_mem = None

        if temperature == 0.0:
            z = mu_p
        else:
            z = mu_p + (0.5 * logvar_p).exp() * temperature * torch.randn_like(mu_p)

        dec_flat = self.decoder(
            self._decode_tok(z, x_rep, obs_rep), pad_rep, memory=prior_mem,
        )                                                     # (B*N, T, JC·P)
        head_out = self._reshape_head_out(
            dec_flat, B * n_samples, T, J, C,
        )

        x_recon = self.head.sample(head_out, temperature=temperature)  # (B*N, T, J, C)

        obs_f = obs_rep.to(x_rep.dtype)
        x_filled = x_rep * obs_f + x_recon * (1.0 - obs_f)
        return x_filled

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @torch.no_grad()
    def count_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only) or p.requires_grad
        )

    @torch.no_grad()
    def count_parameters_breakdown(self) -> dict[str, int]:
        """Structural breakdown of param counts (ignores requires_grad)."""
        return {
            "full_encoder":  sum(p.numel() for p in self.full_encoder.parameters()),
            "prior_encoder": sum(p.numel() for p in self.prior_encoder.parameters()),
            "decoder":       sum(p.numel() for p in self.decoder.parameters()),
            "output_head":   sum(p.numel() for p in self.head.parameters()),
            "total":         sum(p.numel() for p in self.parameters()),
        }
