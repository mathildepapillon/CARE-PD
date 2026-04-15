"""
Recurrent VAE for CARE-PD pose sequences.

Input convention
----------------
All public methods operate on tensors of shape ``(B, T, input_dim)`` where
``input_dim = n_joints * 3`` (default 51 for 17 H36M joints).

Architecture
------------
Encoder
    Bidirectional LSTM → concat last-step hidden states (both directions)
    → FC → two heads for ``mu`` and ``log_var``

Decoder
    ``z`` is projected to hidden_dim, repeated T times, then processed
    by a deep unidirectional LSTM stack.  A final single-layer LSTM
    maps directly to the pose dimension per timestep.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

N_JOINTS = 17
FEAT_PER_JOINT = 3


class LstmEncoder(nn.Module):
    """Bidirectional LSTM encoder that maps a pose sequence to (mu, log_var).

    Parameters
    ----------
    input_dim:
        Flattened pose dimensionality per timestep (17 joints × 3 = 51).
    hidden_dim:
        Hidden size per direction of the BiLSTM.
    num_layers:
        Number of stacked LSTM layers.
    latent_dim:
        Dimensionality of the latent space.
    dropout:
        Dropout probability applied between LSTM layers (0 disables).
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Collapse both directions into a single vector
        fc_in = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(fc_in // 2, latent_dim)
        self.fc_log_var = nn.Linear(fc_in // 2, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : ``(B, T, input_dim)``

        Returns
        -------
        mu, log_var : each ``(B, latent_dim)``
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, hidden_dim) — take last layer, both dirs
        fwd = h_n[-2]  # (B, hidden_dim)
        bwd = h_n[-1]  # (B, hidden_dim)
        h = torch.cat([fwd, bwd], dim=-1)  # (B, hidden_dim * 2)
        h = self.fc(h)
        return self.fc_mu(h), self.fc_log_var(h)


class MaskedLstmEncoder(nn.Module):
    """Masked encoder that outputs a Gaussian mixture distribution.

    Same BiLSTM body as :class:`LstmEncoder`, but with two additions:

    1.  **Learnable mask tokens** — unobserved joints (spatial) or frames
        (temporal) are replaced with per-player learned embeddings before
        the BiLSTM processes the sequence.
    2.  **Mixture-of-Gaussians head** — outputs ``K`` component means/logvars
        plus log-mixing-weights, so the posterior can be multimodal when
        information is missing.

    Parameters
    ----------
    input_dim:
        Flattened pose dimensionality (``17 * 3 = 51``).
    hidden_dim:
        BiLSTM hidden size per direction.
    num_layers:
        Number of stacked LSTM layers.
    latent_dim:
        Latent space dimensionality.
    dropout:
        Dropout between LSTM layers.
    n_mix:
        Number of Gaussian mixture components K.
    n_joints:
        Number of skeleton joints (for spatial mask tokens).
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 2,
        latent_dim: int = 256,
        dropout: float = 0.1,
        n_mix: int = 10,
        n_joints: int = N_JOINTS,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mix = n_mix
        self.n_joints = n_joints

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        fc_in = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(inplace=True),
        )
        mid = fc_in // 2

        self.fc_mu = nn.Linear(mid, n_mix * latent_dim)
        self.fc_logvar = nn.Linear(mid, n_mix * latent_dim)
        self.fc_pi = nn.Linear(mid, n_mix)

        self.mask_token_spatial = nn.Parameter(torch.zeros(n_joints, FEAT_PER_JOINT))
        self.mask_token_temporal = nn.Parameter(torch.zeros(input_dim))

    def forward(
        self, x: Tensor, coalition_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : ``(B, T, input_dim)``
        coalition_mask : ``(B, n_joints)`` bool for spatial or ``(B, T)``
            bool for temporal.  ``True`` = observed.

        Returns
        -------
        log_pi  : ``(B, K)``
        mu      : ``(B, K, latent_dim)``
        logvar  : ``(B, K, latent_dim)``
        """
        x_masked = self._apply_mask(x, coalition_mask)

        _, (h_n, _) = self.lstm(x_masked)
        fwd = h_n[-2]
        bwd = h_n[-1]
        h = torch.cat([fwd, bwd], dim=-1)
        h = self.fc(h)

        B = x.size(0)
        mu = self.fc_mu(h).view(B, self.n_mix, self.latent_dim)
        logvar = self.fc_logvar(h).view(B, self.n_mix, self.latent_dim)
        log_pi = F.log_softmax(self.fc_pi(h), dim=-1)
        return log_pi, mu, logvar

    def _apply_mask(self, x: Tensor, coalition_mask: Tensor) -> Tensor:
        B, T, D = x.shape
        if coalition_mask.shape[-1] == self.n_joints:
            # Spatial: replace unobserved joints with learnable tokens
            x = x.view(B, T, self.n_joints, FEAT_PER_JOINT)
            mask_vals = self.mask_token_spatial[None, None].expand(B, T, -1, -1)
            obs = coalition_mask[:, None, :, None].expand_as(x)
            x = torch.where(obs, x, mask_vals)
            return x.reshape(B, T, D)
        else:
            # Temporal: replace unobserved frames with learnable token
            mask_vals = self.mask_token_temporal[None, None].expand(B, T, -1)
            obs = coalition_mask[:, :, None].expand(B, T, D)
            return torch.where(obs, x, mask_vals)

    def build_mixture(
        self, log_pi: Tensor, mu: Tensor, logvar: Tensor,
    ) -> MixtureSameFamily:
        """Build a ``torch.distributions`` mixture from the encoder output."""
        mix = Categorical(logits=log_pi)
        comp = Independent(Normal(mu, torch.exp(0.5 * logvar)), 1)
        return MixtureSameFamily(mix, comp)

    def sample(
        self, log_pi: Tensor, mu: Tensor, logvar: Tensor,
    ) -> Tensor:
        """Draw one sample per batch element from the mixture (reparameterised)."""
        # Gumbel-softmax select a component, then sample from that Gaussian
        idx = Categorical(logits=log_pi).sample()          # (B,)
        B = mu.size(0)
        mu_k = mu[torch.arange(B, device=mu.device), idx]  # (B, D)
        lv_k = logvar[torch.arange(B, device=mu.device), idx]
        std = torch.exp(0.5 * lv_k)
        return mu_k + std * torch.randn_like(std)


def kl_full_vs_mixture(
    mu_full: Tensor,
    logvar_full: Tensor,
    log_pi: Tensor,
    mu_mix: Tensor,
    logvar_mix: Tensor,
    n_mc_samples: int = 10,
) -> Tensor:
    """Monte-Carlo estimate of KL(q_φ ‖ r_ψ).

    q_φ = N(mu_full, σ_full²)  (full encoder, single Gaussian)
    r_ψ = Σ_k π_k N(mu_k, σ_k²)  (masked encoder, mixture)

    Uses reparameterised samples from q for a low-variance gradient estimate.
    """
    q = Independent(Normal(mu_full, torch.exp(0.5 * logvar_full)), 1)
    mix = Categorical(logits=log_pi)
    comp = Independent(Normal(mu_mix, torch.exp(0.5 * logvar_mix)), 1)
    r = MixtureSameFamily(mix, comp)

    z = q.rsample((n_mc_samples,))     # (S, B, D)
    log_q = q.log_prob(z)              # (S, B)
    log_r = r.log_prob(z)              # (S, B)
    kl = (log_q - log_r).mean()        # scalar
    return torch.clamp(kl, min=0.0)


class LstmDecoder(nn.Module):
    """Deep unidirectional LSTM decoder.

    ``z`` is projected to ``hidden_dim``, repeated T times, then processed by
    a deep LSTM stack.  The LSTM's recurrence across many layers generates
    temporal variation from the constant input — deeper stacks produce richer
    dynamics.  A final single-layer LSTM maps directly to ``output_dim`` per
    timestep (following the pirounet design).

    Parameters
    ----------
    latent_dim:
        Dimensionality of the latent space (same as encoder output).
    hidden_dim:
        LSTM hidden size.
    num_layers:
        Number of stacked LSTM layers in the main block.
    output_dim:
        Flattened pose dimensionality per timestep (17 joints × 3 = 51).
    seq_len:
        Target sequence length T.
    dropout:
        Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        output_dim: int = 51,
        seq_len: int = 80,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.fc_in = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.lstm_out = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=output_dim,
            batch_first=True,
        )

    def forward(self, z: Tensor, seq_len: int | None = None, **_kw) -> Tensor:
        """
        Parameters
        ----------
        z : ``(B, latent_dim)``
        seq_len : override the default sequence length.

        Returns
        -------
        recon : ``(B, T, output_dim)``
        """
        T = seq_len if seq_len is not None else self.seq_len
        B = z.size(0)

        h = self.fc_in(z)                                        # (B, hidden_dim)
        h = h.unsqueeze(1).expand(B, T, -1)                      # (B, T, hidden_dim)

        h, _ = self.lstm(h)                                       # (B, T, hidden_dim)
        out, _ = self.lstm_out(h)                                 # (B, T, output_dim)
        return out


class LstmVAE(nn.Module):
    """Recurrent VAE with LSTM encoder and decoder.

    When ``n_mix > 0`` a :class:`MaskedLstmEncoder` is created alongside
    the full encoder.  Call :meth:`forward_masked` to obtain the mixture
    parameters from a masked input, and use :func:`kl_full_vs_mixture`
    to regularise the masked encoder against the full encoder.

    Parameters
    ----------
    input_dim:
        Flattened pose dimensionality (default 51 = 17 joints × 3).
    encoder_hidden_dim / decoder_hidden_dim:
        LSTM hidden sizes for the encoder and decoder (can differ).
    encoder_num_layers / decoder_num_layers:
        Number of LSTM layers.
    latent_dim:
        Latent space dimensionality.
    seq_len:
        Canonical sequence length; stored for convenience (decoder uses it).
    dropout:
        Shared dropout probability.
    n_mix:
        Number of Gaussian mixture components for the masked encoder.
        Set to 0 to disable the masked encoder entirely.
    """

    def __init__(
        self,
        input_dim: int = 51,
        encoder_hidden_dim: int = 256,
        encoder_num_layers: int = 2,
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 4,
        latent_dim: int = 256,
        seq_len: int = 80,
        dropout: float = 0.1,
        n_mix: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder = LstmEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = LstmDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            output_dim=input_dim,
            seq_len=seq_len,
            dropout=dropout,
        )

        self.masked_encoder: MaskedLstmEncoder | None = None
        if n_mix > 0:
            self.masked_encoder = MaskedLstmEncoder(
                input_dim=input_dim,
                hidden_dim=encoder_hidden_dim,
                num_layers=encoder_num_layers,
                latent_dim=latent_dim,
                dropout=dropout,
                n_mix=n_mix,
            )

    # ------------------------------------------------------------------
    # Core VAE operations
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mu, log_var)`` for input ``x: (B, T, input_dim)``."""
        return self.encoder(x)

    def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample z using the reparameterisation trick (no-op at eval time)."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, seq_len: int | None = None) -> Tensor:
        """Reconstruct a sequence from latent ``z: (B, latent_dim)``."""
        return self.decoder(z, seq_len=seq_len)

    def forward(self, x: Tensor, **_kw) -> tuple[Tensor, Tensor, Tensor]:
        """Full encode → reparameterise → decode pass.

        Parameters
        ----------
        x : ``(B, T, input_dim)``

        Returns
        -------
        recon : ``(B, T, input_dim)``
        mu    : ``(B, latent_dim)``
        log_var : ``(B, latent_dim)``
        """
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decode(z, seq_len=x.size(1))
        return recon, mu, log_var

    def forward_masked(
        self, x: Tensor, coalition_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the masked encoder on ``x`` with the given ``coalition_mask``.

        Returns ``(log_pi, mu, logvar)`` of the Gaussian mixture.
        Raises ``RuntimeError`` if the masked encoder was not created.
        """
        if self.masked_encoder is None:
            raise RuntimeError("Masked encoder not initialised (n_mix=0).")
        return self.masked_encoder(x, coalition_mask)

    def sample_masked(
        self, x: Tensor, coalition_mask: Tensor,
    ) -> Tensor:
        """Sample a latent from the masked encoder's mixture and decode."""
        log_pi, mu, logvar = self.forward_masked(x, coalition_mask)
        z = self.masked_encoder.sample(log_pi, mu, logvar)
        return self.decode(z, seq_len=x.size(1))
