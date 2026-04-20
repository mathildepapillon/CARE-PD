"""heads.py — Pluggable decoder output heads for VAEAC.

Two heads are provided:

* :class:`GaussianScalarHead` — the original single learnable-scalar σ Gaussian
  that ``model.vaeac.vaeac.VAEAC`` shipped with.  Kept as the ``gaussian_scalar``
  config option for backward compatibility / ablation.

* :class:`GaussianIvanovHead` — faithful re-implementation of the continuous
  output head from Ivanov et al. 2019 (``tigvarts/vaeac`` repo,
  ``prob_utils.GaussianLoss``) and inherited by Olsen et al. 2022.  For each
  continuous coordinate the decoder emits ``(μ, σ_param)``; the scale is
  ``σ = softplus(σ_param).clamp_min(min_sigma)`` (``min_sigma=1e-2`` for the
  Gaussian-only case).  The NLL is ``-Normal(μ, σ).log_prob(x) * hid_mask``.
  Sampling is ``μ + temperature · σ · ε``.

Both heads expose an identical public API so :class:`model.vaeac.vaeac.VAEAC`
can swap them without knowing the head type:

* ``out_dim_per_feat`` — number of decoder outputs per scalar feature.
* ``nll(x_target, head_out, hid_mask)`` — per-element NLL, averaged over
  hidden entries only.  Returns ``(nll_scalar, stats_dict)``.
* ``sample(head_out, temperature=1.0)`` — sample ``x`` of the same spatial
  shape as ``x_target``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["GaussianScalarHead", "GaussianIvanovHead", "make_head"]


# ---------------------------------------------------------------------------
# Scalar-σ head (current behaviour)
# ---------------------------------------------------------------------------

class GaussianScalarHead(nn.Module):
    """Diagonal Gaussian with a single learned scalar log σ shared across all
    features.  This is the head the original ``VAEAC`` class used before the
    Olsen-style refactor.  Kept for ablation.
    """

    out_dim_per_feat: int = 1

    def __init__(self) -> None:
        super().__init__()
        self.log_sigma_x = nn.Parameter(torch.zeros(1))

    def nll(
        self,
        x_target: torch.Tensor,          # (B, T, J, C)
        head_out: torch.Tensor,          # (B, T, J, C)   — decoder mean
        hid_mask: torch.Tensor,          # (B, T, J, C) bool, True = hidden
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if head_out.shape != x_target.shape:
            raise ValueError(
                f"head_out shape {tuple(head_out.shape)} != x_target "
                f"shape {tuple(x_target.shape)}"
            )
        sigma2 = (2.0 * self.log_sigma_x).exp()
        nll_per = (
            0.5 * ((x_target - head_out) ** 2) / sigma2
            + self.log_sigma_x
            + 0.5 * math.log(2.0 * math.pi)
        )
        hid_f = hid_mask.to(nll_per.dtype)
        n_hid = hid_f.sum().clamp(min=1.0)
        nll = (nll_per * hid_f).sum() / n_hid
        return nll, {
            "log_sigma_x":       self.log_sigma_x.squeeze(),
            "mean_sigma":        self.log_sigma_x.exp().squeeze(),
        }

    def sample(
        self, head_out: torch.Tensor, temperature: float = 1.0,
    ) -> torch.Tensor:
        sigma = self.log_sigma_x.exp()
        if temperature == 0.0:
            return head_out
        return head_out + temperature * sigma * torch.randn_like(head_out)

    def regularization(self, head_out: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=head_out.device, dtype=head_out.dtype)


# ---------------------------------------------------------------------------
# Ivanov / Olsen heteroscedastic diagonal Gaussian
# ---------------------------------------------------------------------------

class GaussianIvanovHead(nn.Module):
    """Per-feature heteroscedastic diagonal Gaussian — faithful replication of
    ``prob_utils.GaussianLoss`` from the Ivanov / Olsen reference code.

    The decoder is expected to emit ``2 · n_features`` numbers per frame.
    The first half are interpreted as means ``μ``; the second half are
    passed through ``softplus`` and clamped at ``min_sigma`` to give scales
    ``σ``.  NLL = ``-Normal(μ, σ).log_prob(x) * hid_mask``, averaged over
    hidden entries.

    Notes
    -----
    * ``min_sigma=1e-2`` matches the Gaussian-only regime in
      ``train.py`` of Ivanov's repo.  For mixed continuous+categorical data
      they use ``1e-4`` — we don't need that here.
    * No mixture: the published Olsen/Ivanov VAEAC uses a single Gaussian
      component per feature.  See
      ``https://github.com/tigvarts/vaeac/blob/master/prob_utils.py``.
    * Reshapes the last two dims ``(J, C) → JC`` internally so it slots
      into our ``(B, T, J, C)`` layout without the caller having to think
      about where the extra channel dimension lives.
    """

    def __init__(self, min_sigma: float = 1e-2) -> None:
        super().__init__()
        self.min_sigma = float(min_sigma)

    @property
    def out_dim_per_feat(self) -> int:
        return 2

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def _parse(self, head_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split ``(..., 2C)`` into ``(μ, σ)`` each ``(..., C)``.

        Uses the same convention as ``normal_parse_params`` in Ivanov's
        ``prob_utils.py`` — first half mean, second half pre-softplus scale.
        """
        if head_out.shape[-1] % 2 != 0:
            raise ValueError(
                f"GaussianIvanovHead expects trailing dim divisible by 2; "
                f"got {tuple(head_out.shape)}"
            )
        d = head_out.shape[-1] // 2
        mu = head_out[..., :d]
        sigma_param = head_out[..., d:]
        sigma = F.softplus(sigma_param).clamp_min(self.min_sigma)
        return mu, sigma

    # ------------------------------------------------------------------
    # NLL / sample
    # ------------------------------------------------------------------

    def nll(
        self,
        x_target: torch.Tensor,          # (B, T, J, C)
        head_out: torch.Tensor,          # (B, T, J, 2C) — raw decoder output
        hid_mask: torch.Tensor,          # (B, T, J, C) bool
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mu, sigma = self._parse(head_out)
        if mu.shape != x_target.shape:
            raise ValueError(
                f"GaussianIvanovHead: parsed μ shape {tuple(mu.shape)} must "
                f"equal x_target shape {tuple(x_target.shape)}"
            )
        # Normal.log_prob per element:
        # -0.5 * ((x-μ)/σ)^2 - log σ - 0.5·log(2π)
        inv_sigma = 1.0 / sigma
        nll_per = (
            0.5 * ((x_target - mu) * inv_sigma) ** 2
            + sigma.log()
            + 0.5 * math.log(2.0 * math.pi)
        )
        hid_f = hid_mask.to(nll_per.dtype)
        n_hid = hid_f.sum().clamp(min=1.0)
        nll = (nll_per * hid_f).sum() / n_hid
        # Stats for logging — computed on hidden entries only.
        with torch.no_grad():
            sigma_hid = (sigma * hid_f).sum() / n_hid
            log_sigma_hid = (sigma.log() * hid_f).sum() / n_hid
        return nll, {
            "mean_sigma":     sigma_hid.detach(),
            "mean_log_sigma": log_sigma_hid.detach(),
        }

    def sample(
        self, head_out: torch.Tensor, temperature: float = 1.0,
    ) -> torch.Tensor:
        mu, sigma = self._parse(head_out)
        if temperature == 0.0:
            return mu
        return mu + temperature * sigma * torch.randn_like(mu)

    def regularization(self, head_out: torch.Tensor) -> torch.Tensor:
        # Ivanov has no output-head regulariser for continuous features —
        # the only regulariser is on the *latent* prior, which is added
        # separately by VAEAC.prior_regularization.
        return torch.zeros((), device=head_out.device, dtype=head_out.dtype)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_head(name: str, **kwargs) -> nn.Module:
    """Return a configured head by name."""
    name = name.lower()
    if name in ("gaussian_scalar", "scalar"):
        return GaussianScalarHead()
    if name in ("gaussian_ivanov", "ivanov", "olsen"):
        return GaussianIvanovHead(min_sigma=kwargs.get("min_sigma", 1e-2))
    raise ValueError(
        f"Unknown decoder_head '{name}'. "
        f"Expected 'gaussian_scalar' or 'gaussian_ivanov'."
    )
