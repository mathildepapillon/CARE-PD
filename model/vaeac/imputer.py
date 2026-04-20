"""imputer.py — KernelSHAP-compatible conditional imputer backed by VAEAC.

Public contract mirrors :class:`model.flow_shap.imputer.FlowImputer` exactly
so any evaluator that already calls ``FlowImputer.sample_completions`` can
swap in a :class:`VAEACImputer` with a one-line change.

Key parity points with the flow-matching imputer
-------------------------------------------------
* Input layout is classifier-native ``(1, J, F, T)``.
* Supported ``coalition_mask`` shapes: ``(1, T)`` / ``(1, J)`` / ``(1, J, 1)``.
* Optional pelvis-centered z-score via ``stats_mean`` / ``stats_std``
  (``(J, C)`` each), standardised into VAEAC space before sampling and
  de-standardised back to classifier space after.
* Observed entries are *guaranteed* to be preserved bit-for-bit in the
  returned completions (VAEAC overwrites hidden entries only).
* Returns a Python list of ``n_samples`` completions each shaped
  ``(1, J, F, T)`` — identical to ``FlowImputer.sample_completions``.

There is **no** RePaint ODE loop here — VAEAC is an amortised posterior, so
drawing a completion is just (prior encoder → sample z → decoder), giving
a massive speed-up over the 50–100 velocity-net calls per imputation in the
flow imputer.  This is itself a useful empirical comparison point.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .vaeac import VAEAC


__all__ = ["VAEACImputer"]


class VAEACImputer:
    """Drop-in replacement for :class:`FlowImputer` backed by a trained :class:`VAEAC`.

    Parameters
    ----------
    vaeac:
        A trained :class:`VAEAC` module (in-place ``.to(device)`` on init).
    device:
        Torch device to run inference on.
    stats_mean, stats_std:
        Optional per-joint z-score statistics of shape ``(J, C)`` or
        ``(1, 1, J, C)``.  If provided, input is standardised before feeding
        into VAEAC and de-standardised afterwards.  Use identical stats to
        whatever the flow imputer uses for a like-for-like comparison.
    temperature:
        Latent-sampling temperature; ``1.0`` = full prior variance,
        ``0.0`` = deterministic mean-field output.  Defaults to ``1.0`` so
        KernelSHAP Monte-Carlo averaging gets proper posterior variance.
    """

    def __init__(
        self,
        vaeac: VAEAC,
        device: torch.device,
        *,
        stats_mean: Optional[Tensor] = None,
        stats_std:  Optional[Tensor] = None,
        temperature: float = 1.0,
    ) -> None:
        self._model = vaeac
        self._device = torch.device(device)
        self._temperature = float(temperature)
        self._model.to(self._device)
        self._model.eval()

        self._stats_mean = self._prep_stats(stats_mean)
        self._stats_std  = self._prep_stats(stats_std)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prep_stats(self, s: Optional[Tensor]) -> Optional[Tensor]:
        if s is None:
            return None
        if not torch.is_tensor(s):
            s = torch.as_tensor(s)
        if s.dim() == 2:                                         # (J, C)
            s = s.view(1, 1, s.shape[0], s.shape[1])
        elif s.dim() == 4 and s.shape[:2] == (1, 1):
            pass
        else:
            raise ValueError(
                f"stats tensor must be (J, C) or (1, 1, J, C); got {tuple(s.shape)}"
            )
        return s.to(device=self._device, dtype=torch.float32)

    @staticmethod
    def _broadcast_coalition_mask(
        cm: Tensor, N: int, T: int, J: int, F: int,
    ) -> Tensor:
        """Accept ``(1, T)`` / ``(1, J)`` / ``(1, J, 1)`` and return ``(N, T, J, F)``."""
        if cm.dim() == 2:
            B, D = cm.shape
            if B != 1:
                raise ValueError(
                    f"coalition_mask must have leading dim 1; got {tuple(cm.shape)}"
                )
            if D == T:
                return cm.view(1, T, 1, 1).expand(N, T, J, F).contiguous()
            if D == J:
                return cm.view(1, 1, J, 1).expand(N, T, J, F).contiguous()
            raise ValueError(
                f"coalition_mask last dim must be T={T} or J={J}; "
                f"got {tuple(cm.shape)}"
            )
        if cm.dim() == 3:
            if cm.shape == (1, J, 1):
                return cm.view(1, 1, J, 1).expand(N, T, J, F).contiguous()
            raise ValueError(
                f"3-D coalition_mask must be (1, J={J}, 1); got {tuple(cm.shape)}"
            )
        raise ValueError(
            f"Unsupported coalition_mask shape {tuple(cm.shape)}; "
            f"expected (1, T), (1, J), or (1, J, 1)."
        )

    # ------------------------------------------------------------------
    # Public API — matches FlowImputer.sample_completions verbatim
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_completions(
        self,
        x: Tensor,                                # (1, J, F, T) classifier layout
        y: Optional[Tensor],                      # ignored
        mask: Tensor,                             # (1, T) pad mask
        lengths: Optional[Tensor],                # ignored
        coalition_mask: Tensor,                   # (1, T) / (1, J) / (1, J, 1)
        n_samples: int = 20,
    ) -> list[Tensor]:
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                f"VAEACImputer expects (1, J, F, T); got {tuple(x.shape)}"
            )
        _, J, F, T = x.shape
        device = self._device
        x = x.to(device)
        pad = mask.to(device).bool()
        cm = coalition_mask.to(device).bool()

        # (1, J, F, T) -> (1, T, J, C) VAEAC layout.
        x1 = x.permute(0, 3, 1, 2).contiguous()

        # Optional z-score into VAEAC space.
        if self._stats_mean is not None and self._stats_std is not None:
            x1_n = (x1 - self._stats_mean) / self._stats_std
        else:
            x1_n = x1

        # Broadcast coalition to an element-wise obs mask.
        # We need (1, T, J, C) for VAEAC; sample_completions handles the
        # ``n_samples`` repeat internally.
        obs_1 = self._broadcast_coalition_mask(cm, 1, T, J, F)

        pad_1 = pad if pad.shape[0] == 1 else pad[:1]

        completions = self._model.sample_completions(
            x=x1_n, obs=obs_1, pad_mask=pad_1,
            n_samples=n_samples, temperature=self._temperature,
        )                                                         # (N, T, J, C)

        # De-z-score back to classifier space.
        if self._stats_mean is not None and self._stats_std is not None:
            completions = completions * self._stats_std + self._stats_mean

        # (N, T, J, C) -> (N, J, C, T) classifier layout.
        actor = completions.permute(0, 2, 3, 1).contiguous()
        return [actor[i : i + 1] for i in range(n_samples)]

    # ------------------------------------------------------------------
    # Batched API — matches FlowImputer.sample_completions_batched
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_completions_batched(
        self,
        x: Tensor,                              # (1, J, F, T)
        mask: Tensor,                           # (1, T) pad mask
        coalition_masks: Tensor,                # (B, J) spatial
        n_samples: int = 20,
    ) -> Tensor:
        """Batched imputation over ``B`` spatial coalitions, ``(B, n_samples, J, F, T)``."""
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                f"VAEACImputer expects (1, J, F, T); got {tuple(x.shape)}"
            )
        if coalition_masks.dim() != 2:
            raise ValueError(
                f"coalition_masks must be (B, J); got {tuple(coalition_masks.shape)}"
            )
        _, J, F, T = x.shape
        B, Jc = coalition_masks.shape
        if Jc != J:
            raise ValueError(
                f"coalition_masks J dim {Jc} != x J dim {J}"
            )
        device = self._device
        x = x.to(device)
        pad = mask.to(device).bool()
        cm = coalition_masks.to(device).bool()

        # Classifier -> VAEAC layout.
        x1 = x.permute(0, 3, 1, 2).contiguous()                   # (1, T, J, C)
        if self._stats_mean is not None and self._stats_std is not None:
            x1 = (x1 - self._stats_mean) / self._stats_std

        # Expand x to (B, T, J, C) — same x for every coalition.
        x_B = x1.expand(B, -1, -1, -1).contiguous()               # (B, T, J, C)
        obs_B = (
            cm.view(B, 1, J, 1).expand(B, T, J, F).contiguous()
        )
        pad_B = pad.expand(B, -1).contiguous() if pad.shape[0] == 1 else pad

        completions = self._model.sample_completions(
            x=x_B, obs=obs_B, pad_mask=pad_B,
            n_samples=n_samples, temperature=self._temperature,
        )                                                           # (B*N, T, J, C)

        if self._stats_mean is not None and self._stats_std is not None:
            completions = completions * self._stats_std + self._stats_mean

        # (B*N, T, J, C) -> (B*N, J, C, T) -> (B, N, J, F, T)
        actor = completions.permute(0, 2, 3, 1).contiguous()
        return actor.view(B, n_samples, J, F, T)
