"""imputer.py — shared RePaint-style conditional flow imputer.

Both the synthetic and real-world SHAP pipelines need a way to condition a
trained flow-matching velocity field on an *observed* subset of a clip and
draw completions for the hidden subset.  This module provides the single
implementation used by every caller.

Compared to the original ``FlowSyntheticWrapper`` (in
``evaluate_shap_synthetic.py``) this generalised version:

* Accepts optional ``stats_mean`` / ``stats_std`` per-joint pelvis-centered
  z-score statistics (``(J, C)``) so real-world clips can be normalised /
  de-normalised around the ODE.  When ``stats_mean=0, stats_std=1`` (the
  synthetic default) the behaviour is bit-identical to the original wrapper.
* Accepts ``coalition_mask`` in three shapes:

  - ``(1, T)``       temporal mask (observed time frames)
  - ``(1, J)``       spatial mask (observed joints; all frames, all coords)
  - ``(1, J, 1)``    spatial mask broadcast over coords (same as ``(1, J)``
    but explicit; a single coalition player = one full joint across all
    coords and all frames)

  The ``(1, J, 1)`` shape is what the imputer-based KernelSHAP driver in
  ``scripts/compute_flow_shap_imputer.py`` produces for 17-joint spatial
  Shapley on real-world clips.
* Exposes ``sample_completions(x, y, mask, lengths, coalition_mask,
  n_samples)`` with the exact same signature used everywhere else in the
  codebase so existing evaluators can drop this in with a one-line import.
* Defaults to ``solver="midpoint"``, ``num_steps=100`` — the
  synthetic-validated settings that yielded the best benchmark numbers.

Algorithm (per clip)::

    x0 ~ N(0, I)                                  # n_samples noise starts
    for k in range(K):
        # 1. velocity step (midpoint or euler)
        v1 = v_theta(x_k,   t_k)
        if midpoint:
            x_mid = x_k + 0.5 dt v1
            x_mid = harmonize(x_mid, t_mid)       # CondOT path on observed
            v2    = v_theta(x_mid, t_mid)
            x_next = x_k + dt v2
        else:
            x_next = x_k + dt v1
        # 2. per-step RePaint harmonisation: observed entries follow the
        #    same linear CondOT path the net was trained on.
        x_k = harmonize(x_next, t_{k+1})
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor


__all__ = ["FlowImputer"]


class FlowImputer:
    """RePaint-style conditional flow imputer.

    Usage:
        >>> imp = FlowImputer(velocity_net, device, stats_mean=mean,
        ...                   stats_std=std, num_steps=100, solver="midpoint")
        >>> completions = imp.sample_completions(
        ...     x=x_jft, y=None, mask=pad_mask, lengths=None,
        ...     coalition_mask=obs_mask, n_samples=20,
        ... )

    The caller is expected to pass ``x`` in the classifier's native
    ``(1, J, F, T)`` layout (matches ActorSHAP, LstmVAE, the existing
    synthetic wrapper).  Internally we permute to ``(1, T, J, C)`` which is
    the VelocityNet layout.
    """

    _ALLOWED_SOLVERS = ("euler", "midpoint")

    def __init__(
        self,
        velocity_net: torch.nn.Module,
        device: torch.device,
        *,
        stats_mean: Optional[Tensor] = None,
        stats_std:  Optional[Tensor] = None,
        num_steps: int = 100,
        solver: str = "midpoint",
    ) -> None:
        if solver not in self._ALLOWED_SOLVERS:
            raise ValueError(
                f"solver must be one of {self._ALLOWED_SOLVERS}; got {solver!r}"
            )
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2; got {num_steps}")
        self._model = velocity_net
        self._device = torch.device(device)
        self._num_steps = int(num_steps)
        self._solver = solver
        self._model.to(self._device)
        self._model.eval()

        # Cache per-joint z-score stats as (1, 1, J, C) for broadcasting.
        # ``None`` is treated as identity (synthetic default).
        self._stats_mean = self._prep_stats(stats_mean)
        self._stats_std  = self._prep_stats(stats_std)

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_completions(
        self,
        x: Tensor,                                # (1, J, F, T)
        y: Optional[Tensor],                      # ignored
        mask: Tensor,                             # (1, T) pad mask
        lengths: Optional[Tensor],                # ignored
        coalition_mask: Tensor,                   # (1, T) / (1, J) / (1, J, 1)
        n_samples: int = 20,
    ) -> list[Tensor]:
        """Return ``n_samples`` conditional completions in ``(1, J, F, T)`` layout.

        The input ``x`` is assumed to be in *classifier* space (what
        ActorSHAP / LstmVAE use).  When this imputer was built with
        ``stats_mean`` / ``stats_std`` the input is first standardised into
        flow space, the ODE runs in flow space, then the result is
        de-standardised back to classifier space before returning.
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                f"FlowImputer expects (1, J, F, T); got {tuple(x.shape)}"
            )
        _, J, F, T = x.shape
        device = self._device
        x = x.to(device)
        cm = coalition_mask.to(device).bool()
        pad = mask.to(device).bool()

        # ``(1, J, F, T)`` classifier layout  ->  ``(1, T, J, C)`` flow layout.
        x1 = x.permute(0, 3, 1, 2).contiguous()                  # (1, T, J, C)

        # Optional z-score into flow space (pelvis-centered normalisation).
        if self._stats_mean is not None and self._stats_std is not None:
            x1 = (x1 - self._stats_mean) / self._stats_std
        x1_n = x1.expand(n_samples, -1, -1, -1).contiguous()     # (N, T, J, C)

        # Build observed-entry mask ``(N, T, J, C)`` from any of the
        # supported ``coalition_mask`` shapes.
        obs_mask = self._broadcast_coalition_mask(cm, n_samples, T, J, F)

        # Noise starts.
        x0 = torch.randn(n_samples, T, J, F, device=device, dtype=x1_n.dtype)

        pad_for_net = (
            pad.expand(n_samples, -1).contiguous()
            if pad.shape[0] == 1 else pad
        )

        K = self._num_steps
        dt = 1.0 / K

        def v_fn(xk: Tensor, t: float) -> Tensor:
            t_batch = torch.full((n_samples,), t, device=device, dtype=xk.dtype)
            return self._model(xk, t_batch, mask=pad_for_net)     # (N, T, J, C)

        def harmonize(xk: Tensor, t: float) -> Tensor:
            cond_path = (1.0 - t) * x0 + t * x1_n
            return torch.where(obs_mask, cond_path, xk)

        # At t=0 the CondOT path reduces to x0 everywhere; observed entries
        # already agree with the path so no initial harmonise is needed.
        x_k = x0.clone()
        for k in range(K):
            t_k = k * dt
            t_mid = t_k + 0.5 * dt
            t_next = (k + 1) * dt
            if self._solver == "euler":
                v1 = v_fn(x_k, t_k)
                x_next = x_k + dt * v1
            else:                                                 # midpoint (RK2)
                v1 = v_fn(x_k, t_k)
                x_mid = x_k + 0.5 * dt * v1
                x_mid = harmonize(x_mid, t_mid)
                v2 = v_fn(x_mid, t_mid)
                x_next = x_k + dt * v2
            x_k = harmonize(x_next, t_next)

        # De-z-score back to classifier space.
        if self._stats_mean is not None and self._stats_std is not None:
            x_k = x_k * self._stats_std + self._stats_mean

        # ``(N, T, J, C)`` flow layout  ->  ``(N, J, C, T)`` classifier layout.
        actor = x_k.permute(0, 2, 3, 1).contiguous()
        return [actor[i : i + 1] for i in range(n_samples)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _broadcast_coalition_mask(
        cm: Tensor, N: int, T: int, J: int, F: int,
    ) -> Tensor:
        """Accept ``(1, T)`` / ``(1, J)`` / ``(1, J, 1)`` and return ``(N, T, J, F)``.

        Raises ``ValueError`` on any other shape so upstream bugs surface
        immediately rather than silently broadcasting incorrectly.
        """
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
