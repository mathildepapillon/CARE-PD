"""attribution.py — OTFlow-SHAP axiomatic Shapley values via path integration.

Implements equation (7) of *Axiomatic On-Manifold Shapley via Optimal
Generative Flows* (Zhang et al., arXiv:2603.05093) for the CARE-PD stack:

.. math::

    \\Psi_i(f, x^*) \\;=\\; \\int_0^1 \\frac{\\partial f}{\\partial x_i}
        \\!\\bigl(\\gamma(t)\\bigr)\\, v_{\\theta,i}\\!\\bigl(\\gamma(t)\\bigr)\\, dt,
    \\qquad \\gamma(1) = x^*, \\;\\dot\\gamma = v_\\theta(\\gamma, t).

Discretised with the trapezoidal rule on ``K+1`` support points, this gives
per-feature attributions that sum to the classifier logit difference
``f(x^*) - f(gamma(0))`` up to quadrature error.

The algorithm runs in two passes:

1. **Backward ODE solve** (no-grad): integrate ``dx/dt = v_theta(x, t)`` from
   ``t = 1`` backward to ``t = 0`` starting at ``x_star``. Stores the full
   trajectory ``{x_k}_{k=0..K}`` on device (memory is
   ``(K+1) * (B, T, J, C)`` floats, small for realistic ``K``).
2. **Trapezoidal integration**: at each support point compute
   ``v_k = v_theta(x_k, t_k)`` under ``no_grad``, then open a fresh autograd
   graph on a *detached clone* of ``x_k`` and call
   ``g_k = autograd.grad(classifier_fn(x_k, ctx), x_k)`` — that keeps memory
   bounded by a single graph's worth of activations per step, not
   ``O(K)`` graphs in flight simultaneously.

Pelvis attribution is zeroed by default: since the flow's velocity at joint
0 lies inside a pelvis-centered subspace (``v_theta[..., 0, :] ≈ 0``) and the
adapter replaces joint 0 with the *fixed original pelvis trajectory*, any
residual joint-0 attribution is numerical noise. Setting ``zero_pelvis=False``
lets callers audit the leak size without modifying the output.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from flow_matching.solver import ODESolver
from flow_matching.utils.model_wrapper import ModelWrapper

from model.flow_matching import VelocityNet


# ---------------------------------------------------------------------------
# Mask-aware ModelWrapper so the ODE solver sees the pad mask
# ---------------------------------------------------------------------------

class _MaskedVelocityWrapper(ModelWrapper):
    """Wrap VelocityNet with a frozen pad mask suited for ODESolver.sample().

    ``ODESolver.sample`` calls ``model(x, t)`` without extras; we close over
    ``mask`` here so the transformer sees the right ``src_key_padding_mask``
    at every integration step.
    """

    def __init__(self, model: VelocityNet, mask: Optional[Tensor]) -> None:
        super().__init__(model)
        self._mask = mask

    def forward(self, x: Tensor, t: Tensor, **extras: Any) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim > 1:
            t = t.reshape(-1)
        return self.model(x, t, mask=self._mask)


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def compute_flow_shap(
    *,
    velocity_net: VelocityNet,
    classifier_fn: Callable[[Tensor, Dict[str, Any]], Tensor],
    x_star_flow: Tensor,                 # (B, T, 17, 3) pelvis-centered z-score
    ctx: Dict[str, Any],
    num_steps: int = 100,
    solver_method: str = "rk4",
    zero_pelvis: bool = True,
    return_trajectory: bool = False,
) -> Dict[str, Any]:
    """Compute OTFlow-SHAP attributions for a batch of flow-space clips.

    Args:
        velocity_net: Trained ``VelocityNet`` — weights are frozen and used
            only in ``eval`` mode.
        classifier_fn: ``(x_flow, ctx) -> (B,)`` differentiable scalar head.
            Typically built with :func:`classifier_adapter.build_classifier_fn`.
        x_star_flow: ``(B, T, 17, 3)`` flow-space clip whose attributions we
            want. Must be on ``velocity_net``'s device.
        ctx: side info forwarded to ``classifier_fn`` (at least
            ``{"pelvis_world", "mask"}``; see ``classifier_adapter``).
        num_steps: Number of ODE integration sub-steps ``K``. Paper default 50;
            we use 100 so trapezoidal completeness error stays <1 %.
        solver_method: Any method ``flow_matching.solver.ODESolver`` accepts
            (``"euler"``, ``"midpoint"``, ``"rk4"``, ``"dopri5"``, ...).
            Default ``"rk4"`` matches OTFlow-SHAP Sec. 5.1's numerical-precision
            target (median relative completeness residual <1 % at K=100).
        zero_pelvis: If ``True`` (default) force ``psi[..., 0, :] = 0`` after
            integration. The pre-zero values are returned under ``"pelvis_leak_abs"``
            for diagnostics.
        return_trajectory: If ``True`` also return the ``(K+1,)`` list of
            ``x_k`` tensors for debugging. Disabled by default to save memory.

    Returns:
        Dict with:

        - ``psi``: ``(B, T, 17, 3)`` per-feature attributions.
        - ``x0_hat``: ``(B, T, 17, 3)`` induced noise baseline ``gamma(0)``.
        - ``f_xstar``: ``(B,)`` logit at the data end of the path.
        - ``f_x0``:   ``(B,)`` logit at the noise end of the path.
        - ``completeness_residual_rel``: ``(B,)`` — see
          :func:`diagnostics.completeness_residual`.
        - ``pelvis_leak_abs``: sum of ``|psi[..., 0, :]|`` per clip before
          zeroing — expected to be tiny.
        - ``t_grid``: ``(K+1,)`` support times.
        - ``trajectory`` (optional): list of ``K+1`` clones of ``x_k``.
    """
    if x_star_flow.ndim != 4 or x_star_flow.shape[-2:] != (17, 3):
        raise ValueError(
            f"x_star_flow must be (B, T, 17, 3); got {tuple(x_star_flow.shape)}"
        )
    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2; got {num_steps}")

    device = x_star_flow.device
    dtype  = x_star_flow.dtype
    B, T, J, C = x_star_flow.shape
    K = int(num_steps)
    dt = 1.0 / K

    velocity_net.eval()
    mask = ctx["mask"].to(device=device)
    wrapper = _MaskedVelocityWrapper(velocity_net, mask.bool())
    odesolver = ODESolver(velocity_model=wrapper)

    # ------------------------------------------------------------------
    # 1. Backward ODE solve (no grad) — t: 1 → 0.
    # ``ODESolver.sample`` has a required positional ``step_size`` arg. We
    # set it to ``None`` so ``torchdiffeq`` falls back to using the
    # ``time_grid`` deltas as step sizes — this is the single source of
    # truth for step spacing and avoids a silent clash with ``time_grid``
    # if we also specified ``step_size=dt``.
    # ------------------------------------------------------------------
    time_grid_back = torch.linspace(1.0, 0.0, K + 1, device=device, dtype=dtype)
    with torch.no_grad():
        back_out = odesolver.sample(
            x_init=x_star_flow.detach(),
            step_size=None,
            method=solver_method,
            time_grid=time_grid_back,
            return_intermediates=True,
        )
    # Returned shape: (K+1, B, T, J, C) with index 0 = t=1, index K = t=0.
    if torch.is_tensor(back_out):
        x_traj_back = [back_out[i] for i in range(K + 1)]
    else:
        x_traj_back = list(back_out)
    if len(x_traj_back) != K + 1:
        raise RuntimeError(
            f"ODE solver returned {len(x_traj_back)} intermediates; expected {K + 1}"
        )

    # Reverse so index k corresponds to t_k = k/K (forward time).
    x_traj_fwd = list(reversed(x_traj_back))
    assert torch.allclose(x_traj_fwd[-1], x_star_flow.detach(), atol=1e-4), \
        "x_traj_fwd[-1] should equal x_star_flow (endpoint of backward solve)"
    x0_hat = x_traj_fwd[0].detach()

    t_grid = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 2. Per-support-point integrand: h_k = g_k ⊙ v_k.
    # ------------------------------------------------------------------
    def integrand_at(k: int) -> Tensor:
        """Compute g_k ⊙ v_k at support index k (B, T, J, C)."""
        x_k = x_traj_fwd[k].detach()
        t_k = t_grid[k].expand(B)

        with torch.no_grad():
            v_k = velocity_net(x_k, t_k, mask=mask.bool())

        # Fresh autograd graph on a clone of x_k.
        x_k_grad = x_k.clone().requires_grad_(True)
        logit_k = classifier_fn(x_k_grad, ctx)
        if logit_k.ndim != 1 or logit_k.shape[0] != B:
            raise ValueError(
                f"classifier_fn must return (B,) logits; got {tuple(logit_k.shape)}"
            )
        # grad_outputs = ones makes torch.autograd.grad compute per-sample
        # gradients summed only across the batch dim (independent samples).
        g_k, = torch.autograd.grad(
            outputs=logit_k.sum(),
            inputs=x_k_grad,
            create_graph=False,
            retain_graph=False,
        )
        return (g_k.detach() * v_k.detach()).to(dtype)

    # ------------------------------------------------------------------
    # 3. Trapezoidal accumulation: rolling two-point buffer.
    # ------------------------------------------------------------------
    psi = torch.zeros((B, T, J, C), device=device, dtype=dtype)
    h_prev = integrand_at(0)
    for k in range(1, K + 1):
        h_curr = integrand_at(k)
        psi = psi + 0.5 * (h_prev + h_curr) * dt
        h_prev = h_curr
    h_K = h_prev

    # ------------------------------------------------------------------
    # 4. Completeness bookkeeping.
    # ------------------------------------------------------------------
    with torch.no_grad():
        f_xstar = classifier_fn(x_star_flow.detach(), ctx).detach()
        f_x0    = classifier_fn(x0_hat.detach(),     ctx).detach()
    delta_f = f_xstar - f_x0                                               # (B,)

    # Real-frame mask for completeness: psi on padded frames is not meaningful,
    # the classifier ignores them anyway.
    m = mask.to(dtype=dtype)[..., None, None]                              # (B, T, 1, 1)
    psi_sum = (psi * m).sum(dim=(1, 2, 3))                                 # (B,)
    comp_res_rel = (psi_sum - delta_f).abs() / delta_f.abs().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # 5. Pelvis-leak bookkeeping + optional zeroing.
    # ------------------------------------------------------------------
    pelvis_leak_abs = psi[..., 0, :].abs().sum(dim=(1, 2))                # (B,)
    if zero_pelvis:
        psi = psi.clone()
        psi[..., 0, :] = 0.0

    out: Dict[str, Any] = {
        "psi":                        psi,
        "x0_hat":                     x0_hat,
        "f_xstar":                    f_xstar,
        "f_x0":                       f_x0,
        "completeness_residual_rel":  comp_res_rel,
        "pelvis_leak_abs":            pelvis_leak_abs,
        "t_grid":                     t_grid,
        "integrand_last":             h_K.detach(),
    }
    if return_trajectory:
        out["trajectory"] = [x.detach().clone() for x in x_traj_fwd]
    return out
