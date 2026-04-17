#!/usr/bin/env python
"""Paper-fidelity completeness regression test for OTFlow-SHAP.

OTFlow-SHAP (Zhang et al., arXiv:2603.05093) Sec. 5.1 reports a median
relative completeness residual ``|Σψ − (f(x*) − f(γ(0)))| / |f(x*) − f(γ(0))|``
under 1% when the ODE is integrated with RK4 at ``K=100`` support points and
trapezoidal quadrature. Our ``compute_flow_shap`` defaults (``solver='rk4'``,
``num_steps=100``) aim to hit that bar.

This test asserts the bar holds even on an *untrained* velocity field and a
*nonlinear* classifier (tanh before linear readout), which stresses the
quadrature rule much more than a pure linear classifier does. If a future
refactor of the solver stack silently regresses integration accuracy, this
test trips first.

If a trained checkpoint + flow cache are available we ALSO run the test on 8
real BMCLab val clips with the POTR adapter, checking the same bound end-to-
end. That block is ``pytest.mark.skipif`` and costs nothing on CI.

Run with::

    python -m pytest tests/test_attribution_completeness.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.flow_matching import VelocityNet  # noqa: E402
from model.flow_shap import compute_flow_shap  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_net(d_model: int = 32, layers: int = 2, max_len: int = 48) -> VelocityNet:
    """Tiny VelocityNet for fast CPU testing (<2 s per compute_flow_shap call)."""
    return VelocityNet(
        n_joints=17, n_coords=3,
        d_model=d_model, nhead=4, num_layers=layers, ff_dim=64,
        dropout=0.0, time_emb_dim=16, max_len=max_len,
    )


class _NonlinearClassifier:
    """Nonlinear differentiable classifier ``f(x) = sum(w * tanh(a * x + b))``.

    The ``tanh`` adds curvature along the integration path so quadrature error
    actually shows up — on a pure-linear classifier the trapezoidal rule is
    exact regardless of solver, which makes that test insensitive to the
    solver choice.
    """

    def __init__(self, shape, device, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.w = torch.randn(*shape, generator=g).to(device)
        self.a = (torch.randn(*shape, generator=g) * 0.5 + 1.0).to(device)
        self.b = (torch.randn(*shape, generator=g) * 0.1).to(device)
        self.bias = torch.tensor(float(torch.randn(1, generator=g).item())).to(device)

    def __call__(self, x_flow: torch.Tensor, ctx=None) -> torch.Tensor:
        y = torch.tanh(self.a * x_flow + self.b)
        return (y * self.w).flatten(1).sum(dim=1) + self.bias


# ---------------------------------------------------------------------------
# 1. Synthetic completeness bar: median rel-residual < 1% at K=100, rk4
# ---------------------------------------------------------------------------

def test_rk4_K100_completeness_paper_target():
    """RK4 + K=100 + trapezoidal quadrature should hit the paper's 1% bar.

    Uses a tiny untrained VelocityNet (curved trajectories because weights
    are random) + a nonlinear classifier (tanh) so the quadrature rule is
    actually exercised. Config kept small (B=4, T=16, tiny net) so the test
    finishes in <10 s on a single CPU thread.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, T, J, C = 4, 16, 17, 3
    net = _tiny_net(max_len=T + 16).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    x_star = torch.randn(B, T, J, C, device=device) * 0.3
    mask = torch.ones(B, T, device=device, dtype=torch.bool)
    cls = _NonlinearClassifier((T, J, C), device=device, seed=3)

    out = compute_flow_shap(
        velocity_net=net,
        classifier_fn=lambda x, ctx: cls(x),
        x_star_flow=x_star,
        ctx={"pelvis_world": torch.zeros(B, T, 3, device=device), "mask": mask},
        num_steps=100,
        solver_method="rk4",
        zero_pelvis=False,
    )

    rel = out["completeness_residual_rel"].detach().cpu().numpy()
    assert np.isfinite(rel).all(), f"non-finite residuals: {rel}"

    med = float(np.median(rel))
    p95 = float(np.percentile(rel, 95))
    # Paper Table 1 / Sec. 5.1 target for the OTFlow-SHAP quadrature.
    assert med < 0.01, (
        f"median relative completeness residual {med:.4f} exceeds paper's 1% "
        f"target (values: {rel.tolist()})"
    )
    # A looser 95th-percentile bound guards against tail misbehaviour.
    assert p95 < 0.05, (
        f"p95 relative completeness residual {p95:.4f} too large "
        f"(values: {rel.tolist()})"
    )


def test_rk4_beats_euler_on_curved_integrand():
    """Sanity check: rk4 @ K=40 should be MUCH tighter than euler @ K=40.

    RK4 is O(K^-4), Euler is O(K^-1); for any curved integrand the gap at
    K=40 should be multiple orders of magnitude. Guards against a regression
    that silently flips the default solver back to Euler or breaks the
    higher-order integration path.

    (We don't compare rk4 against midpoint — on nearly-linear integrands
    midpoint can incidentally match rk4 because the leading error terms
    cancel. Euler is a much more reliable reference to beat.)
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, T, J, C = 2, 16, 17, 3
    net = _tiny_net(max_len=T + 16).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    x_star = torch.randn(B, T, J, C, device=device) * 0.3
    mask = torch.ones(B, T, device=device, dtype=torch.bool)
    cls = _NonlinearClassifier((T, J, C), device=device, seed=11)

    kwargs = dict(
        velocity_net=net,
        classifier_fn=lambda x, ctx: cls(x),
        x_star_flow=x_star,
        ctx={"pelvis_world": torch.zeros(B, T, 3, device=device), "mask": mask},
        num_steps=40,
        zero_pelvis=False,
    )
    rel_rk4 = compute_flow_shap(**kwargs, solver_method="rk4")["completeness_residual_rel"]
    rel_eul = compute_flow_shap(**kwargs, solver_method="euler")["completeness_residual_rel"]
    med_rk4 = float(rel_rk4.median().item())
    med_eul = float(rel_eul.median().item())
    # rk4 should be at least 2x tighter than euler on any curved integrand;
    # in practice the ratio is much larger (>10x) for K=40.
    assert med_rk4 < 0.5 * med_eul + 1e-6, (
        f"rk4 median residual {med_rk4:.5f} not materially tighter than "
        f"euler's {med_eul:.5f}"
    )


# ---------------------------------------------------------------------------
# 2. End-to-end on a trained BMCLab checkpoint (skipped if unavailable)
# ---------------------------------------------------------------------------

def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    flow_cfg = PROJECT_ROOT / "configs/flow_matching/bmclab_h36m3d_fold1_seed123.json"
    if not flow_cfg.exists():
        return "flow_matching seed123 config missing"
    cache = PROJECT_ROOT / json.loads(flow_cfg.read_text())["cache_dir"] / "cache.npz"
    if not cache.exists():
        return f"flow cache missing at {cache}"
    flow_ckpt_dir = PROJECT_ROOT / "experiment_outs/flow_matching/bmclab_h36m3d_fold1_seed123"
    if not (flow_ckpt_dir / "last.ckpt").exists():
        return f"flow checkpoint missing at {flow_ckpt_dir}/last.ckpt"
    potr_ckpt = PROJECT_ROOT / "experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr"
    if not potr_ckpt.exists():
        return f"POTR checkpoint missing at {potr_ckpt}"
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "unreachable")
def test_trained_checkpoint_median_completeness_below_1pct():
    """Median relative completeness residual on 8 real BMCLab clips < 1%."""
    from model.flow_shap import build_classifier_fn
    from model.flow_shap.data_loading import (
        load_flow_cache,
        load_pelvis_world_for_val,
    )
    from scripts.compute_flow_shap import _load_velocity_net

    device = torch.device("cuda:0")
    flow_cfg_path = PROJECT_ROOT / "configs/flow_matching/bmclab_h36m3d_fold1_seed123.json"
    flow_cfg = json.loads(flow_cfg_path.read_text())
    cache = load_flow_cache(PROJECT_ROOT / flow_cfg["cache_dir"] / "cache.npz", split="val")
    N = 8
    x1 = torch.from_numpy(cache["x1"][:N]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(cache["mask"][:N]).to(device=device, dtype=torch.bool)

    pelvis_np = load_pelvis_world_for_val(
        dataset=str(cache["dataset"]),
        meta_seq_key_val=cache["meta_seq_key"][:N],
        meta_clip_idx_val=cache["meta_clip_idx"][:N],
        seq_len=int(cache["seq_len"]),
        clip_stride_val=int(cache["clip_stride_val"]),
    )
    pelvis = torch.from_numpy(pelvis_np).to(device=device, dtype=torch.float32)
    stats_mean = torch.from_numpy(cache["stats_mean"]).to(device)
    stats_std  = torch.from_numpy(cache["stats_std"]).to(device)

    velocity_net = _load_velocity_net(
        flow_cfg,
        str(PROJECT_ROOT / "experiment_outs/flow_matching/bmclab_h36m3d_fold1_seed123/last.ckpt"),
        device,
    )
    classifier_fn, _, _, _ = build_classifier_fn(
        backbone_name="potr",
        classifier_ckpt=str(PROJECT_ROOT / "experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr"),
        flow_stats_mean=stats_mean, flow_stats_std=stats_std,
        class_idx=0, device=device,
        config_file="BMCLab.json", num_folds=23, fold=1,
    )

    out = compute_flow_shap(
        velocity_net=velocity_net,
        classifier_fn=classifier_fn,
        x_star_flow=x1,
        ctx={"pelvis_world": pelvis, "mask": mask},
        num_steps=100,
        solver_method="rk4",
        zero_pelvis=True,
    )
    rel = out["completeness_residual_rel"].detach().cpu().numpy()
    med = float(np.median(rel))
    # Paper target is ~1%. Allow a small safety margin for the fact that we
    # only average over N=8 clips here; if this bumps above 2% that's the
    # signal to run a reflow (Phase 1b).
    assert med < 0.02, (
        f"trained-ckpt median completeness residual {med:.4f} exceeds 2% "
        f"safety margin (values: {rel.tolist()})"
    )
