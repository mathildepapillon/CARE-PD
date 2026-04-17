#!/usr/bin/env python
"""Unit tests for the OTFlow-SHAP attribution pipeline.

Three independent tests:

1. **Linear-classifier efficiency.** For any smooth classifier the trapezoidal
   integrand approximates the line integral whose exact value is
   ``f(x*) - f(gamma(0))``. For a *linear* classifier this holds to
   quadrature precision regardless of the velocity field, so we can assert a
   tight completeness bound without depending on any trained model.

2. **Pelvis Dummy axiom.** With a classifier that is linear in every feature,
   the pelvis-fix-then-flow construction forces the flow's joint-0 velocity
   to be ≈ 0 and the attribution on joint 0 to be tiny relative to the rest.

3. **End-to-end smoke on real POTR** (only when CUDA + the BMCLab checkpoints
   are available — otherwise skipped). Runs 4 real val clips and asserts
   shapes, NaN-freeness, and a loose completeness bound.

Run with::

    python -m pytest tests/test_flow_shap.py -v
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

def _tiny_net(d_model: int = 64, layers: int = 2) -> VelocityNet:
    return VelocityNet(
        n_joints=17, n_coords=3,
        d_model=d_model, nhead=4, num_layers=layers, ff_dim=128,
        dropout=0.0, time_emb_dim=32, max_len=96,
    )


class _LinearClassifier:
    """Simple linear functional ``f(x) = sum(w * x) + b`` acting on flow-space
    ``(B, T, J, 3)``. Returns ``(B,)``. Differentiable, trivial gradient."""

    def __init__(self, shape, device, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.w = torch.randn(*shape, generator=g).to(device)
        self.b = torch.tensor(float(torch.randn(1, generator=g).item())).to(device)

    def __call__(self, x_flow: torch.Tensor, ctx=None) -> torch.Tensor:
        return (x_flow * self.w).flatten(1).sum(dim=1) + self.b


# ---------------------------------------------------------------------------
# 1. Linear classifier ⇒ completeness to quadrature precision
# ---------------------------------------------------------------------------

def test_linear_classifier_efficiency():
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, T, J, C = 2, 24, 17, 3
    net = _tiny_net().to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    x_star = torch.randn(B, T, J, C, device=device) * 0.2
    mask = torch.ones(B, T, device=device, dtype=torch.bool)
    cls = _LinearClassifier((T, J, C), device=device, seed=1)

    out = compute_flow_shap(
        velocity_net=net,
        classifier_fn=lambda x, ctx: cls(x),
        x_star_flow=x_star,
        ctx={"pelvis_world": torch.zeros(B, T, 3, device=device), "mask": mask},
        num_steps=64,
        solver_method="midpoint",
        zero_pelvis=False,     # keep full psi so it sums to delta_f
    )

    psi_sum = out["psi"].sum(dim=(1, 2, 3))
    delta_f = out["f_xstar"] - out["f_x0"]
    rel = (psi_sum - delta_f).abs() / delta_f.abs().clamp(min=1e-6)
    # Linear f + trapezoidal rule + smooth ODE trajectory ⇒ tight residual.
    assert (rel < 1e-2).all(), (
        f"linear-classifier completeness residuals too large: {rel.tolist()}"
    )
    assert torch.isfinite(out["psi"]).all()


# ---------------------------------------------------------------------------
# 2. Dummy axiom: pelvis velocity is zero so joint-0 attribution is tiny
# ---------------------------------------------------------------------------

def test_pelvis_attribution_small():
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, T, J, C = 2, 16, 17, 3
    net = _tiny_net().to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    # Random linear functional acts on every coord — joint 0 still has
    # non-zero gradient, but pelvis velocity integrated over the path is ~0
    # because the flow-space joint-0 channel stays zero along gamma.
    x_star = torch.randn(B, T, J, C, device=device) * 0.2
    # Pin joint 0 to zero since the flow was trained on pelvis-centered clips:
    x_star[..., 0, :] = 0.0
    mask = torch.ones(B, T, device=device, dtype=torch.bool)
    cls = _LinearClassifier((T, J, C), device=device, seed=7)

    out = compute_flow_shap(
        velocity_net=net,
        classifier_fn=lambda x, ctx: cls(x),
        x_star_flow=x_star,
        ctx={"pelvis_world": torch.zeros(B, T, 3, device=device), "mask": mask},
        num_steps=48,
        zero_pelvis=False,
    )

    total = out["psi"].abs().sum(dim=(1, 2, 3))
    pelvis = out["psi"][..., 0, :].abs().sum(dim=(1, 2))
    frac = pelvis / total.clamp(min=1e-12)
    # Randomly initialised flow is not guaranteed to keep joint 0 exactly at 0
    # over the whole trajectory, but the leak should remain well below equal
    # share (1/17 ≈ 5.9 %) because the integrand is a product against v[...,0,:]
    # which receives no explicit training signal to become large.
    assert (frac < 0.10).all(), (
        f"pelvis leak fraction too large: {frac.tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. End-to-end smoke test on real POTR (skipped if unavailable)
# ---------------------------------------------------------------------------

def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    flow_cfg = PROJECT_ROOT / "configs/flow_matching/bmclab_h36m3d_fold1.json"
    if not flow_cfg.exists():
        return "flow_matching config missing"
    cache = PROJECT_ROOT / json.loads(flow_cfg.read_text())["cache_dir"] / "cache.npz"
    if not cache.exists():
        return f"flow cache missing at {cache}"
    flow_ckpt = PROJECT_ROOT / "experiment_outs/flow_matching/bmclab_h36m3d_fold1/flow_matching_BMCLab_fold1_best.ckpt"
    if not flow_ckpt.exists():
        return f"flow checkpoint missing at {flow_ckpt}"
    potr_ckpt = PROJECT_ROOT / "experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr"
    if not potr_ckpt.exists():
        return f"POTR checkpoint missing at {potr_ckpt}"
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "unreachable")
def test_end_to_end_smoke_potr():
    from model.flow_shap import build_classifier_fn
    from model.flow_shap.data_loading import (
        load_flow_cache,
        load_pelvis_world_for_val,
    )
    from scripts.compute_flow_shap import _load_velocity_net

    device = torch.device("cuda:0")
    flow_cfg_path = PROJECT_ROOT / "configs/flow_matching/bmclab_h36m3d_fold1.json"
    flow_cfg = json.loads(flow_cfg_path.read_text())
    cache = load_flow_cache(PROJECT_ROOT / flow_cfg["cache_dir"] / "cache.npz", split="val")
    N_SMOKE = 4
    x1 = torch.from_numpy(cache["x1"][:N_SMOKE]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(cache["mask"][:N_SMOKE]).to(device=device, dtype=torch.bool)

    pelvis_np = load_pelvis_world_for_val(
        dataset=str(cache["dataset"]),
        meta_seq_key_val=cache["meta_seq_key"][:N_SMOKE],
        meta_clip_idx_val=cache["meta_clip_idx"][:N_SMOKE],
        seq_len=int(cache["seq_len"]),
        clip_stride_val=int(cache["clip_stride_val"]),
    )
    pelvis = torch.from_numpy(pelvis_np).to(device=device, dtype=torch.float32)

    stats_mean = torch.from_numpy(cache["stats_mean"]).to(device)
    stats_std  = torch.from_numpy(cache["stats_std"]).to(device)

    velocity_net = _load_velocity_net(
        flow_cfg,
        str(PROJECT_ROOT / "experiment_outs/flow_matching/bmclab_h36m3d_fold1/flow_matching_BMCLab_fold1_best.ckpt"),
        device,
    )
    classifier_fn, _, _, _ = build_classifier_fn(
        backbone_name="potr",
        classifier_ckpt=str(PROJECT_ROOT / "experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr"),
        flow_stats_mean=stats_mean,
        flow_stats_std=stats_std,
        class_idx=0,
        device=device,
        config_file="BMCLab.json",
        num_folds=23,
        fold=1,
    )

    out = compute_flow_shap(
        velocity_net=velocity_net,
        classifier_fn=classifier_fn,
        x_star_flow=x1,
        ctx={"pelvis_world": pelvis, "mask": mask},
        num_steps=40,
        solver_method="midpoint",
        zero_pelvis=True,
    )
    assert out["psi"].shape == x1.shape
    assert torch.isfinite(out["psi"]).all()
    assert torch.isfinite(out["f_xstar"]).all()
    assert torch.isfinite(out["f_x0"]).all()
    # Loose completeness check — endpoint buckets are lossy and this is a
    # real, shallow ODE.
    assert (out["completeness_residual_rel"] < 1.0).all(), (
        f"completeness residuals way too large: {out['completeness_residual_rel'].tolist()}"
    )
