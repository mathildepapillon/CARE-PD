#!/usr/bin/env python
"""Unit tests for ``model.flow_matching.VelocityNet``.

These tests are self-contained (no data dependencies): they verify that the
velocity net forward-pass is shape-correct, conditioning signals actually
affect the output, masking is respected, and gradients flow end-to-end.

Run with::

    python -m pytest tests/test_flow_matching_velocity_net.py -v
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.flow_matching import VelocityNet, sinusoidal_embedding  # noqa: E402


@pytest.fixture
def small_net():
    # Small model for fast tests.
    return VelocityNet(
        n_joints=17, n_coords=3,
        d_model=64, nhead=4, num_layers=2, ff_dim=128,
        dropout=0.0, time_emb_dim=32, max_len=128,
    )


def test_forward_shape(small_net):
    B, T = 3, 40
    x = torch.randn(B, T, 17, 3)
    t = torch.rand(B)
    y = small_net(x, t)
    assert y.shape == x.shape, f"expected {tuple(x.shape)}, got {tuple(y.shape)}"


def test_forward_shape_default_model():
    # Make sure the production-default config is instantiable and forward-valid.
    net = VelocityNet()  # all defaults (d=256, layers=4, ...)
    B, T = 2, 80
    x = torch.randn(B, T, 17, 3)
    t = torch.rand(B)
    y = net(x, t)
    assert y.shape == x.shape
    p = net.count_parameters()
    # Sanity bound: the model shouldn't balloon past ~6M params with defaults.
    assert 1_000_000 < p < 6_000_000, f"unexpected param count: {p}"


def test_output_changes_with_time(small_net):
    B, T = 4, 16
    torch.manual_seed(0)
    x = torch.randn(B, T, 17, 3)
    # Bias the out_proj away from zero so the network isn't degenerate
    with torch.no_grad():
        small_net.out_proj.weight.copy_(0.1 * torch.randn_like(small_net.out_proj.weight))
        small_net.out_proj.bias.copy_(0.1 * torch.randn_like(small_net.out_proj.bias))
    y_t0 = small_net(x, torch.zeros(B))
    y_t1 = small_net(x, torch.ones(B))
    assert not torch.allclose(y_t0, y_t1, atol=1e-4), \
        "output should depend on flow time"


def test_output_changes_with_input(small_net):
    B, T = 4, 16
    torch.manual_seed(1)
    with torch.no_grad():
        small_net.out_proj.weight.copy_(0.1 * torch.randn_like(small_net.out_proj.weight))
        small_net.out_proj.bias.copy_(0.1 * torch.randn_like(small_net.out_proj.bias))
    x1 = torch.randn(B, T, 17, 3)
    x2 = torch.randn(B, T, 17, 3)
    t = torch.full((B,), 0.5)
    y1 = small_net(x1, t)
    y2 = small_net(x2, t)
    assert not torch.allclose(y1, y2, atol=1e-4), \
        "output should depend on x_t"


def test_mask_runs_without_error(small_net):
    B, T = 3, 20
    x = torch.randn(B, T, 17, 3)
    t = torch.rand(B)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 15:] = False  # first sample has 5 padded frames
    mask[2, 10:] = False
    y = small_net(x, t, mask=mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all(), "NaN/Inf in output under masking"


def test_mask_does_not_affect_real_frame_outputs(small_net):
    """Padded frames must not influence outputs at real frame positions
    (within numerical tolerance). We check this by replacing the padded
    tail with pure noise and ensuring the real-frame outputs are stable."""
    B, T = 2, 20
    t = torch.rand(B)
    x = torch.randn(B, T, 17, 3)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 12:] = False  # last 8 frames are pad

    x_alt = x.clone()
    x_alt[:, 12:] = 1_000.0 * torch.randn(B, T - 12, 17, 3)  # completely different pad

    with torch.no_grad():
        y1 = small_net(x, t, mask=mask)
        y2 = small_net(x_alt, t, mask=mask)

    # Real-frame outputs should be identical because padded positions are
    # ignored by attention.
    diff = (y1[:, :12] - y2[:, :12]).abs().max().item()
    assert diff < 1e-5, f"real-frame outputs differ by {diff} under different pad content"


def test_gradient_flow(small_net):
    B, T = 2, 24
    x = torch.randn(B, T, 17, 3, requires_grad=False)
    t = torch.rand(B)
    target = torch.randn(B, T, 17, 3)
    y = small_net(x, t)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    n_with_grad = 0
    for p in small_net.parameters():
        if p.requires_grad:
            assert p.grad is not None, "parameter with no gradient"
            if p.grad.abs().sum().item() > 0:
                n_with_grad += 1
    assert n_with_grad > 5, "too few parameters received a gradient"


def test_sinusoidal_embedding_shape_and_finite():
    values = torch.linspace(0.0, 1.0, 17)
    emb = sinusoidal_embedding(values, dim=64)
    assert emb.shape == (17, 64)
    assert torch.isfinite(emb).all()
    # Different t -> different embedding (trivial monotone check)
    assert not torch.allclose(emb[0], emb[-1])


def test_sinusoidal_embedding_rejects_odd_dim():
    with pytest.raises(ValueError):
        sinusoidal_embedding(torch.tensor([0.1]), dim=65)


def test_max_len_exceeded_raises(small_net):
    # small_net has max_len=128
    B, T = 1, 200
    x = torch.randn(B, T, 17, 3)
    t = torch.rand(B)
    with pytest.raises(ValueError):
        small_net(x, t)


def test_joint_shape_mismatch_raises(small_net):
    B, T = 1, 10
    x = torch.randn(B, T, 16, 3)  # wrong number of joints
    t = torch.rand(B)
    with pytest.raises(ValueError):
        small_net(x, t)


if __name__ == "__main__":
    # Allow ``python tests/test_flow_matching_velocity_net.py`` for quick checks.
    import pytest as _pt
    sys.exit(_pt.main([__file__, "-v"]))
