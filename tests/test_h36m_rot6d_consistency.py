"""Numerical consistency checks for H36M exp-map → 6D → PyTorch FK vs NumPy FK.

See plan: rot6d saved in prepare_h36m_dataset must match H36MRotation2xyz and
the NumPy reference forward kinematics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_prepare_h36m():
    """Load scripts/prepare_h36m_dataset.py as a module (not on PYTHONPATH)."""
    path = PROJECT_ROOT / "scripts" / "prepare_h36m_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_h36m_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["prepare_h36m_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


def _mpjpe_torch_m(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-joint L2 error (metres). Shapes (T, J, 3)."""
    return float(torch.linalg.norm(a - b, dim=-1).mean().item())


@pytest.fixture(scope="module")
def ph():
    return _load_prepare_h36m()


def test_rotation_6d_matrix_roundtrip_random_so3():
    """Zhou 6D: encode with first two rows, decode with Gram–Schmidt → original R."""
    from model.actor.rotation2xyz import matrix_to_rotation_6d, rotation_6d_to_matrix

    torch.manual_seed(0)
    n = 64
    a = torch.randn(n, 3, 3)
    q, _ = torch.linalg.qr(a)
    # Ensure proper rotations (det +1) per batch element
    dets = torch.linalg.det(q)
    flip = dets < 0
    if flip.any():
        q = q.clone()
        q[flip, :, 2] *= -1
    d6 = matrix_to_rotation_6d(q)
    q2 = rotation_6d_to_matrix(d6)
    assert torch.allclose(q, q2, atol=1e-5, rtol=1e-5)


def test_numpy_vs_carepd_rot6d_backend_close(ph):
    """Legacy NumPy Rodrigues vs CARE-PD PyTorch chain should match to ~1e-5."""
    _parent, _offset, expmap_ind = ph._h36m_skeleton_variables()
    rng = np.random.default_rng(0)
    T = 8
    raw = np.zeros((T, 99), dtype=np.float64)
    raw[:, 6:] = rng.standard_normal((T, 93)) * 0.2
    raw[:, :6] = 0.0

    r_np = ph.expmap_to_rot6d_32(raw, expmap_ind, backend="numpy")
    r_cp = ph.expmap_to_rot6d_32(raw, expmap_ind, backend="carepd")
    max_abs = float(np.abs(r_np - r_cp).max())
    assert max_abs < 1e-5, f"numpy vs carepd rot6d max_abs={max_abs}"


def test_torch_fk_matches_numpy_expmap_reference(ph):
    """Same raw (after [:6]=0): rot6d → H36MRotation2xyz ≈ expmap_to_xyz_17."""
    from model.actor.h36m_rotation2xyz import H36MRotation2xyz

    parent, offset, expmap_ind = ph._h36m_skeleton_variables()
    rng = np.random.default_rng(42)
    T = 16
    raw = np.zeros((T, 99), dtype=np.float64)
    # Post-canonicalisation indices: only 6..98 are free ([:6] cleared in pipeline).
    raw[:, 6:] = rng.standard_normal((T, 93)) * 0.15
    raw[:, :6] = 0.0

    xyz_np = ph.expmap_to_xyz_17(raw, parent, offset, expmap_ind)
    rot6d = ph.expmap_to_rot6d_32(raw, expmap_ind)

    # Avoid torch.from_numpy (breaks when PyTorch is built without NumPy ABI).
    x_bjft = torch.tensor(rot6d.tolist(), dtype=torch.float32).permute(1, 2, 0).unsqueeze(0)
    # (1, 32, 6, T)
    mask = torch.ones(1, T, dtype=torch.bool)
    fk = H36MRotation2xyz()
    xyz_t = fk(x_bjft, mask).permute(0, 3, 1, 2)[0]
    xyz_ref = torch.tensor(xyz_np.tolist(), dtype=torch.float32)

    err = _mpjpe_torch_m(xyz_ref, xyz_t)
    assert err < 1e-4, f"MPJPE {err} m — NumPy vs Torch FK mismatch"


@pytest.mark.parametrize("seq_key", ["S1__walking_1"])
def test_rot6d_npz_fk_matches_xyz_npz_if_present(seq_key):
    """Prepared rot6d NPZ and XYZ NPZ from the same run must agree under Torch FK."""
    from model.actor.h36m_rotation2xyz import H36MRotation2xyz

    rot_path = (
        PROJECT_ROOT / "assets" / "datasets" / "6D_ROTATIONS" / "H36M"
        / "h36m_rot6d_32j_30f_or_longer.npz"
    )
    xyz_path = (
        PROJECT_ROOT / "assets" / "datasets" / "h36m" / "H36M"
        / "h36m_3d_world_floorXZZplus_30f_or_longer.npz"
    )
    if not rot_path.is_file() or not xyz_path.is_file():
        pytest.skip("H36M rot6d or XYZ NPZ not in assets/datasets")
    rotz = np.load(str(rot_path), allow_pickle=True)
    xyz_npz = np.load(str(xyz_path), allow_pickle=True)
    if seq_key not in rotz or seq_key not in xyz_npz:
        pytest.skip(f"sequence {seq_key} not in both NPZ files")

    rot6d = rotz[seq_key]
    xyz_gt = xyz_npz[seq_key]
    T = min(rot6d.shape[0], xyz_gt.shape[0])
    rot6d = rot6d[:T]
    xyz_gt = xyz_gt[:T]

    x_bjft = torch.tensor(rot6d.tolist(), dtype=torch.float32).permute(1, 2, 0).unsqueeze(0)
    mask = torch.ones(1, T, dtype=torch.bool)
    fk = H36MRotation2xyz()
    xyz_t = fk(x_bjft, mask).permute(0, 3, 1, 2)[0]
    xyz_ref = torch.tensor(xyz_gt.tolist(), dtype=torch.float32)

    err = _mpjpe_torch_m(xyz_ref, xyz_t)
    assert err < 1e-4, f"MPJPE {err} m — stored rot6d FK vs stored XYZ NPZ"
