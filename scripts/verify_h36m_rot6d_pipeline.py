#!/usr/bin/env python3
"""Verify the H36M exp-map → rotation matrix → Zhou rot6d → Torch FK pipeline.

Run from the repository root so ``model`` resolves::

    cd /path/to/CARE-PD && python scripts/verify_h36m_rot6d_pipeline.py

CI / automated tests: ``pytest tests/test_h36m_rot6d_consistency.py`` (same numerical checks).

Optional flags: ``--require-npz`` fails if the asset NPZ check cannot run; ``--rot6d-npz``,
``--xyz-npz``, ``--seq-key`` override default paths under ``assets/datasets/``.

Sections:
  1. Zhou 6D encode/decode round-trip on random SO(3) matrices.
  2. ``_expmap2rotmat`` vs SciPy ``Rotation.from_rotvec`` (max error).
  3. SO(3) membership for those matrices (orthogonality + det).
  4. Synthetic sequence: NumPy FK vs rot6d + ``H36MRotation2xyz`` (MPJPE).
  5. Optional: stored rot6d NPZ vs XYZ NPZ under ``assets/datasets`` (SKIP if missing).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciPyRotation

# ---------------------------------------------------------------------------
# Repo layout: scripts/verify_*.py → project root is parent.parent
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ROT6D_NPZ = (
    PROJECT_ROOT
    / "assets"
    / "datasets"
    / "6D_ROTATIONS"
    / "H36M"
    / "h36m_rot6d_32j_30f_or_longer.npz"
)
DEFAULT_XYZ_NPZ = (
    PROJECT_ROOT
    / "assets"
    / "datasets"
    / "h36m"
    / "H36M"
    / "h36m_3d_world_floorXZZplus_30f_or_longer.npz"
)


def _load_prepare_h36m():
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


def _fro_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def check_zhou_roundtrip(seed: int) -> tuple[bool, str]:
    from model.actor.rotation2xyz import matrix_to_rotation_6d, rotation_6d_to_matrix

    torch.manual_seed(seed)
    n = 64
    a = torch.randn(n, 3, 3)
    q, _ = torch.linalg.qr(a)
    dets = torch.linalg.det(q)
    flip = dets < 0
    if flip.any():
        q = q.clone()
        q[flip, :, 2] *= -1
    d6 = matrix_to_rotation_6d(q)
    q2 = rotation_6d_to_matrix(d6)
    ok = torch.allclose(q, q2, atol=1e-5, rtol=1e-5)
    max_err = float((q - q2).abs().max().item())
    msg = f"max_abs_err={max_err:.3e} (threshold allclose atol=1e-5 rtol=1e-5)"
    return ok, msg


def check_expmap_vs_scipy(ph, seed: int, n_samples: int = 500) -> tuple[bool, str]:
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    vecs: list[np.ndarray] = [np.zeros(3), np.array([1e-10, 0.0, 0.0]), np.array([1e-8] * 3)]
    for _ in range(n_samples):
        vecs.append(rng.standard_normal(3) * 0.8)

    for v in vecs:
        R_np = ph._expmap2rotmat(v.astype(np.float64))
        R_sp = SciPyRotation.from_rotvec(v).as_matrix()
        diff = np.abs(R_np - R_sp).max()
        max_diff = max(max_diff, float(diff))

    ok = max_diff < 1e-6
    msg = f"max|R_numpy - R_scipy|={max_diff:.3e} (pass if < 1e-6)"
    return ok, msg


def check_so3(ph, seed: int, n_samples: int = 200) -> tuple[bool, str]:
    rng = np.random.default_rng(seed + 1)
    max_ortho = 0.0
    max_det_err = 0.0
    for _ in range(n_samples):
        v = rng.standard_normal(3) * 0.7
        R = ph._expmap2rotmat(v)
        ortho = _fro_norm(R @ R.T - np.eye(3))
        det_err = abs(float(np.linalg.det(R)) - 1.0)
        max_ortho = max(max_ortho, ortho)
        max_det_err = max(max_det_err, det_err)

    # float64 Rodrigues typically leaves ~1e-7 residual on orthogonality/det
    ok = max_ortho < 1e-6 and max_det_err < 1e-6
    msg = (
        f"max_frobenius(RR^T-I)={max_ortho:.3e} max|det(R)-1|={max_det_err:.3e} "
        f"(pass if both < 1e-6)"
    )
    return ok, msg


def check_synthetic_fk(ph, seed: int) -> tuple[bool, str]:
    from model.actor.h36m_rotation2xyz import H36MRotation2xyz

    parent, offset, expmap_ind = ph._h36m_skeleton_variables()
    rng = np.random.default_rng(seed)
    T = 16
    raw = np.zeros((T, 99), dtype=np.float64)
    raw[:, 6:] = rng.standard_normal((T, 93)) * 0.15
    raw[:, :6] = 0.0

    xyz_np = ph.expmap_to_xyz_17(raw, parent, offset, expmap_ind)
    rot6d = ph.expmap_to_rot6d_32(raw, expmap_ind)

    x_bjft = torch.tensor(rot6d.tolist(), dtype=torch.float32).permute(1, 2, 0).unsqueeze(0)
    mask = torch.ones(1, T, dtype=torch.bool)
    fk = H36MRotation2xyz()
    xyz_t = fk(x_bjft, mask).permute(0, 3, 1, 2)[0]
    xyz_ref = torch.tensor(xyz_np.tolist(), dtype=torch.float32)

    err = _mpjpe_torch_m(xyz_ref, xyz_t)
    threshold = 1e-4
    ok = err < threshold
    msg = f"MPJPE={err:.6e} m (pass if < {threshold})"
    return ok, msg


def check_npz_assets(
    rot_path: Path,
    xyz_path: Path,
    seq_key: str,
) -> tuple[str, bool, str]:
    """Returns (status, ok_or_na, message). status is OK | FAIL | SKIP."""

    if not rot_path.is_file() or not xyz_path.is_file():
        return "SKIP", True, f"missing files (rot={rot_path.is_file()} xyz={xyz_path.is_file()})"

    from model.actor.h36m_rotation2xyz import H36MRotation2xyz

    rotz = np.load(str(rot_path), allow_pickle=True)
    xyz_npz = np.load(str(xyz_path), allow_pickle=True)
    if seq_key not in rotz or seq_key not in xyz_npz:
        return "SKIP", True, f"sequence {seq_key!r} not in both NPZ files"

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
    threshold = 1e-4
    ok = err < threshold
    msg = f"seq={seq_key!r} T={T} MPJPE={err:.6e} m (pass if < {threshold})"
    return ("OK" if ok else "FAIL"), ok, msg


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify H36M rot6d pipeline (Zhou round-trip, SciPy, FK, optional NPZ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for checks 1–4")
    ap.add_argument(
        "--require-npz",
        action="store_true",
        help="Fail if optional NPZ asset check cannot run (files or seq missing)",
    )
    ap.add_argument(
        "--rot6d-npz",
        type=Path,
        default=DEFAULT_ROT6D_NPZ,
        help="Path to h36m_rot6d_32j_30f_or_longer.npz",
    )
    ap.add_argument(
        "--xyz-npz",
        type=Path,
        default=DEFAULT_XYZ_NPZ,
        help="Path to matching XYZ npz",
    )
    ap.add_argument(
        "--seq-key",
        type=str,
        default="S1__walking_1",
        help="Sequence key for NPZ check",
    )
    args = ap.parse_args()

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    ph = _load_prepare_h36m()

    all_ok = True
    lines: list[str] = []

    # 1 Zhou
    ok, msg = check_zhou_roundtrip(args.seed)
    all_ok = all_ok and ok
    lines.append(f"[1] Zhou rot6d round-trip ...... {'OK' if ok else 'FAIL'}  {msg}")

    # 2 Exp-map vs SciPy
    ok, msg = check_expmap_vs_scipy(ph, args.seed)
    all_ok = all_ok and ok
    lines.append(f"[2] expmap2rotmat vs SciPy .... {'OK' if ok else 'FAIL'}  {msg}")

    # 3 SO(3)
    ok, msg = check_so3(ph, args.seed)
    all_ok = all_ok and ok
    lines.append(f"[3] SO(3) orthogonality/det ... {'OK' if ok else 'FAIL'}  {msg}")

    # 4 Synthetic FK
    ok, msg = check_synthetic_fk(ph, args.seed)
    all_ok = all_ok and ok
    lines.append(f"[4] Synthetic NumPy vs Torch FK {'OK' if ok else 'FAIL'}  {msg}")

    # 5 NPZ
    status, npz_ok, msg = check_npz_assets(args.rot6d_npz, args.xyz_npz, args.seq_key)
    if status == "SKIP":
        lines.append(f"[5] NPZ rot6d vs XYZ .......... SKIP  {msg}")
        if args.require_npz:
            all_ok = False
            lines.append("    (--require-npz: treating SKIP as failure)")
    else:
        all_ok = all_ok and npz_ok
        lines.append(f"[5] NPZ rot6d vs XYZ .......... {status}  {msg}")

    print()
    for line in lines:
        print(line)
    print()

    if all_ok:
        print("All executed checks passed.")
        return 0
    print("One or more checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
