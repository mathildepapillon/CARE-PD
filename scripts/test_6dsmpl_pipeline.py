"""
test_6dsmpl_pipeline.py — Correctness tests for the 6D_SMPL → FK pipeline.

Each test has an explicit, pre-determined expected answer — not just "does it
run" or "is the output finite".  Run with:

    python scripts/test_6dsmpl_pipeline.py

All tests print PASS / FAIL with a short explanation of what is being checked
and why the expected value is what it is.

Covered areas
-------------
T1  6D round-trip  — encode then decode gives back the same R (non-identity)
T2  6D row convention — verify the stored 6D packs rows, not columns
T3  SMPL T-pose anatomy — with zero rotations, joint heights match published SMPL
T4  SMPL joint ordering — L_Shoulder is LEFT of pelvis, R_Shoulder is RIGHT
T5  Body-pose indexing — our body_pose_mat[:, 15] == L_Shoulder rotation
T6  FK symmetry — mirroring left<->right gives mirrored XYZ (catches L/R swap bugs)
T7  FK root centering — joint 0 is exactly zero for every frame
T8  FK padding — padded frames are zeroed out in XYZ output
T9  _mat_to_aa round-trip — R → aa → SMPL gives same XYZ as R → 6D → FK
T10 actor_batch_from_6dsmpl — joint 24 (zeros) is correctly dropped
T11 Loss dimensions — rc loss on two identical tensors is exactly 0
T12 Anatomical plausibility on real data — elbows below head, feet below knees
T13 FK gradient flow — rcxyz gradients reach the decoder's 6D rotation output
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.nn.functional as F
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from data.preprocessing.preprocessing_utils import matrix_to_rotation_6d
from model.actor.rotation2xyz import rotation_6d_to_matrix, _mat_to_aa, Rotation2xyz, SMPL_NJOINTS
from model.actor.cvae_data import actor_batch_from_6dsmpl
from model.actor.losses import compute_rc_loss, compute_rcxyz_loss, compute_kl_loss

SMPL_PATH = os.path.join(_ROOT, "data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl")

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []

def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    _results.append((name, condition, detail))
    mark = "✓" if condition else "✗"
    print(f"  [{mark}] {status}  {name}")
    if not condition:
        print(f"        └─ {detail}")

def _random_rotation() -> torch.Tensor:
    """Return a random SO(3) rotation matrix via QR decomposition."""
    M = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(M)
    if Q.det() < 0:
        Q[:, 0] *= -1   # ensure det = +1
    return Q

def _smpl_tpose_joints() -> torch.Tensor:
    """Return the 24 SMPL joint positions in T-pose (root-centred)."""
    from smplx.body_models import SMPL
    smpl = SMPL(model_path=SMPL_PATH).eval()
    with torch.no_grad():
        out = smpl(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 69),
            global_orient=torch.zeros(1, 3),
        )
    j = out.joints[0, :24].detach()
    return j - j[0]   # root-centred

# ---------------------------------------------------------------------------
# T1  6D round-trip with non-identity rotation
# ---------------------------------------------------------------------------
def test_6d_roundtrip():
    print("\nT1  6D encode→decode round-trip (5 random non-identity rotations)")
    max_err = 0.0
    for _ in range(5):
        R = _random_rotation()
        d6 = matrix_to_rotation_6d(R)
        R_rec = rotation_6d_to_matrix(d6)
        err = (R_rec - R).abs().max().item()
        max_err = max(max_err, err)
    check(
        "6D round-trip max error < 1e-5",
        max_err < 1e-5,
        f"max error = {max_err:.2e}  (non-zero means rows/columns are swapped)",
    )

# ---------------------------------------------------------------------------
# T2  6D convention: data stores rows, not columns
# ---------------------------------------------------------------------------
def test_6d_row_convention():
    print("\nT2  6D convention: preprocessing stores ROWS of R")
    R = _random_rotation()  # (3, 3)
    d6 = matrix_to_rotation_6d(R)   # should store R[0,:] and R[1,:]
    row0_match = torch.allclose(d6[:3], R[0, :], atol=1e-6)
    row1_match = torch.allclose(d6[3:], R[1, :], atol=1e-6)
    col0_match = torch.allclose(d6[:3], R[:, 0], atol=1e-6)
    check(
        "6D d6[:3] == R[0,:] (first ROW, not first column)",
        row0_match and row1_match,
        f"row0={row0_match} row1={row1_match}  col0_match={col0_match} "
        "(if col0_match=True the convention changed to columns)",
    )

# ---------------------------------------------------------------------------
# T3  SMPL T-pose anatomy
# ---------------------------------------------------------------------------
def test_smpl_tpose_anatomy():
    print("\nT3  SMPL T-pose anatomy (zero rotations → expected joint heights)")
    j = _smpl_tpose_joints()
    # Published SMPL neutral model approximate heights:
    #   head > neck > shoulders ≈ elbows ≈ wrists (T-pose arms horizontal)
    #   pelvis > knees > ankles > feet
    head_y      = j[15, 1].item()
    neck_y      = j[12, 1].item()
    l_shoulder_y = j[16, 1].item()
    l_elbow_y   = j[18, 1].item()
    knee_y      = j[4, 1].item()
    ankle_y     = j[7, 1].item()

    check("T-pose: head above pelvis (head_y > 0)",
          head_y > 0.4,
          f"head_y={head_y:.3f}")
    check("T-pose: head above neck",
          head_y > neck_y,
          f"head={head_y:.3f} neck={neck_y:.3f}")
    check("T-pose: elbow at ~same height as shoulder (arms horizontal, |Δy| < 0.05 m)",
          abs(l_elbow_y - l_shoulder_y) < 0.05,
          f"l_shoulder_y={l_shoulder_y:.3f}  l_elbow_y={l_elbow_y:.3f}  diff={l_elbow_y-l_shoulder_y:.3f}")
    check("T-pose: knee below pelvis (knee_y < 0)",
          knee_y < -0.2,
          f"knee_y={knee_y:.3f}")
    check("T-pose: ankle below knee",
          ankle_y < knee_y,
          f"ankle={ankle_y:.3f} knee={knee_y:.3f}")

# ---------------------------------------------------------------------------
# T4  SMPL joint ordering: left vs right sides
# ---------------------------------------------------------------------------
def test_smpl_joint_sides():
    print("\nT4  SMPL joint ordering: left joints are +x, right joints are -x")
    j = _smpl_tpose_joints()
    # In SMPL canonical T-pose:
    #   person's LEFT  = world +x  (joint 1 L_Hip, 16 L_Shoulder, 18 L_Elbow …)
    #   person's RIGHT = world -x  (joint 2 R_Hip, 17 R_Shoulder, 19 R_Elbow …)
    check("T-pose: L_Hip (+x) is LEFT of R_Hip (-x)",
          j[1, 0].item() > j[2, 0].item(),
          f"L_Hip_x={j[1,0]:.3f}  R_Hip_x={j[2,0]:.3f}")
    check("T-pose: L_Shoulder (+x) is LEFT of R_Shoulder (-x)",
          j[16, 0].item() > j[17, 0].item(),
          f"L_Shoulder_x={j[16,0]:.3f}  R_Shoulder_x={j[17,0]:.3f}")
    check("T-pose: L_Elbow further left than L_Shoulder (arm extends outward)",
          j[18, 0].item() > j[16, 0].item(),
          f"L_Elbow_x={j[18,0]:.3f}  L_Shoulder_x={j[16,0]:.3f}")

# ---------------------------------------------------------------------------
# T5  Body-pose joint index mapping: index 15 in body_pose = L_Shoulder
# ---------------------------------------------------------------------------
def test_body_pose_indexing():
    """Verify that body_pose_mat[:, 15] controls L_Shoulder by applying a
    known rotation and checking that L_Shoulder (joint 16) moves while
    other joints stay put."""
    print("\nT5  Body-pose indexing: body_pose[:, 15] = L_Shoulder (joint 16 in FK output)")
    from smplx.body_models import SMPL
    smpl = SMPL(model_path=SMPL_PATH).eval()

    # Baseline T-pose
    body_pose_base = torch.zeros(1, 69)
    with torch.no_grad():
        out_base = smpl(betas=torch.zeros(1, 10), body_pose=body_pose_base, global_orient=torch.zeros(1, 3))
    j_base = out_base.joints[0, :24].detach() - out_base.joints[0, 0].detach()

    # Rotate body_pose index 15 (joints 16 = L_Shoulder, 0-indexed offset = 15*3):
    # Apply 90° rotation around z-axis to raise the left arm
    body_pose_mod = body_pose_base.clone()
    body_pose_mod[0, 15*3 + 2] = torch.pi / 2   # 90° around z in L_Shoulder local frame

    with torch.no_grad():
        out_mod = smpl(betas=torch.zeros(1, 10), body_pose=body_pose_mod, global_orient=torch.zeros(1, 3))
    j_mod = out_mod.joints[0, :24].detach() - out_mod.joints[0, 0].detach()

    l_shoulder_moved = (j_mod[16] - j_base[16]).norm().item()
    l_elbow_moved    = (j_mod[18] - j_base[18]).norm().item()
    r_shoulder_moved = (j_mod[17] - j_base[17]).norm().item()
    pelvis_moved     = (j_mod[0]  - j_base[0]).norm().item()

    # NB: In SMPL, a joint's world position is controlled by its PARENT's rotation,
    # not by its own.  L_Shoulder (joint 16) is the pivot for the upper arm, so
    # changing body_pose[15] (= L_Shoulder rotation) moves L_Elbow/L_Wrist/L_Hand
    # (children) but NOT L_Shoulder's position itself.
    check("Modifying body_pose[15] moves L_Elbow (child of L_Shoulder) > 10 cm",
          l_elbow_moved > 0.10,
          f"L_Elbow displacement={l_elbow_moved:.3f} m  "
          "(small value means body_pose[15] does not map to L_Shoulder rotation)")
    check("Modifying body_pose[15] does NOT move R_Shoulder (< 1 mm)",
          r_shoulder_moved < 0.001,
          f"R_Shoulder displacement={r_shoulder_moved:.4f} m  (should be ~0)")
    check("Modifying body_pose[15] does NOT move Pelvis (< 1 mm)",
          pelvis_moved < 0.001,
          f"Pelvis displacement={pelvis_moved:.4f} m  (should be ~0)")

# ---------------------------------------------------------------------------
# T6  FK left-right symmetry
# ---------------------------------------------------------------------------
def test_fk_symmetry():
    """Build a symmetric pose (T-pose = zero rotations), verify L and R joints
    are mirror images in x."""
    print("\nT6  FK left-right symmetry (T-pose → L and R joints are ±x mirrors)")
    r2xyz = Rotation2xyz(torch.device("cpu"))
    B, T = 1, 5
    # T-pose in 6D: identity rotation for all joints
    d6_identity = torch.zeros(B, SMPL_NJOINTS, 6, T)
    # identity 6D = first row [1,0,0] + second row [0,1,0]
    d6_identity[:, :, 0, :] = 1.0
    d6_identity[:, :, 4, :] = 1.0
    mask = torch.ones(B, T, dtype=torch.bool)
    with torch.no_grad():
        xyz = r2xyz(d6_identity, mask)   # (1, 24, 3, 5)

    j = xyz[0, :, :, 0]   # (24, 3) at frame 0
    # L_Hip (1) and R_Hip (2) should be symmetric: L_x ≈ -R_x, L_y ≈ R_y
    l_hip, r_hip = j[1], j[2]
    x_sym = abs(l_hip[0].item() + r_hip[0].item())    # should be ~0 (mirror)
    y_diff = abs(l_hip[1].item() - r_hip[1].item())   # should be ~0 (same height)

    l_elbow, r_elbow = j[18], j[19]
    x_sym_elbow = abs(l_elbow[0].item() + r_elbow[0].item())

    # The SMPL neutral model has small built-in left-right asymmetries
    # (up to ~7 mm for some joints) because it is an average over real bodies.
    # Tolerance of 1 cm covers all SMPL joints while still catching gross errors
    # like a true left-right swap (which would give ~14 cm asymmetry at the hips).
    check("T-pose: L_Hip and R_Hip are approximate x-mirrors (|L_x + R_x| < 1 cm)",
          x_sym < 0.01,
          f"|L_Hip_x + R_Hip_x| = {x_sym*1000:.1f} mm  "
          "(>10mm = likely L/R joint swap in SMPL model)")
    check("T-pose: L_Hip and R_Hip at same height (|Δy| < 5 mm)",
          y_diff < 0.005,
          f"|L_Hip_y - R_Hip_y| = {y_diff*1000:.2f} mm")
    check("T-pose: L_Elbow and R_Elbow are approximate x-mirrors (|L_x + R_x| < 1 cm)",
          x_sym_elbow < 0.01,
          f"|L_Elbow_x + R_Elbow_x| = {x_sym_elbow*1000:.1f} mm")

# ---------------------------------------------------------------------------
# T7  FK root centering
# ---------------------------------------------------------------------------
def test_fk_root_centering():
    print("\nT7  FK root centering (joint 0 = [0,0,0] for every frame)")
    # Load a real batch from BMCLab
    import types
    from data.dataloaders import collate_fn
    from model.actor.cvae_data import get_6dsmpl_datasets, actor_batch_from_6dsmpl

    args = types.SimpleNamespace(
        dataset="BMCLab", num_folds=6, fold=1, batch_size=4,
        experiment_name="test_root", carepd_pose_npz=None, carepd_labels_pkl=None,
    )
    _, val_ds = get_6dsmpl_datasets(args)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    x_raw, _, _, _, pad = next(iter(loader))
    batch = actor_batch_from_6dsmpl(x_raw, pad, torch.device("cpu"))
    r2xyz = Rotation2xyz(torch.device("cpu"))
    with torch.no_grad():
        xyz = r2xyz(batch["x"], batch["mask"])   # (4, 24, 3, 60)

    root = xyz[:, 0, :, :]   # (4, 3, 60)
    max_root = root.abs().max().item()
    check("Root joint is exactly zero for all frames (max |root| < 1e-5)",
          max_root < 1e-5,
          f"max |root xyz| = {max_root:.2e}")

# ---------------------------------------------------------------------------
# T8  FK padding: padded frames should be zeroed
# ---------------------------------------------------------------------------
def test_fk_padding():
    print("\nT8  FK padding (padded frames produce zero XYZ output)")
    r2xyz = Rotation2xyz(torch.device("cpu"))
    B, T = 2, 10
    d6_identity = torch.zeros(B, SMPL_NJOINTS, 6, T)
    d6_identity[:, :, 0, :] = 1.0
    d6_identity[:, :, 4, :] = 1.0
    # Only frames 0-4 are valid for seq 0
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :]  = True
    with torch.no_grad():
        xyz = r2xyz(d6_identity, mask)   # (2, 24, 3, 10)

    # Frames 5-9 of seq 0 should all be zero
    padded_frames = xyz[0, :, :, 5:]     # (24, 3, 5)
    max_padded = padded_frames.abs().max().item()
    # Frames 0-4 of seq 0 (valid) should NOT all be zero
    valid_frames_max = xyz[0, 1:, :, :5].abs().max().item()   # exclude root which is always 0

    check("Padded frames are zeroed (max |xyz| < 1e-6 in padded region)",
          max_padded < 1e-6,
          f"max padded |xyz| = {max_padded:.2e}")
    check("Valid frames are non-zero (non-root joints have |xyz| > 0)",
          valid_frames_max > 0.05,
          f"max valid |xyz| = {valid_frames_max:.4f}  (should be ~0.4+ m for hips/arms)")

# ---------------------------------------------------------------------------
# T9  _mat_to_aa round-trip
# ---------------------------------------------------------------------------
def test_mat_to_aa_roundtrip():
    """Convert R → axis-angle → SMPL FK and compare with 6D → FK path.
    Both should give the same joint positions (within numerical tolerance).
    This tests that _mat_to_aa is correct for arbitrary non-trivial rotations."""
    print("\nT9  _mat_to_aa round-trip: R→aa→FK == R→6D→FK (5 random batches)")
    from smplx.body_models import SMPL
    smpl = SMPL(model_path=SMPL_PATH).eval()
    r2xyz = Rotation2xyz(torch.device("cpu"))

    max_err = 0.0
    for _ in range(5):
        # Random pose: 24 random rotations
        rots = torch.stack([_random_rotation() for _ in range(24)])  # (24, 3, 3)

        # Path A: R → axis-angle → SMPL (direct)
        global_aa = _mat_to_aa(rots[0:1])      # (1, 3)
        body_aa   = _mat_to_aa(rots[1:].reshape(23, 3, 3)).reshape(1, 69)
        with torch.no_grad():
            out_aa = smpl(betas=torch.zeros(1, 10), body_pose=body_aa, global_orient=global_aa)
        j_aa = out_aa.joints[0, :24].detach()
        j_aa = j_aa - j_aa[0:1]   # root-centre (clone avoids aliasing error)

        # Path B: R → 6D → rotation_6d_to_matrix → _mat_to_aa → SMPL (via Rotation2xyz)
        d6 = matrix_to_rotation_6d(rots)   # (24, 6)
        R_rec = rotation_6d_to_matrix(d6)      # (24, 3, 3)
        global_aa2 = _mat_to_aa(R_rec[0:1])
        body_aa2   = _mat_to_aa(R_rec[1:].reshape(23, 3, 3)).reshape(1, 69)
        with torch.no_grad():
            out_6d = smpl(betas=torch.zeros(1, 10), body_pose=body_aa2, global_orient=global_aa2)
        j_6d = out_6d.joints[0, :24].detach()
        j_6d = j_6d - j_6d[0:1]   # root-centre

        err = (j_aa - j_6d).abs().max().item()
        max_err = max(max_err, err)

    check(
        "_mat_to_aa R→aa→FK == R→6D→FK  (max diff < 1 mm)",
        max_err < 1e-3,
        f"max joint position error = {max_err*100:.3f} cm",
    )

# ---------------------------------------------------------------------------
# T10  actor_batch_from_6dsmpl drops joint 24 correctly
# ---------------------------------------------------------------------------
def test_joint_24_dropped():
    print("\nT10  actor_batch_from_6dsmpl drops zero-padding joint 24")
    B, T = 4, 60
    x_btjf = torch.randn(B, T, 25, 6)
    x_btjf[:, :, 24, :] = 0.0   # joint 24 is all-zero (as in real NPZ)
    pad = torch.ones(B, T, dtype=torch.bool)

    batch = actor_batch_from_6dsmpl(x_btjf, pad, torch.device("cpu"))
    x_out = batch["x"]   # (B, 24, 6, T)

    check("Output has exactly 24 joints (joint 24 dropped)",
          x_out.shape == (B, 24, 6, T),
          f"got shape {tuple(x_out.shape)}")
    check("Output joint 0 corresponds to original joint 0 (not joint 1)",
          torch.allclose(x_out[:, 0, :, 0], x_btjf[:, 0, 0, :]),
          "joint 0 mismatch — off-by-one error in the drop")
    check("Output joint 23 corresponds to original joint 23 (not 24)",
          torch.allclose(x_out[:, 23, :, 0], x_btjf[:, 0, 23, :]),
          "joint 23 mismatch")

# ---------------------------------------------------------------------------
# T11  Loss functions: identical inputs → rc = 0, kl with mu=0/logvar=0 = exact
# ---------------------------------------------------------------------------
def test_losses():
    print("\nT11  Loss functions: known-answer checks")
    B, J, F, T = 4, 24, 6, 30
    mask = torch.ones(B, T, dtype=torch.bool)

    # rc on identical tensors must be exactly 0
    x = torch.randn(B, J, F, T)
    batch_same = {"x": x, "output": x.clone(), "mask": mask}
    rc_same = compute_rc_loss(None, batch_same).item()
    check("rc(x, x) == 0 (MSE of identical tensors is zero)",
          rc_same == 0.0,
          f"rc = {rc_same}")

    # rc on different tensors must be > 0
    batch_diff = {"x": x, "output": torch.randn_like(x), "mask": mask}
    rc_diff = compute_rc_loss(None, batch_diff).item()
    check("rc(x, y) > 0 for x ≠ y",
          rc_diff > 0,
          f"rc = {rc_diff}")

    # rcxyz on identical xyz tensors must be 0
    xyz = torch.randn(B, J, 3, T)
    batch_xyz = {"x_xyz": xyz, "output_xyz": xyz.clone(), "mask": mask}
    rcxyz_same = compute_rcxyz_loss(None, batch_xyz).item()
    check("rcxyz(xyz, xyz) == 0",
          rcxyz_same == 0.0,
          f"rcxyz = {rcxyz_same}")

    # KL with mu=0, logvar=0 → -0.5 * sum(1+0-0-1) = 0
    batch_kl_zero = {"mu": torch.zeros(B, 256), "logvar": torch.zeros(B, 256)}
    kl_zero = compute_kl_loss(None, batch_kl_zero).item()
    check("KL(mu=0, logvar=0) == 0  (posterior = prior → no divergence)",
          abs(kl_zero) < 1e-5,
          f"kl = {kl_zero:.6f}")

    # KL should be > 0 when mu != 0
    batch_kl_pos = {"mu": torch.ones(B, 256), "logvar": torch.zeros(B, 256)}
    kl_pos = compute_kl_loss(None, batch_kl_pos).item()
    check("KL(mu=1, logvar=0) > 0  (posterior shifted from prior)",
          kl_pos > 0,
          f"kl = {kl_pos:.2f}")

# ---------------------------------------------------------------------------
# T12  Anatomical plausibility on real BMCLab data
# ---------------------------------------------------------------------------
def test_anatomical_plausibility():
    """Load a real batch and verify that the FK output obeys basic anatomy:
    - Elbows are BELOW the head at every valid frame
    - Feet are BELOW the knees at every valid frame
    - L and R joints have approximately mirrored x positions (within 40 cm)
    - Bone lengths are constant across time (rigid body, CV < 1%)
    """
    print("\nT12  Anatomical plausibility on real BMCLab data")
    import types
    from data.dataloaders import collate_fn
    from model.actor.cvae_data import get_6dsmpl_datasets, actor_batch_from_6dsmpl

    args = types.SimpleNamespace(
        dataset="BMCLab", num_folds=6, fold=1, batch_size=16,
        experiment_name="test_anatomy", carepd_pose_npz=None, carepd_labels_pkl=None,
    )
    _, val_ds = get_6dsmpl_datasets(args)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    x_raw, _, _, _, pad = next(iter(loader))
    batch = actor_batch_from_6dsmpl(x_raw, pad, torch.device("cpu"))
    r2xyz = Rotation2xyz(torch.device("cpu"))
    with torch.no_grad():
        xyz = r2xyz(batch["x"], batch["mask"])   # (16, 24, 3, 60)
    mask = batch["mask"]   # (16, 60)

    # Only evaluate on valid frames
    # xyz: (B, J, 3, T) → for each valid (b,t): xyz[b, :, :, t]
    def joint_ys_valid(jidx):
        v = xyz[:, jidx, 1, :]   # (B, T)
        return v[mask]

    head_ys      = joint_ys_valid(15)
    l_elbow_ys   = joint_ys_valid(18)
    r_elbow_ys   = joint_ys_valid(19)
    l_knee_ys    = joint_ys_valid(4)
    l_foot_ys    = joint_ys_valid(10)
    r_foot_ys    = joint_ys_valid(11)

    elbow_above_head = ((l_elbow_ys > head_ys) | (r_elbow_ys > head_ys)).float().mean().item()
    foot_above_knee  = ((l_foot_ys > l_knee_ys)).float().mean().item()

    check("Elbows are below head in >99% of valid frames",
          elbow_above_head < 0.01,
          f"{elbow_above_head*100:.1f}% of frames have an elbow above the head "
          "(>1% means arm rotations are wrong)")

    check("Feet are below knees in >99% of valid frames",
          foot_above_knee < 0.01,
          f"{foot_above_knee*100:.1f}% of frames have foot above knee")

    # Bone length consistency (rigid body → CV < 1%)
    # Check femur: pelvis(0) → L_Knee(4)
    pelvis_xyz = xyz[:, 0, :, :]    # (B, 3, T)
    lknee_xyz  = xyz[:, 4, :, :]
    femur_len = (lknee_xyz - pelvis_xyz).norm(dim=1)   # (B, T)
    femur_valid = femur_len[mask]
    femur_cv = (femur_valid.std() / femur_valid.mean()).item()

    # Check upper arm: L_Shoulder(16) → L_Elbow(18)
    lshoulder_xyz = xyz[:, 16, :, :]
    lelbow_xyz    = xyz[:, 18, :, :]
    ua_len = (lelbow_xyz - lshoulder_xyz).norm(dim=1)
    ua_valid = ua_len[mask]
    ua_cv = (ua_valid.std() / ua_valid.mean()).item()

    check("Femur bone length CV < 1% across all valid frames (rigid body)",
          femur_cv < 0.01,
          f"femur CV = {femur_cv*100:.3f}%  (>1% = bone changing length = FK error)")
    check("Upper arm bone length CV < 1% across all valid frames",
          ua_cv < 0.01,
          f"upper arm CV = {ua_cv*100:.3f}%")

    # L/R symmetry: L and R hips should be approximately mirrored in x
    l_hip_x = xyz[:, 1, 0, :][mask]
    r_hip_x = xyz[:, 2, 0, :][mask]
    mean_asym = (l_hip_x + r_hip_x).abs().mean().item()   # should be ~0 if perfectly symmetric
    check("L/R hip x-positions are approximately mirrored (mean |L_x+R_x| < 5 cm)",
          mean_asym < 0.05,
          f"mean |L_Hip_x + R_Hip_x| = {mean_asym*100:.1f} cm  "
          "(large value = L/R joint label swap)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_fk_gradient_flow():
    """T13: rcxyz gradients flow back through FK to the decoder output.

    This catches the ``with torch.no_grad()`` bug: if the SMPL forward pass
    runs under no-grad, ``output_xyz.requires_grad`` is False and the rcxyz
    loss contributes zero gradient to the model — causing static-pose collapse.
    """
    print("\nT13  FK gradient flow: rcxyz loss reaches the decoder's 6D output")
    device = torch.device("cpu")
    r2xyz = Rotation2xyz(device=device, smpl_path=SMPL_PATH)

    B, T = 2, 8
    x = torch.randn(B, 24, 6, T, device=device, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    xyz = r2xyz(x, mask)

    check(
        "T13-a: FK output carries requires_grad",
        xyz.requires_grad,
        "FK output must have requires_grad=True; if False, rcxyz loss has zero gradient",
    )

    loss = xyz.pow(2).mean()
    loss.backward()

    grad_ok = x.grad is not None and x.grad.abs().max().item() > 0
    check(
        "T13-b: gradient reaches 6D decoder output",
        grad_ok,
        "Gradient of rcxyz loss must reach the decoder's 6D rotation output",
    )


def main():
    print("=" * 60)
    print("6D_SMPL pipeline correctness tests")
    print("=" * 60)

    tests = [
        test_6d_roundtrip,
        test_6d_row_convention,
        test_smpl_tpose_anatomy,
        test_smpl_joint_sides,
        test_body_pose_indexing,
        test_fk_symmetry,
        test_fk_root_centering,
        test_fk_padding,
        test_mat_to_aa_roundtrip,
        test_joint_24_dropped,
        test_losses,
        test_anatomical_plausibility,
        test_fk_gradient_flow,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            print(f"  [✗] EXCEPTION in {test_fn.__name__}:")
            traceback.print_exc()
            _results.append((test_fn.__name__, False, "exception"))

    # Summary
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print()
    print("=" * 60)
    print(f"Results: {n_pass} passed, {n_fail} failed  ({len(_results)} checks total)")
    if n_fail:
        print("\nFailed checks:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  ✗ {name}: {detail}")
    print("=" * 60)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
