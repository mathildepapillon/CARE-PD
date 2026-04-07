"""
rotation2xyz.py — Forward kinematics replicating ACTOR's Rotation2xyz exactly.

Taken directly from:
  https://github.com/Mathux/ACTOR/blob/master/src/models/rotation2xyz.py
  https://github.com/Mathux/ACTOR/blob/master/src/models/smpl.py

Key design choices (from ACTOR, not invented here):
  - Uses SMPLLayer (not SMPL) from smplx.
  - Passes rotation matrices DIRECTLY to SMPLLayer as (N, 3, 3) and (N, 23, 3, 3).
    No conversion to axis-angle — that conversion has unstable gradients near
    identity rotations (which is where most gait joints live).
  - Root-centres the output: x_xyz = x_xyz - x_xyz[:, [0], :, :]  (ACTOR line).
  - rotation_6d_to_matrix is copied verbatim from ACTOR's rotation_conversions.py
    (which is in turn from PyTorch3D).
"""

import contextlib

import torch
import torch.nn.functional as F
from smplx import SMPLLayer as _SMPLLayer

import os as _os
_DEFAULT_SMPL_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "data", "preprocessing", "common", "body_models", "smpl", "SMPL_NEUTRAL.pkl",
)

SMPL_NJOINTS = 24


# ---------------------------------------------------------------------------
# rotation_6d_to_matrix — copied verbatim from ACTOR / PyTorch3D
# ---------------------------------------------------------------------------

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation → 3×3 rotation matrix (Gram–Schmidt, Zhou et al. CVPR 2019).

    Args:
        d6: (..., 6)
    Returns:
        (..., 3, 3)  — rows are the three orthonormal basis vectors.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)   # stack as rows → (*, 3, 3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix → 6D representation (first two rows).

    Args:
        matrix: (..., 3, 3)
    Returns:
        (..., 6)
    """
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


# ---------------------------------------------------------------------------
# SMPL wrapper — matches ACTOR's smpl.py (simplified: smpl jointstype only)
# ---------------------------------------------------------------------------

class _SMPL(_SMPLLayer):
    """Thin SMPLLayer wrapper that accepts rotation matrices directly.

    ACTOR passes body_pose as (N, 23, 3, 3) and global_orient as (N, 3, 3).
    SMPLLayer handles this without any explicit pose2rot flag.
    """

    def __init__(self, model_path: str):
        with contextlib.redirect_stdout(None):
            super().__init__(model_path=model_path)
        self.eval()
        self.requires_grad_(False)

    def forward(self, global_orient, body_pose, betas=None):
        if betas is None:
            betas = torch.zeros(
                global_orient.shape[0], self.num_betas,
                dtype=global_orient.dtype, device=global_orient.device,
            )
        out = super().forward(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
        )
        # Return only the 24 canonical SMPL joints
        return out.joints[:, :SMPL_NJOINTS, :]   # (N, 24, 3)


# ---------------------------------------------------------------------------
# Rotation2xyz — matches ACTOR's Rotation2xyz.__call__ for rot6d + smpl joints
# ---------------------------------------------------------------------------

class Rotation2xyz:
    """Convert (B, 24, 6, T) 6D rotations → (B, 24, 3, T) root-centred XYZ.

    Replicates ACTOR's Rotation2xyz for the ``pose_rep="rot6d"`` /
    ``jointstype="smpl"`` case used in CARE-PD.

    The SMPL model is loaded lazily on the first call (or to a specified device)
    so importing this module is cheap.
    """

    def __init__(self, device: torch.device, smpl_path: str | None = None):
        self.device = device
        self.smpl_path = smpl_path or _DEFAULT_SMPL_PATH
        self._smpl: _SMPL | None = None

    def _ensure_smpl(self) -> None:
        if self._smpl is None:
            self._smpl = _SMPL(self.smpl_path).to(self.device)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, 24, 6, T)  — 6D rotation features, ACTOR layout.
            mask: (B, T)         — True for valid frames.

        Returns:
            (B, 24, 3, T)  — root-centred XYZ joint positions.
        """
        self._ensure_smpl()
        B, J, F, T = x.shape
        assert J == 24 and F == 6, f"Expected (B,24,6,T), got {x.shape}"

        # ---- Step 1: permute → (B, T, 24, 6) then flatten time & batch ----
        x_btj6 = x.permute(0, 3, 1, 2)              # (B, T, 24, 6)

        # ACTOR only runs FK on *masked* (valid) frames to save compute.
        # We mimic that: gather valid (b, t) pairs, run SMPL, scatter back.
        mask_flat = mask.reshape(B * T)              # (B*T,)
        x_flat    = x_btj6.reshape(B * T, J, F)     # (B*T, 24, 6)
        x_masked  = x_flat[mask_flat]               # (N_valid, 24, 6)

        # ---- Step 2: 6D → 3×3 rotation matrices (ACTOR's exact function) --
        rot_mat   = rotation_6d_to_matrix(x_masked)  # (N_valid, 24, 3, 3)

        # ---- Step 3: separate global_orient / body_pose and call SMPL ------
        # ACTOR passes rotation matrices directly — no axis-angle conversion.
        #   global_orient : (N_valid, 3, 3)   ← joint 0
        #   body_pose     : (N_valid, 23, 3, 3) ← joints 1-23
        global_orient = rot_mat[:, 0]                # (N_valid, 3, 3)
        body_pose     = rot_mat[:, 1:]               # (N_valid, 23, 3, 3)

        joints_masked = self._smpl(
            global_orient=global_orient,
            body_pose=body_pose,
        )                                            # (N_valid, 24, 3)

        # ---- Step 4: scatter back into (B*T, 24, 3) with zeros for padding -
        joints_flat = x.new_zeros(B * T, J, 3)
        joints_flat[mask_flat] = joints_masked

        # ---- Step 5: reshape → (B, T, 24, 3) then root-centre -------------
        joints = joints_flat.view(B, T, J, 3)       # (B, T, 24, 3)

        # ACTOR: x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]  (rootindex=0)
        root   = joints[:, :, 0:1, :]               # (B, T, 1, 3)
        joints = joints - root                       # (B, T, 24, 3)

        # Zero padded frames (they were zeroed pre-root-centre; re-zero post)
        pad_expand = mask.unsqueeze(-1).unsqueeze(-1)   # (B, T, 1, 1)
        joints = joints * pad_expand.float()

        # ---- Step 6: permute to ACTOR's (B, J, 3, T) convention -----------
        return joints.permute(0, 2, 3, 1).contiguous()  # (B, 24, 3, T)
