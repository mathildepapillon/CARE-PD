"""motion_utils.py — Utilities for converting between rooted and global motion representations.

The CARE-PD / ActorSHAP pipeline uses a *global-pelvis* representation:

  - Joint 0 (pelvis): absolute world-space position.
  - Joints 1–16: position **relative to pelvis** at each frame.

This lets the CVAE model learn local body shape (joints 1–16) independently of
global trajectory (joint 0), while still allowing full global motion to be
recovered at any time via ``unroot_to_global``.

Conventions
-----------
All functions operate on ``(B, T, J, 3)`` tensors where:

  B = batch size
  T = sequence length (frames)
  J = number of joints
  F = 3 (XYZ)

Use ``permute(0, 3, 1, 2)`` / ``permute(0, 2, 3, 1)`` to convert between this
layout and ACTOR's ``(B, J, F, T)`` storage format.
"""

import torch
from torch import Tensor


def unroot_to_global(x: Tensor) -> Tensor:
    """Recover full global 3D from global-pelvis representation.

    Args:
        x: ``(B, T, J, 3)`` tensor where joint 0 contains the global pelvis
           position and joints 1..J-1 contain positions relative to the pelvis.

    Returns:
        ``(B, T, J, 3)`` tensor where **all** joints are in global world space.
        Joint 0 is unchanged; joint k is ``x[..., k, :] + x[..., 0, :]``.
    """
    pelvis = x[:, :, 0:1, :]   # (B, T, 1, 3) — global pelvis trajectory
    out = x.clone()
    out[:, :, 1:, :] = x[:, :, 1:, :] + pelvis
    return out


def root_with_global_pelvis(x: Tensor) -> Tensor:
    """Convert global 3D sequences to global-pelvis representation.

    Args:
        x: ``(B, T, J, 3)`` tensor where all joints are in global world space.

    Returns:
        ``(B, T, J, 3)`` tensor where joint 0 retains the global pelvis
        position and joints 1..J-1 are expressed relative to the pelvis.
        Calling ``unroot_to_global`` on the result recovers the original ``x``.
    """
    pelvis = x[:, :, 0:1, :]   # (B, T, 1, 3) — global pelvis trajectory
    out = x.clone()
    out[:, :, 1:, :] = x[:, :, 1:, :] - pelvis
    return out
