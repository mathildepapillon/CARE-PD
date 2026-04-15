"""h36m_rotation2xyz.py — Differentiable FK for the H36M 32-joint skeleton.

Converts (B, 32, 6, T) 6D rotation features → (B, 17, 3, T) root-centred
XYZ positions, mirroring the NumPy FK in scripts/prepare_h36m_dataset.py
but fully differentiable (for the rcxyz loss).

No SMPL body model is needed — the H36M skeleton topology (parent array,
bone offsets) is hard-coded from the standard GGMotion / una-dinosauria
constants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.actor.rotation2xyz import rotation_6d_to_matrix

H36M_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# Per-joint mean 6D rotation velocity computed from the full H36M training set.
# Joints with 0.0 are completely static (root, fingers, toes) and should be
# excluded from rotation-space velocity losses.
H36M_32_JOINT_MEAN_VEL = [
    0.000000, 0.031911, 0.028671, 0.033217, 0.010987, 0.000000,  # 0-5
    0.032637, 0.029288, 0.032380, 0.011244, 0.000000, 0.016438,  # 6-11
    0.021278, 0.031630, 0.018268, 0.000000, 0.017981, 0.041014,  # 12-17
    0.025002, 0.056252, 0.000000, 0.000000, 0.000000, 0.000000,  # 18-23
    0.019579, 0.045580, 0.026925, 0.066384, 0.000000, 0.000000,  # 24-29
    0.000000, 0.000000,                                            # 30-31
]


def h36m_vel_joint_weights(njoints: int = 32) -> torch.Tensor:
    """Per-joint velocity weight vector for the 32-joint H36M rotation model.

    Static joints get weight 0; active joints get weight proportional to
    their empirical mean velocity, normalised so that the mean weight over
    active joints is 1.0.  This concentrates the velocity gradient on joints
    that actually move.
    """
    w = torch.tensor(H36M_32_JOINT_MEAN_VEL[:njoints], dtype=torch.float32)
    active = w > 0
    if active.any():
        w[active] = w[active] / w[active].mean()
    return w

_PARENT_RAW = [
    0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
    17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31,
]
PARENT = [p - 1 for p in _PARENT_RAW]  # -1 = root

OFFSET_FLAT = [
    0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000,
    0.000000, -442.894612, 0.000000, 0.000000, -454.206447, 0.000000,
    0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437,
    132.948826, 0.000000, 0.000000, 0.000000, -442.894413, 0.000000,
    0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
    0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000,
    0.000000, 233.383263, 0.000000, 0.000000, 257.077681, 0.000000,
    0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000,
    0.000000, 257.077681, 0.000000, 0.000000, 151.034226, 0.000000,
    0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627,
    0.000000, 100.000188, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000,
    0.000000, 278.892924, 0.000000, 0.000000, 251.728680, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
    0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000,
]


class H36MRotation2xyz(torch.nn.Module):
    """Convert (B, 32, 6, T) H36M 6D rotations → (B, 17, 3, T) XYZ in metres.

    The skeleton constants are stored as persistent buffers so they move
    automatically with ``.to(device)``.
    """

    def __init__(self):
        super().__init__()
        offset = torch.tensor(OFFSET_FLAT, dtype=torch.float32).view(32, 3)
        self.register_buffer("offset", offset)  # (32, 3) in mm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, 32, 6, T) — per-joint 6D rotations.
            mask: (B, T)        — True for valid frames.
        Returns:
            (B, 17, 3, T) — root-centred XYZ positions in metres.
        """
        B, J, F, T = x.shape
        assert J == 32 and F == 6, f"Expected (B,32,6,T), got {x.shape}"

        # (B, T, 32, 6) → (N, 32, 6) where N = B*T
        x_bt = x.permute(0, 3, 1, 2).reshape(B * T, J, F)
        mask_flat = mask.reshape(B * T)

        x_valid = x_bt[mask_flat]            # (N_valid, 32, 6)
        rot = rotation_6d_to_matrix(x_valid)  # (N_valid, 32, 3, 3)

        xyz_valid = self._fk(rot)            # (N_valid, 32, 3)

        xyz_flat = x.new_zeros(B * T, 32, 3)
        xyz_flat[mask_flat] = xyz_valid

        xyz = xyz_flat.view(B, T, 32, 3)
        xyz_17 = xyz[:, :, H36M_32_TO_17, :]   # (B, T, 17, 3)

        # root-centre (pelvis = index 0 of the 17-joint set)
        root = xyz_17[:, :, 0:1, :]
        xyz_17 = xyz_17 - root

        xyz_17 = xyz_17 * mask.unsqueeze(-1).unsqueeze(-1).float()

        return xyz_17.permute(0, 2, 3, 1).contiguous()  # (B, 17, 3, T)

    def _fk(self, rot: torch.Tensor) -> torch.Tensor:
        """Run FK for a batch of frames.

        Args:
            rot: (N, 32, 3, 3) — per-joint rotation matrices.
        Returns:
            (N, 32, 3) — world XYZ in metres.
        """
        N = rot.shape[0]
        # Avoid in-place writes so autograd can back-propagate through the FK.
        world_R_list: list[torch.Tensor] = [torch.empty(0)] * 32
        world_pos_list: list[torch.Tensor] = [torch.empty(0)] * 32

        for i in range(32):
            p = PARENT[i]
            R_local = rot[:, i]   # (N, 3, 3)
            if p == -1:
                world_R_list[i] = R_local
                world_pos_list[i] = self.offset[i].unsqueeze(0).expand(N, -1)
            else:
                # xyz[i] = offset[i] @ world_R[parent] + xyz[parent]
                world_pos_list[i] = (
                    torch.einsum("j,njk->nk", self.offset[i], world_R_list[p])
                    + world_pos_list[p]
                )
                # world_R[i] = R_local @ world_R[parent]
                world_R_list[i] = R_local @ world_R_list[p]

        # (N, 32, 3), mm → metres
        return torch.stack(world_pos_list, dim=1) / 1000.0
