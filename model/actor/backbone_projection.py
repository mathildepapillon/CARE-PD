"""backbone_projection.py — Project ActorSHAP 3D completions into each backbone's input space.

ActorSHAP generates sequences in raw root-centred 3D space (pelvis at origin,
mm-scale, standard H36M joint order 0-16).  Each CARE-PD backbone was trained
on a different representation of the same underlying skeleton.  This module
provides a single ``project_for_backbone`` entry point that converts ActorSHAP
output into the correct input format for each backbone, enabling apple-to-apple
SHAP comparison across all CARE-PD feature encoders.

Supported backbones
-------------------
potr            root-centred world 3D → z-score normalise → reorder joints
                (MIRRORED order) → merge to (B, T, 51).

motionbert      root-centred world 3D → 40° Y-rotation (backright view) →
motionagformer  world→cam → crop_scale xy to [-1,1] → replace z with
                confidence=1 → (B, T, J, 3).

mixste          root-centred world 3D → 40° Y-rotation (backright view) →
poseformerv2    world→cam → perspective project → screen normalise to [-1,1]
                → (B, T, J, 2).

Not supported (different representation than 3D xyz)
-------------------
momask          HumanML3D 263-dim features (requires full SMPL mesh)
motionclip      6D SMPL rotation matrices (requires SMPL body model)

Camera parameters
-----------------
All 2D projections use the CARE-PD ``backright`` view, whose parameters are
hard-coded in ``data/preprocessing/smpl2h36m.py``:

  image resolution    1100 × 1100 px
  focal length        fx = fy = 700
  principal point     cx = cy = 550
  world→cam rotation  R = [[-1,0,0],[0,-1,0],[0,0,1]]
  world→cam trans     t = [[0, 1, 2]]   (camera 1 m above, 2 m behind)
  backright pre-rot   40° around the Y-axis (applied before world→cam)

Z-score statistics
------------------
POTR computes normalisation stats over ALL sequences in the dataset
(train + test, before folding).  Pass sequences from both splits to
``compute_zscore_stats`` for the most faithful approximation.
In practice, using only the training pool introduces negligible error for
large datasets like BMCLab.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Joint ordering constants (from data/dataloaders.py)
# ---------------------------------------------------------------------------

# Standard H36M order — identity permutation (joints 0–16 unchanged).
_STANDARD_JOINTS = list(range(17))

# POTR uses a left-right mirrored joint order.
_MIRRORED_JOINTS = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

# ---------------------------------------------------------------------------
# Camera constants for the CARE-PD backright view (smpl2h36m.py)
# ---------------------------------------------------------------------------

_IMG_WH = 1100       # image width = height in pixels
_FX = _FY = 700.0   # focal length (pixels)
_CX = _CY = _IMG_WH / 2.0  # principal point = 550.0

# World→camera rotation for the backright view.
_R_BACKRIGHT = torch.tensor(
    [[-1., 0., 0.],
     [ 0., -1., 0.],
     [ 0.,  0., 1.]],
    dtype=torch.float32,
)

# World→camera translation (camera height 1 m, 2 m behind subject).
_T_BACKRIGHT = torch.tensor([[0., 1., 2.]], dtype=torch.float32)  # (1, 3)

# Additional 40° Y-axis pre-rotation applied to world coords for backright.
_BACKRIGHT_ANGLE_DEG = 40.0


# ---------------------------------------------------------------------------
# Z-score utilities (POTR)
# ---------------------------------------------------------------------------

def compute_zscore_stats(
    sequences: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-joint per-feature mean and std from root-centred sequences.

    POTR computes these stats over all sequences in the dataset (before fold
    splitting).  Pass ``torch.cat([train_pool, test_pool], dim=0)`` for the
    most faithful approximation, or just ``train_pool`` as an approximation.

    Args:
        sequences: ``(N, J, F, T)`` float32 tensor in standard H36M joint
                   order, root-centred (pelvis at origin).

    Returns:
        mean: ``(J, F)`` per-joint per-feature mean.
        std:  ``(J, F)`` per-joint per-feature std, clamped to ≥ 1e-6.
    """
    N, J, F, T = sequences.shape
    # (N, J, F, T) → (N, T, J, F) → (N*T, J, F)
    flat = sequences.permute(0, 3, 1, 2).reshape(N * T, J, F)
    mean = flat.mean(dim=0)                    # (J, F)
    std  = flat.std(dim=0).clamp(min=1e-6)    # (J, F)
    return mean, std


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _rotation_y(angle_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a (3, 3) rotation matrix for a *angle_deg* rotation around Y."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor(
        [[c, 0., s], [0., 1., 0.], [-s, 0., c]],
        dtype=dtype, device=device,
    )


def _world_to_cam(
    x: torch.Tensor,    # (B, T, J, 3)
    R: torch.Tensor,    # (3, 3)
    t: torch.Tensor,    # (1, 3)
) -> torch.Tensor:
    """Apply rigid world→camera transform X_cam = X_world @ Rᵀ + t.

    Mirrors ``smpl2h36m.world_to_camera`` (which uses np.dot(X_w, R.T) + t).
    """
    return x @ R.T + t   # broadcast: (B,T,J,3) @ (3,3) + (1,1,3)


def _perspective_project(
    x_cam: torch.Tensor,   # (B, T, J, 3)
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """Pinhole perspective projection; returns (B, T, J, 2) image coordinates."""
    z = x_cam[..., 2:3].clamp(min=1e-4)       # (B, T, J, 1)
    u = x_cam[..., 0:1] / z * fx + cx
    v = x_cam[..., 1:2] / z * fy + cy
    return torch.cat([u, v], dim=-1)           # (B, T, J, 2)


def _crop_scale(x_cam: torch.Tensor) -> torch.Tensor:
    """Normalise (B, T, J, 3) camera-frame coords to [-1, 1] via bounding box.

    Mirrors ``DataPreprocessor.crop_scale``:
      - Valid joints: those with z ≠ 0.
      - Scale = max(xmax-xmin, ymax-ymin) across valid joints in the clip.
      - xy is mapped to [-1, 1]; z channel is preserved unchanged.
      - Returns zeros for degenerate clips (< 4 valid xy points).
    """
    B, T, J, _ = x_cam.shape
    result = x_cam.clone()

    for b in range(B):
        clip = x_cam[b]                              # (T, J, 3)
        valid = clip[..., 2] != 0                    # (T, J) bool
        valid_xy = clip[valid][:, :2]                # (N_valid, 2)

        if valid_xy.shape[0] < 4:
            result[b] = 0.
            continue

        xmin, ymin = valid_xy.min(dim=0).values
        xmax, ymax = valid_xy.max(dim=0).values
        scale = max((xmax - xmin).item(), (ymax - ymin).item())

        if scale == 0:
            result[b] = 0.
            continue

        xs = ((xmin + xmax) / 2 - scale / 2)
        ys = ((ymin + ymax) / 2 - scale / 2)
        result[b, :, :, 0] = (clip[:, :, 0] - xs) / scale
        result[b, :, :, 1] = (clip[:, :, 1] - ys) / scale
        # Map [0, 1] → [-1, 1] and clamp
        result[b, :, :, :2] = (result[b, :, :, :2] - 0.5) * 2
        result[b, :, :, :2] = result[b, :, :, :2].clamp(-1., 1.)

    return result


def _screen_normalize(
    x_img: torch.Tensor,   # (B, T, J, 2)  image pixel coordinates
    w: int = _IMG_WH,
    h: int = _IMG_WH,
) -> torch.Tensor:
    """Map pixel coordinates [0, w] → [-1, 1] preserving aspect ratio.

    Mirrors ``MixSTEPreprocessor.normalize_screen_coordinates``::
        (xy / w) * 2 - [1, h/w]
    """
    offset = torch.tensor([1., h / w], dtype=x_img.dtype, device=x_img.device)
    return (x_img / w) * 2 - offset


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def project_for_backbone(
    x_actor: torch.Tensor,                    # (B, J=17, F=3, T)
    backbone_name: str,
    zscore_mean: Optional[torch.Tensor] = None,   # (J, F)  — POTR only
    zscore_std:  Optional[torch.Tensor] = None,   # (J, F)  — POTR only
    pad_mask: Optional[torch.Tensor] = None,       # (B, T) bool — for confidence
) -> torch.Tensor:
    """Project root-centred 3D sequences into a CARE-PD backbone's input space.

    All projections start from ``x_actor`` in the same root-centred world
    frame (pelvis at origin, standard H36M joint order 0–16).  The function
    applies the exact same sequence of transforms that each backbone's
    preprocessor applies to the raw stored NPZ, so that ActorSHAP completions
    and all SHAP baselines (zero / mean / marginal) are evaluated on a
    classifier that sees in-distribution input.

    Args:
        x_actor:     ``(B, J=17, F=3, T)`` root-centred sequences.
        backbone_name: One of ``'potr'``, ``'motionbert'``,
                       ``'motionagformer'``, ``'mixste'``, ``'poseformerv2'``.
        zscore_mean: ``(J, F)`` training-set mean  (required for POTR).
        zscore_std:  ``(J, F)`` training-set std   (required for POTR).
        pad_mask:    ``(B, T)`` bool mask, True = valid frame.  Used to set
                     the confidence channel to 0 on padded frames for
                     MotionBERT / MotionAGFormer.  Defaults to all-ones.

    Returns:
        Tensor in the backbone's native input format:

        =========== ===================== =========================================
        Backbone    Output shape          Description
        =========== ===================== =========================================
        potr        (B, T, J*F=51)        z-scored, MIRRORED joint order, merged
        motionbert  (B, T, J=17, 3)       crop-scaled xy + confidence (0/1)
        motionagf.  (B, T, J=17, 3)       same as motionbert
        mixste      (B, T, J=17, 2)       screen-normalised image coords
        poseform.   (B, T, J=17, 2)       same as mixste
        =========== ===================== =========================================

    Raises:
        ValueError:
            For unsupported backbones (momask, motionclip) or if zscore
            stats are missing for POTR.
    """
    bn = backbone_name.lower()
    B, J, F, T = x_actor.shape
    device = x_actor.device
    dtype  = x_actor.dtype

    # (B, J, F, T) → (B, T, J, F=3)
    x_bttj = x_actor.permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------
    # POTR: z-score + joint reorder + flatten
    # ------------------------------------------------------------------
    if bn == 'potr':
        if zscore_mean is None or zscore_std is None:
            raise ValueError(
                "zscore_mean and zscore_std are required for backbone='potr'. "
                "Compute them from the training pool with compute_zscore_stats()."
            )
        m = zscore_mean.to(device=device, dtype=dtype)  # (J, F)
        s = zscore_std.to(device=device, dtype=dtype)   # (J, F)
        # Broadcast: (1, 1, J, F)
        x_norm = (x_bttj - m) / s                       # (B, T, J, F)
        # Reorder joints to MIRRORED order (as POTR was trained on).
        x_norm = x_norm[:, :, _MIRRORED_JOINTS, :]      # (B, T, J, F)
        return x_norm.reshape(B, T, J * F)               # (B, T, 51)

    # ------------------------------------------------------------------
    # MotionBERT / MotionAGFormer: world→cam→crop_scale + confidence
    # ------------------------------------------------------------------
    if bn in ('motionbert', 'motionagformer'):
        # 1. Apply 40° Y-rotation (backright-view pre-rotation).
        R_y = _rotation_y(_BACKRIGHT_ANGLE_DEG, device, dtype)
        x_rot = x_bttj @ R_y.T                          # (B, T, J, 3)

        # 2. World→camera (R, t from smpl2h36m.py backright).
        R = _R_BACKRIGHT.to(device=device, dtype=dtype)
        t = _T_BACKRIGHT.to(device=device, dtype=dtype)
        x_cam = _world_to_cam(x_rot, R, t)              # (B, T, J, 3)

        # 3. crop_scale: bbox-normalise xy to [-1, 1]; z preserved.
        x_cs = _crop_scale(x_cam)                       # (B, T, J, 3)

        # 4. Replace z with per-frame confidence score (1 = valid, 0 = pad).
        if pad_mask is not None:
            # (B, T) → (B, T, 1, 1) → (B, T, J, 1)
            conf = pad_mask.to(dtype=dtype, device=device)
            conf = conf.unsqueeze(-1).unsqueeze(-1).expand(B, T, J, 1)
        else:
            conf = torch.ones(B, T, J, 1, dtype=dtype, device=device)

        return torch.cat([x_cs[..., :2], conf], dim=-1)  # (B, T, J, 3)

    # ------------------------------------------------------------------
    # MixSTE / PoseFormerV2: world→cam→perspective→screen_normalise
    # ------------------------------------------------------------------
    if bn in ('mixste', 'poseformerv2'):
        # 1. Apply 40° Y-rotation (backright-view pre-rotation).
        R_y = _rotation_y(_BACKRIGHT_ANGLE_DEG, device, dtype)
        x_rot = x_bttj @ R_y.T                          # (B, T, J, 3)

        # 2. World→camera.
        R = _R_BACKRIGHT.to(device=device, dtype=dtype)
        t = _T_BACKRIGHT.to(device=device, dtype=dtype)
        x_cam = _world_to_cam(x_rot, R, t)              # (B, T, J, 3)

        # 3. Pinhole perspective projection → image pixel coords.
        x_img = _perspective_project(x_cam, _FX, _FY, _CX, _CY)  # (B, T, J, 2)

        # 4. Screen normalise to [-1, 1].
        return _screen_normalize(x_img)                  # (B, T, J, 2)

    # ------------------------------------------------------------------
    # Unsupported backbones
    # ------------------------------------------------------------------
    if bn in ('momask', 'motionclip'):
        raise ValueError(
            f"Backbone '{backbone_name}' is incompatible with ActorSHAP's 3D XYZ "
            "output.  MoMask requires HumanML3D 263-dim features; MotionCLIP "
            "requires 6D SMPL rotation matrices.  Both require a SMPL body model "
            "fit and cannot be derived from H36M joint positions alone."
        )

    raise ValueError(
        f"Unknown backbone '{backbone_name}'.  Supported for SHAP: "
        "potr, motionbert, motionagformer, mixste, poseformerv2."
    )
