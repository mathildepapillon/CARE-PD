"""backbone_projection.py — Project ActorSHAP 3D completions into each backbone's input space.

ActorSHAP generates sequences in the *global-pelvis* representation (joint 0 =
global world position, joints 1–16 = relative to pelvis; see
``model.actor.motion_utils``).  Callers are expected to call
``unroot_to_global`` **before** invoking ``project_for_backbone`` so that this
module always receives full global 3D world-space sequences.

Each CARE-PD backbone was trained on a different representation of the same
underlying skeleton.  This module provides a single ``project_for_backbone``
entry point that converts global 3D output into the correct input format for
each backbone, enabling apple-to-apple SHAP comparison across all CARE-PD
feature encoders.

Supported backbones
-------------------
potr            global world 3D → root-centre (pelvis subtracted) →
                z-score normalise → reorder joints (MIRRORED order) →
                merge to (B, T, 51).

motionbert      global world 3D → 40° Y-rotation (backright view) →
motionagformer  world→cam → crop_scale xy to [-1,1] → replace z with
                confidence=1 → (B, T, J, 3).

mixste          global world 3D → 40° Y-rotation (backright view) →
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

crop_scale note
---------------
The 2D backbones (MotionBERT / MotionAGFormer / MixSTE / PoseFormerV2) were
trained on full global walking trajectories, so their crop_scale bounding boxes
spanned the complete walk.  By passing unrooted global sequences into this
function those backbones now receive in-distribution bounding-box extents,
matching what they saw during training.
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

# POTR and MotionAGFormer use a left-right mirrored joint order
# (matches BACKBONES_WITH_MIRRORED_JOINTS in const.py).
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
        std:  ``(J, F)`` per-joint per-feature std — features with std
              below ``1e-4`` are set to ``1.0`` (matching the training
              ``_MIN_STD`` convention in ``data/dataloaders.py``).
    """
    N, J, F, T = sequences.shape
    # (N, J, F, T) → (N, T, J, F) → (N*T, J, F)
    flat = sequences.permute(0, 3, 1, 2).reshape(N * T, J, F)
    mean = flat.mean(dim=0)                    # (J, F)
    std  = flat.std(dim=0)                     # (J, F)
    # Match dataloaders.py: std[std < _MIN_STD] = 1 (not clamped to _MIN_STD)
    std  = torch.where(std < 1e-4, torch.ones_like(std), std)
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


def _crop_scale(
    x_cam: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalise (B, T, J, 3) camera-frame coords to [-1, 1] via bounding box.

    Mirrors ``DataPreprocessor.crop_scale``:
      - Scale = max(xmax-xmin, ymax-ymin) across valid joints in the clip.
      - xy is mapped to [-1, 1]; z channel is preserved unchanged.
      - Returns zeros for degenerate clips (< 4 valid xy points).

    Valid-frame detection
    ~~~~~~~~~~~~~~~~~~~~~
    When ``pad_mask`` is provided (``(B, T)`` bool, True = valid frame), it is
    used to exclude padded frames from the bounding-box computation.  This is
    the correct behaviour for the Actor pipeline: padded frames are
    ``(0,0,0)`` in root-centred world space, but after ``_world_to_cam`` they
    land at ``z = t_z = 2`` (non-zero), so the fallback ``z != 0`` check would
    wrongly include them and inflate the bbox.

    When ``pad_mask`` is ``None`` (e.g. for camera-space data where padded
    frames have literal ``z = 0``), the original ``z != 0`` fallback is used,
    preserving backward compatibility.

    Implementation note: fully vectorised over the batch dimension to avoid
    per-sample Python loops on CUDA tensors (which are extremely slow for
    large batches due to per-kernel-launch overhead).
    """
    B, T, J, _ = x_cam.shape
    result = x_cam.clone()

    # Build per-sample, per-joint validity mask: (B, T, J) bool
    if pad_mask is not None:
        # pad_mask: (B, T) → (B, T, J)
        valid = pad_mask.unsqueeze(-1).expand(B, T, J)
    else:
        # Camera-space fallback: padded joints have z = 0
        valid = x_cam[..., 2] != 0   # (B, T, J)

    # Count valid joints per sequence: (B,)
    n_valid = valid.sum(dim=[1, 2])

    # Build sentinel-filled xy for vectorised min/max
    _INF = 1e9
    xy = x_cam[..., :2]                                    # (B, T, J, 2)
    valid2 = valid.unsqueeze(-1).expand_as(xy)             # (B, T, J, 2)
    xy_for_min = torch.where(valid2, xy, xy.new_full(xy.shape, _INF))
    xy_for_max = torch.where(valid2, xy, xy.new_full(xy.shape, -_INF))

    xy_min_flat = xy_for_min.reshape(B, -1, 2)
    xy_max_flat = xy_for_max.reshape(B, -1, 2)
    xmin = xy_min_flat[..., 0].min(dim=1).values           # (B,)
    xmax = xy_max_flat[..., 0].max(dim=1).values
    ymin = xy_min_flat[..., 1].min(dim=1).values
    ymax = xy_max_flat[..., 1].max(dim=1).values

    scale = torch.maximum(xmax - xmin, ymax - ymin)        # (B,)

    # Degenerate: too few valid joints, or flat bbox
    degenerate = (n_valid < 4) | (scale <= 0)

    # Avoid divide-by-zero for degenerate rows (they'll be zeroed out anyway)
    safe_scale = scale.masked_fill(degenerate, 1.0)

    # Bounding-box origin shifted so bbox spans [0, 1]
    xs = ((xmin + xmax) / 2 - safe_scale / 2)[:, None, None]   # (B, 1, 1)
    ys = ((ymin + ymax) / 2 - safe_scale / 2)[:, None, None]
    s  = safe_scale[:, None, None]

    # Normalise to [0, 1] then remap to [-1, 1] in a single fused step
    result[..., 0] = ((result[..., 0] - xs) / s - 0.5) * 2
    result[..., 1] = ((result[..., 1] - ys) / s - 0.5) * 2
    result[..., :2] = result[..., :2].clamp(-1., 1.)

    # Zero out degenerate sequences (broadcast over T, J, 3)
    result[degenerate] = 0.

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
    """Project global 3D sequences into a CARE-PD backbone's input space.

    ``x_actor`` must be in **global world-space** (all joints in the lab frame,
    standard H36M joint order 0–16).  Callers should call
    ``motion_utils.unroot_to_global`` before invoking this function when the
    source sequences are in global-pelvis format.

    The function applies the exact same sequence of transforms that each
    backbone's preprocessor applies to the raw stored NPZ, so that ActorSHAP
    completions and all SHAP baselines are evaluated on a classifier that sees
    in-distribution input.

    Args:
        x_actor:     ``(B, J=17, F=3, T)`` global world-space sequences.
        backbone_name: One of ``'potr'``, ``'motionbert'``,
                       ``'motionagformer'``, ``'mixste'``, ``'poseformerv2'``.
        zscore_mean: ``(J, F)`` training-set mean  (required for POTR).
        zscore_std:  ``(J, F)`` training-set std   (required for POTR).
        pad_mask:    ``(B, T)`` bool mask, True = valid frame.  Used both to
                     exclude padded frames from the crop_scale bounding box
                     (MotionBERT / MotionAGFormer) and to set the confidence
                     channel to 0 on padded frames.  Defaults to all-ones.

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
    # POTR: root-centre → z-score + joint reorder + flatten
    # ------------------------------------------------------------------
    if bn == 'potr':
        if zscore_mean is None or zscore_std is None:
            raise ValueError(
                "zscore_mean and zscore_std are required for backbone='potr'. "
                "Compute them from the training pool with compute_zscore_stats()."
            )
        # POTR was trained on root-centred sequences (pelvis at origin).
        # Subtract the pelvis before z-scoring.
        x_bttj = x_bttj - x_bttj[:, :, 0:1, :]         # (B, T, J, F)
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
        # Pass pad_mask so padded frames (z=2 in cam space after translation)
        # are excluded from the bounding-box computation.
        x_cs = _crop_scale(x_cam, pad_mask=pad_mask)    # (B, T, J, 3)

        # 4. Replace z with per-frame confidence score (1 = valid, 0 = pad).
        if pad_mask is not None:
            # (B, T) → (B, T, 1, 1) → (B, T, J, 1)
            conf = pad_mask.to(dtype=dtype, device=device)
            conf = conf.unsqueeze(-1).unsqueeze(-1).expand(B, T, J, 1)
        else:
            conf = torch.ones(B, T, J, 1, dtype=dtype, device=device)

        out = torch.cat([x_cs[..., :2], conf], dim=-1)   # (B, T, J, 3)

        # 5. MotionAGFormer was trained with mirrored joint order
        #    (BACKBONES_WITH_MIRRORED_JOINTS in const.py).
        if bn == 'motionagformer':
            out = out[:, :, _MIRRORED_JOINTS, :]

        return out

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
