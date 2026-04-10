"""
Shared CARE-PD data helpers for Actor CVAE (no Lightning dependency).

Supported data modes
--------------------
``carepd``
    CARE-PD clinical H36M-style XYZ joint positions (17 joints × 3).
    Loaded via ``dataset_factory`` using the project's standard fold splits.
    Batch shape: ``(B, T, 17, 3)`` → ACTOR dict ``x: (B, 17, 3, T)``.

``6dsmpl``
    6D rotation representation of SMPL body pose from the 6D_SMPL pipeline.

    The stored NPZ has shape ``(T, 25, 6)`` per sequence, where:
    - Joints 0–23 are the 6D rotation matrices (from ``axis_angle_to_6d``)
      of SMPL's 24 joints (global_orient + 23 body joints).
    - Joint 24 is all-zeros — a placeholder added during preprocessing that
      was intended for translation but is unused.  We drop it.

    After dropping joint 24 the working shape is ``(T, 24, 6)``.  The
    dataloader returns ``(B, source_seq_len, 25, 6)``; we strip the last
    joint before building the ACTOR batch dict.

    ACTOR batch dict: ``x: (B, 24, 6, T)``.

    Rotation values are unit-orthonormal 6D vectors (first two columns of
    a rotation matrix); they are NOT unit quaternions or axis-angles.
"""

from __future__ import annotations

import os

import torch
from const import path
from data.dataloaders import dataset_factory

_SUPPORTED = ["BMCLab", "T-SDU-PD", "PD-GaM", "3DGait"]

# Number of valid SMPL joints in our 6D_SMPL data (drop the zero-padding joint).
SMPL_NJOINTS = 24


def build_motionclip_params(
    dataset_name,
    num_folds,
    batch_size,
    experiment_name,
    *,
    source_seq_len: int = 81,
    pose_npz: str | None = None,
    labels_pkl: str | None = None,
):
    """Return a param dict for the H36M (XYZ) data loader — the legacy path."""
    pose_default = path.POSE_AND_LABEL[dataset_name]["h36m"]["PATH_POSES"]["3D"]["preprocessed"]
    labels_default = path.POSE_AND_LABEL[dataset_name]["6DSMPL"]["PATH_LABELS"]
    return {
        "backbone": "motionclip",
        "dataset": dataset_name,
        "data_type": "h36m",
        "experiment_name": experiment_name,
        "in_data_dim": 3,
        "source_seq_len": source_seq_len,
        "num_folds": num_folds,
        "data_centered": False,
        "merge_last_dim": False,
        "simulate_confidence_score": False,
        "data_norm": False,
        "select_middle": False,
        "views": [""],
        "LODO": False,
        "hypertune": False,
        "cross_dataset_test": False,
        "AID": False,
        "medication": False,
        "metadata": [],
        "rotation_range": [-10, 10],
        "noise_std": 0.005,
        "mirror_prob": 0.0,
        "rotation_prob": 0.0,
        "noise_prob": 0.0,
        "axis_mask_prob": 0.0,
        "data_path": [pose_npz or pose_default],
        "labels_path": labels_pkl or labels_default,
        "batch_size": batch_size,
        "dim_rep": 512,
        "model_checkpoint_path": "",
    }


def build_6dsmpl_params(
    dataset_name: str,
    num_folds: int,
    batch_size: int,
    experiment_name: str,
    *,
    source_seq_len: int = 81,
    pose_npz: str | None = None,
    labels_pkl: str | None = None,
) -> dict:
    """Return a param dict for the 6D_SMPL data loader.

    The ``dataset_factory`` machinery is reused unchanged; the only
    differences from the H36M path are:
    - ``data_type="6DSMPL"``
    - ``in_data_dim=6`` (six-component rotation)
    - data_path points to the 6D_SMPL NPZ file
    """
    pose_default = path.POSE_AND_LABEL[dataset_name]["6DSMPL"]["PATH_POSES"]
    labels_default = path.POSE_AND_LABEL[dataset_name]["6DSMPL"]["PATH_LABELS"]
    return {
        "backbone": "motionclip",
        "dataset": dataset_name,
        "data_type": "6DSMPL",
        "experiment_name": experiment_name,
        "in_data_dim": 6,
        "source_seq_len": source_seq_len,
        "num_folds": num_folds,
        "data_centered": False,
        "merge_last_dim": False,
        "simulate_confidence_score": False,
        "data_norm": False,
        "select_middle": False,
        "views": [""],
        "LODO": False,
        "hypertune": False,
        "cross_dataset_test": False,
        "AID": False,
        "medication": False,
        "metadata": [],
        "rotation_range": [-10, 10],
        "noise_std": 0.0,    # no noise on rotation features
        "mirror_prob": 0.0,
        "rotation_prob": 0.0,
        "noise_prob": 0.0,
        "axis_mask_prob": 0.0,
        "data_path": [pose_npz or pose_default],
        "labels_path": labels_pkl or labels_default,
        "batch_size": batch_size,
        "dim_rep": 512,
        "model_checkpoint_path": "",
    }


def get_carepd_datasets(args):
    """Load CARE-PD H36M (XYZ) fold datasets for the given args namespace."""
    if args.dataset == "all":
        raise NotImplementedError("Use a single dataset for ACTOR CVAE debugging.")
    assert args.dataset in _SUPPORTED
    p = build_motionclip_params(
        args.dataset,
        args.num_folds,
        args.batch_size,
        args.experiment_name,
        source_seq_len=getattr(args, "source_seq_len", 81),
        pose_npz=getattr(args, "carepd_pose_npz", None),
        labels_pkl=getattr(args, "carepd_labels_pkl", None),
    )
    pose_path = p["data_path"][0]
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(
            f"CARE-PD H36M pose npz not found:\n  {pose_path}\n"
            "Place your preprocessed file at that path (see const/path.py), or pass "
            "--carepd_pose_npz /absolute/path/to/h36m_3d_world_floorXZZplus_30f_or_longer.npz"
        )
    labels_path = p["labels_path"]
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"CARE-PD labels pkl not found:\n  {labels_path}\n"
            "Pass --carepd_labels_pkl /path/to/{dataset}.pkl if it lives elsewhere."
        )
    return dataset_factory(p, "motionclip", args.fold)


def get_6dsmpl_datasets(args):
    """Load CARE-PD 6D_SMPL fold datasets for the given args namespace.

    Returns ``(train_dataset, val_dataset)`` via ``dataset_factory``.  The
    returned batches have shape ``(B, source_seq_len, 25, 6)``; the caller is
    responsible for dropping the trailing zero-padding joint (index 24) via
    ``actor_batch_from_6dsmpl``.
    """
    if args.dataset == "all":
        raise NotImplementedError("Use a single dataset for ACTOR CVAE debugging.")
    assert args.dataset in _SUPPORTED, f"Unsupported dataset: {args.dataset}"
    p = build_6dsmpl_params(
        args.dataset,
        args.num_folds,
        args.batch_size,
        args.experiment_name,
        source_seq_len=getattr(args, "source_seq_len", 81),
        pose_npz=getattr(args, "carepd_pose_npz", None),
        labels_pkl=getattr(args, "carepd_labels_pkl", None),
    )
    pose_path = p["data_path"][0]
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(
            f"CARE-PD 6D_SMPL pose npz not found:\n  {pose_path}\n"
            "Check const/path.py for the expected location, or pass "
            "--carepd_pose_npz /path/to/6D_SMPL_30f_or_longer.npz"
        )
    labels_path = p["labels_path"]
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"CARE-PD labels pkl not found:\n  {labels_path}\n"
            "Pass --carepd_labels_pkl /path/to/{dataset}.pkl if it lives elsewhere."
        )
    return dataset_factory(p, "motionclip", args.fold)


def actor_batch_from_carepd(
    x_bttf: torch.Tensor,
    pad_mask: torch.Tensor,
    num_classes: int,
    device: torch.device,
    y: torch.Tensor | None = None,
) -> dict:
    """Convert H36M XYZ batch ``(B, T, J, 3)`` to ACTOR batch dict.

    The raw CARE-PD H36M data is in absolute world coordinates (the person
    walks forward through space).  We root-centre every frame by subtracting
    joint 0 (pelvis), matching ACTOR's treatment of XYZ data.  Without this,
    the rc loss is dominated by the global translation and the model collapses
    to the mean pose (skeleton "glides" forward without any gait oscillation).

    Args:
        x_bttf:    ``(B, T, J, F)`` float tensor from the H36M data loader.
        pad_mask:  ``(B, T)`` bool mask, True = valid (non-padded) frame.
        num_classes: unused; kept for API compatibility.
        device:    target device.
        y:         optional ``(B,)`` int64 class label tensor (e.g. UPDRS score).
                   If None, defaults to all-zeros (single-class mode).

    Returns:
        dict with keys ``x``, ``y``, ``mask``, ``lengths`` per ACTOR convention.
        ``x`` has shape ``(B, 17, 3, T)`` in root-centred coordinates.
    """
    b, t, j, f = x_bttf.shape
    x_bttf = x_bttf.to(device)
    pad_mask = pad_mask.bool().to(device)

    # Root-centre: subtract pelvis (joint 0) from every joint every frame.
    # x_bttf[:, :, 0:1, :] has shape (B, T, 1, 3) — broadcasts over J.
    x_bttf = x_bttf - x_bttf[:, :, 0:1, :]

    if y is None:
        y_out = torch.zeros(b, dtype=torch.long, device=device)
    else:
        y_out = y.long().to(device)
    lengths = pad_mask.sum(dim=-1).long()
    # ACTOR stores motion as (B, njoints, nfeats, nframes)
    x_bjft = x_bttf.permute(0, 2, 3, 1).contiguous()
    return {
        "x": x_bjft,
        "y": y_out,
        "mask": pad_mask,
        "lengths": lengths,
    }


def actor_batch_from_6dsmpl(
    x_btjf: torch.Tensor,
    pad_mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """Convert 6D_SMPL batch ``(B, T, 25, 6)`` to ACTOR batch dict.

    The 6D_SMPL NPZ stores 25 joints per frame:
    - Joints 0–23: valid 6D rotation vectors (first two columns of rotation matrix).
    - Joint 24:    all-zeros placeholder (added during preprocessing for
                   translation; we drop it here).

    After dropping joint 24 the tensor has shape ``(B, T, 24, 6)``, which is
    permuted to ACTOR's ``(B, njoints, nfeats, nframes)`` = ``(B, 24, 6, T)``.

    Args:
        x_btjf:   ``(B, T, 25, 6)`` float tensor from the 6D_SMPL data loader.
        pad_mask: ``(B, T)`` bool mask, True = valid (non-padded) frame.
        device:   target device.

    Returns:
        dict with keys ``x``, ``y``, ``mask``, ``lengths``.
        ``x`` has shape ``(B, 24, 6, T)``.
    """
    b, t, j, f = x_btjf.shape
    assert j == 25, f"Expected 25 joints from 6D_SMPL loader, got {j}"
    assert f == 6, f"Expected 6 rotation features, got {f}"

    # Drop the zero-padding joint (index 24).
    x_valid = x_btjf[:, :, :SMPL_NJOINTS, :]  # (B, T, 24, 6)

    x_bjft = x_valid.permute(0, 2, 3, 1).contiguous().to(device)   # (B, 24, 6, T)
    pad_mask = pad_mask.bool().to(device)
    y = torch.zeros(b, dtype=torch.long, device=device)
    lengths = pad_mask.sum(dim=-1).long()

    return {
        "x": x_bjft,          # (B, 24, 6, T)
        "y": y,                # (B,) — class label (unused; set to 0)
        "mask": pad_mask,      # (B, T) — True = valid frame
        "lengths": lengths,    # (B,)  — number of valid frames per sequence
    }
