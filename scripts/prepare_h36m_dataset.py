#!/usr/bin/env python
"""prepare_h36m_dataset.py — Convert raw Human3.6M exp-map .txt files to CARE-PD format.

Reads the GGMotion-style H36M directory layout::

    <h36m_root>/S{subject}/{action}_{trial}.txt

Applies forward kinematics (exp-map → 32 joint XYZ), selects the standard
17-joint subset used by CARE-PD, filters out unwanted actions, downsamples,
and writes:

    1. An NPZ file with one ``(T, 17, 3)`` array per sequence (key = sequence name).
    2. A labels pickle matching the BMCLabReader interface.
    3. A fold-split pickle for cross-validation (subjects as participants).

Usage::

    python scripts/prepare_h36m_dataset.py \\
        --h36m_root ~/code/manifoldshap/data/h36m/h36m \\
        --output_dir assets/datasets \\
        --sample_rate 2

The script requires no GPU — forward kinematics is done in NumPy.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

# ---------------------------------------------------------------------------
# H36M skeleton constants (from GGMotion / una-dinosauria)
# ---------------------------------------------------------------------------

H36M_ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog",
    "walkingtogether",
]

ACTIONS_TO_EXCLUDE = {"sitting", "sittingdown"}

SUBJECTS = [1, 5, 6, 7, 8, 9, 11]

# 32 → 17 joint selection.  The resulting order matches CARE-PD's H36M_JOINT_NAMES:
#   Pelvis, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle,
#   Spine, Thorax, Neck, Head,
#   LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist.
H36M_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]


# ---------------------------------------------------------------------------
# Forward kinematics (NumPy, no CUDA)
# ---------------------------------------------------------------------------

def _h36m_skeleton_variables():
    """Return parent, offset, expmapInd for the 32-joint H36M skeleton."""
    parent = np.array([
        0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
        17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31,
    ]) - 1

    offset = np.array([
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
    ]).reshape(-1, 3)

    expmap_ind = np.split(np.arange(4, 100) - 1, 32)
    return parent, offset, expmap_ind


def _expmap2rotmat(r: np.ndarray) -> np.ndarray:
    """Convert a single 3-D exponential-map vector to a 3×3 rotation matrix."""
    theta = np.linalg.norm(r)
    r0 = r / (theta + np.finfo(np.float32).eps)
    r0x = np.array([
        [0, -r0[2], r0[1]],
        [r0[2], 0, -r0[0]],
        [-r0[1], r0[0], 0],
    ])
    return (
        np.eye(3)
        + np.sin(theta) * r0x
        + (1 - np.cos(theta)) * r0x @ r0x
    )


def _fk_numpy(angles: np.ndarray, parent, offset, expmap_ind) -> np.ndarray:
    """Forward kinematics for a single frame (99-D exp-map → 32×3 XYZ)."""
    assert angles.shape == (99,)
    njoints = 32
    xyz = np.zeros((njoints, 3))
    rotations = [None] * njoints

    for i in range(njoints):
        if i == 0:
            this_pos = angles[:3]
        else:
            this_pos = np.zeros(3)

        R = _expmap2rotmat(angles[expmap_ind[i]])

        if parent[i] == -1:
            rotations[i] = R
            xyz[i] = offset[i] + this_pos
        else:
            xyz[i] = (offset[i] + this_pos) @ rotations[parent[i]] + xyz[parent[i]]
            rotations[i] = R @ rotations[parent[i]]

    return xyz


def expmap_to_xyz_17(raw_frames: np.ndarray, parent, offset, expmap_ind) -> np.ndarray:
    """Convert (T, 99) exp-map sequence to (T, 17, 3) XYZ in meters."""
    T = raw_frames.shape[0]
    xyz_32 = np.zeros((T, 32, 3), dtype=np.float64)
    for t in range(T):
        xyz_32[t] = _fk_numpy(raw_frames[t], parent, offset, expmap_ind)
    xyz_17 = xyz_32[:, H36M_32_TO_17, :]
    # H36M FK output is in millimeters; convert to meters.
    xyz_17 = xyz_17 / 1000.0
    return xyz_17.astype(np.float32)


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def read_expmap_txt(path: str) -> np.ndarray:
    """Read a GGMotion-style comma-separated exp-map text file → (T, 99)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                vals = line.split(",")
            else:
                vals = line.split()
            rows.append([float(v) for v in vals])
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert raw H36M exp-map data to CARE-PD format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--h36m_root", type=str,
        default=os.path.expanduser("~/code/manifoldshap/data/h36m/h36m"),
        help="Root directory with S1/ S5/ … sub-folders.",
    )
    ap.add_argument(
        "--output_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "datasets",
        ),
        help="CARE-PD assets/datasets directory.",
    )
    ap.add_argument("--sample_rate", type=int, default=2,
                    help="Frame sub-sampling rate (2 = 50fps→25fps).")
    ap.add_argument("--min_frames", type=int, default=30,
                    help="Discard sequences shorter than this after down-sampling.")
    args = ap.parse_args()

    parent, offset, expmap_ind = _h36m_skeleton_variables()

    pose_dict: dict[str, np.ndarray] = {}
    labels_dict: dict[str, int] = {}
    participant_for_seq: dict[str, str] = {}
    action_for_seq: dict[str, str] = {}

    # Build a label index for the non-excluded actions.
    kept_actions = [a for a in H36M_ACTIONS if a not in ACTIONS_TO_EXCLUDE]
    action_to_label = {a: i for i, a in enumerate(kept_actions)}
    n_classes = len(kept_actions)

    total_kept = 0
    total_skipped = 0

    for subj in SUBJECTS:
        subj_dir = os.path.join(args.h36m_root, f"S{subj}")
        if not os.path.isdir(subj_dir):
            print(f"[WARN] Subject directory not found: {subj_dir}")
            continue

        for action in H36M_ACTIONS:
            if action in ACTIONS_TO_EXCLUDE:
                for trial in [1, 2]:
                    total_skipped += 1
                    print(f"  SKIP S{subj}/{action}_{trial} (excluded action)")
                continue

            for trial in [1, 2]:
                txt_path = os.path.join(subj_dir, f"{action}_{trial}.txt")
                if not os.path.isfile(txt_path):
                    print(f"  [WARN] missing: {txt_path}")
                    continue

                raw = read_expmap_txt(txt_path)
                n_raw = raw.shape[0]

                # Downsample.
                raw = raw[::args.sample_rate]

                # Zero out global translation and rotation (first 6 dims).
                raw[:, :6] = 0.0

                xyz_17 = expmap_to_xyz_17(raw, parent, offset, expmap_ind)

                if xyz_17.shape[0] < args.min_frames:
                    print(f"  SKIP S{subj}/{action}_{trial} "
                          f"(too short: {xyz_17.shape[0]} < {args.min_frames})")
                    total_skipped += 1
                    continue

                seq_name = f"S{subj}__{action}_{trial}"
                pose_dict[seq_name] = xyz_17
                labels_dict[seq_name] = action_to_label[action]
                participant_for_seq[seq_name] = f"S{subj}"
                action_for_seq[seq_name] = action
                total_kept += 1
                print(f"  OK  S{subj}/{action}_{trial}  "
                      f"raw={n_raw}→{xyz_17.shape[0]} frames  "
                      f"shape={xyz_17.shape}")

    print(f"\nTotal: {total_kept} sequences kept, {total_skipped} skipped")
    print(f"Actions kept ({n_classes}): {kept_actions}")
    print(f"Subjects: {sorted(set(participant_for_seq.values()))}")

    # ---- Save NPZ ----------------------------------------------------------
    h36m_dir = os.path.join(args.output_dir, "h36m", "H36M")
    os.makedirs(h36m_dir, exist_ok=True)
    npz_path = os.path.join(h36m_dir, "h36m_3d_world_floorXZZplus_30f_or_longer.npz")
    np.savez(npz_path, **pose_dict)
    print(f"\nSaved pose NPZ → {npz_path}")

    # ---- Save labels pickle ------------------------------------------------
    # Structure matches what H36MReader expects: {seq_name: label_int}.
    # Also store the action→label mapping and participant mapping for
    # the reader to reconstruct participant IDs.
    labels_bundle = {
        "labels": labels_dict,
        "participant": participant_for_seq,
        "action_to_label": action_to_label,
        "n_classes": n_classes,
    }
    pkl_path = os.path.join(args.output_dir, "H36M.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(labels_bundle, f)
    print(f"Saved labels pkl → {pkl_path}")

    # ---- Save fold-split pickle --------------------------------------------
    # Standard H36M convention:
    #   Fold 1: train = S1,S6,S7,S8,S9,S11  eval = S5
    #   Fold 2: train = S1,S5,S7,S8,S9,S11  eval = S6
    #   (LOSO over 7 subjects)
    subject_strs = sorted(set(participant_for_seq.values()))
    n_subjects = len(subject_strs)
    cv_folds = {}
    for fold_idx, eval_subj in enumerate(subject_strs, start=1):
        train_subjs = [s for s in subject_strs if s != eval_subj]
        cv_folds[fold_idx] = {"train": train_subjs, "eval": [eval_subj]}

    folds_dir = os.path.join(args.output_dir, "folds", "UPDRS_Datasets")
    os.makedirs(folds_dir, exist_ok=True)
    folds_path = os.path.join(
        folds_dir, f"H36M_{n_subjects}fold_participants.pkl"
    )
    with open(folds_path, "wb") as f:
        pickle.dump(cv_folds, f)
    print(f"Saved {n_subjects}-fold split → {folds_path}")

    # Print summary stats per subject.
    print("\nPer-subject summary:")
    for subj_str in subject_strs:
        subj_seqs = [k for k, v in participant_for_seq.items() if v == subj_str]
        total_frames = sum(pose_dict[k].shape[0] for k in subj_seqs)
        print(f"  {subj_str}: {len(subj_seqs)} sequences, {total_frames} frames")


if __name__ == "__main__":
    main()
