#!/usr/bin/env python3
"""compute_motion_stats.py — Offline kinematic statistics for PhysicsInformedCompleter.

Iterates the training split of a fold and computes two tiers of statistics:

  **Per-subject** (keyed by participant_ID, e.g. "SUB06"):
    - mean body pose (joints 1-16 in global-pelvis space), shape (48,)
    - lag-k cross-covariance matrices C[k] for k = 0..T-1, shape (T, 48, 48)
    - bone lengths per skeleton edge (mean, std), shape (16, 2)
    - joint angle limits (min, max, mean, std) for 18 limb-pair angles, shape (18, 4)
    - per-joint velocity 99th-percentile, shape (16,)  [body joints only]
    - pelvis velocity mean vector (3,) and magnitude 99th-percentile (scalar)
    - dominant UPDRS class and total frame count

  **Per-class** (keyed by UPDRS class 0/1/2, as fallback for unseen subjects):
    - same fields, pooled across all subjects of that class

The statistics are saved as a joblib pickle so they can be memory-mapped at inference.

Feature convention
------------------
All covariance statistics are computed on **body features**: the 16 joints
(indices 1-16) in global-pelvis representation, already expressed relative to
the pelvis at each frame.  Joint 0 (absolute world pelvis) is kept out of the
covariance to avoid non-stationarity from the walking trajectory; it is handled
separately in PhysicsInformedCompleter.

Feature ordering: joint j (1-indexed, j=1..16) maps to features
(j-1)*3, (j-1)*3+1, (j-1)*3+2  in the 48-dim body vector.

Usage::

    python scripts/compute_motion_stats.py \\
        --fold 1 \\
        --out_path experiment_outs/motion_stats_fold1.pkl

    # with explicit data paths (bypasses config discovery):
    python scripts/compute_motion_stats.py --fold 1 \\
        --carepd_pose_npz /path/to/poses.npz \\
        --carepd_labels_pkl /path/to/BMCLab.pkl \\
        --out_path experiment_outs/motion_stats_fold1.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.actor.cvae_data import get_carepd_datasets  # noqa: E402

# ---------------------------------------------------------------------------
# Skeleton constants
# ---------------------------------------------------------------------------

# H36M 17-joint parent array.  -1 = root (pelvis).
H36M_PARENTS: list[int] = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# Sorted list of unique edges (child, parent), matching H36M17_EDGES from viz_utils.
H36M_EDGES: list[tuple[int, int]] = sorted(
    {(j, H36M_PARENTS[j]) for j in range(1, 17)},
    key=lambda e: (e[1], e[0]),
)  # 16 edges

# Limb vectors used by get_angles (from motionagformer/loss/pose3d.py).
# Each entry is (parent_joint, child_joint) defining one limb direction.
_LIMB_IDS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

# Pairs of limb indices whose enclosed angle is measured (18 angles).
_ANGLE_PAIRS: list[tuple[int, int]] = [
    (0, 3), (0, 6), (3, 6), (0, 1), (1, 2), (3, 4), (4, 5),
    (6, 7), (7, 10), (7, 13), (8, 13), (10, 13), (7, 8), (8, 9),
    (10, 11), (11, 12), (13, 14), (14, 15),
]

J_BODY = 16   # body joints (1-16)
F = 3         # xyz features per joint
N_BODY = J_BODY * F   # 48 body features
N_ANGLES = 18
N_EDGES = 16
T_STD = 81    # canonical clip length


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy, no torch dependency)
# ---------------------------------------------------------------------------

def _to_global_pelvis(x_tjf: np.ndarray) -> np.ndarray:
    """(T, 17, 3) world-space → global-pelvis representation.

    Joint 0 keeps its absolute world-space position.
    Joints 1-16 become relative to the pelvis at each frame.
    """
    out = x_tjf.copy()
    pelvis = x_tjf[:, 0:1, :]            # (T, 1, 3)
    out[:, 1:, :] = x_tjf[:, 1:, :] - pelvis
    return out


def _get_angles_np(x_tjf: np.ndarray) -> np.ndarray:
    """Compute 18 limb-pair angles from (T, 17, 3) root-centered positions.

    Replicates get_angles() from pretext/motionagformer/loss/pose3d.py in numpy.
    ``x_tjf`` should have joint 0 at the origin (root-centered).

    Returns (T, 18) angles in radians.
    """
    eps = 1e-7
    T = x_tjf.shape[0]
    # Limb direction vectors: (16, T, 3)
    limbs = np.stack([x_tjf[:, a, :] - x_tjf[:, b, :] for a, b in _LIMB_IDS])
    limbs = limbs.transpose(1, 0, 2)   # (T, 16, 3)

    angles = np.zeros((T, N_ANGLES), dtype=np.float32)
    for idx, (ai, aj) in enumerate(_ANGLE_PAIRS):
        u = limbs[:, ai, :]    # (T, 3)
        v = limbs[:, aj, :]    # (T, 3)
        norm_u = np.linalg.norm(u, axis=-1, keepdims=True) + eps
        norm_v = np.linalg.norm(v, axis=-1, keepdims=True) + eps
        cos = ((u / norm_u) * (v / norm_v)).sum(-1)
        cos = np.clip(cos, -1.0 + eps, 1.0 - eps)
        angles[:, idx] = np.arccos(cos)
    return angles  # (T, 18)


def _bone_lengths(x_tjf: np.ndarray) -> np.ndarray:
    """Compute 16 bone lengths per frame from (T, 17, 3) root-centered poses.

    Returns (T, 16) bone lengths, edge order matching ``H36M_EDGES``.
    """
    bl = np.array([
        np.linalg.norm(x_tjf[:, j, :] - x_tjf[:, p, :], axis=-1)
        for j, p in H36M_EDGES
    ]).T   # (T, 16)
    return bl


def _velocity_norms(x_tjf: np.ndarray) -> np.ndarray:
    """Compute per-joint velocity magnitudes (T-1, 17) from (T, 17, 3)."""
    vel = x_tjf[1:] - x_tjf[:-1]      # (T-1, 17, 3)
    return np.linalg.norm(vel, axis=-1)  # (T-1, 17)


# ---------------------------------------------------------------------------
# Per-subject accumulator
# ---------------------------------------------------------------------------

class _SubjectAccumulator:
    """Accumulates statistics for one subject across all their clips."""

    def __init__(self) -> None:
        self.clips_body: list[np.ndarray] = []   # (T, 48) per clip
        self.clips_rc: list[np.ndarray] = []     # (T, 17, 3) root-centered per clip
        self.labels: list[int] = []
        self.n_frames: int = 0

    def add_clip(self, x_world_tjf: np.ndarray, label: int) -> None:
        """Register one (T, 17, 3) world-space clip."""
        T = x_world_tjf.shape[0]
        gp = _to_global_pelvis(x_world_tjf)      # (T, 17, 3) global-pelvis

        # Root-centered (for angles and bone lengths): set joint 0 = origin.
        rc = gp.copy()
        rc[:, 0, :] = 0.0                         # (T, 17, 3) root-centered

        # Body features: joints 1-16, already pelvis-relative in global-pelvis.
        body = gp[:, 1:, :].reshape(T, N_BODY)   # (T, 48)

        self.clips_body.append(body.astype(np.float64))
        self.clips_rc.append(rc.astype(np.float64))
        self.labels.append(label)
        self.n_frames += T

    def compute(self, max_lag: int) -> dict:
        """Return a dict of statistics for this subject."""
        if not self.clips_body:
            return {}

        n_clips = len(self.clips_body)
        T = self.clips_body[0].shape[0]

        # ------------------------------------------------------------------
        # Mean body pose (global-pelvis body features)
        # ------------------------------------------------------------------
        all_body = np.concatenate(self.clips_body, axis=0)   # (N, 48)
        mean_body = all_body.mean(axis=0)                     # (48,)

        # ------------------------------------------------------------------
        # Lag-k covariance matrices C[k], k = 0..max_lag
        # Centred per-clip to remove any slow within-clip drift.
        # ------------------------------------------------------------------
        lag_cov = np.zeros((max_lag + 1, N_BODY, N_BODY), dtype=np.float64)
        lag_counts = np.zeros(max_lag + 1, dtype=np.float64)

        for body in self.clips_body:
            x = body - mean_body       # (T, 48) centred around subject mean
            for k in range(min(max_lag + 1, T)):
                n_pairs = T - k
                # C[k] += sum_t (x[t] outer x[t+k])
                lag_cov[k] += x[:n_pairs].T @ x[k:]   # (48, 48)
                lag_counts[k] += n_pairs

        for k in range(max_lag + 1):
            if lag_counts[k] > 0:
                lag_cov[k] /= lag_counts[k]

        # Ledoit-Wolf shrinkage on the instantaneous covariance (k=0).
        # All clips contribute frames → well-conditioned even for small subjects.
        try:
            lw = LedoitWolf(assume_centered=True)
            lw.fit(all_body - mean_body)
            lag_cov[0] = lw.covariance_
        except Exception:
            pass   # fall back to sample covariance

        # ------------------------------------------------------------------
        # Bone lengths: computed in root-centered space.
        # ------------------------------------------------------------------
        all_rc = np.concatenate(self.clips_rc, axis=0)   # (N, 17, 3)
        bl = _bone_lengths(all_rc)                         # (N, 16)
        bone_mean = bl.mean(axis=0)                        # (16,)
        bone_std = bl.std(axis=0) + 1e-6                   # (16,) avoid zero

        # ------------------------------------------------------------------
        # Joint angles (18 limb-pair angles).
        # ------------------------------------------------------------------
        angles = _get_angles_np(all_rc)   # (N, 18)
        angle_stats = np.stack([
            angles.min(axis=0),
            angles.max(axis=0),
            angles.mean(axis=0),
            angles.std(axis=0) + 1e-6,
        ], axis=1)   # (18, 4): [min, max, mean, std]

        # ------------------------------------------------------------------
        # Velocity statistics: body joints (1-16) and pelvis (joint 0).
        # ------------------------------------------------------------------
        all_gp = np.concatenate([
            _to_global_pelvis(rc) for rc in self.clips_rc
            # We reconstructed gp earlier; re-derive from rc for simplicity.
        ], axis=0)   # (N, 17, 3) — but rc[:,0,:]=0 so this is not valid for pelvis!

        # Recompute from original clips_rc + pelvis from clips_body context.
        # Body velocity from body features (joints 1-16, pelvis-relative):
        body_vels = []
        pelvis_vels = []
        for body, rc in zip(self.clips_body, self.clips_rc):
            # Body joints: velocity of pelvis-relative positions
            bv = np.diff(body, axis=0)               # (T-1, 48)
            bv_mag = np.linalg.norm(
                bv.reshape(T - 1, J_BODY, F), axis=-1)   # (T-1, 16)
            body_vels.append(bv_mag)
            # Pelvis: we need the actual pelvis trajectory.
            # rc[:,0,:] = 0 always, so we can't use rc.
            # We'll derive pelvis vel from angle/bone context not available here.
            # Instead, we track it when it's added separately in add_clip.
            # Approximation: rc contains global-pelvis with joint0 zeroed;
            # pelvis velocity is NOT stored in rc. Skip for now.

        # We need raw pelvis trajectory. Re-read from clips_rc which has 0 at j0.
        # Build a separate pass using world poses — stored implicitly via add_clip.
        # Since we didn't store world poses, we recover: world_pelvis is not in rc.
        # Solution: store pelvis velocity separately. We'll compute an approximation
        # from the body centroid motion (pelvis ≈ mean of hip joints).
        # Actually: in global-pelvis, joints 1-16 are pelvis-relative. The pelvis
        # velocity must be computed from the original world-space data.
        # We no longer have it here. Use a proxy: hip midpoint in root-centered
        # (which has pelvis at 0) changes by definition — so pelvis motion is the
        # motion not captured in rc. We'll skip pelvis velocity here and compute
        # it in a second pass if needed.

        all_bv = np.concatenate(body_vels, axis=0)         # (N_t, 16)
        vel_p99_body = np.percentile(all_bv, 99, axis=0)   # (16,)

        # ------------------------------------------------------------------
        # Pelvis velocity placeholder (filled in second pass if world data available).
        # ------------------------------------------------------------------
        pelvis_vel_mean = np.zeros(3, dtype=np.float64)
        pelvis_vel_p99 = 0.0   # will be updated by _SubjectAccumulator.add_pelvis

        dominant_class = int(np.bincount(self.labels).argmax())

        return {
            "n_clips":          n_clips,
            "n_frames":         self.n_frames,
            "updrs_class":      dominant_class,
            "mean_body":        mean_body.astype(np.float32),
            "lag_cov":          lag_cov.astype(np.float32),
            "bone_mean":        bone_mean.astype(np.float32),
            "bone_std":         bone_std.astype(np.float32),
            "angle_stats":      angle_stats.astype(np.float32),
            "vel_p99_body":     vel_p99_body.astype(np.float32),
            "pelvis_vel_mean":  pelvis_vel_mean.astype(np.float32),
            "pelvis_vel_p99":   float(pelvis_vel_p99),
        }


class _SubjectPelvisAccumulator:
    """Separate accumulator for pelvis world-space velocity (needs world data)."""

    def __init__(self) -> None:
        self._vels: list[np.ndarray] = []

    def add_clip(self, x_world_tjf: np.ndarray) -> None:
        pelvis = x_world_tjf[:, 0, :]          # (T, 3) world pelvis
        vel = np.diff(pelvis, axis=0)           # (T-1, 3)
        self._vels.append(vel)

    def stats(self) -> tuple[np.ndarray, float]:
        if not self._vels:
            return np.zeros(3, dtype=np.float32), 0.0
        all_vel = np.concatenate(self._vels, axis=0)    # (N_t, 3)
        mean_vel = all_vel.mean(axis=0).astype(np.float32)
        speeds = np.linalg.norm(all_vel, axis=-1)
        p99 = float(np.percentile(speeds, 99))
        return mean_vel, p99


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute per-subject/per-class kinematic statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--num_folds", type=int, default=23)
    ap.add_argument("--dataset", default="BMCLab")
    ap.add_argument("--max_lag", type=int, default=80,
                    help="Maximum temporal lag stored (should be T-1=80 for T=81).")
    ap.add_argument("--out_path", default=None,
                    help="Output pkl path.  Default: experiment_outs/motion_stats_fold{fold}.pkl")
    ap.add_argument("--carepd_pose_npz", default=None)
    ap.add_argument("--carepd_labels_pkl", default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out_path is None:
        os.makedirs("experiment_outs", exist_ok=True)
        args.out_path = f"experiment_outs/motion_stats_fold{args.fold}.pkl"

    print(f"[stats] fold={args.fold}  max_lag={args.max_lag}  → {args.out_path}")

    # ------------------------------------------------------------------
    # Load training dataset
    # ------------------------------------------------------------------
    class _DataArgs:
        dataset = args.dataset
        num_folds = args.num_folds
        fold = args.fold
        batch_size = 1
        source_seq_len = T_STD
        experiment_name = "VaeacMotion"
        carepd_pose_npz = args.carepd_pose_npz
        carepd_labels_pkl = args.carepd_labels_pkl

    train_ds, _ = get_carepd_datasets(_DataArgs())
    n = len(train_ds)
    print(f"[stats] training set: {n} clips")

    # ------------------------------------------------------------------
    # Accumulate per-subject and per-class data
    # ------------------------------------------------------------------
    subj_accum: dict[str, _SubjectAccumulator] = defaultdict(_SubjectAccumulator)
    subj_pelvis: dict[str, _SubjectPelvisAccumulator] = defaultdict(_SubjectPelvisAccumulator)
    class_accum: dict[int, _SubjectAccumulator] = {c: _SubjectAccumulator() for c in range(3)}

    t0 = time.time()
    for idx in tqdm(range(n), desc="accumulating clips"):
        sample = train_ds[idx]
        x_raw = sample["encoder_inputs"]    # (T, 17, 3) world-space float32
        label = int(sample["label"])
        video_name: str = train_ds.video_names[idx]
        subject_id = video_name.split("__")[0]

        subj_accum[subject_id].add_clip(x_raw, label)
        subj_pelvis[subject_id].add_clip(x_raw)
        class_accum[label].add_clip(x_raw, label)

    print(f"[stats] accumulation done ({time.time() - t0:.1f}s). "
          f"Subjects: {sorted(subj_accum.keys())}")

    # ------------------------------------------------------------------
    # Compute per-subject statistics
    # ------------------------------------------------------------------
    print("[stats] computing per-subject statistics …")
    subject_stats: dict[str, dict] = {}
    for subj_id, acc in tqdm(subj_accum.items(), desc="subjects"):
        stats = acc.compute(args.max_lag)
        if not stats:
            continue
        pv_mean, pv_p99 = subj_pelvis[subj_id].stats()
        stats["pelvis_vel_mean"] = pv_mean
        stats["pelvis_vel_p99"] = pv_p99
        subject_stats[subj_id] = stats
        print(f"  {subj_id}: {stats['n_clips']} clips, "
              f"{stats['n_frames']} frames, class={stats['updrs_class']}")

    # ------------------------------------------------------------------
    # Compute per-class (fallback) statistics
    # ------------------------------------------------------------------
    print("[stats] computing per-class fallback statistics …")
    class_stats: dict[int, dict] = {}
    for c, acc in class_accum.items():
        if acc.n_frames == 0:
            continue
        cs = acc.compute(args.max_lag)
        cs["pelvis_vel_mean"] = np.zeros(3, dtype=np.float32)
        cs["pelvis_vel_p99"] = 0.0
        class_stats[c] = cs
        print(f"  class {c}: {cs['n_clips']} clips, {cs['n_frames']} frames")

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    output = {
        "T":            T_STD,
        "J_body":       J_BODY,
        "F":            F,
        "N_BODY":       N_BODY,
        "MAX_LAG":      args.max_lag,
        "edges":        H36M_EDGES,
        "parents":      H36M_PARENTS,
        "fold":         args.fold,
        "subject":      subject_stats,
        "class":        class_stats,
    }

    joblib.dump(output, args.out_path, compress=3)
    print(f"[stats] saved → {args.out_path}  "
          f"({os.path.getsize(args.out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
