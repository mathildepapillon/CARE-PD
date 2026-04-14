"""physics_completer.py — Physics-Informed Motion Completion for SHAP.

Drop-in replacement for ``ActorSHAP.sample_completions`` that generates
diverse, manifold-constrained completions of partially-observed gait sequences
using empirical biomechanical statistics rather than a neural network.

Algorithm overview (per call to ``sample_completions``)
-------------------------------------------------------
1. Extract the T×48 body feature matrix (joints 1-16, global-pelvis space).
2. Look up the subject's precomputed statistics (or fall back to UPDRS class).
3. Build the block-Toeplitz temporal covariance Σ (with Bartlett taper) for
   the observed/held-out partition of body joints.
4. Compute the conditional Gaussian p(x_held | x_obs) in one shot across ALL
   T frames simultaneously (Option C: full temporal block sampling).
5. Draw K trajectories from the conditional distribution using L_cond.
6. Apply biomechanical projection: bone-length correction, velocity clamping.
7. Predict pelvis (joint 0) via constant-velocity if it is held out.
8. Paste observed joints back (``paste_observed=True`` default).

Coordinate system
-----------------
Input ``x`` is in **global-pelvis** format ``(B, 17, 3, T)`` — the same
representation used by VaeacMotion / ActorSHAP:
  - Joint 0: absolute world-space pelvis position.
  - Joints 1-16: position relative to the pelvis (body shape).

All covariance statistics are computed on the 48-dim body feature vector
(joints 1-16 in pelvis-relative space) which is approximately stationary.

Usage::

    from model.actor.physics_completer import PhysicsInformedCompleter

    completer = PhysicsInformedCompleter(
        stats_path="experiment_outs/motion_stats_fold1.pkl",
        device=torch.device("cuda:0"),
    )
    completer.set_subject("SUB06")   # optional but recommended

    completions = completer.sample_completions(
        x, y, mask, lengths, coalition_mask, n_samples=20
    )
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import joblib
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, cholesky

# ---------------------------------------------------------------------------
# Skeleton constants (mirrors compute_motion_stats.py)
# ---------------------------------------------------------------------------

H36M_PARENTS: list[int] = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# BFS topological order from root (pelvis = 0), levels 0-5.
# Used for bone-length correction: parent must be corrected before child.
H36M_TOPO_ORDER: list[int] = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 11, 14, 10, 12, 15, 13, 16]

# Edges from compute_motion_stats: sorted list of (child, parent).
# Index into this list matches the stored bone_mean / bone_std arrays.
H36M_EDGES: list[tuple[int, int]] = sorted(
    {(j, H36M_PARENTS[j]) for j in range(1, 17)},
    key=lambda e: (e[1], e[0]),
)   # 16 edges

J_BODY = 16
F_FEAT = 3
N_BODY = J_BODY * F_FEAT    # 48

_EPS_PD = 1e-5              # ridge added to diagonal before Cholesky


# ---------------------------------------------------------------------------
# Block-Toeplitz helpers
# ---------------------------------------------------------------------------

def _taper_weights(T: int) -> np.ndarray:
    """Bartlett (linear) taper: w[k] = max(0, 1 - k/T)."""
    k = np.arange(T, dtype=np.float64)
    return np.maximum(0.0, 1.0 - k / T)


def _build_sym_block(lag_cov_k: np.ndarray, feat_idx: np.ndarray,
                     T: int, taper: np.ndarray) -> np.ndarray:
    """Build symmetric (T*n, T*n) block-Toeplitz from lag-k sub-matrices.

    Parameters
    ----------
    lag_cov_k : (T, 48, 48) lag-k covariance matrices.
    feat_idx  : indices into [0, 48) selecting the feature subset.
    T         : number of frames.
    taper     : (T,) Bartlett weights per lag.

    Returns
    -------
    (T*n, T*n) ndarray, symmetric, dtype float64.
    """
    n = len(feat_idx)
    # Sub-matrices for each lag, pre-tapered.
    C = np.array([lag_cov_k[k][np.ix_(feat_idx, feat_idx)] * taper[k]
                  for k in range(T)])    # (T, n, n)

    S = np.zeros((T * n, T * n), dtype=np.float64)
    for lag in range(T):
        Ck = C[lag]
        for t1 in range(T - lag):
            t2 = t1 + lag
            S[t1*n:(t1+1)*n, t2*n:(t2+1)*n] = Ck
            if lag > 0:
                S[t2*n:(t2+1)*n, t1*n:(t1+1)*n] = Ck.T
    return S


def _build_cross_block(lag_cov_k: np.ndarray,
                       obs_idx: np.ndarray, held_idx: np.ndarray,
                       T: int, taper: np.ndarray) -> np.ndarray:
    """Build (T*nO, T*nH) cross-block-Toeplitz Σ_{OH}.

    Block(t1, t2):
      t2 >= t1  →  C[t2-t1][obs_idx, :][:, held_idx]
      t1 > t2   →  C[t1-t2][held_idx, :][:, obs_idx].T

    Parameters
    ----------
    lag_cov_k : (T, 48, 48) lag-k covariance matrices.
    obs_idx   : observed feature indices.
    held_idx  : held-out feature indices.
    T         : number of frames.
    taper     : (T,) Bartlett weights.
    """
    nO, nH = len(obs_idx), len(held_idx)
    C_oh = np.array([lag_cov_k[k][np.ix_(obs_idx, held_idx)] * taper[k]
                     for k in range(T)])    # (T, nO, nH)
    C_ho = np.array([lag_cov_k[k][np.ix_(held_idx, obs_idx)] * taper[k]
                     for k in range(T)])    # (T, nH, nO)

    S = np.zeros((T * nO, T * nH), dtype=np.float64)
    for lag in range(T):
        for t1 in range(T - lag):
            t2 = t1 + lag
            S[t1*nO:(t1+1)*nO, t2*nH:(t2+1)*nH] = C_oh[lag]
            if lag > 0:
                # (t2, t1) block: t1 is "later" in the second position → negative lag
                S[t2*nO:(t2+1)*nO, t1*nH:(t1+1)*nH] = C_ho[lag].T
    return S


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PhysicsInformedCompleter:
    """Conditional Gaussian motion completer with biomechanical projection.

    Parameters
    ----------
    stats_path : path to the joblib pickle produced by compute_motion_stats.py.
    device     : torch device (used for output tensors only; Cholesky runs on CPU).
    max_cache  : max number of (subject, coalition) Cholesky factors to cache.
    """

    def __init__(
        self,
        stats_path: str,
        device: torch.device,
        max_cache: int = 256,
    ) -> None:
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Motion stats not found: {stats_path}")
        self._stats = joblib.load(stats_path)
        self._device = device
        self._subject_id: Optional[str] = None
        self._cache: dict[tuple, tuple] = {}
        self._max_cache = max_cache
        self._T = int(self._stats.get("T", 81))
        print(f"[PhysicsInformedCompleter] loaded stats from {stats_path}  "
              f"(subjects: {sorted(self._stats['subject'].keys())})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_subject(self, subject_id: str) -> None:
        """Set the active subject for subsequent sample_completions calls."""
        self._subject_id = subject_id

    @torch.no_grad()
    def sample_completions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        coalition_mask: torch.Tensor,
        n_samples: int = 20,
        paste_observed: bool = True,
        subject_id: Optional[str] = None,
    ) -> list[torch.Tensor]:
        """Generate K diverse, manifold-constrained completions.

        Parameters
        ----------
        x              : (1, 17, 3, T) global-pelvis tensor.
        y              : (1,) UPDRS class label tensor.
        mask           : (1, T) bool pad mask (True = valid frame).
        lengths        : (1,) sequence lengths.
        coalition_mask : (1, 17) bool, True = joint is observed.
        n_samples      : K, number of completions to draw.
        paste_observed : if True, overwrite held-out joints with GT values
                         (standard SHAP behaviour).
        subject_id     : override for subject lookup; falls back to
                         self._subject_id then per-class stats.

        Returns
        -------
        List of K tensors, each shape (1, 17, 3, T).
        """
        # ------------------------------------------------------------------
        # Unpack input
        # ------------------------------------------------------------------
        B, J, Fj, T_raw = x.shape
        assert B == 1, "PhysicsInformedCompleter only supports batch size 1."
        T = int(lengths[0].item())   # valid frames

        # (T, 17, 3) global-pelvis float64
        x_gp = x[0].permute(2, 0, 1).cpu().numpy().astype(np.float64)[:T]
        cm = coalition_mask[0].cpu().numpy().astype(bool)   # (17,) observed flags
        updrs_class = int(y[0].item())

        # Subject lookup
        sid = subject_id or self._subject_id
        stats = self._get_stats(sid, updrs_class)

        # ------------------------------------------------------------------
        # Split body (joints 1-16) from pelvis (joint 0)
        # ------------------------------------------------------------------
        # Body features: (T, 48) — pelvis-relative positions of joints 1-16.
        x_body = x_gp[:, 1:, :].reshape(T, N_BODY)   # (T, 48)
        x_pelvis = x_gp[:, 0, :]                       # (T, 3)

        cm_body = cm[1:]   # (16,) observed flags for body joints
        pelvis_observed = bool(cm[0])

        obs_joints = np.where(cm_body)[0]      # indices in [0, 15]
        held_joints = np.where(~cm_body)[0]

        # Feature indices within the 48-dim body vector
        obs_feat = np.array([j * 3 + d for j in obs_joints for d in range(3)])
        held_feat = np.array([j * 3 + d for j in held_joints for d in range(3)])

        # ------------------------------------------------------------------
        # Conditional Gaussian sampling (Option C: full temporal block)
        # ------------------------------------------------------------------
        mean_body = stats["mean_body"].astype(np.float64)   # (48,)

        if len(held_feat) == 0:
            # All body joints observed — sample trivially (no uncertainty)
            body_samples = np.tile(x_body[np.newaxis], (n_samples, 1, 1))   # (K, T, 48)
        else:
            body_samples = self._sample_body(
                x_body, obs_feat, held_feat, mean_body,
                stats["lag_cov"].astype(np.float64),
                T, n_samples, sid, updrs_class,
            )    # (K, T, 48)

        # ------------------------------------------------------------------
        # Handle pelvis (joint 0)
        # ------------------------------------------------------------------
        pelvis_samples = self._sample_pelvis(
            x_pelvis, pelvis_observed, stats, n_samples, T
        )    # (K, T, 3)

        # ------------------------------------------------------------------
        # Biomechanical projection
        # ------------------------------------------------------------------
        body_samples = self._biomech_project(
            body_samples, x_body, cm_body, stats
        )    # (K, T, 48)

        # ------------------------------------------------------------------
        # Assemble completions
        # ------------------------------------------------------------------
        completions: list[torch.Tensor] = []
        for k in range(n_samples):
            # Build (T, 17, 3): joint 0 = pelvis, joints 1-16 = body
            x_comp = np.zeros((T, 17, 3), dtype=np.float32)
            x_comp[:, 0, :] = pelvis_samples[k]
            x_comp[:, 1:, :] = body_samples[k].reshape(T, J_BODY, F_FEAT)

            # Paste observed joints back (standard SHAP convention)
            if paste_observed:
                x_comp[:, cm, :] = x_gp[:, cm, :].astype(np.float32)

            # Pad back to T_raw if needed and convert to ACTOR format (1, 17, 3, T).
            out_np = np.zeros((T_raw, 17, 3), dtype=np.float32)
            out_np[:T] = x_comp
            if T < T_raw:
                out_np[T:] = x_comp[-1]   # repeat last valid frame

            out_t = torch.from_numpy(
                out_np.transpose(1, 2, 0)[np.newaxis]   # (1, 17, 3, T_raw)
            ).to(self._device)
            completions.append(out_t)

        return completions

    # ------------------------------------------------------------------
    # Stats lookup
    # ------------------------------------------------------------------

    def _get_stats(self, subject_id: Optional[str], updrs_class: int) -> dict:
        """Return per-subject stats, falling back to per-class."""
        subj_db = self._stats["subject"]
        class_db = self._stats["class"]
        if subject_id and subject_id in subj_db:
            return subj_db[subject_id]
        # Fall back to per-class
        if updrs_class in class_db:
            return class_db[updrs_class]
        # Last resort: first available class
        return next(iter(class_db.values()))

    # ------------------------------------------------------------------
    # Conditional Gaussian sampling (Option C)
    # ------------------------------------------------------------------

    def _sample_body(
        self,
        x_body: np.ndarray,           # (T, 48)
        obs_feat: np.ndarray,          # feature indices in [0, 48) observed
        held_feat: np.ndarray,         # feature indices in [0, 48) held out
        mean_body: np.ndarray,         # (48,)
        lag_cov: np.ndarray,           # (MAX_LAG+1, 48, 48)
        T: int,
        n_samples: int,
        subject_id: Optional[str],
        updrs_class: int,
    ) -> np.ndarray:                   # (K, T, 48)
        """Draw K joint trajectories for held-out body joints given observed ones."""

        nH = len(held_feat)
        nO = len(obs_feat)

        # Build / retrieve cached Cholesky factors
        cache_key = (subject_id or "_class_" + str(updrs_class),
                     tuple(int(i) for i in held_feat), T)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._build_conditional_factors(
                lag_cov, obs_feat, held_feat, T
            )
            # Evict oldest entry if cache is full
            if len(self._cache) > self._max_cache:
                self._cache.pop(next(iter(self._cache)))

        L_cond, K_mat_T = self._cache[cache_key]
        # L_cond  : (T*nH, T*nH) lower Cholesky of Sigma_cond
        # K_mat_T : (T*nH, T*nO) = Sigma_OH.T @ Sigma_OO_inv

        # Centre observed features
        mu_obs_flat = np.tile(mean_body[obs_feat], T) if nO > 0 else np.array([])
        mu_held_flat = np.tile(mean_body[held_feat], T)

        if nO > 0:
            # (T, nO) → (T*nO,)
            x_obs_flat = x_body[:, obs_feat].ravel(order="C")
            dev_obs = x_obs_flat - mu_obs_flat
            mu_cond = mu_held_flat + K_mat_T @ dev_obs   # (T*nH,)
        else:
            # No observed body features: sample from marginal
            mu_cond = mu_held_flat

        # Draw K samples: sample = mu_cond + L_cond @ z, z ~ N(0, I)
        rng = np.random.default_rng()
        z = rng.standard_normal((nH * T, n_samples))   # (T*nH, K)
        draws = mu_cond[:, np.newaxis] + L_cond @ z    # (T*nH, K)
        # draws[:, k] is already in the original (uncentred) feature space
        # because mu_cond = mu_held_flat + K_mat_T @ dev_obs
        # where mu_held_flat = tile(mean_body[held_feat], T) includes the mean.

        # Assemble full body feature matrix (K, T, 48).
        # Start from x_body (observed joints will be overwritten later by paste_observed).
        body_samples = np.tile(x_body[np.newaxis], (n_samples, 1, 1))   # (K, T, 48)
        for k in range(n_samples):
            draw_tjf = draws[:, k].reshape(T, nH)   # (T, nH) actual feature values
            # Scatter held features back into the full feature vector.
            # Use per-feature scalar indexing to avoid numpy fancy-index shape pitfall.
            for fi, feat_idx in enumerate(held_feat):
                body_samples[k, :, feat_idx] = draw_tjf[:, fi]

        return body_samples

    def _build_conditional_factors(
        self,
        lag_cov: np.ndarray,    # (MAX_LAG+1, 48, 48)
        obs_feat: np.ndarray,
        held_feat: np.ndarray,
        T: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build and return (L_cond, K_mat_T) for the given obs/held partition.

        L_cond  : (T*nH, T*nH) lower Cholesky of the conditional covariance.
        K_mat_T : (T*nH, T*nO) = Σ_{HO} Σ_{OO}⁻¹ used to compute conditional mean.
        """
        nO = len(obs_feat)
        nH = len(held_feat)
        MAX_LAG = lag_cov.shape[0] - 1
        # Clip to T-1 if stored MAX_LAG > T-1
        n_lag = min(MAX_LAG, T - 1) + 1
        taper = _taper_weights(T)

        # Use only lags 0..T-1
        lag_cov_T = lag_cov[:n_lag]

        # Always build Sigma_HH
        Sigma_HH = _build_sym_block(lag_cov_T, held_feat, T, taper)
        Sigma_HH += _EPS_PD * np.eye(T * nH)

        if nO == 0:
            # Unconditional: sample from marginal p(x_held)
            L_cond = cholesky(Sigma_HH, lower=True)
            K_mat_T = np.zeros((T * nH, 0))
            return L_cond, K_mat_T

        Sigma_OO = _build_sym_block(lag_cov_T, obs_feat, T, taper)
        Sigma_OO += _EPS_PD * np.eye(T * nO)

        Sigma_OH = _build_cross_block(lag_cov_T, obs_feat, held_feat, T, taper)

        # Solve Sigma_OO @ K_mat = Sigma_OH  →  K_mat = Sigma_OO^{-1} Sigma_OH
        try:
            L_OO_factor = cho_factor(Sigma_OO, lower=True, check_finite=False)
            K_mat = cho_solve(L_OO_factor, Sigma_OH,
                              check_finite=False)   # (T*nO, T*nH)
        except np.linalg.LinAlgError:
            # Fallback: increase ridge and retry
            Sigma_OO += 1e-3 * np.eye(T * nO)
            L_OO_factor = cho_factor(Sigma_OO, lower=True, check_finite=False)
            K_mat = cho_solve(L_OO_factor, Sigma_OH, check_finite=False)

        # Conditional covariance: Sigma_cond = Sigma_HH - Sigma_OH.T @ K_mat
        Sigma_cond = Sigma_HH - Sigma_OH.T @ K_mat    # (T*nH, T*nH)
        # Symmetrise (small numerical errors may break symmetry)
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
        # Ensure PD
        Sigma_cond += _EPS_PD * np.eye(T * nH)

        try:
            L_cond = cholesky(Sigma_cond, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # Increase ridge until PD
            for alpha in (1e-4, 1e-3, 1e-2, 1e-1):
                try:
                    L_cond = cholesky(
                        Sigma_cond + alpha * np.eye(T * nH),
                        lower=True, check_finite=False,
                    )
                    break
                except np.linalg.LinAlgError:
                    continue
            else:
                # Last resort: use diagonal (independence) approximation
                L_cond = np.diag(np.sqrt(np.maximum(np.diag(Sigma_cond), _EPS_PD)))

        K_mat_T = K_mat.T    # (T*nH, T*nO)
        return L_cond, K_mat_T

    # ------------------------------------------------------------------
    # Pelvis handling
    # ------------------------------------------------------------------

    def _sample_pelvis(
        self,
        x_pelvis: np.ndarray,   # (T, 3) observed pelvis (if observed)
        pelvis_observed: bool,
        stats: dict,
        n_samples: int,
        T: int,
    ) -> np.ndarray:             # (K, T, 3)
        """Return K pelvis trajectories.

        If observed: all K samples are identical (the GT pelvis).
        If held out: use a constant-velocity model based on the subject's
        mean pelvis velocity (repeated K times with small Gaussian noise).
        """
        if pelvis_observed:
            return np.tile(x_pelvis[np.newaxis], (n_samples, 1, 1))

        # Constant-velocity prediction
        vel = stats["pelvis_vel_mean"].astype(np.float64)    # (3,)
        p0 = np.zeros(3, dtype=np.float64)                   # start at origin
        traj = p0[np.newaxis] + vel[np.newaxis] * np.arange(T)[:, np.newaxis]

        # Add small noise proportional to velocity std
        p99 = float(stats.get("pelvis_vel_p99", 0.01))
        noise_std = p99 * 0.1
        rng = np.random.default_rng()
        noise = rng.standard_normal((n_samples, T, 3)) * noise_std

        return (traj[np.newaxis] + noise).astype(np.float64)

    # ------------------------------------------------------------------
    # Biomechanical projection
    # ------------------------------------------------------------------

    def _biomech_project(
        self,
        body_samples: np.ndarray,    # (K, T, 48)
        x_body_gt: np.ndarray,       # (T, 48) observed body (GT)
        cm_body: np.ndarray,         # (16,) bool, True = joint observed
        stats: dict,
        n_iters: int = 2,
    ) -> np.ndarray:                 # (K, T, 48)
        """Apply bone-length correction and velocity clamping to sampled body."""

        bone_mean = stats["bone_mean"].astype(np.float64)   # (16,)
        bone_std  = stats["bone_std"].astype(np.float64)    # (16,)
        vel_p99   = stats["vel_p99_body"].astype(np.float64)  # (16,)

        n_samples, T, _ = body_samples.shape
        result = body_samples.copy()

        for k in range(n_samples):
            x_k = result[k].reshape(T, J_BODY, F_FEAT).copy()   # (T, 16, 3)

            for _iter in range(n_iters):
                # ---- Bone length correction (all T frames simultaneously) ----
                # Work in root-relative space: joint 0 (pelvis) = origin.
                # x_k[:, j, :] is already pelvis-relative (= x_body for joint j+1).
                for edge_idx, (child, parent) in enumerate(H36M_EDGES):
                    j_c = child - 1     # 0-indexed into J_BODY
                    j_p = parent - 1    # -1 if parent is pelvis (index 0)

                    # Skip if the child joint is observed (GT will be pasted back)
                    if cm_body[j_c]:
                        continue

                    target = float(bone_mean[edge_idx])
                    tol    = 2.0 * float(bone_std[edge_idx])

                    if parent == 0:
                        # Parent is pelvis = origin in pelvis-relative space
                        vec = x_k[:, j_c, :]                      # (T, 3)
                        current_len = np.linalg.norm(vec, axis=-1, keepdims=True)  # (T, 1)
                    else:
                        # Both child and parent are body joints
                        vec = x_k[:, j_c, :] - x_k[:, j_p, :]    # (T, 3)
                        current_len = np.linalg.norm(vec, axis=-1, keepdims=True)

                    # Avoid division by zero
                    safe_len = np.maximum(current_len, 1e-8)
                    unit = vec / safe_len                           # (T, 3)
                    error = np.abs(current_len - target)           # (T, 1)
                    # Correct frames where error exceeds tolerance
                    needs_fix = (error > tol).squeeze(-1)          # (T,) bool

                    if needs_fix.any():
                        if parent == 0:
                            x_k[needs_fix, j_c, :] = unit[needs_fix] * target
                        else:
                            x_k[needs_fix, j_c, :] = (
                                x_k[needs_fix, j_p, :] + unit[needs_fix] * target
                            )

                # ---- Velocity clamping (body joints, pelvis-relative) --------
                vel = np.diff(x_k, axis=0)                    # (T-1, 16, 3)
                speeds = np.linalg.norm(vel, axis=-1)         # (T-1, 16)
                for j in range(J_BODY):
                    if cm_body[j]:
                        continue
                    max_speed = vel_p99[j]
                    if max_speed <= 0:
                        continue
                    over = speeds[:, j] > max_speed           # (T-1,) bool
                    if over.any():
                        scale = max_speed / np.maximum(speeds[over, j], 1e-8)
                        vel[over, j, :] *= scale[:, np.newaxis]
                        # Re-integrate from frame 0
                        x_k[1:, j, :] = x_k[0, j, :] + np.cumsum(vel[:, j, :], axis=0)

            result[k] = x_k.reshape(T, N_BODY)

        return result


# ---------------------------------------------------------------------------
# Convenience loader (mirrors load_vaeac_from_ckpt in sandbox eval script)
# ---------------------------------------------------------------------------

def load_physics_completer(stats_path: str, device: torch.device) -> PhysicsInformedCompleter:
    """Instantiate a PhysicsInformedCompleter from a stats file."""
    return PhysicsInformedCompleter(stats_path, device)
