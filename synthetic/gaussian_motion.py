"""gaussian_motion.py — Gaussian motion benchmark for flow-SHAP validation.

Mirrors Olsen et al. JMLR 2022 Section 4.2 (Simulation Study: Continuous Data)
adapted for temporal motion sequences, and extended with a spatial-players
variant where the J joints themselves are the SHAP players.

DATA MODEL
----------
    x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)

    Sigma_joints[j,j'] = rho  (j ≠ j')
    Sigma_joints[j,j]  = 1
    Sigma_time[t,t']   = alpha^|t-t'|        (AR(1))

PLAYERS
-------
Temporal: M = K equal-length windows, K must be a multiple of 4 so the
Olsen-style interaction label can be tiled cleanly.  2^K coalitions → exact
Shapley tractable by enumeration when K is small (K <= 12 or so); for K == 4
this reduces to Olsen et al. (2022) Eq. (12).

Spatial:  M = J joints (typically 17).  Coalition space 2^17 is too large for
enumeration so ground-truth Shapley is estimated via KernelSHAP with the
exact joint-conditional Gaussian as imputer (same oracle used for "true" in
the temporal case).  Label depends on 4 designated "signal" joints via an
Olsen-style term; the remaining joints are nuisance (uncorrelated with label
except indirectly via joint equi-correlation).

TRUE CONDITIONAL (key simplification via Kronecker structure)
--------------------------------------------------------------
For temporal coalition S (observed windows) with observed time indices t_obs and
hidden indices t_hid:

    W             = Sigma_time[t_hid, t_obs] @ inv(Sigma_time[t_obs, t_obs])
    Sigma_hid_cond = Sigma_time[t_hid,t_hid] - W @ Sigma_time[t_obs, t_hid]

    mu_hid[j,f,:]  = W @ x[j,f,t_obs]        (same W for all joints, features)
    Sigma_cond      = Sigma_joints ⊗ I_F ⊗ Sigma_hid_cond

Sampling from the conditional is done per (joint, feature) via:
    1. z[j,f,:]  ~ N(0, Sigma_hid_cond)  → apply L_time_cond per (j,f)
    2. z_corr[j] = L_joints @ z[j,:]      → apply L_joints per feature/time
    3. x_hid[j,f,:] = mu_hid[j,f,:] + z_corr[j,f,:]

This is O(J·F·T/K) per sample — very fast.

CLASSIFIER
----------
A small MLP trained on window-level summary statistics (per-window joint mean
positions) to make temporal imputation quality matter for SHAP accuracy.

USAGE
-----
    bench = GaussianMotionBenchmark(J=17, F=3, T=81, rho=0.5, alpha=0.8)
    x_train, y_train = bench.sample(2000, seed=0)   # (N, J, F, T), (N,)
    x_test,  y_test  = bench.sample(100,  seed=1)

    # Train classifier
    clf = SyntheticMLPClassifier(J=17, F=3, T=81, K=4, num_classes=3)
    clf.fit(x_train, y_train, epochs=50)

    # True v(S) for a single test sequence
    x_single = torch.tensor(x_test[0:1])   # (1, J, F, T)
    v_true = bench.compute_v_true_all_coalitions(x_single, clf, K_mc=1000)

    # True Shapley values from v_true
    phi_true = bench.compute_true_shapley(v_true)   # (M=4,)
"""

from __future__ import annotations

import itertools
import pickle
from math import comb
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import ndtr as _ndtr  # standard normal CDF Φ
from torch.utils.data import DataLoader, TensorDataset

PlayerMode = Literal["temporal", "spatial"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ar1_cov(T: int, alpha: float) -> np.ndarray:
    """Return T×T AR(1) covariance matrix: C[t,t'] = alpha^|t-t'|."""
    t = np.arange(T, dtype=float)
    return alpha ** np.abs(t[:, None] - t[None, :])


def _equicorr(J: int, rho: float) -> np.ndarray:
    """Return J×J equicorrelation matrix."""
    return rho * np.ones((J, J)) + (1.0 - rho) * np.eye(J)


def _shapley_kernel_weight(s: int, M: int) -> float:
    if s == 0 or s == M:
        return 0.0
    return (M - 1) / (comb(M, s) * s * (M - s))


def _enumerate_temporal_coalitions(K: int = 4):
    """Return all 2^K (K,)-binary rows and their kernel weights."""
    if K > 20:
        raise ValueError(
            f"K={K} would enumerate 2^{K} = {2**K} coalitions; refusing. "
            "Use KernelSHAP sampling instead."
        )
    coalitions = np.array(list(itertools.product([0, 1], repeat=K)), dtype=int)
    weights = np.array([_shapley_kernel_weight(int(c.sum()), K) for c in coalitions])
    return coalitions, weights


def _sample_kernelshap_coalitions(
    M: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``2 * n_pairs`` KernelSHAP coalitions with their kernel weights.

    Paired sampling (Covert & Lee 2021) draws a subset size s from the
    SHAP kernel size-distribution and emits BOTH z and its complement to
    halve variance.  Boundary coalitions (empty and full) are emitted
    separately by the caller (they carry constraints, not regression
    weights).

    Returns
    -------
    coalitions : (2*n_pairs, M) binary array
    weights    : (2*n_pairs,) SHAP kernel weights (all non-zero by
                 construction since 0 < s < M)
    """
    sizes = np.arange(1, M)                       # valid coalition sizes
    size_weights = np.array([
        _shapley_kernel_weight(int(s), M) * comb(M, int(s))
        for s in sizes
    ], dtype=np.float64)
    p = size_weights / size_weights.sum()

    coalitions = np.zeros((2 * n_pairs, M), dtype=int)
    weights = np.zeros(2 * n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        s = int(rng.choice(sizes, p=p))
        idx = rng.choice(M, size=s, replace=False)
        z = np.zeros(M, dtype=int)
        z[idx] = 1
        coalitions[2 * i]     = z
        coalitions[2 * i + 1] = 1 - z
        w = _shapley_kernel_weight(int(z.sum()), M)
        weights[2 * i]     = w
        weights[2 * i + 1] = _shapley_kernel_weight(int((1 - z).sum()), M)
    return coalitions, weights


def _solve_shapley_wls(
    coalitions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    v_empty: float,
    v_full: float,
) -> np.ndarray:
    """Constrained WLS Shapley solve (Lundberg & Lee 2017)."""
    M = coalitions.shape[1]
    _BIG = 1e6
    boundary_z = np.array([[0] * M, [1] * M])
    boundary_v = np.array([v_empty, v_full])
    boundary_w = np.full(2, _BIG)
    c = np.vstack([coalitions, boundary_z])
    v = np.concatenate([values, boundary_v])
    w = np.concatenate([weights, boundary_w])

    keep = w > 0
    Z = np.column_stack([np.ones(keep.sum()), c[keep]])
    sq = np.sqrt(w[keep])[:, None]
    A = (Z * sq).T @ (Z * sq) + 1e-8 * np.eye(M + 1)
    b = (Z * sq).T @ (v[keep] * sq[:, 0])
    theta = np.linalg.solve(A, b)
    return theta[1:]


# ---------------------------------------------------------------------------
# Nonlinear response function — Olsen et al. (JMLR 2022) Eq. (12) adaptation
# ---------------------------------------------------------------------------

def _per_window_grand_means(
    x: np.ndarray,
    window_assignments: list[list[int]],
) -> np.ndarray:
    """Per-window scalar grand means: mean over all (j, f) in each window.

    Parameters
    ----------
    x : (N, J, F, T)
    window_assignments : K lists of frame indices

    Returns
    -------
    w : (N, K) float64
    """
    return np.stack(
        [x[:, :, :, frames].mean(axis=(1, 2, 3)) for frames in window_assignments],
        axis=1,
    )


def _olsen_term(
    u: np.ndarray,
    feat_idx: Sequence[int],
    coeffs: Sequence[float],
) -> np.ndarray:
    """One Olsen-style interaction term across 4 feature indices.

    Given standardised uniform features ``u`` (shape (N, P)) and four feature
    indices ``(a, b, c, d)``, returns::

        c1 · sin(π · u_a · u_b) + c2 · u_c · exp(c3 · u_c · u_d)

    All four features appear in the output so each one has a well-defined
    Shapley contribution (neither the sin term nor the exp term factorises
    across its pair, guaranteeing a non-trivial interaction effect).
    """
    a, b, c, d = feat_idx
    c1, c2, c3 = coeffs
    return (
        c1 * np.sin(np.pi * u[:, a] * u[:, b])
        + c2 * u[:, c] * np.exp(c3 * u[:, c] * u[:, d])
    )


def nonlinear_olsen_score(
    x: np.ndarray,
    window_assignments: list[list[int]],
    sigma_k: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Nonlinear response adapted from Olsen et al. (JMLR 2022) Eq. (12).

    Generalised to arbitrary K divisible by 4 by tiling the base Olsen term
    over consecutive groups of 4 windows.  For K=4 this reduces exactly to

        score = c1·sin(π·u0·u1) + c2·u2·exp(c3·u2·u3)

    (one term, three coefficients).  For K=8 we add a second term over
    windows 4..7, and so on.

    Parameters
    ----------
    x : (N, J, F, T)
    window_assignments : K lists of frame indices.
    sigma_k : (K,) per-window standard deviations for standardisation.
    coeffs  : (n_terms, 3) coefficient matrix where ``n_terms = K // 4``.

    Returns
    -------
    score : (N,) float64
    """
    w = _per_window_grand_means(x, window_assignments)        # (N, K)
    u = _ndtr(w / (sigma_k[np.newaxis] + 1e-12))              # (N, K)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim == 1:
        # Back-compat: a flat (3,) vector means a single Olsen term.
        coeffs = coeffs.reshape(1, 3)
    K = u.shape[1]
    n_terms = coeffs.shape[0]
    if K != 4 * n_terms:
        raise ValueError(
            f"K={K} must equal 4 * n_terms={4 * n_terms} (coeffs has "
            f"{n_terms} rows)."
        )
    score = np.zeros(u.shape[0], dtype=np.float64)
    for t_idx in range(n_terms):
        offset = 4 * t_idx
        score += _olsen_term(u, (offset, offset + 1, offset + 2, offset + 3), coeffs[t_idx])
    return score


def spatial_olsen_score(
    x: np.ndarray,
    signal_joints: Sequence[int],
    sigma_j: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Spatial variant: single Olsen term over 4 designated "signal" joints.

    Per-joint grand means (averaged over F and T) play the role of u_k in the
    temporal case.  Only the 4 signal joints enter the label; the remaining
    J-4 joints are nuisance (but still correlated with signal via Sigma_joints,
    so they can leak information to imperfect imputers).

    Parameters
    ----------
    x : (N, J, F, T)
    signal_joints : tuple of 4 joint indices in [0, J).
    sigma_j : (J,) per-joint standard deviations for standardisation.
    coeffs  : (3,) = [c1, c2, c3] for the single Olsen term.

    Returns
    -------
    score : (N,) float64
    """
    if len(signal_joints) != 4:
        raise ValueError("spatial_olsen_score requires exactly 4 signal joints.")
    w = x.mean(axis=(2, 3))                                    # (N, J)
    u = _ndtr(w / (sigma_j[np.newaxis] + 1e-12))               # (N, J)
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    return _olsen_term(u, tuple(signal_joints), tuple(coeffs[:3]))


# ---------------------------------------------------------------------------
# Core benchmark class
# ---------------------------------------------------------------------------

class GaussianMotionBenchmark:
    """Synthetic Gaussian motion benchmark.

    Parameters
    ----------
    J, F, T : int
        Joints (17), features per joint (3), frames per sequence (81).
    rho : float
        Off-diagonal equicorrelation between joints.
    alpha : float
        AR(1) temporal autocorrelation.
    K : int
        Number of temporal windows (only used when player_mode == "temporal").
        Must be a multiple of 4 so the Olsen-style label tiles cleanly; larger
        K means smaller windows and more SHAP players (2^K coalitions).
    player_mode : {"temporal", "spatial"}
        * "temporal": SHAP players are K temporal windows.  Label function is
          the generalised Olsen score.  Ground-truth Shapley is exact by
          enumeration of 2^K coalitions.
        * "spatial":  SHAP players are J joints.  Label function is a single
          Olsen term over 4 signal joints.  Ground-truth Shapley is estimated
          with KernelSHAP using the exact joint-conditional Gaussian as
          imputer (2^J is too large to enumerate).
    signal_joints : tuple of 4 joint indices, optional
        Only used when player_mode == "spatial".  Defaults to (0, 1, 2, 3).
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        rho: float = 0.5,
        alpha: float = 0.8,
        K: int = 4,
        player_mode: PlayerMode = "temporal",
        signal_joints: Sequence[int] | None = None,
    ):
        if player_mode not in ("temporal", "spatial"):
            raise ValueError(
                f"player_mode must be 'temporal' or 'spatial'; got {player_mode!r}"
            )
        if K % 4 != 0 or K <= 0:
            raise ValueError(
                f"K must be a positive multiple of 4; got {K}"
            )
        if player_mode == "temporal" and K > 12:
            raise ValueError(
                f"K={K} would enumerate 2^{K} coalitions; refuse for tractability."
            )
        if player_mode == "spatial":
            if signal_joints is None:
                signal_joints = (0, 1, 2, 3)
            signal_joints = tuple(int(j) for j in signal_joints)
            if len(signal_joints) != 4 or len(set(signal_joints)) != 4:
                raise ValueError("signal_joints must be 4 distinct indices.")
            if not all(0 <= j < J for j in signal_joints):
                raise ValueError(f"signal_joints out of range [0, {J}).")

        self.J = J
        self.F = F
        self.T = T
        self.rho = rho
        self.alpha = alpha
        self.K = K
        self.player_mode: PlayerMode = player_mode
        self.signal_joints: tuple[int, ...] | None = (
            tuple(signal_joints) if player_mode == "spatial" else None
        )

        self.Sigma_joints = _equicorr(J, rho)      # (J, J)
        self.Sigma_time   = _ar1_cov(T, alpha)      # (T, T)
        self.L_joints     = np.linalg.cholesky(
            self.Sigma_joints + 1e-8 * np.eye(J)
        )  # (J, J)
        self.L_time       = np.linalg.cholesky(
            self.Sigma_time + 1e-8 * np.eye(T)
        )  # (T, T)

        quarter = T // K
        self.window_assignments: list[list[int]] = [
            list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
            for k in range(K)
        ]

        # Conditional-covariance caches.  Temporal entries use keys
        # ("temporal", s_obs_windows, s_hid_windows); spatial entries use keys
        # ("spatial",  s_obs_joints,  s_hid_joints).
        self._cond_cache: dict[tuple, tuple] = {}
        if player_mode == "temporal":
            self._precompute_conditionals()
        # For spatial mode we build the cache lazily — 2^J is too large to
        # enumerate so we only materialise the coalitions KernelSHAP samples.

        # Label function config (set by setup_label_fn; None until then).
        self.label_config: dict | None = None

    # ------------------------------------------------------------------
    # Label function — Olsen et al. (2022) Eq. (12) adaptation
    # ------------------------------------------------------------------

    def setup_label_fn(
        self,
        n_calib: int = 5000,
        seed: int = 999,
        noise_std: float = 0.0,
    ) -> None:
        """Fit and store the canonical nonlinear label function.

        For temporal mode: one Olsen term per group of 4 windows (so
        ``n_terms = K // 4``), each with its own freshly-sampled (c1, c2, c3).
        For spatial mode: a single Olsen term over the 4 signal joints.
        """
        rng = np.random.default_rng(seed)
        x_cal = self.sample(n_calib, seed=int(rng.integers(2**30)))

        def _sample_term_coeffs() -> list[float]:
            # c1, c2 in [0.5, 2.0]; c3 in [0.5, 1.0] so exp(c3·u·v) stays bounded.
            return [
                float(rng.uniform(0.5, 2.0)),
                float(rng.uniform(0.5, 2.0)),
                float(rng.uniform(0.5, 1.0)),
            ]

        if self.player_mode == "temporal":
            w_cal = _per_window_grand_means(x_cal, self.window_assignments)  # (n_calib, K)
            sigma_features = w_cal.std(axis=0).clip(min=1e-8)                # (K,)
            n_terms = self.K // 4
            coeffs = np.array([_sample_term_coeffs() for _ in range(n_terms)])  # (n_terms, 3)
            self.label_config = {
                "mode":             "temporal",
                "sigma_features":   sigma_features,
                "coeffs":           coeffs,
                "noise_std":        noise_std,
                "seed":             seed,
                "n_terms":          n_terms,
            }
        else:                                                                 # spatial
            w_cal = x_cal.mean(axis=(2, 3))                                   # (n_calib, J)
            sigma_features = w_cal.std(axis=0).clip(min=1e-8)                 # (J,)
            coeffs = np.array([_sample_term_coeffs()])                        # (1, 3)
            self.label_config = {
                "mode":             "spatial",
                "sigma_features":   sigma_features,
                "coeffs":           coeffs,
                "noise_std":        noise_std,
                "seed":             seed,
                "n_terms":          1,
                "signal_joints":    self.signal_joints,
            }

    def canonical_label_fn(
        self,
        x: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply the stored nonlinear label function to sequences x (N, J, F, T).

        Raises ``RuntimeError`` if ``setup_label_fn`` has not been called.
        """
        if self.label_config is None:
            raise RuntimeError(
                "Call bench.setup_label_fn() before using canonical_label_fn."
            )
        cfg = self.label_config
        mode = cfg.get("mode", "temporal")
        # Back-compat: old pickles stored "sigma_k" and a flat (3,) coeff vector.
        sigma_features = cfg.get("sigma_features", cfg.get("sigma_k"))
        coeffs = np.asarray(cfg["coeffs"], dtype=np.float64)
        if mode == "temporal":
            score = nonlinear_olsen_score(
                x, self.window_assignments, sigma_features, coeffs,
            )
        else:
            score = spatial_olsen_score(
                x, cfg["signal_joints"], sigma_features, coeffs.reshape(-1),
            )
        if cfg["noise_std"] > 0:
            _rng = rng if rng is not None else np.random.default_rng(cfg["seed"] + 1)
            score = score + _rng.normal(0, cfg["noise_std"], size=len(score))
        q33, q67 = np.percentile(score, [33, 67])
        return np.where(score < q33, 0, np.where(score < q67, 1, 2)).astype(np.int64)

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _window_frames(self, window_indices: list[int]) -> np.ndarray:
        return np.concatenate([self.window_assignments[k] for k in window_indices])

    def _precompute_conditionals(self) -> None:
        """Temporal mode: cache (L_cond_time, W_mean) for all non-trivial coalitions."""
        K = self.K
        coalitions, _ = _enumerate_temporal_coalitions(K)
        seen: set[tuple] = set()
        for z in coalitions:
            s_obs = tuple(k for k in range(K) if z[k] == 1)
            s_hid = tuple(k for k in range(K) if z[k] == 0)
            if not s_obs or not s_hid:
                continue
            key = ("temporal", s_obs, s_hid)
            if key in seen:
                continue
            seen.add(key)
            self._cond_cache[key] = self._compute_cond_params(s_obs, s_hid)

    def _compute_cond_params(
        self, s_obs: tuple[int, ...], s_hid: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (L_cond_time, W_mean) for the given window partition.

        Returns
        -------
        L_cond_time : (n_hid, n_hid)
            Cholesky factor of the temporal conditional covariance.
        W_mean : (n_hid, n_obs)
            Linear weight for the conditional mean:
            mu_hid[j,f,:] = W_mean @ x[j,f,t_obs]
        """
        t_obs = self._window_frames(list(s_obs))
        t_hid = self._window_frames(list(s_hid))

        Soo = self.Sigma_time[np.ix_(t_obs, t_obs)]
        Shh = self.Sigma_time[np.ix_(t_hid, t_hid)]
        Sho = self.Sigma_time[np.ix_(t_hid, t_obs)]

        # W = Sigma_time[hid,obs] @ inv(Sigma_time[obs,obs])
        # Use lstsq for numerical stability.
        W = Sho @ np.linalg.solve(Soo + 1e-10 * np.eye(len(t_obs)), np.eye(len(t_obs)))

        # Conditional temporal covariance.
        Sigma_cond = Shh - W @ Sho.T
        Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2.0  # symmetrise
        Sigma_cond += 1e-8 * np.eye(len(t_hid))
        L_cond = np.linalg.cholesky(Sigma_cond)

        return L_cond, W

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, N: int, seed: int | None = None) -> tuple[np.ndarray, None]:
        """Sample N sequences from N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time).

        Returns
        -------
        x : (N, J, F, T) float32 array
        """
        rng = np.random.default_rng(seed)
        # Draw standard normals then apply Kronecker Cholesky.
        z = rng.standard_normal((N, self.J, self.F, self.T)).astype(np.float64)
        # Apply temporal Cholesky: x_t = L_time @ z_t  (per j, f)
        x = np.einsum("tT,njfT->njft", self.L_time, z)
        # Apply joint Cholesky: x_j = L_joints @ x_J  (per f, t)
        x = np.einsum("jJ,nJft->njft", self.L_joints, x)
        return x.astype(np.float32)

    def conditional_sample(
        self,
        x: np.ndarray,
        s_obs: tuple[int, ...],
        s_hid: tuple[int, ...],
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw n_samples from p(x_hid | x_obs) exactly.

        Parameters
        ----------
        x : (J, F, T) array of the conditioning sequence.
        s_obs : tuple of observed window indices.
        s_hid : tuple of hidden window indices.
        n_samples : int
        rng : numpy Generator

        Returns
        -------
        x_completed : (n_samples, J, F, T) array with observed frames copied
                      from x and hidden frames drawn from the exact conditional.
        """
        if rng is None:
            rng = np.random.default_rng()
        J, F, T = self.J, self.F, self.T
        t_obs = self._window_frames(list(s_obs))
        t_hid = self._window_frames(list(s_hid))
        n_hid = len(t_hid)

        key = ("temporal", s_obs, s_hid)
        if key not in self._cond_cache:
            self._cond_cache[key] = self._compute_cond_params(s_obs, s_hid)
        L_cond, W_mean = self._cond_cache[key]

        # Conditional mean: shape (J, F, n_hid)
        mu = np.einsum("ht,jft->jfh", W_mean, x[:, :, t_obs])

        # Draw noise: z_time ~ N(0, I_{n_hid}), z_joints ~ N(0, I_J)
        # Resulting covariance: Sigma_joints ⊗ I_F ⊗ Sigma_hid_cond
        noise_t = rng.standard_normal((n_samples, J, F, n_hid)).astype(np.float64)
        # Apply temporal Cholesky: (n_samples, J, F, n_hid)
        noise_t = np.einsum("tT,njfT->njft", L_cond, noise_t)
        # Apply joint Cholesky: (n_samples, J, F, n_hid)
        noise_t = np.einsum("jJ,nJft->njft", self.L_joints, noise_t)

        # Completed sequences.
        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)  # (n_samples, J, F, T)
        out[:, :, :, t_hid] = mu[None] + noise_t
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # Spatial conditional sampling (joints as players)
    # ------------------------------------------------------------------

    def _compute_cond_params_spatial(
        self,
        j_obs: tuple[int, ...],
        j_hid: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Joint-conditional Gaussian parameters.

        Under ``x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)`` the joint-marginal
        covariance is ``Sigma_joints`` (for any fixed f, t), so conditioning
        x_{j_hid} | x_{j_obs} is a standard multivariate Gaussian conditional::

            W             = Sigma_joints[hid, obs] @ inv(Sigma_joints[obs, obs])
            Sigma_hid_cond = Sigma_joints[hid, hid] - W @ Sigma_joints[obs, hid]

        Because the Kronecker structure is ``I_F ⊗ Sigma_time`` across (f, t),
        the same W and Sigma_hid_cond apply independently to every (f, t).

        Returns
        -------
        L_cond_j : (n_hid, n_hid) Cholesky factor of Sigma_hid_cond
        W        : (n_hid, n_obs) conditional-mean weight matrix
        """
        j_obs_a = np.asarray(j_obs, dtype=int)
        j_hid_a = np.asarray(j_hid, dtype=int)
        Soo = self.Sigma_joints[np.ix_(j_obs_a, j_obs_a)]
        Shh = self.Sigma_joints[np.ix_(j_hid_a, j_hid_a)]
        Sho = self.Sigma_joints[np.ix_(j_hid_a, j_obs_a)]
        W = Sho @ np.linalg.solve(
            Soo + 1e-10 * np.eye(len(j_obs_a)),
            np.eye(len(j_obs_a)),
        )
        Sigma_cond = Shh - W @ Sho.T
        Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2.0
        Sigma_cond += 1e-8 * np.eye(len(j_hid_a))
        L_cond = np.linalg.cholesky(Sigma_cond)
        return L_cond, W

    def conditional_sample_spatial(
        self,
        x: np.ndarray,
        j_obs: tuple[int, ...],
        j_hid: tuple[int, ...],
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw n_samples from p(x_{j_hid} | x_{j_obs}) exactly.

        Parameters
        ----------
        x : (J, F, T) array of the conditioning sequence.
        j_obs, j_hid : tuples of joint indices (disjoint, non-empty).
        n_samples : int
        rng : numpy Generator

        Returns
        -------
        x_completed : (n_samples, J, F, T) array with x[j_obs, :, :] copied
                      from the input and x[j_hid, :, :] drawn from the
                      exact joint-conditional.
        """
        if rng is None:
            rng = np.random.default_rng()
        J, F, T = self.J, self.F, self.T
        j_obs = tuple(int(j) for j in j_obs)
        j_hid = tuple(int(j) for j in j_hid)
        n_hid = len(j_hid)
        n_obs = len(j_obs)
        if n_hid == 0 or n_obs == 0:
            raise ValueError("j_obs and j_hid must both be non-empty.")

        key = ("spatial", j_obs, j_hid)
        if key not in self._cond_cache:
            self._cond_cache[key] = self._compute_cond_params_spatial(j_obs, j_hid)
        L_cond_j, W = self._cond_cache[key]

        j_obs_a = np.asarray(j_obs, dtype=int)
        j_hid_a = np.asarray(j_hid, dtype=int)

        # Conditional mean: (n_hid, F, T) = W_{hid,obs} @ x[obs, :, :]
        mu = np.einsum("ho,oft->hft", W, x[j_obs_a, :, :])       # (n_hid, F, T)

        # Noise with Kronecker covariance Sigma_hid_cond ⊗ I_F ⊗ Sigma_time.
        noise = rng.standard_normal((n_samples, n_hid, F, T)).astype(np.float64)
        # Temporal Cholesky over t (per n, hid, f).
        noise = np.einsum("tT,nhfT->nhft", self.L_time, noise)
        # Joint-conditional Cholesky over hidden-joint axis (per n, f, t).
        noise = np.einsum("hH,nHft->nhft", L_cond_j, noise)

        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
        out[:, j_hid_a, :, :] = mu[None] + noise
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # True value function
    # ------------------------------------------------------------------

    def compute_v_true_all_coalitions(
        self,
        x: torch.Tensor,
        classifier_fn: Callable,
        K_mc: int = 1000,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> dict[tuple, float]:
        """Compute v_true(S) for all 2^K coalitions via exact Gaussian sampling.

        Parameters
        ----------
        x : (1, J, F, T) torch tensor — a single test sequence.
        classifier_fn : callable(Tensor(B,J,F,T)) → Tensor(B,) scalar or Tensor(B,C) logits.
                        The function should return the SHAP target (e.g. class probability).
        K_mc : int — Monte Carlo samples per coalition for v_true approximation.
        device : torch.device

        Returns
        -------
        dict mapping coalition tuple (e.g. (1,0,1,1)) to v_true(S) float.
        """
        if device is None:
            device = x.device
        rng = np.random.default_rng(seed)
        x_np = x[0].cpu().numpy()  # (J, F, T)

        coalitions, _ = _enumerate_temporal_coalitions(self.K)
        v_true: dict[tuple, float] = {}

        for z in coalitions:
            s_obs = tuple(k for k in range(self.K) if z[k] == 1)
            s_hid = tuple(k for k in range(self.K) if z[k] == 0)

            if not s_hid:
                # All observed: v = f(x*)
                with torch.no_grad():
                    val = float(_eval_classifier(classifier_fn, x.to(device)).mean().item())
            elif not s_obs:
                # Nothing observed: v = E[f(x)] — average over marginal (use K_mc uncond samples).
                # For efficiency: sample K_mc complete sequences from N(0, Sigma).
                x_marg = self.sample(K_mc, seed=rng.integers(1 << 31))
                x_marg_t = torch.tensor(x_marg, device=device)
                with torch.no_grad():
                    val = float(_eval_classifier(classifier_fn, x_marg_t).mean().item())
            else:
                samps = self.conditional_sample(x_np, s_obs, s_hid, n_samples=K_mc, rng=rng)
                x_samp_t = torch.tensor(samps, device=device)
                with torch.no_grad():
                    val = float(_eval_classifier(classifier_fn, x_samp_t).mean().item())

            v_true[tuple(z.tolist())] = val

        return v_true

    def compute_true_shapley(self, v_all: dict[tuple, float]) -> np.ndarray:
        """Compute exact temporal Shapley values from v_all.

        Parameters
        ----------
        v_all : output of compute_v_true_all_coalitions.

        Returns
        -------
        phi : (K,) array of Shapley values for windows 0..K-1.
        """
        K = self.K
        coalitions, weights = _enumerate_temporal_coalitions(K)
        values = np.array([v_all[tuple(z.tolist())] for z in coalitions])
        z_empty = tuple([0] * K)
        z_full  = tuple([1] * K)
        return _solve_shapley_wls(
            coalitions, values, weights,
            v_empty=v_all[z_empty],
            v_full=v_all[z_full],
        )

    # ------------------------------------------------------------------
    # Spatial ground-truth Shapley via KernelSHAP sampling
    # ------------------------------------------------------------------

    def sample_spatial_coalitions(
        self,
        n_pairs: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw 2*n_pairs spatial (J-dim) KernelSHAP coalitions with weights."""
        if rng is None:
            rng = np.random.default_rng()
        return _sample_kernelshap_coalitions(self.J, n_pairs, rng)

    def compute_v_spatial(
        self,
        x: torch.Tensor,
        classifier_fn: Callable,
        coalitions: np.ndarray,
        K_mc: int = 500,
        device: torch.device | None = None,
        seed: int | None = None,
        v_empty_samples: int | None = None,
    ) -> np.ndarray:
        """Evaluate the true value function v(S) = E[f(x) | x_S = x_S*] for a
        list of spatial coalitions using the exact joint-conditional Gaussian.

        Parameters
        ----------
        x : (1, J, F, T) torch tensor — a single test sequence.
        classifier_fn : callable (B, J, F, T) -> (B,) scalar target.
        coalitions : (N, J) int/bool array (1 = observed joint).
        K_mc : MC samples per coalition for the conditional expectation.
        v_empty_samples : MC samples for the empty coalition (draws from
                          the unconditional marginal). Defaults to K_mc.

        Returns
        -------
        v : (N,) float64 array.
        """
        if device is None:
            device = x.device
        if v_empty_samples is None:
            v_empty_samples = K_mc
        rng = np.random.default_rng(seed)
        x_np = x[0].cpu().numpy().astype(np.float64)             # (J, F, T)
        N = coalitions.shape[0]
        v = np.zeros(N, dtype=np.float64)

        for i in range(N):
            z = coalitions[i].astype(bool)
            j_obs = tuple(int(j) for j in np.nonzero(z)[0])
            j_hid = tuple(int(j) for j in np.nonzero(~z)[0])
            if not j_hid:
                with torch.no_grad():
                    v[i] = float(_eval_classifier(classifier_fn, x.to(device)).mean().item())
            elif not j_obs:
                x_marg = self.sample(v_empty_samples, seed=int(rng.integers(1 << 31)))
                x_marg_t = torch.tensor(x_marg, device=device, dtype=torch.float32)
                with torch.no_grad():
                    v[i] = float(_eval_classifier(classifier_fn, x_marg_t).mean().item())
            else:
                samps = self.conditional_sample_spatial(
                    x_np, j_obs, j_hid, n_samples=K_mc, rng=rng,
                )
                samps_t = torch.tensor(samps, device=device, dtype=torch.float32)
                with torch.no_grad():
                    v[i] = float(_eval_classifier(classifier_fn, samps_t).mean().item())
        return v

    def compute_true_shapley_spatial(
        self,
        x: torch.Tensor,
        classifier_fn: Callable,
        coalitions: np.ndarray,
        weights: np.ndarray,
        K_mc: int = 500,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Estimate spatial Shapley φ via KernelSHAP with oracle conditionals.

        Returns
        -------
        phi     : (J,) oracle Shapley estimate.
        v       : (N,) oracle contribution values on the sampled coalitions.
        v_empty : float, E[f(x)] under the unconditional marginal.
        v_full  : float, f(x*).
        """
        rng = np.random.default_rng(seed)
        v = self.compute_v_spatial(
            x, classifier_fn, coalitions, K_mc=K_mc,
            device=device, seed=int(rng.integers(1 << 31)),
        )
        empty_c = np.zeros((1, self.J), dtype=int)
        full_c  = np.ones((1, self.J),  dtype=int)
        v_ef = self.compute_v_spatial(
            x, classifier_fn, np.vstack([empty_c, full_c]), K_mc=K_mc,
            device=device, seed=int(rng.integers(1 << 31)),
        )
        v_empty = float(v_ef[0])
        v_full  = float(v_ef[1])
        phi = _solve_shapley_wls(
            coalitions, v, weights,
            v_empty=v_empty, v_full=v_full,
        )
        return phi, v, v_empty, v_full

    # ------------------------------------------------------------------
    # PyTorch dataset builder
    # ------------------------------------------------------------------

    def build_pytorch_dataset(
        self,
        N: int,
        label_fn: Callable | None = None,
        seed: int | None = None,
    ) -> TensorDataset:
        """Build a TensorDataset with (x_bttf, y, pad_mask) compatible with ACTOR format.

        Parameters
        ----------
        N : int — number of sequences.
        label_fn : optional callable(x: (N, J, F, T) ndarray) → (N,) int32 labels.
                   If None, labels are assigned by quantile-binning of the first joint's
                   mean x-position (simple proxy for demonstrating SHAP differences).
        seed : int

        Returns
        -------
        TensorDataset of (x_bttf, y, pad_mask):
            x_bttf   : (N, T, J, F) float32 — ACTOR format (T first)
            y        : (N,) int64 labels in {0, 1, 2}
            pad_mask : (N, T) bool — all True (no padding)
        """
        x = self.sample(N, seed=seed)   # (N, J, F, T)

        if label_fn is None:
            if self.label_config is not None:
                # Use the stored nonlinear canonical label function.
                y = self.canonical_label_fn(x)
            else:
                # Fallback: quantile binning of joint-0 grand mean (legacy).
                score = x[:, 0, 0, :].mean(axis=-1)  # (N,)
                q33, q67 = np.percentile(score, [33, 67])
                y = np.where(score < q33, 0, np.where(score < q67, 1, 2)).astype(np.int64)
        else:
            y = label_fn(x).astype(np.int64)

        # Convert (N, J, F, T) → (N, T, J, F)
        x_bttf = torch.tensor(x).permute(0, 3, 1, 2).contiguous()  # (N, T, J, F)
        y_t    = torch.tensor(y, dtype=torch.long)
        pm     = torch.ones(N, self.T, dtype=torch.bool)
        return TensorDataset(x_bttf, y_t, pm)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the benchmark (covariance params, window assignments, cache)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "GaussianMotionBenchmark":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Synthetic MLP classifier
# ---------------------------------------------------------------------------

class SyntheticMLPClassifier(nn.Module):
    """Black-box MLP classifier for the Gaussian motion benchmark.

    Two feature-extraction paths:

    * ``player_mode='temporal'``: per-window grand means ``(B, K)``. Mirrors
      Olsen et al. (2022): the classifier sees the same K aggregates the
      nonlinear label depends on, making it a genuine black-box for SHAP.

    * ``player_mode='spatial'``: per-joint grand means ``(B, J)``. The
      classifier sees every joint individually; the label depends on 4
      signal joints but the classifier does not know which, so any
      reasonable imputer must preserve joint correlations to approximate
      v(S) well.

    Architecture: input_dim → BatchNorm → hidden → ReLU → hidden → ReLU → num_classes
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        K: int = 4,
        num_classes: int = 3,
        hidden: int = 64,
        player_mode: PlayerMode = "temporal",
    ):
        super().__init__()
        if player_mode not in ("temporal", "spatial"):
            raise ValueError(f"player_mode must be 'temporal' or 'spatial'; got {player_mode!r}")
        self.player_mode: PlayerMode = player_mode
        self.J = J
        self.F = F
        self.T = T
        self.K = K

        if player_mode == "temporal":
            quarter = T // K
            self.window_starts = [k * quarter for k in range(K)]
            self.window_ends   = [(k + 1) * quarter if k < K - 1 else T for k in range(K)]
            input_dim = K
        else:
            self.window_starts = None
            self.window_ends = None
            input_dim = J

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, J, F, T) → (B, K) or (B, J) grand means."""
        if self.player_mode == "temporal":
            feats = [
                x[:, :, :, s:e].mean(dim=(1, 2, 3))
                for s, e in zip(self.window_starts, self.window_ends)
            ]
            return torch.stack(feats, dim=1)                     # (B, K)
        return x.mean(dim=(2, 3))                                # (B, J)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._extract_features(x))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 80,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> list[float]:
        """Train the classifier on numpy arrays.

        Parameters
        ----------
        x_train : (N, J, F, T)
        y_train : (N,) int

        Returns
        -------
        train_losses : list of per-epoch cross-entropy loss values.
        """
        torch.manual_seed(seed)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        X = torch.tensor(x_train, dtype=torch.float32)
        Y = torch.tensor(y_train, dtype=torch.long)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(xb)
            losses.append(epoch_loss / len(X))
        self.eval()
        return losses

    def class_prob_fn(self, class_idx: int) -> Callable:
        """Return a callable (B,J,F,T) → (B,) giving probability for class_idx."""
        def fn(x: torch.Tensor) -> torch.Tensor:
            dev = next(self.parameters()).device
            logits = self(x.to(dev))
            return torch.softmax(logits, dim=-1)[:, class_idx]
        return fn


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _eval_classifier(
    classifier_fn: Callable,
    x: torch.Tensor,
    chunk: int = 512,
) -> torch.Tensor:
    """Run classifier in chunks to avoid OOM; returns (B,) float tensor."""
    results = []
    for i in range(0, len(x), chunk):
        out = classifier_fn(x[i : i + chunk])
        if out.ndim > 1:
            raise ValueError(
                "classifier_fn must return a 1-D tensor of scalars (e.g. a class "
                "probability), not logits.  Use model.class_prob_fn(class_idx)."
            )
        results.append(out.float())
    return torch.cat(results)


def build_gaussian_benchmark_and_classifier(
    rho: float = 0.5,
    alpha: float = 0.8,
    J: int = 17,
    F: int = 3,
    T: int = 81,
    K: int = 4,
    n_train: int = 2000,
    n_val: int = 1000,
    n_test: int = 500,
    clf_epochs: int = 80,
    seed: int = 0,
    device: torch.device | None = None,
    player_mode: PlayerMode = "temporal",
    signal_joints: Sequence[int] | None = None,
) -> tuple[
    "GaussianMotionBenchmark",
    "SyntheticMLPClassifier",
    TensorDataset,
    TensorDataset,
    TensorDataset,
]:
    """Convenience factory: build benchmark, sample splits, train classifier."""
    bench = GaussianMotionBenchmark(
        J=J, F=F, T=T, rho=rho, alpha=alpha, K=K,
        player_mode=player_mode, signal_joints=signal_joints,
    )
    bench.setup_label_fn(n_calib=5000, seed=seed + 99)

    x_tr = bench.sample(n_train, seed=seed)
    x_va = bench.sample(n_val,   seed=seed + 1)
    x_te = bench.sample(n_test,  seed=seed + 2)

    y_tr = bench.canonical_label_fn(x_tr)
    y_va = bench.canonical_label_fn(x_va)
    y_te = bench.canonical_label_fn(x_te)

    def _make_ds(x_arr, y_arr):
        xt = torch.tensor(x_arr).permute(0, 3, 1, 2).contiguous()
        yt = torch.tensor(y_arr, dtype=torch.long)
        pm = torch.ones(len(x_arr), T, dtype=torch.bool)
        return TensorDataset(xt, yt, pm)

    train_ds = _make_ds(x_tr, y_tr)
    val_ds   = _make_ds(x_va, y_va)
    test_ds  = _make_ds(x_te, y_te)

    clf = SyntheticMLPClassifier(
        J=J, F=F, T=T, K=K, num_classes=3, player_mode=player_mode,
    )
    clf.fit(x_tr, y_tr, epochs=clf_epochs, device=device, seed=seed)

    return bench, clf, train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Temporal K=4 smoke test -------------------------------------------
    bench = GaussianMotionBenchmark(J=5, F=1, T=16, rho=0.5, alpha=0.8, K=4)
    x = bench.sample(10, seed=0)
    assert x.shape == (10, 5, 1, 16), x.shape

    x0 = x[0]
    samps = bench.conditional_sample(x0, s_obs=(0, 2), s_hid=(1, 3), n_samples=100)
    assert samps.shape == (100, 5, 1, 16), samps.shape
    for k in [0, 2]:
        for t in bench.window_assignments[k]:
            assert np.allclose(samps[:, :, :, t], x0[:, :, t][None]), f"window {k} frame {t} changed"
    print("temporal K=4 conditional_sample OK")

    clf = SyntheticMLPClassifier(J=5, F=1, T=16, K=4, num_classes=3)
    x_np = bench.sample(200, seed=1)
    y_np = np.where(x_np[:, 0, 0, :].mean(-1) < 0, 0,
                    np.where(x_np[:, 1, 0, :].mean(-1) > 0, 2, 1)).astype(np.int64)
    clf.fit(x_np, y_np, epochs=5)
    print("temporal SyntheticMLPClassifier.fit OK")

    x_t = torch.tensor(x[:1])
    prob_fn = clf.class_prob_fn(0)
    with torch.no_grad():
        assert prob_fn(x_t).shape == (1,)
    v_all = bench.compute_v_true_all_coalitions(x_t, prob_fn, K_mc=50, seed=7)
    assert len(v_all) == 2 ** 4
    phi = bench.compute_true_shapley(v_all)
    assert phi.shape == (4,)
    print(f"temporal K=4 true φ = {phi}")

    # ---- Temporal K=8 smoke test -------------------------------------------
    bench8 = GaussianMotionBenchmark(J=5, F=1, T=32, rho=0.5, alpha=0.8, K=8)
    bench8.setup_label_fn(n_calib=500, seed=0)
    x8 = bench8.sample(5, seed=0)
    y8 = bench8.canonical_label_fn(x8)
    assert y8.shape == (5,) and y8.dtype == np.int64
    # label_config should hold 2 Olsen terms (K=8 → K//4 = 2).
    assert bench8.label_config["n_terms"] == 2
    print("temporal K=8 label + n_terms OK")

    # ---- Spatial smoke test ------------------------------------------------
    benchS = GaussianMotionBenchmark(
        J=5, F=1, T=16, rho=0.5, alpha=0.8, K=4,
        player_mode="spatial", signal_joints=(0, 1, 2, 3),
    )
    benchS.setup_label_fn(n_calib=500, seed=0)
    xS = benchS.sample(5, seed=0)
    yS = benchS.canonical_label_fn(xS)
    assert yS.shape == (5,)
    # Conditional sample over joints.
    xS0 = xS[0]
    j_obs = (0, 2, 4)
    j_hid = (1, 3)
    samps_j = benchS.conditional_sample_spatial(xS0, j_obs, j_hid, n_samples=20)
    assert samps_j.shape == (20, 5, 1, 16)
    for j in j_obs:
        assert np.allclose(samps_j[:, j], xS0[j][None]), f"observed joint {j} changed"
    print("spatial conditional_sample_spatial OK")
    print("Smoke test PASSED")
