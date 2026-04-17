"""gaussian_motion.py — Gaussian temporal motion benchmark for ActorSHAP validation.

Mirrors Olsen et al. JMLR 2022 Section 4.2 (Simulation Study: Continuous Data)
adapted for temporal motion sequences.

DATA MODEL
----------
    x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)

    Sigma_joints[j,j'] = rho  (j ≠ j')
    Sigma_joints[j,j]  = 1
    Sigma_time[t,t']   = alpha^|t-t'|        (AR(1))

PLAYERS
-------
Temporal: M = K = 4 equal-length windows, exactly 2^4 = 16 coalitions → exact
Shapley values tractable without approximation.

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
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import ndtr as _ndtr  # standard normal CDF Φ
from torch.utils.data import DataLoader, TensorDataset


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
    coalitions = np.array(list(itertools.product([0, 1], repeat=K)), dtype=int)
    weights = np.array([_shapley_kernel_weight(int(c.sum()), K) for c in coalitions])
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


def nonlinear_olsen_score(
    x: np.ndarray,
    window_assignments: list[list[int]],
    sigma_k: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Nonlinear response adapted from Olsen et al. (JMLR 2022) Eq. (12), K=4 windows.

    Maps sequences (N, J, F, T) → scalar score (N,) via:

        w_k  = mean of all (j, f, t) in window k          (per-window grand mean)
        u_k  = Φ(w_k / σ_k)                               (CDF → approx. Uniform[0,1])
        score = c₁·sin(π·u₀·u₁) + c₂·u₂·exp(c₃·u₂·u₃)  (nonlinear + interactions)

    Windows 0&1 interact in the sin term; windows 2&3 interact in the exp term.
    All four windows influence the score, with different structural roles.

    Parameters
    ----------
    x : (N, J, F, T)
    window_assignments : K=4 lists of frame indices (from GaussianMotionBenchmark)
    sigma_k : (K,) per-window standard deviations for standardisation
    coeffs  : (3,) = [c1, c2, c3]

    Returns
    -------
    score : (N,) float64
    """
    w = _per_window_grand_means(x, window_assignments)  # (N, K)
    u = _ndtr(w / (sigma_k[np.newaxis] + 1e-12))        # (N, K), ≈ Uniform[0,1]
    c1, c2, c3 = coeffs
    return (
        c1 * np.sin(np.pi * u[:, 0] * u[:, 1])
        + c2 * u[:, 2] * np.exp(c3 * u[:, 2] * u[:, 3])
    )


# ---------------------------------------------------------------------------
# Core benchmark class
# ---------------------------------------------------------------------------

class GaussianMotionBenchmark:
    """Synthetic Gaussian temporal motion benchmark.

    Parameters
    ----------
    J, F, T : int
        Joints (17), features per joint (3), frames per sequence (81).
    rho : float
        Off-diagonal equicorrelation between joints.  Controls how much
        knowing joint j helps predict joint j'.
    alpha : float
        AR(1) temporal autocorrelation.  Controls how much past frames
        predict future frames.
    K : int
        Number of temporal windows (players for SHAP). Must be 4.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        rho: float = 0.5,
        alpha: float = 0.8,
        K: int = 4,
    ):
        assert K == 4, "Only K=4 temporal windows supported."
        self.J = J
        self.F = F
        self.T = T
        self.rho = rho
        self.alpha = alpha
        self.K = K

        # Build parametric covariance matrices.
        self.Sigma_joints = _equicorr(J, rho)      # (J, J)
        self.Sigma_time   = _ar1_cov(T, alpha)      # (T, T)
        self.L_joints     = np.linalg.cholesky(
            self.Sigma_joints + 1e-8 * np.eye(J)
        )  # (J, J)
        self.L_time       = np.linalg.cholesky(
            self.Sigma_time + 1e-8 * np.eye(T)
        )  # (T, T)

        # Window frame assignments (equal quarters).
        quarter = T // K
        self.window_assignments: list[list[int]] = [
            list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
            for k in range(K)
        ]

        # Precompute Cholesky + mean-weight matrices for all 14 non-trivial coalitions.
        self._cond_cache: dict[tuple, tuple] = {}
        self._precompute_conditionals()

        # Label function config (set by setup_label_fn; None until then).
        # Stored in the pickle so train/val/test always use identical labelling.
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

        Draws a calibration sample to estimate per-window standard deviations
        (σ_k) and samples the three response coefficients (c1, c2, c3).  The
        result is saved in ``self.label_config`` and persisted with the pickle.

        Call this once when creating the benchmark; all subsequent calls to
        ``canonical_label_fn`` and ``build_pytorch_dataset`` will use the
        stored config so that train / val / test splits see identical labels.

        Parameters
        ----------
        n_calib : int
            Number of calibration sequences used to estimate σ_k.
        seed : int
            RNG seed for coefficient sampling and calibration data.
        noise_std : float
            Gaussian noise added to the score before quantile-binning
            (0 = deterministic labels, consistent with the paper's default).
        """
        rng = np.random.default_rng(seed)

        # Calibration sample to estimate per-window std σ_k.
        x_cal = self.sample(n_calib, seed=int(rng.integers(2**30)))
        w_cal = _per_window_grand_means(x_cal, self.window_assignments)  # (n_calib, K)
        sigma_k = w_cal.std(axis=0).clip(min=1e-8)  # (K,)

        # Coefficients following the paper's spirit: c1 in [0.5, 2], c2 in [0.5, 2],
        # c3 in [0.5, 1.0] (kept small so exp(c3·u·v) ≤ e ≈ 2.7 stays bounded).
        c1 = float(rng.uniform(0.5, 2.0))
        c2 = float(rng.uniform(0.5, 2.0))
        c3 = float(rng.uniform(0.5, 1.0))

        self.label_config = {
            "sigma_k":   sigma_k,
            "coeffs":    np.array([c1, c2, c3]),
            "noise_std": noise_std,
            "seed":      seed,
        }

    def canonical_label_fn(
        self,
        x: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply the stored nonlinear label function to sequences x (N, J, F, T).

        Raises ``RuntimeError`` if ``setup_label_fn`` has not been called.

        Returns
        -------
        y : (N,) int64 in {0, 1, 2} (quantile-binned score).
        """
        if self.label_config is None:
            raise RuntimeError(
                "Call bench.setup_label_fn() before using canonical_label_fn."
            )
        cfg = self.label_config
        score = nonlinear_olsen_score(
            x, self.window_assignments, cfg["sigma_k"], cfg["coeffs"],
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
        """Cache (L_cond_time, W_mean) for each unique (S_obs_tuple, S_hid_tuple)."""
        K = self.K
        coalitions, _ = _enumerate_temporal_coalitions(K)
        seen: set[tuple] = set()
        for z in coalitions:
            s_obs = tuple(k for k in range(K) if z[k] == 1)
            s_hid = tuple(k for k in range(K) if z[k] == 0)
            if not s_obs or not s_hid:
                continue  # skip empty or full coalition
            key = (s_obs, s_hid)
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

        key = (s_obs, s_hid)
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

    Architecture mirrors Olsen et al. (2022) Section 4.2: the classifier sees
    the same K per-window grand-mean features that determine the nonlinear
    Shapley label, making it a genuine black-box the SHAP methods must explain.

    Feature extraction
    ------------------
    For each of the K temporal windows:
        w_k = mean of x over all (joint, feature, time) in that window  → scalar
    Input to the MLP: (w_0, w_1, w_2, w_3) — K=4 dimensional.

    Why grand means?  The nonlinear response
        score = c1·sin(π·u0·u1) + c2·u2·exp(c3·u2·u3)
    depends only on these K per-window aggregates.  Using higher-dimensional
    per-(joint, feature) means would give the MLP spurious noise dimensions.

    Architecture: K → BatchNorm → 64 → ReLU → 64 → ReLU → num_classes
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        K: int = 4,
        num_classes: int = 3,
        hidden: int = 64,
    ):
        super().__init__()
        quarter = T // K
        self.window_starts = [k * quarter for k in range(K)]
        self.window_ends   = [(k + 1) * quarter if k < K - 1 else T for k in range(K)]
        self.net = nn.Sequential(
            nn.BatchNorm1d(K),
            nn.Linear(K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, J, F, T) → (B, K) per-window grand means."""
        feats = [
            x[:, :, :, s:e].mean(dim=(1, 2, 3))  # (B,)
            for s, e in zip(self.window_starts, self.window_ends)
        ]
        return torch.stack(feats, dim=1)  # (B, K)

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
) -> tuple[
    "GaussianMotionBenchmark",
    "SyntheticMLPClassifier",
    TensorDataset,
    TensorDataset,
    TensorDataset,
]:
    """Convenience factory: build benchmark, sample splits, train classifier.

    Returns
    -------
    bench, classifier, train_ds, val_ds, test_ds
    """
    bench = GaussianMotionBenchmark(J=J, F=F, T=T, rho=rho, alpha=alpha, K=K)
    bench.setup_label_fn(n_calib=5000, seed=seed + 99)

    x_tr = bench.sample(n_train, seed=seed)
    x_va = bench.sample(n_val,   seed=seed + 1)
    x_te = bench.sample(n_test,  seed=seed + 2)

    y_tr = bench.canonical_label_fn(x_tr)
    y_va = bench.canonical_label_fn(x_va)
    y_te = bench.canonical_label_fn(x_te)

    # Build datasets in ACTOR (T-first) format.
    def _make_ds(x_arr, y_arr):
        xt = torch.tensor(x_arr).permute(0, 3, 1, 2).contiguous()  # (N, T, J, F)
        yt = torch.tensor(y_arr, dtype=torch.long)
        pm = torch.ones(len(x_arr), T, dtype=torch.bool)
        return TensorDataset(xt, yt, pm)

    train_ds = _make_ds(x_tr, y_tr)
    val_ds   = _make_ds(x_va, y_va)
    test_ds  = _make_ds(x_te, y_te)

    # Train classifier — use majority class as the target class for SHAP.
    clf = SyntheticMLPClassifier(J=J, F=F, T=T, K=K, num_classes=3)
    clf.fit(x_tr, y_tr, epochs=clf_epochs, device=device, seed=seed)

    return bench, clf, train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bench = GaussianMotionBenchmark(J=5, F=1, T=16, rho=0.5, alpha=0.8, K=4)
    x = bench.sample(10, seed=0)
    assert x.shape == (10, 5, 1, 16), x.shape

    x0 = x[0]  # (J, F, T)
    samps = bench.conditional_sample(x0, s_obs=(0, 2), s_hid=(1, 3), n_samples=1000)
    assert samps.shape == (100, 5, 1, 16), samps.shape
    # Observed windows should be unchanged.
    for k in [0, 2]:
        for t in bench.window_assignments[k]:
            assert np.allclose(samps[:, :, :, t], x0[:, :, t][None]), f"window {k} frame {t} changed"
    print("conditional_sample OK")

    clf = SyntheticMLPClassifier(J=5, F=1, T=16, K=4, num_classes=3)
    x_np = bench.sample(200, seed=1)
    y_np = (x_np[:, 0, 0, :].mean(-1) > 0).astype(np.int64)  # dummy labels
    # map to 3 classes
    y_np = np.where(y_np == 0, 0, np.where(x_np[:, 1, 0, :].mean(-1) > 0, 1, 2))
    clf.fit(x_np, y_np, epochs=5)
    print("SyntheticMLPClassifier.fit OK")

    x_t = torch.tensor(x[:1])  # (1, 5, 1, 16)
    prob_fn = clf.class_prob_fn(0)
    with torch.no_grad():
        p = prob_fn(x_t)
    assert p.shape == (1,), p.shape
    print("class_prob_fn OK")

    # True v(S).
    v_all = bench.compute_v_true_all_coalitions(x_t, prob_fn, K_mc=50, seed=7)
    assert len(v_all) == 2 ** 4  # 16
    phi = bench.compute_true_shapley(v_all)
    assert phi.shape == (4,)
    print(f"True Shapley values: {phi}")
    print("Smoke test PASSED")
