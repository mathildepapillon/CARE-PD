"""burr_tabular.py — Multivariate Burr tabular benchmark for flow-SHAP validation.

Implements the exact Olsen et al. (JMLR 2022) Section 4.2 Simulation Study:
Continuous Data, adapted for the flow-SHAP pipeline.

DATA MODEL
----------
M scalar features (w_1,...,w_M) are drawn jointly from the M-dimensional
Burr(κ, b, r) distribution via the compound Weibull representation:

    V  ~ Gamma(κ, 1)
    Z_m ~ Exp(1)  independently for m=1,...,M
    w_m = (Z_m / (r_m * V))^(1/b_m)

κ=2.0 gives avg Kendall τ ≈ 0.20 (moderate dependence, matching the paper).
b_m ∈ {2.00, 2.25,...,6.00} and r_m ∈ {1.00, 1.25,...,5.00} are sampled once
at benchmark construction.

PIPELINE EMBEDDING
------------------
The flow-matching model (VelocityNet) requires (J=17, F=3, T) tensors.
We embed the M scalar Burr features as:

    x[j, f, t] = w_t    for all j ∈ {0,...,16}, f ∈ {0,1,2}, t ∈ {0,...,M-1}

So T=M and each "time step" carries exactly one Burr feature replicated across
all 17×3 spatial channels.  Per-window mean recovers the Burr scalar exactly.

PLAYERS
-------
M = K temporal windows of size 1 frame each.  Each of the M features is one
SHAP player.  For M ≤ 20: enumerate all 2^M coalitions.  For M > 20: sample
NS=1000 coalitions (KernelSHAP weighted), matching the paper's Step 4.

LABEL (Eq. 12/16)
-----------------
M must be divisible by 5.  The label uses M/5 interaction groups (non-overlapping):

    y = Σ_{k=0}^{M/5-1} [
            c_{3k+1}·sin(π·u_{5k+1}·u_{5k+2})
          + c_{3k+2}·u_{5k+3}·exp(c_{3k+3}·u_{5k+4}·u_{5k+5})
        ]
        + ε · (1/(M/5)) · Σ_{k=1}^{M/5} v_k

where u_m = F_m(w_m) ∈ (0,1) (marginal CDF), c_l ∈ {0.1,0.2,...,2.0} (fixed
at setup), ε ~ N(0,1) and v_k ~ U[0,1] are drawn fresh per observation.

CLASSIFIER
----------
sklearn RandomForestRegressor(n_estimators=500) fitted on the M raw Burr
features.  This is a regression model; SHAP values explain the continuous
prediction f(x).

EXPERIMENT CONFIGS (from the paper)
------------------------------------
  Low-dim  (M ∈ {5, 10}):  all 2^M coalitions, N_train ∈ {100, 1000, 5000},
                            N_test=100, R=20 repetitions per setting.
  High-dim (M ∈ {25, 50, 100, 250}): NS=1000 coalitions, R=10 repetitions.

USAGE
-----
    bench = BurrTabularBenchmark(M=10, kappa=2.0, seed=0)
    x_train = bench.sample(1000, seed=0)           # (1000, 17, 3, 10)
    y_train = bench.canonical_label_fn(x_train)    # (1000,) float regression targets

    clf = BurrRFWrapper(M=10); clf.fit(x_train, y_train)

    x_test = bench.sample(100, seed=1)
    v_true = bench.compute_v_true_all_coalitions(
        torch.tensor(x_test[0:1]), clf.class_prob_fn(0), K_mc=5000)
    phi_true = bench.compute_true_shapley(v_true)  # (10,)
"""

from __future__ import annotations

import itertools
import pickle
from math import comb
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import TensorDataset

# Reuse helpers from the Gaussian benchmark — no code duplication.
from synthetic.gaussian_motion import (
    _enumerate_temporal_coalitions,
    _sample_kernelshap_coalitions,
    _shapley_kernel_weight,
    _solve_shapley_wls,
)

_NS_THRESHOLD = 20   # For M <= 20: enumerate all 2^M; else sample NS=1000


# ---------------------------------------------------------------------------
# BurrTabularBenchmark
# ---------------------------------------------------------------------------

class BurrTabularBenchmark:
    """Multivariate Burr tabular SHAP benchmark (Olsen et al. JMLR 2022, §4.2).

    M scalar features, exact conditional distributions, regression RF black-box.
    """

    def __init__(
        self,
        M: int = 10,
        kappa: float = 2.0,
        J: int = 17,
        F: int = 3,
        seed: int = 42,
    ) -> None:
        if M % 5 != 0:
            raise ValueError(
                f"M={M} must be divisible by 5 (paper Eq. 16 groups 5 features per term)."
            )
        self.M      = M
        self.kappa  = float(kappa)
        self.J      = J
        self.F      = F
        # Pipeline dimensions: T=M so one time step per Burr feature.
        self.T      = M
        self.K      = M

        self.player_mode      = "temporal"
        self.signal_joints    = None          # not used in temporal mode
        self.label_config: dict | None = None

        # One temporal window per feature (window k = time step k, size 1).
        self.window_assignments: list[list[int]] = [[m] for m in range(M)]

        # Sample b_m and r_m once at construction (re-drawn per seed).
        rng = np.random.default_rng(seed)
        b_choices = np.arange(2.00, 6.25, 0.25)   # {2.00, 2.25,...,6.00}
        r_choices = np.arange(1.00, 5.25, 0.25)   # {1.00, 1.25,...,5.00}
        self.b = rng.choice(b_choices, size=M, replace=True).astype(np.float64)
        self.r = rng.choice(r_choices, size=M, replace=True).astype(np.float64)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_burr(
        self, N: int, kappa: float, b: np.ndarray, r: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample N draws from len(b)-dimensional Burr(kappa, b, r).

        Returns (N, len(b)) float64 array.
        """
        M = len(b)
        V = rng.gamma(shape=kappa, scale=1.0, size=N)         # (N,)
        Z = rng.exponential(scale=1.0, size=(N, M))            # (N, M)
        W = (Z / (r[None, :] * V[:, None])) ** (1.0 / b[None, :])  # (N, M)
        return W

    def sample(self, N: int, seed: int | None = None) -> np.ndarray:
        """Sample N unconditioned sequences of shape (N, J, F, M) float32.

        Each sample contains M Burr features replicated across all J*F spatial
        channels so that x[n, j, f, m] = w_m for all j, f.
        """
        rng = np.random.default_rng(seed)
        W = self._sample_burr(N, self.kappa, self.b, self.r, rng)  # (N, M)
        # Replicate across J, F: shape (N, J, F, M)
        x = np.broadcast_to(
            W[:, None, None, :], (N, self.J, self.F, self.M)
        ).copy().astype(np.float32)
        return x

    # ------------------------------------------------------------------
    # Conditional sampling (exact Burr conditional, Appendix E.1)
    # ------------------------------------------------------------------

    def conditional_sample(
        self,
        x: np.ndarray,
        s_obs: tuple[int, ...],
        s_hid: tuple[int, ...],
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw n_samples from p(x_hidden | x_observed) using exact Burr conditional.

        Parameters
        ----------
        x : (J, F, M) conditioning sequence
        s_obs : observed window/feature indices
        s_hid : hidden window/feature indices

        Returns (n_samples, J, F, M) float32.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract observed Burr scalars (exact, no noise: x[0,0,m] == w_m).
        w_obs = np.array([float(x[0, 0, m]) for m in s_obs], dtype=np.float64)

        # Conditional Burr parameters (Appendix E.1).
        sum_obs = sum(
            self.r[m] * (float(x[0, 0, m]) ** self.b[m])
            for m in s_obs
        )
        kappa_tilde = self.kappa + len(s_obs)
        b_hid = self.b[list(s_hid)]
        r_hid = self.r[list(s_hid)] / (1.0 + sum_obs)

        # Sample from Burr(kappa_tilde, b_hid, r_hid).
        W_hid = self._sample_burr(n_samples, kappa_tilde, b_hid, r_hid, rng)  # (n_samples, |s_hid|)

        # Construct completions: copy observed, fill hidden.
        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
        for i, m in enumerate(s_hid):
            w_m = W_hid[:, i]                        # (n_samples,)
            out[:, :, :, m] = w_m[:, None, None]     # broadcast over J, F

        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # Label function
    # ------------------------------------------------------------------

    def _burr_cdf(self, w: np.ndarray) -> np.ndarray:
        """CDF transform: u_m = 1 - (1 + r_m * w_m^{b_m})^{-κ}, element-wise.

        Parameters
        ----------
        w : (..., M) array of raw Burr values

        Returns (..., M) array of u values in (0, 1).
        """
        return 1.0 - (1.0 + self.r * w ** self.b) ** (-self.kappa)

    def _extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract M Burr scalar features from (N, J, F, M) tensor.

        Returns (N, M) via per-timestep mean over J, F axes (= w_m exactly).
        """
        return x.mean(axis=(1, 2))  # (N, M)

    def setup_label_fn(
        self,
        n_calib: int = 5000,
        seed: int = 999,
        noise_std: float = 1.0,
    ) -> None:
        """Sample and store label function coefficients.

        Parameters
        ----------
        n_calib : not used (label function has no data-dependent calibration
                  beyond the coefficient draw — kept for interface parity).
        seed : seed for coefficient sampling.
        noise_std : standard deviation scale for the heteroscedastic noise
                    (set to 0.0 to disable noise for debugging). Default=1.0.
        """
        rng = np.random.default_rng(seed)
        n_groups = self.M // 5
        # 3 coefficients per group, sampled from {0.1, 0.2,..., 2.0}.
        c_choices = np.arange(0.1, 2.05, 0.1)
        coeffs = rng.choice(c_choices, size=3 * n_groups, replace=True)
        self.label_config = {
            "coeffs": coeffs.tolist(),
            "n_groups": n_groups,
            "noise_std": float(noise_std),
        }

    def _olsen_score(self, x: np.ndarray) -> np.ndarray:
        """Compute the noiseless Olsen score for (N, J, F, M) input.

        Returns (N,) float64 array.
        """
        if self.label_config is None:
            raise RuntimeError("Call setup_label_fn() first.")

        w = self._extract_features(x).astype(np.float64)  # (N, M)
        u = self._burr_cdf(w)                              # (N, M), ∈ (0,1)

        n_groups = self.label_config["n_groups"]
        coeffs = np.asarray(self.label_config["coeffs"], dtype=np.float64)
        N = len(x)
        score = np.zeros(N, dtype=np.float64)

        for k in range(n_groups):
            c1 = coeffs[3 * k]
            c2 = coeffs[3 * k + 1]
            c3 = coeffs[3 * k + 2]
            u1 = u[:, 5 * k]
            u2 = u[:, 5 * k + 1]
            u3 = u[:, 5 * k + 2]
            u4 = u[:, 5 * k + 3]
            u5 = u[:, 5 * k + 4]
            score += c1 * np.sin(np.pi * u1 * u2) + c2 * u3 * np.exp(c3 * u4 * u5)

        return score

    def canonical_label_fn(
        self,
        x: np.ndarray,
        rng: np.random.Generator | None = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Map (N, J, F, M) sequences to (N,) float32 regression targets.

        Adds heteroscedastic noise per Appendix E.1:
            noise_i = ε_i · (1/(M/5)) · Σ_{k=1}^{M/5} v_k
        where ε_i ~ N(0,1) and v_k ~ U[0,1] per observation.
        """
        if self.label_config is None:
            raise RuntimeError("Call setup_label_fn() first.")

        score = self._olsen_score(x)  # (N,) noiseless

        if add_noise and self.label_config["noise_std"] > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            N = len(x)
            n_groups = self.label_config["n_groups"]
            noise_std = self.label_config["noise_std"]
            eps = rng.standard_normal(N)                          # (N,)
            v = rng.uniform(0.0, 1.0, size=(N, n_groups))        # (N, M/5)
            noise = noise_std * eps * v.mean(axis=1)              # (N,)
            score = score + noise

        return score.astype(np.float32)

    # ------------------------------------------------------------------
    # True value function & Shapley values
    # ------------------------------------------------------------------

    def _eval_regression_fn(
        self,
        regression_fn: Callable,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate regression_fn on (B, J, F, M) tensor, returning (B,)."""
        return regression_fn(x)

    def compute_v_true_all_coalitions(
        self,
        x: torch.Tensor,
        classifier_fn: Callable,
        K_mc: int = 5000,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> dict[tuple, float]:
        """Compute v_true(S) for all 2^M (or 1000 sampled) coalitions.

        Parameters
        ----------
        x : (1, J, F, M) torch tensor — a single test sample.
        classifier_fn : callable (B, J, F, M) → (B,) scalar values.
        K_mc : Monte Carlo samples per coalition.

        Returns dict mapping coalition tuple to v_true(S) float.
        """
        if device is None:
            device = x.device
        rng = np.random.default_rng(seed)
        x_np = x[0].cpu().numpy()  # (J, F, M)

        # Choose coalitions: enumerate all 2^M if feasible, else sample 1000.
        if self.M <= _NS_THRESHOLD:
            coalitions, _ = _enumerate_temporal_coalitions(self.M)
        else:
            # KernelSHAP paired sampling (1000 pairs = 2000 coalitions).
            # Also add empty and full explicitly.
            sample_cols, _ = _sample_kernelshap_coalitions(self.M, 500, rng)
            empty = np.zeros((1, self.M), dtype=int)
            full  = np.ones((1, self.M), dtype=int)
            coalitions = np.vstack([empty, full, sample_cols])
            # Deduplicate.
            coalitions = np.unique(coalitions, axis=0)

        v_true: dict[tuple, float] = {}

        for z in coalitions:
            s_obs = tuple(k for k in range(self.M) if z[k] == 1)
            s_hid = tuple(k for k in range(self.M) if z[k] == 0)

            if not s_hid:
                # v = f(x*)
                with torch.no_grad():
                    val = float(classifier_fn(x.to(device)).mean().item())
            elif not s_obs:
                # v = E[f(x)] — marginal average.
                x_marg = self.sample(K_mc, seed=int(rng.integers(1 << 31)))
                with torch.no_grad():
                    val = float(classifier_fn(
                        torch.tensor(x_marg, device=device)
                    ).mean().item())
            else:
                samps = self.conditional_sample(x_np, s_obs, s_hid, K_mc, rng)
                with torch.no_grad():
                    val = float(classifier_fn(
                        torch.tensor(samps, device=device)
                    ).mean().item())

            v_true[tuple(z.tolist())] = val

        return v_true

    def compute_true_shapley(self, v_all: dict[tuple, float]) -> np.ndarray:
        """Convert v_all dict to (M,) Shapley values via WLS solve."""
        if self.M <= _NS_THRESHOLD:
            coalitions, weights = _enumerate_temporal_coalitions(self.M)
        else:
            # Reconstruct from dict keys.
            keys = np.array(list(v_all.keys()), dtype=int)
            coalitions = keys
            weights = np.array([
                _shapley_kernel_weight(int(k.sum()), self.M)
                for k in keys
            ])

        values = np.array([v_all[tuple(z.tolist())] for z in coalitions])
        z_empty = tuple([0] * self.M)
        z_full  = tuple([1] * self.M)
        return _solve_shapley_wls(
            coalitions, values, weights,
            v_empty=v_all[z_empty],
            v_full=v_all[z_full],
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BurrTabularBenchmark":
        with open(path, "rb") as f:
            return pickle.load(f)

    def build_pytorch_dataset(
        self,
        N: int,
        seed: int = 0,
        add_noise: bool = True,
    ) -> TensorDataset:
        """Sample N sequences, compute labels, return TensorDataset.

        Tensors stored as (N, T, J, F) — the pipeline's ACTOR convention.
        """
        rng = np.random.default_rng(seed)
        x = self.sample(N, seed=seed)                    # (N, J, F, M)
        y = self.canonical_label_fn(x, rng=rng, add_noise=add_noise)
        xt = torch.tensor(x).permute(0, 3, 1, 2).contiguous()  # (N, T, J, F)
        yt = torch.tensor(y, dtype=torch.float32)
        pm = torch.ones(N, self.M, dtype=torch.bool)
        return TensorDataset(xt, yt, pm)


# ---------------------------------------------------------------------------
# BurrRFWrapper — sklearn RF regressor with pipeline-compatible interface
# ---------------------------------------------------------------------------

class BurrRFWrapper:
    """Wraps sklearn RandomForestRegressor to match the SyntheticMLPClassifier
    external interface expected by evaluate_shap_synthetic_gaussian.py.

    Attributes
    ----------
    task : str = "regression"  — detected by evaluate script for special handling.
    """

    task: str = "regression"

    def __init__(self, M: int, n_estimators: int = 500, num_classes: int = 1) -> None:
        self.M = M
        self.n_estimators = n_estimators
        self.num_classes = 1  # regression output
        self.rf = None

    def _extract_features(self, x: np.ndarray) -> np.ndarray:
        """(N, J, F, M) → (N, M) Burr scalars via per-timestep mean."""
        return x.mean(axis=(1, 2))  # (N, M)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train RandomForestRegressor on (N, J, F, M) data and (N,) labels."""
        from sklearn.ensemble import RandomForestRegressor
        feats = self._extract_features(x_train)  # (N, M)
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=0,
            n_jobs=-1,
        )
        self.rf.fit(feats, y_train)

    def predict_np(self, x: np.ndarray) -> np.ndarray:
        """(N, J, F, M) → (N,) regression predictions (numpy)."""
        if self.rf is None:
            raise RuntimeError("Call fit() first.")
        return self.rf.predict(self._extract_features(x)).astype(np.float32)

    def class_prob_fn(self, class_idx: int = 0) -> Callable:
        """Return a callable (x: Tensor (B,J,F,T)) → Tensor (B,) predictions.

        ``class_idx`` is ignored — this is a regression model.
        """
        rf = self.rf
        M = self.M

        def _fn(x: torch.Tensor) -> torch.Tensor:
            xnp = x.cpu().numpy()  # (B, J, F, M) or (B, J, F, T)
            feats = xnp.mean(axis=(1, 2))          # (B, M)
            preds = rf.predict(feats).astype(np.float32)
            return torch.tensor(preds, dtype=torch.float32)

        return _fn

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BurrRFWrapper":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------

def build_burr_benchmark_and_classifier(
    M: int = 10,
    kappa: float = 2.0,
    J: int = 17,
    F: int = 3,
    n_train: int = 1000,
    n_val: int = 200,
    n_test: int = 100,
    n_estimators: int = 500,
    seed: int = 0,
    noise_std: float = 1.0,
) -> tuple[
    BurrTabularBenchmark,
    BurrRFWrapper,
    TensorDataset,
    TensorDataset,
    TensorDataset,
]:
    """Convenience factory: build benchmark, sample splits, train RF.

    Parameters
    ----------
    M : number of Burr features (SHAP players). Must be divisible by 5.
    kappa : Burr shape parameter (controls dependence). Default 2.0.
    n_train/n_val/n_test : split sizes.
    n_estimators : trees in the random forest.
    seed : random seed; b_m/r_m parameters and data are all derived from this.
    noise_std : scale for label noise (0.0 = no noise).
    """
    bench = BurrTabularBenchmark(M=M, kappa=kappa, J=J, F=F, seed=seed)
    bench.setup_label_fn(seed=seed + 99, noise_std=noise_std)

    rng_tr = np.random.default_rng(seed)
    rng_va = np.random.default_rng(seed + 1)
    rng_te = np.random.default_rng(seed + 2)

    x_tr = bench.sample(n_train, seed=seed)
    x_va = bench.sample(n_val,   seed=seed + 1)
    x_te = bench.sample(n_test,  seed=seed + 2)

    y_tr = bench.canonical_label_fn(x_tr, rng=rng_tr)
    y_va = bench.canonical_label_fn(x_va, rng=rng_va)
    y_te = bench.canonical_label_fn(x_te, rng=rng_te)

    def _make_ds(x_arr, y_arr):
        xt = torch.tensor(x_arr).permute(0, 3, 1, 2).contiguous()  # (N, T, J, F)
        yt = torch.tensor(y_arr, dtype=torch.float32)
        pm = torch.ones(len(x_arr), M, dtype=torch.bool)
        return TensorDataset(xt, yt, pm)

    train_ds = _make_ds(x_tr, y_tr)
    val_ds   = _make_ds(x_va, y_va)
    test_ds  = _make_ds(x_te, y_te)

    clf = BurrRFWrapper(M=M, n_estimators=n_estimators)
    clf.fit(x_tr, y_tr)

    return bench, clf, train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    print("BurrTabularBenchmark smoke test …")
    try:
        M_test = 5  # use 5 (minimum valid M) for speed; M=10 takes ~20s
        bench = BurrTabularBenchmark(M=M_test, kappa=2.0, J=17, F=3, seed=0)

        # --- sample ---
        x = bench.sample(50, seed=1)
        assert x.shape == (50, 17, 3, M_test), f"sample shape wrong: {x.shape}"
        assert np.all(x >= 0), "Burr values must be non-negative"
        print(f"  sample: OK  shape={x.shape}  min={x.min():.4f}  max={x.max():.4f}")

        # --- label ---
        bench.setup_label_fn(seed=42, noise_std=1.0)
        y = bench.canonical_label_fn(x)
        assert y.shape == (50,), f"label shape wrong: {y.shape}"
        assert np.isfinite(y).all(), "Labels contain NaN/Inf"
        print(f"  canonical_label_fn: OK  mean={y.mean():.3f}  std={y.std():.3f}")

        # --- conditional_sample ---
        x0 = x[0]  # (J, F, M)
        samps = bench.conditional_sample(x0, s_obs=(0, 1), s_hid=(2, 3, 4), n_samples=5)
        assert samps.shape == (5, 17, 3, M_test), f"conditional_sample shape: {samps.shape}"
        # Observed frames must be preserved exactly.
        for k in (0, 1):
            assert np.allclose(samps[:, :, :, k], x0[:, :, k][None]), f"frame {k} not preserved"
        print(f"  conditional_sample: OK  shape={samps.shape}")

        # --- BurrRFWrapper ---
        clf = BurrRFWrapper(M=M_test, n_estimators=10)
        clf.fit(x, y)
        prob_fn = clf.class_prob_fn(0)
        preds = prob_fn(torch.tensor(x))
        assert preds.shape == (50,), f"prob_fn output shape: {preds.shape}"
        print(f"  BurrRFWrapper.fit + class_prob_fn: OK  preds shape={preds.shape}")

        # --- v_true and Shapley ---
        x_single = torch.tensor(x[0:1])  # (1, J, F, M)
        v_true = bench.compute_v_true_all_coalitions(
            x_single, prob_fn, K_mc=20, device=torch.device("cpu"), seed=7
        )
        assert len(v_true) == 2 ** M_test, f"Expected {2**M_test} coalitions, got {len(v_true)}"
        phi = bench.compute_true_shapley(v_true)
        assert phi.shape == (M_test,), f"phi shape: {phi.shape}"
        # Efficiency: sum(phi) + v_empty ≈ v_full
        z_empty = tuple([0] * M_test)
        z_full  = tuple([1] * M_test)
        resid = abs(v_true[z_full] - (v_true[z_empty] + phi.sum()))
        assert resid < 1.0, f"Efficiency axiom violated: residual={resid:.4f}"
        print(f"  compute_v_true + compute_true_shapley: OK  phi={phi}")

        # --- save/load round-trip ---
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            bench.save(os.path.join(tmpdir, "bench.pkl"))
            bench2 = BurrTabularBenchmark.load(os.path.join(tmpdir, "bench.pkl"))
            x2 = bench2.sample(3, seed=99)
            assert x2.shape == (3, 17, 3, M_test)
            clf.save(os.path.join(tmpdir, "clf.pkl"))
            clf2 = BurrRFWrapper.load(os.path.join(tmpdir, "clf.pkl"))
            p2 = clf2.class_prob_fn(0)(torch.tensor(x2))
            assert p2.shape == (3,)
        print("  save/load round-trip: OK")

        print("\nSmoke test PASSED")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
