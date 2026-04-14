"""evaluate_shap_synthetic.py — Compare SHAP methods against true Shapley values.

Two sub-commands:

  gaussian
      Loads a trained ActorSHAP checkpoint + the Gaussian motion benchmark
      saved by train_actor_shap_synthetic.py.  For each test sequence enumerates
      all 2^4 = 16 temporal coalitions, computes v(S) for five methods, and
      computes the true v_true(S) via exact Gaussian conditional sampling.

      Metrics (Olsen et al. JMLR 2022, Sec 4.1):
        EC1 = mean |phi_j_true - phi_j_method|  (MAE of Shapley values)
        EC2 = mean (v_true(S) - v_hat(S))^2      (MSE of contribution functions)
        EC3 = mean (f(x*) - v_hat(S))^2          (EPE — no v_true needed)

      Methods compared:
        actor           — ActorSHAP manifold-constrained completions (our method)
        zero            — replace hidden windows with zeros
        mean            — replace with training-set mean trajectory
        marginal        — replace with random training sequence (ignores x_S)
        gaussian_temporal — empirical AR structure per joint (ignores joint correlations)
        gaussian_full   — Ledoit-Wolf shrinkage on full (J·F·T) covariance

  diagnostic
      Loads a trained ActorSHAP checkpoint + the diagnostic gait benchmark.
      Runs spatial KernelSHAP (M=17 joints) for all methods and compares against
      the analytically exact true Shapley values.

      Metrics:
        EC1               — MAE of per-joint Shapley values vs true
        Top-k recovery    — fraction of true top-k joints in estimated top-k
        Spearman ρ        — rank correlation of estimated vs true Shapley values

USAGE
-----
    # Gaussian mode:
    python evaluate_shap_synthetic.py gaussian \\
        --ckpt_dir experiment_outs/actor_shap_synthetic/<run> \\
        --device cuda:0 --K_mc_true 1000 --n_completion_samples 20

    # Diagnostic mode:
    python evaluate_shap_synthetic.py diagnostic \\
        --ckpt_dir experiment_outs/actor_shap_diagnostic/<run> \\
        --device cuda:0 --n_kernel_samples 500 --n_completion_samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.actor.actor_shap import ActorSHAP, CoalitionFullEncoder, MaskedActorEncoder
from model.actor.shap_masking import (
    H36M_JOINT_NAMES,
    build_spatial_shap_mask,
    build_temporal_shap_mask,
)
from model.actor.shap_compute import (
    _enumerate_all_coalitions,
    _sample_kernel_coalitions,
    _shapley_kernel_weight,
    _solve_shapley_wls,
)
from model.actor.transformer_arch import Decoder_TRANSFORMER

from synthetic.gaussian_motion import (
    GaussianMotionBenchmark,
    SyntheticMLPClassifier,
    _enumerate_temporal_coalitions,
)
from synthetic.diagnostic_motion import LinearDiagnosticClassifier, DIAGNOSTIC_JOINTS


# ---------------------------------------------------------------------------
# Shared: model loading
# ---------------------------------------------------------------------------

def load_actor_shap(ckpt_dir: str, device: torch.device) -> ActorSHAP:
    """Rebuild ActorSHAP from config.json + last checkpoint in ckpt_dir."""
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    J = cfg.get("J", 17)
    F = cfg.get("F", 3)
    common = dict(
        modeltype="cvae",
        njoints=J, nfeats=F,
        num_frames=0, num_classes=cfg.get("num_classes", 3),
        translation=True, pose_rep="xyz",
        glob=True, glob_rot=[3.141592653589793, 0, 0],
        latent_dim=cfg["latent_dim"], ff_size=cfg["ff_size"],
        num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
        dropout=cfg.get("dropout", 0.1), ablation=None, activation="gelu",
    )
    encoder        = CoalitionFullEncoder(**common)
    masked_encoder = MaskedActorEncoder(**common)
    decoder        = Decoder_TRANSFORMER(**common)
    model = ActorSHAP(
        encoder, masked_encoder, decoder,
        latent_dim=cfg["latent_dim"], device=device,
        pose_rep="xyz", num_classes=cfg.get("num_classes", 3),
    ).to(device)

    ckpt_path = os.path.join(ckpt_dir, "actor_shap_synthetic_last.ckpt")
    if not os.path.exists(ckpt_path):
        # Fallback: find any .ckpt file, prefer the most recently modified one.
        ckpts = [
            os.path.join(ckpt_dir, f)
            for f in os.listdir(ckpt_dir)
            if f.endswith(".ckpt")
        ]
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_dir}\n"
                "Make sure training completed (check logs for 'Saved final checkpoint')."
            )
        ckpt_path = max(ckpts, key=os.path.getmtime)
    print(f"[Load] ActorSHAP ← {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Remove Lightning module prefix if present.
    state = {k.removeprefix("model."): v for k, v in state.items()
             if k.startswith("model.") or not k.startswith(("lr", "lambda", "phase"))}
    # Try strict=False in case extra keys are present.
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Shared: value function helpers
# ---------------------------------------------------------------------------

def actor_value_fn(
    actor_shap: ActorSHAP,
    prob_fn,
    x: torch.Tensor,          # (1, J, F, T)
    y: torch.Tensor,          # (1,)
    mask: torch.Tensor,       # (1, T)
    lengths: torch.Tensor,    # (1,)
    coalition_mask: torch.Tensor,  # (1, T) temporal  or  (1, J) spatial
    n_samples: int = 20,
) -> float:
    """Average classifier probability over n_samples ActorSHAP completions."""
    comps = actor_shap.sample_completions(x, y, mask, lengths, coalition_mask, n_samples=n_samples)
    x_batch = torch.cat(comps, dim=0)  # (n_samples, J, F, T)
    with torch.no_grad():
        probs = prob_fn(x_batch)
    return float(probs.mean().item())


def zero_value_fn(
    prob_fn,
    x: torch.Tensor,
    coalition_mask: torch.Tensor,  # (1, T) or (1, J)
    is_temporal: bool,
) -> float:
    x_imp = x.clone()
    if is_temporal:
        # coalition_mask: (1, T) — True = observed
        hid = ~coalition_mask[0]  # (T,)
        x_imp[0, :, :, hid] = 0.0
    else:
        hid = ~coalition_mask[0]  # (J,)
        x_imp[0, hid] = 0.0
    with torch.no_grad():
        return float(prob_fn(x_imp).item())


def mean_value_fn(
    prob_fn,
    x: torch.Tensor,
    coalition_mask: torch.Tensor,
    is_temporal: bool,
    mean_tensor: torch.Tensor,  # (J, F, T) training mean
) -> float:
    x_imp = x.clone()
    if is_temporal:
        hid = ~coalition_mask[0]
        x_imp[0, :, :, hid] = mean_tensor[:, :, hid]
    else:
        hid = ~coalition_mask[0]
        x_imp[0, hid] = mean_tensor[hid]
    with torch.no_grad():
        return float(prob_fn(x_imp).item())


def marginal_value_fn(
    prob_fn,
    x: torch.Tensor,
    coalition_mask: torch.Tensor,
    is_temporal: bool,
    train_pool: torch.Tensor,   # (N, J, F, T)
    n_samples: int = 20,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    idxs = rng.integers(0, len(train_pool), size=n_samples)
    probs = []
    for idx in idxs:
        x_imp = x.clone()
        donor = train_pool[int(idx)].to(x.device)
        if is_temporal:
            hid = ~coalition_mask[0]
            x_imp[0, :, :, hid] = donor[:, :, hid]
        else:
            hid = ~coalition_mask[0]
            x_imp[0, hid] = donor[hid]
        with torch.no_grad():
            probs.append(float(prob_fn(x_imp).item()))
    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Gaussian_cov baselines
# ---------------------------------------------------------------------------

class GaussianTemporalImputer:
    """Imputes hidden windows using empirical temporal AR structure.

    Fits Sigma_time_hat from training data (averaged over all joints/features).
    At inference uses the Gaussian conditional mean formula for each (j,f)
    independently — ignores cross-joint correlations.
    """

    def __init__(self, T: int, K: int = 4):
        self.T = T
        self.K = K
        self.Sigma_time_hat: np.ndarray | None = None
        self._cond_cache: dict = {}
        quarter = T // K
        self.window_assignments = [
            list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
            for k in range(K)
        ]

    def fit(self, x_train_jft: np.ndarray) -> None:
        """Fit temporal covariance from (N, J, F, T) training array."""
        N, J, F, T = x_train_jft.shape
        # Flatten to (N*J*F, T) — pool over all joints and features.
        x_flat = x_train_jft.reshape(-1, T)
        x_flat = x_flat - x_flat.mean(axis=0, keepdims=True)
        self.Sigma_time_hat = (x_flat.T @ x_flat) / max(len(x_flat) - 1, 1)  # (T, T)
        self.Sigma_time_hat += 1e-6 * np.eye(T)
        self._cond_cache.clear()

    def _cond_params(self, s_obs: tuple, s_hid: tuple):
        key = (s_obs, s_hid)
        if key in self._cond_cache:
            return self._cond_cache[key]
        t_obs = np.concatenate([self.window_assignments[k] for k in s_obs])
        t_hid = np.concatenate([self.window_assignments[k] for k in s_hid])
        Soo = self.Sigma_time_hat[np.ix_(t_obs, t_obs)]
        Sho = self.Sigma_time_hat[np.ix_(t_hid, t_obs)]
        W = Sho @ np.linalg.solve(Soo + 1e-10 * np.eye(len(t_obs)), np.eye(len(t_obs)))
        Shh = self.Sigma_time_hat[np.ix_(t_hid, t_hid)]
        Sigma_cond = Shh - W @ Sho.T
        Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2 + 1e-8 * np.eye(len(t_hid))
        L_cond = np.linalg.cholesky(Sigma_cond)
        self._cond_cache[key] = (W, L_cond, t_obs, t_hid)
        return self._cond_cache[key]

    def sample(
        self,
        x_np: np.ndarray,    # (J, F, T)
        s_obs: tuple,
        s_hid: tuple,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return (n_samples, J, F, T) conditional samples (ignores joint corr)."""
        J, F, T = x_np.shape
        W, L_cond, t_obs, t_hid = self._cond_params(s_obs, s_hid)
        n_hid = len(t_hid)
        mu = np.einsum("ht,jft->jfh", W, x_np[:, :, t_obs])  # (J, F, n_hid)
        noise = rng.standard_normal((n_samples, J, F, n_hid))
        noise = np.einsum("tT,njfT->njft", L_cond, noise)  # apply temporal Cholesky per j,f
        # No joint Cholesky — temporal baseline ignores joint correlations.
        out = np.tile(x_np[None], (n_samples, 1, 1, 1)).astype(np.float32)
        out[:, :, :, t_hid] = (mu[None] + noise).astype(np.float32)
        return out


class GaussianFullImputer:
    """Imputes hidden windows using full Ledoit-Wolf covariance estimate.

    Fits a (J·F·T × J·F·T) covariance matrix on training data.  For large J·F·T
    relative to N_train, the Ledoit-Wolf shrinkage is heavy and the baseline
    degrades — demonstrating ActorSHAP's advantage at CARE-PD training-set scales.
    """

    def __init__(self, J: int = 17, F: int = 3, T: int = 81, K: int = 4):
        self.J = J
        self.F = F
        self.T = T
        self.K = K
        self.D = J * F * T
        self.Sigma_hat: np.ndarray | None = None
        self.mu_hat: np.ndarray | None = None
        self._cond_cache: dict = {}
        quarter = T // K
        self.window_assignments = [
            list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
            for k in range(K)
        ]

    def fit(self, x_train_jft: np.ndarray) -> None:
        """Fit LedoitWolf on (N, J, F, T) → (N, J*F*T) data."""
        N = len(x_train_jft)
        D = self.D
        print(f"  [GaussianFull] Fitting LedoitWolf on ({N}, {D}) data …", end=" ", flush=True)
        X = x_train_jft.reshape(N, -1).astype(np.float64)  # (N, D)
        self.mu_hat = X.mean(axis=0)
        X_c = X - self.mu_hat
        lw = LedoitWolf(assume_centered=True)
        lw.fit(X_c)
        self.Sigma_hat = lw.covariance_  # (D, D)
        self._cond_cache.clear()
        print("done.")

    def _obs_hid_indices(self, s_obs: tuple, s_hid: tuple):
        """Return flat indices for observed and hidden window elements."""
        def _win_flat_idx(window_keys):
            idx = []
            for k in window_keys:
                t_frames = self.window_assignments[k]
                for j in range(self.J):
                    for fv in range(self.F):
                        for t in t_frames:
                            idx.append(j * self.F * self.T + fv * self.T + t)
            return np.array(idx, dtype=int)
        return _win_flat_idx(s_obs), _win_flat_idx(s_hid)

    def _cond_params(self, s_obs: tuple, s_hid: tuple):
        key = (s_obs, s_hid)
        if key in self._cond_cache:
            return self._cond_cache[key]
        obs_idx, hid_idx = self._obs_hid_indices(s_obs, s_hid)
        Soo = self.Sigma_hat[np.ix_(obs_idx, obs_idx)]
        Sho = self.Sigma_hat[np.ix_(hid_idx, obs_idx)]
        W = Sho @ np.linalg.solve(Soo + 1e-10 * np.eye(len(obs_idx)), np.eye(len(obs_idx)))
        Shh = self.Sigma_hat[np.ix_(hid_idx, hid_idx)]
        Sigma_cond = Shh - W @ Sho.T
        Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2 + 1e-8 * np.eye(len(hid_idx))
        L_cond = np.linalg.cholesky(Sigma_cond)
        self._cond_cache[key] = (W, L_cond, obs_idx, hid_idx)
        return self._cond_cache[key]

    def sample(
        self,
        x_np: np.ndarray,  # (J, F, T)
        s_obs: tuple,
        s_hid: tuple,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return (n_samples, J, F, T) conditional samples."""
        J, F, T = self.J, self.F, self.T
        W, L_cond, obs_idx, hid_idx = self._cond_params(s_obs, s_hid)

        x_flat = x_np.reshape(-1) - self.mu_hat  # (D,)
        mu_cond = W @ x_flat[obs_idx]             # (n_hid,)
        noise   = rng.standard_normal((n_samples, len(hid_idx)))  # (n_samp, n_hid)
        noise   = (L_cond @ noise.T).T            # (n_samp, n_hid)

        out = np.tile(x_np[None], (n_samples, 1, 1, 1)).astype(np.float32)  # (n, J, F, T)
        hid_vals = (mu_cond + self.mu_hat[hid_idx])[None] + noise  # (n_samp, n_hid)
        # Scatter back to (J, F, T) indexing.
        out_flat = out.reshape(n_samples, -1)
        out_flat[:, hid_idx] = hid_vals.astype(np.float32)
        return out_flat.reshape(n_samples, J, F, T)


# ---------------------------------------------------------------------------
# Core temporal SHAP evaluation (returns both v(S) and phi)
# ---------------------------------------------------------------------------

def evaluate_temporal_shap_all_methods(
    actor_shap: ActorSHAP,
    bench: GaussianMotionBenchmark,
    prob_fn,                        # callable (B,J,F,T) → (B,)
    x: torch.Tensor,                # (1, J, F, T)
    y: torch.Tensor,                # (1,)
    mask: torch.Tensor,             # (1, T)
    lengths: torch.Tensor,          # (1,)
    train_pool: torch.Tensor,       # (N, J, F, T) for marginal
    train_mean: torch.Tensor,       # (J, F, T) for mean baseline
    gauss_temporal: GaussianTemporalImputer,
    gauss_full: GaussianFullImputer | None,
    window_assignments: list[list[int]],
    K_mc_true: int = 1000,
    n_completion_samples: int = 20,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> dict[str, dict]:
    """Evaluate all SHAP methods for one test sequence.

    Returns
    -------
    results : dict mapping method_name → {"v": array(16,), "phi": array(4,)}
    Also includes "true" key.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if device is None:
        device = x.device

    K = 4
    T = x.shape[-1]
    x_np = x[0].cpu().numpy()  # (J, F, T)

    coalitions, weights = _enumerate_temporal_coalitions(K)
    N_coal = len(coalitions)  # 16

    # Build temporal coalition masks once.
    all_cms: list[torch.Tensor] = []
    for z in coalitions:
        s_obs = [k for k in range(K) if z[k] == 1]
        cm = build_temporal_shap_mask(s_obs, window_assignments, T, device).unsqueeze(0)
        all_cms.append(cm)

    # Helper: build (s_obs_tuple, s_hid_tuple) from coalition vector.
    def _split(z):
        return (
            tuple(k for k in range(K) if z[k] == 1),
            tuple(k for k in range(K) if z[k] == 0),
        )

    results: dict[str, dict] = {}

    # ---- True (exact Gaussian conditional) ----------------------------------
    print("    true ...", end=" ", flush=True)
    v_true_all = bench.compute_v_true_all_coalitions(x, prob_fn, K_mc=K_mc_true,
                                                      device=device, seed=int(rng.integers(2**31)))
    v_true = np.array([v_true_all[tuple(z.tolist())] for z in coalitions])
    phi_true = bench.compute_true_shapley(v_true_all)
    results["true"] = {"v": v_true, "phi": phi_true}

    # ---- Imputation-based methods -------------------------------------------
    def _imputer_to_v(imputer_fn, n_samp: int) -> np.ndarray:
        """imputer_fn(x_np, s_obs, s_hid, n_samp, rng) → (n_samp, J, F, T)"""
        v = np.zeros(N_coal)
        for i, z in enumerate(coalitions):
            s_obs, s_hid = _split(z)
            if not s_hid:
                with torch.no_grad():
                    v[i] = float(prob_fn(x.to(device)).item())
            elif not s_obs:
                # Empty coalition: sample from marginal (random training samples).
                idxs = rng.integers(0, len(train_pool), size=n_samp)
                x_marg = train_pool[idxs].to(device)
                with torch.no_grad():
                    v[i] = float(prob_fn(x_marg).mean().item())
            else:
                samps_np = imputer_fn(x_np, s_obs, s_hid, n_samp, rng)
                samps_t  = torch.tensor(samps_np, device=device)
                with torch.no_grad():
                    v[i] = float(prob_fn(samps_t).mean().item())
        return v

    def _v_to_phi(v: np.ndarray) -> np.ndarray:
        v_empty = float(v[0])   # all-zeros coalition is first in product order
        v_full  = float(v[-1])  # all-ones coalition is last
        return _solve_shapley_wls(coalitions, v, weights, v_empty=v_empty, v_full=v_full)

    # Zero imputation.
    print("zero ...", end=" ", flush=True)
    v_zero = np.zeros(N_coal)
    for i, (z, cm) in enumerate(zip(coalitions, all_cms)):
        s_obs, s_hid = _split(z)
        if not s_hid:
            with torch.no_grad():
                v_zero[i] = float(prob_fn(x.to(device)).item())
        else:
            x_imp = x.clone().to(device)
            hid_t = ~cm[0]  # (T,)
            x_imp[0, :, :, hid_t] = 0.0
            with torch.no_grad():
                v_zero[i] = float(prob_fn(x_imp).item())
    results["zero"] = {"v": v_zero, "phi": _v_to_phi(v_zero)}

    # Mean imputation.
    print("mean ...", end=" ", flush=True)
    v_mean_arr = np.zeros(N_coal)
    for i, (z, cm) in enumerate(zip(coalitions, all_cms)):
        s_obs, s_hid = _split(z)
        if not s_hid:
            with torch.no_grad():
                v_mean_arr[i] = float(prob_fn(x.to(device)).item())
        else:
            x_imp = x.clone().to(device)
            hid_t = ~cm[0]
            x_imp[0, :, :, hid_t] = train_mean[:, :, hid_t].to(device)
            with torch.no_grad():
                v_mean_arr[i] = float(prob_fn(x_imp).item())
    results["mean"] = {"v": v_mean_arr, "phi": _v_to_phi(v_mean_arr)}

    # Marginal imputation.
    print("marginal ...", end=" ", flush=True)

    def _marginal_imp(x_np, s_obs, s_hid, n_samp, rng):
        t_hid = np.concatenate([window_assignments[k] for k in s_hid])
        idxs = rng.integers(0, len(train_pool), size=n_samp)
        donors_np = train_pool[idxs].numpy()  # (n_samp, J, F, T)
        out = np.tile(x_np[None], (n_samp, 1, 1, 1))
        out[:, :, :, t_hid] = donors_np[:, :, :, t_hid]
        return out.astype(np.float32)

    v_marg = _imputer_to_v(_marginal_imp, n_completion_samples)
    results["marginal"] = {"v": v_marg, "phi": _v_to_phi(v_marg)}

    # Gaussian temporal imputation.
    print("gauss_temporal ...", end=" ", flush=True)

    def _gauss_temp_imp(x_np, s_obs, s_hid, n_samp, rng):
        return gauss_temporal.sample(x_np, s_obs, s_hid, n_samp, rng)

    v_gt = _imputer_to_v(_gauss_temp_imp, n_completion_samples)
    results["gaussian_temporal"] = {"v": v_gt, "phi": _v_to_phi(v_gt)}

    # Gaussian full imputation (if available).
    if gauss_full is not None:
        print("gauss_full ...", end=" ", flush=True)

        def _gauss_full_imp(x_np, s_obs, s_hid, n_samp, rng):
            return gauss_full.sample(x_np, s_obs, s_hid, n_samp, rng)

        v_gf = _imputer_to_v(_gauss_full_imp, n_completion_samples)
        results["gaussian_full"] = {"v": v_gf, "phi": _v_to_phi(v_gf)}

    # ActorSHAP method.
    print("actor ...", end=" ", flush=True)
    v_actor = np.zeros(N_coal)
    for i, (z, cm) in enumerate(zip(coalitions, all_cms)):
        s_obs, s_hid = _split(z)
        if not s_hid:
            with torch.no_grad():
                v_actor[i] = float(prob_fn(x.to(device)).item())
        elif not s_obs:
            idxs = rng.integers(0, len(train_pool), size=n_completion_samples)
            x_marg = train_pool[idxs].to(device)
            with torch.no_grad():
                v_actor[i] = float(prob_fn(x_marg).mean().item())
        else:
            comps = actor_shap.sample_completions(
                x.to(device), y.to(device), mask.to(device), lengths.to(device),
                cm.to(device), n_samples=n_completion_samples,
            )
            x_batch = torch.cat(comps, dim=0)
            with torch.no_grad():
                v_actor[i] = float(prob_fn(x_batch).mean().item())
    results["actor"] = {"v": v_actor, "phi": _v_to_phi(v_actor)}
    print("done.")

    return results


# ---------------------------------------------------------------------------
# EC metrics
# ---------------------------------------------------------------------------

def compute_ec_metrics(
    results_per_seq: list[dict],
    methods: list[str],
) -> dict[str, dict[str, float]]:
    """Compute EC1, EC2, EC3 across all test sequences for each method.

    EC1 = mean |phi_j_method - phi_j_true|  (MAE of Shapley values)
    EC2 = mean (v_hat(S) - v_true(S))^2      (MSE of contribution functions;
          averaged over non-trivial coalitions S ∈ P*(M))
    EC3 = mean (f(x) - v_hat(S))^2           (EPE; f(x) = v_true(all-ones))

    Parameters
    ----------
    results_per_seq : list of per-sequence result dicts (output of
                      evaluate_temporal_shap_all_methods).
    methods : list of method names to report (excluding "true").
    """
    ec: dict[str, dict[str, list[float]]] = {m: {"ec1": [], "ec2": [], "ec3": []} for m in methods}

    for res in results_per_seq:
        phi_true = res["true"]["phi"]
        v_true   = res["true"]["v"]
        # f(x*) = v_true at the all-ones coalition (last entry in product order).
        f_x = float(v_true[-1])

        # Non-trivial coalitions: exclude all-zeros (index 0) and all-ones (index -1).
        nontrivial = slice(1, -1)

        for m in methods:
            if m not in res:
                continue
            phi_m = res[m]["phi"]
            v_m   = res[m]["v"]

            ec1 = float(np.abs(phi_m - phi_true).mean())
            ec2 = float(((v_m[nontrivial] - v_true[nontrivial]) ** 2).mean())
            ec3 = float(((f_x - v_m[nontrivial]) ** 2).mean())

            ec[m]["ec1"].append(ec1)
            ec[m]["ec2"].append(ec2)
            ec[m]["ec3"].append(ec3)

    summary: dict[str, dict[str, float]] = {}
    for m in methods:
        if not ec[m]["ec1"]:
            continue
        summary[m] = {
            "EC1_mean": float(np.mean(ec[m]["ec1"])),
            "EC1_std":  float(np.std(ec[m]["ec1"])),
            "EC2_mean": float(np.mean(ec[m]["ec2"])),
            "EC2_std":  float(np.std(ec[m]["ec2"])),
            "EC3_mean": float(np.mean(ec[m]["ec3"])),
            "EC3_std":  float(np.std(ec[m]["ec3"])),
        }
    return summary


def print_ec_table(summary: dict[str, dict[str, float]], rho: float, alpha: float) -> None:
    """Print a LaTeX-style table to stdout."""
    methods = list(summary.keys())
    header = f"Gaussian benchmark  ρ={rho}  α={alpha}"
    print(f"\n{'=' * 72}")
    print(header)
    print(f"{'=' * 72}")
    row_fmt = "{:<22s} {:>10.4f} {:>10.4f} {:>10.4f}"
    print(f"{'Method':<22s} {'EC1':>10s} {'EC2':>10s} {'EC3':>10s}")
    print("-" * 56)
    for m in methods:
        s = summary[m]
        print(row_fmt.format(m, s["EC1_mean"], s["EC2_mean"], s["EC3_mean"]))
    print()

    # LaTeX snippet.
    print("% LaTeX table snippet:")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print("Method & EC1 & EC2 & EC3 \\\\")
    print("\\midrule")
    for m in methods:
        s = summary[m]
        print(
            f"{m} & {s['EC1_mean']:.4f} $\\pm$ {s['EC1_std']:.4f}"
            f" & {s['EC2_mean']:.4f} $\\pm$ {s['EC2_std']:.4f}"
            f" & {s['EC3_mean']:.4f} $\\pm$ {s['EC3_std']:.4f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")


# ---------------------------------------------------------------------------
# ==================== GAUSSIAN MODE ========================================
# ---------------------------------------------------------------------------

def run_gaussian(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt_dir = args.ckpt_dir

    # Load benchmark and classifier.
    bench_path = os.path.join(ckpt_dir, "synthetic_benchmark.pkl")
    with open(bench_path, "rb") as f:
        bench: GaussianMotionBenchmark = pickle.load(f)
    print(f"[Gaussian] Benchmark: rho={bench.rho}, alpha={bench.alpha}, J={bench.J}, T={bench.T}")

    clf_meta_path = os.path.join(ckpt_dir, "synthetic_clf_meta.json")
    with open(clf_meta_path) as f:
        clf_meta = json.load(f)
    clf = SyntheticMLPClassifier(
        J=clf_meta["J"], F=clf_meta["F"], T=clf_meta["T"],
        K=clf_meta.get("K", 4), num_classes=clf_meta["num_classes"],
    )
    clf.load_state_dict(torch.load(os.path.join(ckpt_dir, "synthetic_clf.pt"), map_location="cpu"))
    clf.to(device).eval()

    # Load test sequences.
    test_data = torch.load(os.path.join(ckpt_dir, "synthetic_test.pt"), map_location="cpu")
    x_test_bttf = test_data["x"]   # (N, T, J, F)
    y_test       = test_data["y"]
    pm_test      = test_data["pad_mask"]

    # Convert to (N, J, F, T) format.
    x_test_jft = x_test_bttf.permute(0, 2, 3, 1).contiguous()  # (N, J, F, T)

    # Load train pool for marginal/mean.
    x_train_jft = np.load(os.path.join(ckpt_dir, "x_train_jft.npy"))  # (N_tr, J, F, T)
    train_pool = torch.tensor(x_train_jft)  # (N_tr, J, F, T)
    train_mean = torch.tensor(x_train_jft.mean(axis=0))  # (J, F, T)

    # Fit Gaussian baselines.
    print("[Gaussian] Fitting Gaussian_temporal baseline …")
    gauss_temp = GaussianTemporalImputer(T=bench.T, K=bench.K)
    gauss_temp.fit(x_train_jft)

    gauss_full = None
    if args.use_gaussian_full:
        print("[Gaussian] Fitting Gaussian_full (LedoitWolf) baseline …")
        gauss_full = GaussianFullImputer(J=bench.J, F=bench.F, T=bench.T, K=bench.K)
        gauss_full.fit(x_train_jft)

    # Load ActorSHAP.
    actor_shap = load_actor_shap(ckpt_dir, device)

    # Probability function: class 0 probability.
    class_idx = args.class_idx
    def prob_fn(x_in: torch.Tensor) -> torch.Tensor:
        logits = clf(x_in.to(device))
        return torch.softmax(logits, dim=-1)[:, class_idx]

    # Fixed window assignments (equal quarters — same as training).
    T = bench.T
    K = bench.K
    quarter = T // K
    window_assignments = [
        list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
        for k in range(K)
    ]

    # Iterate over test sequences.
    n_test = min(len(x_test_jft), args.n_test_sequences)
    print(f"[Gaussian] Evaluating {n_test} test sequences …")
    results_per_seq = []
    rng = np.random.default_rng(args.seed)

    for i in range(n_test):
        print(f"  [{i + 1}/{n_test}]", end=" ")
        x_i    = x_test_jft[i : i + 1].to(device)   # (1, J, F, T)
        y_i    = y_test[i : i + 1]
        mask_i = pm_test[i : i + 1]
        lengths_i = torch.tensor([T])

        res = evaluate_temporal_shap_all_methods(
            actor_shap=actor_shap,
            bench=bench,
            prob_fn=prob_fn,
            x=x_i, y=y_i, mask=mask_i, lengths=lengths_i,
            train_pool=train_pool,
            train_mean=train_mean,
            gauss_temporal=gauss_temp,
            gauss_full=gauss_full,
            window_assignments=window_assignments,
            K_mc_true=args.K_mc_true,
            n_completion_samples=args.n_completion_samples,
            device=device,
            rng=rng,
        )
        results_per_seq.append(res)

    # Compute metrics.
    methods = ["actor", "zero", "mean", "marginal", "gaussian_temporal"]
    if gauss_full is not None:
        methods.append("gaussian_full")

    summary = compute_ec_metrics(results_per_seq, methods)
    print_ec_table(summary, rho=bench.rho, alpha=bench.alpha)

    # Save results.
    out_dir = args.output_dir or os.path.join(ckpt_dir, "eval_synthetic_gaussian")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ec_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # Save per-sequence results (phi and v arrays as lists).
    per_seq_save = []
    for res in results_per_seq:
        r = {}
        for m, d in res.items():
            r[m] = {"v": d["v"].tolist(), "phi": d["phi"].tolist()}
        per_seq_save.append(r)
    with open(os.path.join(out_dir, "per_sequence.json"), "w") as f:
        json.dump(per_seq_save, f)
    print(f"\n[Gaussian] Results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# ==================== DIAGNOSTIC MODE ======================================
# ---------------------------------------------------------------------------

def run_diagnostic(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt_dir = args.ckpt_dir

    # Load true Shapley values and test sequences.
    x_test_np   = np.load(os.path.join(ckpt_dir, "x_test.npy"))          # (N, J, F, T)
    phi_true_np = np.load(os.path.join(ckpt_dir, "phi_true_test.npy"))   # (N, J)
    w_true      = np.load(os.path.join(ckpt_dir, "w_true.npy"))          # (J,)
    mu_j_path   = os.path.join(ckpt_dir, "mu_j.npy")
    mu_j        = np.load(mu_j_path) if os.path.exists(mu_j_path) else None

    test_data = torch.load(os.path.join(ckpt_dir, "synthetic_test.pt"), map_location="cpu")
    x_test_bttf = test_data["x"]   # (N, T, J, F)
    y_test       = test_data["y"]
    pm_test      = test_data["pad_mask"]
    x_test_jft   = x_test_bttf.permute(0, 2, 3, 1).contiguous()  # (N, J, F, T)

    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    T = cfg.get("T", 81)
    J = cfg.get("J", 17)
    F = cfg.get("F", 3)

    # Build and load classifier.
    clf = LinearDiagnosticClassifier(w_true)
    clf.load_state_dict(torch.load(os.path.join(ckpt_dir, "synthetic_clf.pt"), map_location="cpu"),
                        strict=False)
    clf.to(device).eval()
    if mu_j is not None:
        clf.register_buffer("mu_j", torch.tensor(mu_j))

    def prob_fn(x_in: torch.Tensor) -> torch.Tensor:
        return clf(x_in.to(device))  # (B,) scalar

    # Train pool for marginal baseline.
    x_train_jft = np.load(os.path.join(ckpt_dir, "x_train_jft.npy"))  # (N_tr, J, F, T)
    train_pool  = torch.tensor(x_train_jft)
    train_mean  = torch.tensor(x_train_jft.mean(axis=0))  # (J, F, T)

    # Load ActorSHAP.
    actor_shap = load_actor_shap(ckpt_dir, device)

    # Spatial SHAP: M=17 players. Run KernelSHAP for each method.
    M = 17
    rng = np.random.default_rng(args.seed)

    n_test = min(len(x_test_np), args.n_test_sequences)
    print(f"[Diagnostic] Evaluating {n_test} test sequences, M={M} spatial joints …")

    all_phi: dict[str, list] = {m: [] for m in ["actor", "zero", "mean", "marginal"]}
    phi_true_list = []
    lengths_t = torch.tensor([T])

    for i in range(n_test):
        print(f"  [{i + 1}/{n_test}]", end=" ", flush=True)
        x_i    = x_test_jft[i : i + 1].to(device)   # (1, J, F, T)
        y_i    = y_test[i : i + 1]
        mask_i = pm_test[i : i + 1]
        phi_t  = phi_true_np[i]                       # (J,)
        phi_true_list.append(phi_t)

        coalitions, weights = _sample_kernel_coalitions(M, args.n_kernel_samples, rng)

        # Build all spatial coalition masks.
        cms_list = []
        for z in coalitions:
            s_obs = np.where(z == 1)[0].tolist()
            cms_list.append(
                build_spatial_shap_mask(s_obs, device, n_joints=M).unsqueeze(0)
            )
        zero_cm = build_spatial_shap_mask([], device, n_joints=M).unsqueeze(0)
        full_cm = build_spatial_shap_mask(list(range(M)), device, n_joints=M).unsqueeze(0)
        all_cms_with_bounds = [zero_cm, full_cm] + cms_list

        # Value function per method.
        for method_name in ["actor", "zero", "mean", "marginal"]:
            vals = np.zeros(len(all_cms_with_bounds))
            for ci, cm in enumerate(all_cms_with_bounds):
                hid_joints = (~cm[0]).nonzero(as_tuple=True)[0].cpu().numpy()
                if len(hid_joints) == 0:
                    # v(full): f(x*)
                    with torch.no_grad():
                        vals[ci] = float(prob_fn(x_i).item())
                elif len(hid_joints) == M:
                    # v(empty): marginal mean
                    idxs = rng.integers(0, len(train_pool), args.n_completion_samples)
                    x_m = train_pool[idxs].to(device)
                    with torch.no_grad():
                        vals[ci] = float(prob_fn(x_m).mean().item())
                elif method_name == "actor":
                    comps = actor_shap.sample_completions(
                        x_i, y_i.to(device), mask_i.to(device), lengths_t.to(device),
                        cm.to(device), n_samples=args.n_completion_samples,
                    )
                    x_b = torch.cat(comps, dim=0)
                    with torch.no_grad():
                        vals[ci] = float(prob_fn(x_b).mean().item())
                elif method_name == "zero":
                    x_imp = x_i.clone()
                    x_imp[0, hid_joints] = 0.0
                    with torch.no_grad():
                        vals[ci] = float(prob_fn(x_imp).item())
                elif method_name == "mean":
                    x_imp = x_i.clone()
                    x_imp[0, hid_joints] = train_mean[hid_joints].to(device)
                    with torch.no_grad():
                        vals[ci] = float(prob_fn(x_imp).item())
                elif method_name == "marginal":
                    probs_m = []
                    for _ in range(args.n_completion_samples):
                        idx = int(rng.integers(0, len(train_pool)))
                        x_imp = x_i.clone()
                        x_imp[0, hid_joints] = train_pool[idx, hid_joints].to(device)
                        with torch.no_grad():
                            probs_m.append(float(prob_fn(x_imp).item()))
                    vals[ci] = float(np.mean(probs_m))

            v_empty = float(vals[0])
            v_full  = float(vals[1])
            phi_m = _solve_shapley_wls(
                coalitions, vals[2:], weights, v_empty=v_empty, v_full=v_full
            )
            all_phi[method_name].append(phi_m)
        print("done.")

    # Compute diagnostics.
    print("\n[Diagnostic] Summary")
    methods = ["actor", "zero", "mean", "marginal"]
    phi_true_arr = np.array(phi_true_list)  # (N, J)

    diag_results: dict[str, dict] = {}
    for m in methods:
        phi_m_arr = np.array(all_phi[m])  # (N, J)
        ec1 = float(np.abs(phi_m_arr - phi_true_arr).mean())

        # Top-k recovery.
        topk_recovery: dict[int, float] = {}
        for k in [3, 5]:
            top_true = set(np.argsort(np.abs(phi_true_arr).mean(axis=0))[-k:])
            top_m    = set(np.argsort(np.abs(phi_m_arr).mean(axis=0))[-k:])
            topk_recovery[k] = len(top_true & top_m) / k

        # Spearman ρ (averaged over test sequences).
        spearman_vals = []
        for n in range(len(phi_true_arr)):
            rho_s, _ = spearmanr(np.abs(phi_true_arr[n]), np.abs(phi_m_arr[n]))
            spearman_vals.append(float(rho_s))
        spearman_mean = float(np.mean(spearman_vals))

        diag_results[m] = {
            "EC1": ec1,
            "top3_recovery": topk_recovery[3],
            "top5_recovery": topk_recovery[5],
            "spearman_rho":  spearman_mean,
        }

    # Print table.
    print(f"\n{'=' * 72}")
    print("Diagnostic benchmark — spatial Shapley value accuracy")
    print(f"{'=' * 72}")
    print(f"True diagnostic joints (H36M): {[H36M_JOINT_NAMES[j] for j in DIAGNOSTIC_JOINTS]}")
    hdr = f"{'Method':<12} {'EC1':>8} {'Top-3':>8} {'Top-5':>8} {'Spearman':>10}"
    print(hdr)
    print("-" * 50)
    for m in methods:
        d = diag_results[m]
        print(f"{m:<12} {d['EC1']:>8.4f} {d['top3_recovery']:>8.2%} "
              f"{d['top5_recovery']:>8.2%} {d['spearman_rho']:>10.4f}")

    # LaTeX snippet.
    print("\n% LaTeX snippet:")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Method & EC1 & Top-3 & Top-5 & Spearman $\\rho$ \\\\")
    print("\\midrule")
    for m in methods:
        d = diag_results[m]
        print(
            f"{m} & {d['EC1']:.4f} & {d['top3_recovery']:.2%} "
            f"& {d['top5_recovery']:.2%} & {d['spearman_rho']:.4f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")

    out_dir = args.output_dir or os.path.join(ckpt_dir, "eval_synthetic_diagnostic")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diagnostic_summary.json"), "w") as f:
        json.dump(diag_results, f, indent=2)
    print(f"\n[Diagnostic] Results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ActorSHAP against true Shapley values on synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- Shared args ----
    def _add_common(sp):
        sp.add_argument("--ckpt_dir", required=True,
                        help="Directory produced by train_actor_shap_synthetic.py.")
        sp.add_argument("--device", default="cuda:0")
        sp.add_argument("--n_test_sequences", type=int, default=100,
                        help="Cap on test sequences to evaluate (0 = all).")
        sp.add_argument("--n_completion_samples", type=int, default=20,
                        help="Stochastic completions per coalition.")
        sp.add_argument("--output_dir", default=None,
                        help="Directory for output JSON files (default: <ckpt_dir>/eval_…).")
        sp.add_argument("--seed", type=int, default=42)

    # ---- Gaussian sub-command ----
    sg = sub.add_parser("gaussian", help="EC1/EC2/EC3 evaluation on Gaussian benchmark.")
    _add_common(sg)
    sg.add_argument("--K_mc_true", type=int, default=1000,
                    help="Monte Carlo samples per coalition for v_true (higher = more accurate).")
    sg.add_argument("--class_idx", type=int, default=0,
                    help="Class index for probability evaluation.")
    sg.add_argument("--use_gaussian_full", action="store_true",
                    help="Also run the full LedoitWolf Gaussian baseline (expensive).")

    # ---- Diagnostic sub-command ----
    sd = sub.add_parser("diagnostic", help="Joint rank recovery evaluation on diagnostic benchmark.")
    _add_common(sd)
    sd.add_argument("--n_kernel_samples", type=int, default=200,
                    help="KernelSHAP coalition pairs per sequence (spatial, M=17).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "gaussian":
        run_gaussian(args)
    elif args.mode == "diagnostic":
        run_diagnostic(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
