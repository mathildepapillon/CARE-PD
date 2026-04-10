"""shap_compute.py — KernelSHAP computation for ActorSHAP and comparison baselines.

Implements KernelSHAP (Lundberg & Lee 2017) over two axes:
  - Spatial:  17 individual H36M joints → ~3000 sampled coalitions, weighted OLS.
  - Temporal: K=4 stride-aligned windows → exact enumeration of all 2^4=16 coalitions.

For spatial SHAP, post-hoc aggregation into anatomical groups is provided by
summing individual joint Shapley values using H36M_GROUPS from shap_masking.

VALUE FUNCTION BASELINES
All four methods use the same KernelSHAP sampling and WLS regression.  Only the
replacement strategy for masked joints differs — this is the core comparison in
manifold SHAP papers (Frye et al. 2021 "Shapley Explainability on the Data Manifold").

  value_fn               — ActorSHAP manifold-constrained completions (our method).
  value_fn_zero          — Replace masked joints with zeros (standard off-manifold baseline).
  value_fn_mean          — Replace masked joints with their training-set mean position.
  value_fn_marginal      — Sample masked joints independently from the training distribution,
                           ignoring conditioning on observed joints.

Corresponding compute functions:
  compute_spatial_shap              — our method.
  compute_spatial_shap_baseline     — runs any of the three baselines above.
  compute_temporal_shap             — our method (exact, K=4 windows).
  compute_temporal_shap_baseline    — same for baselines.

GRADIENT-BASED BASELINES (not implemented here)
GradientSHAP (Lundberg & Lee 2017) and Integrated Gradients (Sundararajan et al. 2017)
operate on the input tensor directly and require a differentiable classifier.  They give
attributions at (J, F, T) resolution and must be summed over F×T per joint for comparison.
These are different in kind from coalition-based SHAP and are provided as wrappers around
the SHAP library in shap_baselines_gradient.py (optional, requires shap>=0.44).

Dependencies:
  model.actor.actor_shap  (ActorSHAP.sample_completions)
  model.actor.shap_masking (H36M_JOINT_NAMES, H36M_GROUPS, build_spatial_shap_mask,
                            build_temporal_shap_mask, build_temporal_windows,
                            detect_stride_period)
"""

from __future__ import annotations

import itertools
from math import comb
from typing import Any, Callable

import numpy as np
import torch

from model.actor.shap_masking import (
    H36M_GROUPS,
    H36M_JOINT_NAMES,
    build_spatial_shap_mask,
    build_temporal_shap_mask,
    build_temporal_windows,
    detect_stride_period,
)


# ---------------------------------------------------------------------------
# KernelSHAP kernel weight
# ---------------------------------------------------------------------------

def _shapley_kernel_weight(s: int, M: int) -> float:
    """Shapley kernel weight for a coalition of size s out of M players.

    Undefined (infinite) at s=0 and s=M; returns 0 for those cases since
    we handle them separately by excluding all-zeros/all-ones coalitions.
    """
    if s == 0 or s == M:
        return 0.0
    return (M - 1) / (comb(M, s) * s * (M - s))


def _sample_kernel_coalitions(
    M: int,
    N: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample N coalitions from the Shapley kernel distribution.

    Samples coalition sizes proportional to C(M,s) * π(s), then samples
    uniformly within each size class.  Each sampled coalition is paired with
    its complement to reduce variance (Lundberg & Lee 2017, Appendix).

    Args:
        M:   number of players.
        N:   number of coalition pairs to sample (total coalitions = 2*N).
        rng: optional numpy random generator for reproducibility.

    Returns:
        coalitions: (2*N, M) int array — each row is a binary coalition vector.
        weights:    (2*N,) float array — Shapley kernel weight for each row.
    """
    if rng is None:
        rng = np.random.default_rng()

    sizes = np.arange(1, M)
    probs = np.array(
        [comb(M, int(s)) * _shapley_kernel_weight(int(s), M) for s in sizes],
        dtype=float,
    )
    probs /= probs.sum()

    coalitions = np.zeros((2 * N, M), dtype=int)
    weights = np.zeros(2 * N)

    for i in range(N):
        s = int(rng.choice(sizes, p=probs))
        chosen = rng.choice(M, size=s, replace=False)
        z = np.zeros(M, dtype=int)
        z[chosen] = 1
        w = _shapley_kernel_weight(s, M)
        coalitions[2 * i] = z
        weights[2 * i] = w
        # Complement coalition.
        coalitions[2 * i + 1] = 1 - z
        weights[2 * i + 1] = _shapley_kernel_weight(M - s, M)

    return coalitions, weights


def _enumerate_all_coalitions(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all 2^M coalitions with their Shapley kernel weights.

    All-zeros and all-ones coalitions receive weight 0 (excluded from
    regression); the constraint Σφⱼzⱼ = f(all_ones) - φ_0 is satisfied
    naturally by the regression with these rows included.

    Args:
        M: number of players.

    Returns:
        coalitions: (2^M, M) int array.
        weights:    (2^M,) float array.
    """
    coalitions = np.array(
        list(itertools.product([0, 1], repeat=M)), dtype=int
    )  # (2^M, M)
    weights = np.array(
        [_shapley_kernel_weight(int(c.sum()), M) for c in coalitions],
        dtype=float,
    )
    return coalitions, weights


def _solve_shapley_wls(
    coalitions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted least squares to estimate Shapley values.

    Fits the linear model f(z) ≈ φ_0 + Σ φⱼ zⱼ via weighted OLS.
    Rows with weight=0 (all-zeros/all-ones coalitions) contribute nothing.

    Args:
        coalitions: (N, M) binary coalition matrix.
        values:     (N,) function values for each coalition.
        weights:    (N,) Shapley kernel weights.

    Returns:
        (M,) array of Shapley values φ_1, …, φ_M (intercept excluded).
    """
    M = coalitions.shape[1]
    keep = weights > 0
    Z = np.column_stack([np.ones(keep.sum()), coalitions[keep]])  # (K, M+1)
    w = np.sqrt(weights[keep])[:, None]
    Zw = Z * w
    fw = values[keep] * np.sqrt(weights[keep])

    A = Zw.T @ Zw + 1e-8 * np.eye(M + 1)
    b_vec = Zw.T @ fw
    theta = np.linalg.solve(A, b_vec)
    return theta[1:]  # exclude intercept


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

@torch.no_grad()
def value_fn(
    model: Any,                      # ActorSHAP
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    coalition_mask: torch.Tensor,
    n_samples: int = 20,
) -> float:
    """Evaluate the expected classifier output on manifold-constrained completions.

    Args:
        model:          ActorSHAP instance with sample_completions method.
        classifier:     callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1, n_classes).
        x:              (1, J, F, T) input sequence.
        y:              (1,) UPDRS label (int64).
        mask:           (1, T) real-frame mask.
        lengths:        (1,) frame count.
        coalition_mask: (1, J) or (1, T) — True = observed.
        n_samples:      number of stochastic completions to average over.

    Returns:
        float — mean predicted probability for the ground-truth class.
    """
    completions = model.sample_completions(
        x, y, mask, lengths, coalition_mask, n_samples=n_samples
    )
    class_idx = int(y[0].item())
    probs = []
    for x_hat in completions:
        logits = classifier(x_hat)
        p = float(torch.softmax(logits, dim=-1)[0, class_idx].item())
        probs.append(p)
    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Baseline value functions (Frye et al. 2021 comparison set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def value_fn_zero(
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    coalition_mask: torch.Tensor,
) -> float:
    """Value function with zero-baseline replacement (off-manifold).

    Masked joints are set to 0 in every frame. For H36M skeletons centred on the
    pelvis this places masked joints at the origin, which is off-manifold for all
    non-pelvis joints.  This is the standard baseline in Shapley papers
    (Lundberg & Lee 2017) and the primary comparison in Frye et al. 2021.

    Args:
        classifier:     callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1,n_classes).
        x:              (1, J, F, T) input sequence.
        y:              (1,) UPDRS label.
        coalition_mask: (1, J) — True = observed.

    Returns:
        float — predicted probability for the ground-truth class.
    """
    x_perturbed = x.clone()
    masked_joints = (~coalition_mask[0]).nonzero(as_tuple=True)[0]
    x_perturbed[0, masked_joints] = 0.0
    class_idx = int(y[0].item())
    logits = classifier(x_perturbed)
    return float(torch.softmax(logits, dim=-1)[0, class_idx].item())


@torch.no_grad()
def value_fn_mean(
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    coalition_mask: torch.Tensor,
    joint_means: torch.Tensor,
) -> float:
    """Value function with mean-baseline replacement.

    Masked joints are replaced with their per-joint training-set mean position,
    broadcast across all T frames.  Closer to the data manifold than zero baseline
    because the mean position reflects typical joint locations, but still ignores
    the correlation between joints in the current sequence.

    joint_means is computed once over the training set:
        joint_means[j, f] = mean(x_train[:, j, f, :])   shape (J, F)

    Args:
        classifier:     callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1,n_classes).
        x:              (1, J, F, T) input sequence.
        y:              (1,) UPDRS label.
        coalition_mask: (1, J) — True = observed.
        joint_means:    (J, F) training-set mean per joint and coordinate.

    Returns:
        float — predicted probability for the ground-truth class.
    """
    x_perturbed = x.clone()
    T = x.shape[-1]
    masked_joints = (~coalition_mask[0]).nonzero(as_tuple=True)[0]
    # joint_means: (J, F) → (J, F, 1) → broadcast to (J, F, T)
    mean_expanded = joint_means[masked_joints].unsqueeze(-1).expand(-1, -1, T)
    x_perturbed[0, masked_joints] = mean_expanded.to(x.device)
    class_idx = int(y[0].item())
    logits = classifier(x_perturbed)
    return float(torch.softmax(logits, dim=-1)[0, class_idx].item())


@torch.no_grad()
def value_fn_marginal(
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    coalition_mask: torch.Tensor,
    train_pool: torch.Tensor,
    rng: np.random.Generator | None = None,
    n_samples: int = 20,
) -> float:
    """Value function with marginal-distribution replacement.

    Masked joints are filled by sampling their positions independently from the
    training distribution.  Concretely: draw n_samples random training sequences,
    copy masked joint tracks from each, run the classifier, and average.  This
    approximates the marginal expectation E_{x_S ~ p(x_S)}[f(x_obs, x_S)] and
    is the standard "marginal SHAP" baseline in Frye et al. 2021.

    NOTE: marginal sampling ignores correlation between observed and masked joints,
    so the resulting sequences are also off-manifold — but less so than zero/mean
    because the marginal positions are drawn from realistic joint trajectories.

    Args:
        classifier:   callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1,n_classes).
        x:            (1, J, F, T) input sequence.
        y:            (1,) UPDRS label.
        coalition_mask: (1, J) — True = observed.
        train_pool:   (N_train, J, F, T) tensor of training sequences (or a large
                      random subset).  Kept on CPU; moved to x.device per sample.
        rng:          optional numpy random generator.
        n_samples:    training sequences drawn to approximate the expectation.

    Returns:
        float — mean predicted probability for the ground-truth class.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = train_pool.shape[0]
    indices = rng.integers(0, N, size=n_samples)
    masked_joints = (~coalition_mask[0]).nonzero(as_tuple=True)[0]
    class_idx = int(y[0].item())
    probs = []
    for idx in indices:
        x_perturbed = x.clone()
        donor = train_pool[int(idx)].to(x.device)  # (J, F, T)
        x_perturbed[0, masked_joints] = donor[masked_joints]
        logits = classifier(x_perturbed)
        p = float(torch.softmax(logits, dim=-1)[0, class_idx].item())
        probs.append(p)
    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Spatial SHAP — 17 individual joints
# ---------------------------------------------------------------------------

def compute_spatial_shap(
    model: Any,
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    n_kernel_samples: int = 3000,
    n_completion_samples: int = 20,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute per-joint Shapley values via KernelSHAP.

    Players are 17 individual H36M joints.  Post-hoc aggregation into
    anatomical groups is provided by summing joint Shapley values within each
    group (H36M_GROUPS).  This preserves the Shapley axioms at the joint level
    while enabling clinical group-level summaries.

    Args:
        model:               ActorSHAP.
        classifier:          callable(x_hat: Tensor) → logits Tensor(1, n_classes).
        x:                   (1, J, F, T) single sequence (B=1).
        y:                   (1,) class label.
        mask:                (1, T) real-frame mask.
        lengths:             (1,) frame count.
        n_kernel_samples:    number of coalition PAIRS sampled from kernel distribution
                             (total coalitions evaluated = 2 * n_kernel_samples).
        n_completion_samples: stochastic completions averaged per coalition.
        seed:                optional random seed for reproducibility.

    Returns:
        dict with keys:
          "{JointName}": individual Shapley value for each of the 17 joints.
          "{group_name}": sum of joint Shapley values for each anatomical group.
    """
    M = 17
    device = x.device
    rng = np.random.default_rng(seed)

    coalitions, weights = _sample_kernel_coalitions(M, n_kernel_samples, rng)

    values = np.zeros(len(coalitions))
    for i, z in enumerate(coalitions):
        observed = np.where(z == 1)[0].tolist()
        if len(observed) == 0:
            # All-masked: replace all joints; value is model output on pure sample.
            observed = []
        cm = build_spatial_shap_mask(observed, device, n_joints=M).unsqueeze(0)
        values[i] = value_fn(model, classifier, x, y, mask, lengths, cm,
                             n_samples=n_completion_samples)

    phi = _solve_shapley_wls(coalitions, values, weights)

    result: dict[str, float] = {
        H36M_JOINT_NAMES[j]: float(phi[j]) for j in range(M)
    }
    for group_name, joint_indices in H36M_GROUPS.items():
        result[group_name] = float(sum(phi[j] for j in joint_indices))

    return result


# ---------------------------------------------------------------------------
# Temporal SHAP — K=4 stride-aligned windows
# ---------------------------------------------------------------------------

def compute_temporal_shap(
    model: Any,
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    window_assignments: list[list[int]] | None = None,
    n_completion_samples: int = 20,
    fps: int = 30,
) -> dict[str, float]:
    """Compute per-window Shapley values via exact enumeration.

    With K=4 windows there are only 2^4=16 coalitions; exact Shapley values
    are computed without approximation.

    Args:
        model:               ActorSHAP.
        classifier:          callable(x_hat: Tensor) → logits Tensor(1, n_classes).
        x:                   (1, J, F, T) single sequence (B=1).
        y:                   (1,) class label.
        mask:                (1, T) real-frame mask.
        lengths:             (1,) frame count.
        window_assignments:  optional pre-computed window frame lists from
                             build_temporal_windows.  If None, stride period is
                             auto-detected from x.
        n_completion_samples: stochastic completions averaged per coalition.
        fps:                 capture frame rate (passed to detect_stride_period).

    Returns:
        dict {"window_0": φ, "window_1": φ, "window_2": φ, "window_3": φ}.
        Window names are replaced with gait-phase labels when stride detection
        succeeds (no fallback).
    """
    K = 4
    T = x.shape[-1]
    device = x.device
    PHASE_LABELS = ["IC_loading", "midstance_terminal", "preswing_initswing", "midswing_terminal"]

    if window_assignments is None:
        # x is (1, J, F, T); convert to (T, J, 3) for stride detection.
        x_np = x[0].permute(2, 0, 1).cpu().numpy()  # (T, J, F)
        stride_period, fallback = detect_stride_period(x_np, fps=fps)
        window_assignments = build_temporal_windows(T, stride_period, K=K)
        window_names = (
            [f"window_{k}" for k in range(K)] if fallback else PHASE_LABELS
        )
    else:
        window_names = PHASE_LABELS

    coalitions, weights = _enumerate_all_coalitions(K)
    values = np.zeros(len(coalitions))

    for i, z in enumerate(coalitions):
        observed = [k for k in range(K) if z[k] == 1]
        cm = build_temporal_shap_mask(observed, window_assignments, T, device).unsqueeze(0)
        values[i] = value_fn(model, classifier, x, y, mask, lengths, cm,
                             n_samples=n_completion_samples)

    phi = _solve_shapley_wls(coalitions, values, weights)
    return {window_names[k]: float(phi[k]) for k in range(K)}


# ---------------------------------------------------------------------------
# Baseline compute wrappers
# ---------------------------------------------------------------------------

def compute_spatial_shap_baseline(
    method: str,
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    joint_means: torch.Tensor | None = None,
    train_pool: torch.Tensor | None = None,
    n_kernel_samples: int = 3000,
    n_marginal_samples: int = 20,
    seed: int | None = None,
) -> dict[str, float]:
    """KernelSHAP with a value-function baseline (not ActorSHAP).

    Identical coalition sampling and WLS regression as compute_spatial_shap.
    Only the replacement strategy for masked joints changes.

    Args:
        method:          "zero" | "mean" | "marginal".
        classifier:      callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1,n_classes).
        x:               (1, J, F, T).
        y:               (1,) label.
        mask:            (1, T) real-frame mask (not used for replacement; kept for
                         interface parity with compute_spatial_shap).
        lengths:         (1,) frame count.
        joint_means:     (J, F) training mean — required for method="mean".
        train_pool:      (N, J, F, T) training sequences — required for method="marginal".
        n_kernel_samples: number of coalition pairs sampled.
        n_marginal_samples: draws per coalition for method="marginal".
        seed:            random seed.

    Returns:
        dict — same structure as compute_spatial_shap: individual joint keys + group keys.
    """
    if method not in {"zero", "mean", "marginal"}:
        raise ValueError(f"method must be 'zero', 'mean', or 'marginal'; got {method!r}")
    if method == "mean" and joint_means is None:
        raise ValueError("joint_means is required for method='mean'")
    if method == "marginal" and train_pool is None:
        raise ValueError("train_pool is required for method='marginal'")

    M = 17
    device = x.device
    rng = np.random.default_rng(seed)

    coalitions, weights = _sample_kernel_coalitions(M, n_kernel_samples, rng)
    values = np.zeros(len(coalitions))

    for i, z in enumerate(coalitions):
        cm = torch.tensor(z, dtype=torch.bool, device=device).unsqueeze(0)  # (1, M)
        if method == "zero":
            values[i] = value_fn_zero(classifier, x, y, cm)
        elif method == "mean":
            values[i] = value_fn_mean(classifier, x, y, cm, joint_means)
        else:
            values[i] = value_fn_marginal(classifier, x, y, cm, train_pool, rng,
                                           n_samples=n_marginal_samples)

    phi = _solve_shapley_wls(coalitions, values, weights)
    result: dict[str, float] = {H36M_JOINT_NAMES[j]: float(phi[j]) for j in range(M)}
    for group_name, joint_indices in H36M_GROUPS.items():
        result[group_name] = float(sum(phi[j] for j in joint_indices))
    return result


def compute_temporal_shap_baseline(
    method: str,
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    window_assignments: list[list[int]] | None = None,
    joint_means: torch.Tensor | None = None,
    train_pool: torch.Tensor | None = None,
    n_marginal_samples: int = 20,
    fps: int = 30,
    seed: int | None = None,
) -> dict[str, float]:
    """Exact temporal SHAP (K=4 windows) with a value-function baseline.

    Mirrors compute_temporal_shap but uses zero/mean/marginal replacement instead
    of ActorSHAP manifold-constrained completions.

    For temporal masking, entire frame ranges (windows) are masked.  The replacement
    fills *all joints* in masked frames, not individual joints.

    Args:
        method:          "zero" | "mean" | "marginal".
        classifier, x, y, mask, lengths, window_assignments, fps: same as
                         compute_temporal_shap.
        joint_means:     (J, F) training mean (method="mean").
        train_pool:      (N, J, F, T) training sequences (method="marginal").
        n_marginal_samples: draws per coalition for method="marginal".
        seed:            random seed for method="marginal".

    Returns:
        {window_name: Shapley value} for K=4 windows.
    """
    if method not in {"zero", "mean", "marginal"}:
        raise ValueError(f"method must be 'zero', 'mean', or 'marginal'; got {method!r}")

    K = 4
    T = x.shape[-1]
    device = x.device
    rng = np.random.default_rng(seed)
    PHASE_LABELS = ["IC_loading", "midstance_terminal", "preswing_initswing", "midswing_terminal"]

    if window_assignments is None:
        x_np = x[0].permute(2, 0, 1).cpu().numpy()
        stride_period, fallback = detect_stride_period(x_np, fps=fps)
        window_assignments = build_temporal_windows(T, stride_period, K=K)
        window_names = [f"window_{k}" for k in range(K)] if fallback else PHASE_LABELS
    else:
        window_names = PHASE_LABELS

    coalitions, weights = _enumerate_all_coalitions(K)
    values = np.zeros(len(coalitions))

    for i, z in enumerate(coalitions):
        observed_windows = [k for k in range(K) if z[k] == 1]
        # Build a (J,) spatial mask where all joints are "observed" in observed windows
        # and "masked" in the remaining windows.  Then apply replacement per frame.
        x_perturbed = x.clone()
        masked_frames = [
            t for k in range(K) if k not in observed_windows
            for t in window_assignments[k]
        ]
        if masked_frames:
            mf = torch.tensor(masked_frames, dtype=torch.long)
            if method == "zero":
                x_perturbed[0, :, :, mf] = 0.0
            elif method == "mean":
                # joint_means: (J, F) → (J, F, 1) → broadcast to (J, F, n_frames)
                mean_exp = joint_means.unsqueeze(-1).expand(-1, -1, len(mf)).to(device)
                x_perturbed[0, :, :, mf] = mean_exp
            else:
                # marginal: sample a random training sequence, copy its frames
                donors = [int(rng.integers(0, train_pool.shape[0]))
                          for _ in range(n_marginal_samples)]
                samples_p = []
                for d in donors:
                    xp = x.clone()
                    xp[0, :, :, mf] = train_pool[d, :, :, mf].to(device)
                    logits = classifier(xp)
                    class_idx = int(y[0].item())
                    samples_p.append(float(torch.softmax(logits, dim=-1)[0, class_idx]))
                values[i] = float(np.mean(samples_p))
                continue  # marginal handles its own averaging

        logits = classifier(x_perturbed)
        class_idx = int(y[0].item())
        values[i] = float(torch.softmax(logits, dim=-1)[0, class_idx].item())

    phi = _solve_shapley_wls(coalitions, values, weights)
    return {window_names[k]: float(phi[k]) for k in range(K)}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    B, J, F, T = 1, 17, 3, 16

    class _MockModel:
        def sample_completions(self, x, y, mask, lengths, coalition_mask,
                               n_samples=20, paste_observed=True):
            return [torch.randn_like(x) for _ in range(n_samples)]

    def _mock_classifier(x_hat):
        return torch.randn(1, 3)

    model = _MockModel()
    x = torch.randn(B, J, F, T)
    y = torch.zeros(B, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.bool)
    lengths = torch.full((B,), T, dtype=torch.long)

    # ActorSHAP spatial SHAP.
    s_result = compute_spatial_shap(
        model, _mock_classifier, x, y, mask, lengths,
        n_kernel_samples=10, n_completion_samples=2, seed=42,
    )
    assert set(s_result.keys()) == set(H36M_JOINT_NAMES) | set(H36M_GROUPS.keys())
    print(f"Spatial SHAP OK — {len(s_result)} keys")

    # ActorSHAP temporal SHAP.
    windows = build_temporal_windows(T, stride_period=8, K=4)
    t_result = compute_temporal_shap(
        model, _mock_classifier, x, y, mask, lengths,
        window_assignments=windows, n_completion_samples=2,
    )
    assert len(t_result) == 4
    print(f"Temporal SHAP OK — {list(t_result.keys())}")

    # Zero-baseline spatial SHAP.
    z_result = compute_spatial_shap_baseline(
        "zero", _mock_classifier, x, y, mask, lengths,
        n_kernel_samples=10, seed=0,
    )
    assert set(z_result.keys()) == set(H36M_JOINT_NAMES) | set(H36M_GROUPS.keys())
    print(f"Zero-baseline spatial SHAP OK — {len(z_result)} keys")

    # Mean-baseline spatial SHAP.
    joint_means = torch.randn(J, F)
    m_result = compute_spatial_shap_baseline(
        "mean", _mock_classifier, x, y, mask, lengths,
        joint_means=joint_means, n_kernel_samples=10, seed=0,
    )
    assert set(m_result.keys()) == set(H36M_JOINT_NAMES) | set(H36M_GROUPS.keys())
    print(f"Mean-baseline spatial SHAP OK")

    # Marginal-baseline spatial SHAP.
    train_pool = torch.randn(8, J, F, T)
    mg_result = compute_spatial_shap_baseline(
        "marginal", _mock_classifier, x, y, mask, lengths,
        train_pool=train_pool, n_kernel_samples=4, n_marginal_samples=2, seed=0,
    )
    assert set(mg_result.keys()) == set(H36M_JOINT_NAMES) | set(H36M_GROUPS.keys())
    print(f"Marginal-baseline spatial SHAP OK")

    # Zero-baseline temporal SHAP.
    zt_result = compute_temporal_shap_baseline(
        "zero", _mock_classifier, x, y, mask, lengths,
        window_assignments=windows,
    )
    assert len(zt_result) == 4
    print(f"Zero-baseline temporal SHAP OK — {list(zt_result.keys())}")

    # Mean-baseline temporal SHAP.
    mt_result = compute_temporal_shap_baseline(
        "mean", _mock_classifier, x, y, mask, lengths,
        window_assignments=windows, joint_means=joint_means,
    )
    assert len(mt_result) == 4
    print(f"Mean-baseline temporal SHAP OK")

    # Marginal-baseline temporal SHAP.
    mgt_result = compute_temporal_shap_baseline(
        "marginal", _mock_classifier, x, y, mask, lengths,
        window_assignments=windows, train_pool=train_pool,
        n_marginal_samples=2, seed=0,
    )
    assert len(mgt_result) == 4
    print(f"Marginal-baseline temporal SHAP OK")

    # Kernel weight sanity.
    for M in [4, 17]:
        for s in range(1, M):
            assert _shapley_kernel_weight(s, M) > 0

    print("OK")
