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
    v_empty: float | None = None,
    v_full: float | None = None,
) -> np.ndarray:
    """Weighted least squares to estimate Shapley values.

    Fits the linear model f(z) ≈ φ_0 + Σ φⱼ zⱼ via weighted OLS.
    Rows with weight=0 (all-zeros/all-ones coalitions) contribute nothing.

    If v_empty and v_full are provided, they are appended as high-weight
    constraint rows (weight=1e6) to enforce the Shapley efficiency property
    Σφⱼ ≈ v(full) − v(∅), following the constrained-WLS formulation of
    Lundberg & Lee (2017).  Without these constraints the unconstrained WLS
    only approximately satisfies efficiency (measured as completeness_error).

    Args:
        coalitions: (N, M) binary coalition matrix.
        values:     (N,) function values for each coalition.
        weights:    (N,) Shapley kernel weights.
        v_empty:    value function on the all-masked coalition v(∅).
        v_full:     value function on the all-observed coalition v(1).

    Returns:
        (M,) array of Shapley values φ_1, …, φ_M (intercept excluded).
    """
    M = coalitions.shape[1]

    if v_empty is not None and v_full is not None:
        # Append boundary coalitions with large weights as soft constraints.
        # Weight 1e6 dominates interior kernel weights (~O(1/M)), enforcing
        # φ_0 ≈ v_empty  and  φ_0 + Σφⱼ ≈ v_full  ⟹  Σφⱼ ≈ v_full − v_empty.
        _BIG = 1e6
        boundary_z = np.array([[0] * M, [1] * M], dtype=int)
        boundary_v = np.array([v_empty, v_full])
        boundary_w = np.full(2, _BIG)
        coalitions = np.vstack([coalitions, boundary_z])
        values     = np.concatenate([values, boundary_v])
        weights    = np.concatenate([weights, boundary_w])

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
# Batched classifier helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batch_classify(
    classifier: Callable,
    x_list: list[torch.Tensor],
    class_idx: int,
    chunk_size: int = 256,
) -> np.ndarray:
    """Classify a list of (1, J, F, T) tensors in a small number of batched GPU calls.

    Splits into chunks of ``chunk_size`` to bound peak GPU memory.  Returns a
    1-D float array of length ``len(x_list)`` with the predicted probability
    for ``class_idx`` for each input.
    """
    if not x_list:
        return np.array([], dtype=np.float32)
    probs: list[np.ndarray] = []
    for start in range(0, len(x_list), chunk_size):
        batch = torch.cat(x_list[start : start + chunk_size], dim=0)
        logits = classifier(batch)
        p = torch.softmax(logits, dim=-1)[:, class_idx].cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


@torch.no_grad()
def _classify_chunked(
    classifier: Callable,
    x_batch: torch.Tensor,  # (N, J, F, T) already on device
    class_idx: int,
    chunk_size: int = 256,
) -> np.ndarray:
    """Classify a pre-stacked (N, J, F, T) tensor in chunks.

    Unlike ``_batch_classify``, takes a single contiguous tensor instead of a
    list — avoids the overhead of ``torch.cat`` on every chunk and is the
    preferred path for vectorised baseline evaluation.
    """
    N = x_batch.shape[0]
    if N == 0:
        return np.array([], dtype=np.float32)
    probs: list[np.ndarray] = []
    for start in range(0, N, chunk_size):
        chunk = x_batch[start : start + chunk_size]
        logits = classifier(chunk)
        p = torch.softmax(logits, dim=-1)[:, class_idx].cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


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
    probs = _batch_classify(classifier, completions, class_idx)
    return float(probs.mean())


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
    batch_chunk_size: int = 256,
) -> dict[str, float]:
    """Compute per-joint Shapley values via KernelSHAP.

    Players are 17 individual H36M joints.  Post-hoc aggregation into
    anatomical groups is provided by summing joint Shapley values within each
    group (H36M_GROUPS).  This preserves the Shapley axioms at the joint level
    while enabling clinical group-level summaries.

    All classifier calls across all coalitions are batched into a small number
    of GPU forward passes (controlled by ``batch_chunk_size``) rather than one
    call per coalition.  VAEAC completion calls are still sequential per
    coalition; batching the classifier typically gives ~10–50× speedup.

    Args:
        model:               ActorSHAP.
        classifier:          callable(x_hat: Tensor(B,J,F,T)) → logits Tensor(B,n_classes).
                             Must support batch size > 1.
        x:                   (1, J, F, T) single sequence (B=1).
        y:                   (1,) class label.
        mask:                (1, T) real-frame mask.
        lengths:             (1,) frame count.
        n_kernel_samples:    number of coalition PAIRS sampled from kernel distribution
                             (total coalitions evaluated = 2 * n_kernel_samples).
        n_completion_samples: stochastic completions averaged per coalition.
        seed:                optional random seed for reproducibility.
        batch_chunk_size:    max sequences per batched classifier forward pass.

    Returns:
        dict with keys:
          "{JointName}": individual Shapley value for each of the 17 joints.
          "{group_name}": sum of joint Shapley values for each anatomical group.
    """
    M = 17
    device = x.device
    rng = np.random.default_rng(seed)
    class_idx = int(y[0].item())

    coalitions, weights = _sample_kernel_coalitions(M, n_kernel_samples, rng)

    # Build all coalition masks including boundaries (empty=index 0, full=index 1).
    zero_cm = build_spatial_shap_mask([], device, n_joints=M).unsqueeze(0)
    full_cm = build_spatial_shap_mask(list(range(M)), device, n_joints=M).unsqueeze(0)
    all_cms: list[torch.Tensor] = [zero_cm, full_cm] + [
        build_spatial_shap_mask(np.where(z == 1)[0].tolist(), device, n_joints=M).unsqueeze(0)
        for z in coalitions
    ]

    # Generate completions for every coalition (VAEAC calls sequential per coalition),
    # then classify all completions in a single batched pass.
    all_completions: list[torch.Tensor] = []
    slice_ends: list[int] = []
    for cm in all_cms:
        comps = model.sample_completions(
            x, y, mask, lengths, cm, n_samples=n_completion_samples
        )
        all_completions.extend(comps)
        slice_ends.append(len(all_completions))

    all_probs = _batch_classify(classifier, all_completions, class_idx,
                                chunk_size=batch_chunk_size)

    # Average probabilities within each coalition's completion block.
    starts = [0] + slice_ends[:-1]
    all_values = np.array([
        float(all_probs[s:e].mean())
        for s, e in zip(starts, slice_ends)
    ])

    v_empty = all_values[0]
    v_full  = all_values[1]
    values  = all_values[2:]

    phi = _solve_shapley_wls(coalitions, values, weights, v_empty=v_empty, v_full=v_full)

    result: dict[str, float] = {
        H36M_JOINT_NAMES[j]: float(phi[j]) for j in range(M)
    }
    for group_name, joint_indices in H36M_GROUPS.items():
        result[group_name] = float(sum(phi[j] for j in joint_indices))
    result["_v_empty"] = float(v_empty)
    result["_v_full"]  = float(v_full)
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
    class_idx = int(y[0].item())

    # Build all coalition masks and generate completions, then batch-classify.
    all_cms = [
        build_temporal_shap_mask(
            [k for k in range(K) if z[k] == 1], window_assignments, T, device
        ).unsqueeze(0)
        for z in coalitions
    ]
    all_completions: list[torch.Tensor] = []
    slice_ends: list[int] = []
    for cm in all_cms:
        comps = model.sample_completions(
            x, y, mask, lengths, cm, n_samples=n_completion_samples
        )
        all_completions.extend(comps)
        slice_ends.append(len(all_completions))

    all_probs = _batch_classify(classifier, all_completions, class_idx)
    starts = [0] + slice_ends[:-1]
    values = np.array([
        float(all_probs[s:e].mean()) for s, e in zip(starts, slice_ends)
    ])

    # itertools.product([0,1], K) produces all-zeros first and all-ones last.
    v_empty = float(values[0])
    v_full  = float(values[-1])

    phi = _solve_shapley_wls(coalitions, values, weights, v_empty=v_empty, v_full=v_full)
    result = {window_names[k]: float(phi[k]) for k in range(K)}
    result["_v_empty"] = v_empty
    result["_v_full"]  = v_full
    return result


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
    J, F, T = x.shape[1], x.shape[2], x.shape[3]
    device = x.device
    rng = np.random.default_rng(seed)
    class_idx = int(y[0].item())

    coalitions, weights = _sample_kernel_coalitions(M, n_kernel_samples, rng)

    # Stack all coalition masks into a single (N_total, J) tensor — avoids
    # per-coalition Python overhead in the vectorised masking below.
    cms_np = np.vstack([
        np.zeros((1, M), dtype=bool),
        np.ones((1, M),  dtype=bool),
        np.array(coalitions, dtype=bool),   # (2*n_kernel_samples, M)
    ])  # (N_total, M)
    N_total = cms_np.shape[0]
    all_cms_t = torch.tensor(cms_np, dtype=torch.bool, device=device)  # (N_total, J)

    # mask_b: (N_total, J, 1, 1) — True = observed (kept), False = masked (replaced)
    mask_b = all_cms_t[:, :, None, None]  # broadcasts over F and T

    if method == "zero":
        # Vectorised: zero out masked joints for every coalition at once.
        # x.expand() is a zero-copy view; torch.where allocates the output once.
        x_batch = torch.where(
            mask_b,
            x.expand(N_total, -1, -1, -1),
            torch.zeros(1, J, F, T, dtype=x.dtype, device=device).expand(N_total, -1, -1, -1),
        )  # (N_total, J, F, T)
        all_values = _classify_chunked(classifier, x_batch, class_idx)

    elif method == "mean":
        mean_fill = joint_means.unsqueeze(-1).expand(-1, -1, T)   # (J, F, T)
        x_batch = torch.where(
            mask_b,
            x.expand(N_total, -1, -1, -1),
            mean_fill.unsqueeze(0).expand(N_total, -1, -1, -1),
        )  # (N_total, J, F, T)
        all_values = _classify_chunked(classifier, x_batch, class_idx)

    else:  # marginal
        # Pre-sample all donor indices upfront: (N_total, n_marginal_samples).
        # Process coalitions in outer chunks so the per-chunk donor tensor fits
        # in GPU memory comfortably.
        N_pool    = train_pool.shape[0]
        all_idx   = rng.integers(0, N_pool, size=(N_total, n_marginal_samples))
        COAL_CHUNK = 64  # coalitions per outer iteration

        all_values_list: list[np.ndarray] = []
        x0 = x[0]  # (J, F, T)

        for c0 in range(0, N_total, COAL_CHUNK):
            c1 = min(c0 + COAL_CHUNK, N_total)
            C  = c1 - c0

            # One bulk CPU→GPU transfer per outer chunk instead of per-sample.
            donors = train_pool[all_idx[c0:c1]].to(device)  # (C, n_marg, J, F, T)

            # Vectorised masking across all C coalitions × n_marg donors.
            # cm_b: (C, 1, J, 1, 1) — broadcasts over (C, n_marg, J, F, T)
            cm_b = all_cms_t[c0:c1][:, None, :, None, None]
            x_out = torch.where(
                cm_b,                            # (C, 1, J, 1, 1) → (C, n_marg, J, F, T)
                x0[None, None, :, :, :],         # (1, 1, J, F, T) → (C, n_marg, J, F, T)
                donors,                           # (C, n_marg, J, F, T)
            )  # (C, n_marg, J, F, T)
            x_flat = x_out.reshape(C * n_marginal_samples, J, F, T)

            probs_c = _classify_chunked(classifier, x_flat, class_idx)  # (C*n_marg,)
            per_coal = probs_c.reshape(C, n_marginal_samples).mean(axis=1)  # (C,)
            all_values_list.append(per_coal)

        all_values = np.concatenate(all_values_list)  # (N_total,)

    v_empty = float(all_values[0])
    v_full  = float(all_values[1])
    values  = np.array(all_values[2:], dtype=np.float64)

    phi = _solve_shapley_wls(coalitions, values, weights, v_empty=v_empty, v_full=v_full)
    result: dict[str, float] = {H36M_JOINT_NAMES[j]: float(phi[j]) for j in range(M)}
    for group_name, joint_indices in H36M_GROUPS.items():
        result[group_name] = float(sum(phi[j] for j in joint_indices))
    result["_v_empty"] = float(v_empty)
    result["_v_full"]  = float(v_full)
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
    class_idx = int(y[0].item())

    # Build all perturbed sequences for every coalition, then batch-classify.
    if method in ("zero", "mean"):
        x_batch_list: list[torch.Tensor] = []
        for z in coalitions:
            observed_windows = [k for k in range(K) if z[k] == 1]
            masked_frames = [
                t for k in range(K) if k not in observed_windows
                for t in window_assignments[k]
            ]
            x_p = x.clone()
            if masked_frames:
                mf = torch.tensor(masked_frames, dtype=torch.long)
                if method == "zero":
                    x_p[0, :, :, mf] = 0.0
                else:
                    mean_exp = joint_means.unsqueeze(-1).expand(-1, -1, len(mf)).to(device)
                    x_p[0, :, :, mf] = mean_exp
            x_batch_list.append(x_p)
        all_probs = _batch_classify(classifier, x_batch_list, class_idx)
        values = all_probs

    else:  # marginal
        N_pool = train_pool.shape[0]
        all_coal_values: list[float] = []

        for z in coalitions:
            observed_windows = [k for k in range(K) if z[k] == 1]
            masked_frames = [
                t for k in range(K) if k not in observed_windows
                for t in window_assignments[k]
            ]
            donor_idx = rng.integers(0, N_pool, size=n_marginal_samples)
            # Bulk transfer donors for this coalition (n_marg, J, F, T).
            donors = train_pool[donor_idx].to(device)   # (n_marg, J, F, T)
            if masked_frames:
                mf = torch.tensor(masked_frames, dtype=torch.long, device=device)
                # x0: (J, F, n_frames) — observed frames stay; masked frames come from donors
                x0 = x[0].clone()                       # (J, F, T)
                x_rep = x0.unsqueeze(0).expand(n_marginal_samples, -1, -1, -1).clone()
                x_rep[:, :, :, mf] = donors[:, :, :, mf]
            else:
                x_rep = x[0].unsqueeze(0).expand(n_marginal_samples, -1, -1, -1)
            probs = _classify_chunked(classifier, x_rep, class_idx)
            all_coal_values.append(float(probs.mean()))

        values = np.array(all_coal_values)

    # itertools.product([0,1], K) produces all-zeros first and all-ones last.
    v_empty = float(values[0])
    v_full  = float(values[-1])

    phi = _solve_shapley_wls(coalitions, values, weights, v_empty=v_empty, v_full=v_full)
    result = {window_names[k]: float(phi[k]) for k in range(K)}
    result["_v_empty"] = v_empty
    result["_v_full"]  = v_full
    return result


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
        return torch.randn(x_hat.shape[0], 3)

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
    _EXPECTED_SPATIAL = set(H36M_JOINT_NAMES) | set(H36M_GROUPS.keys()) | {"_v_empty", "_v_full"}
    assert set(s_result.keys()) == _EXPECTED_SPATIAL
    assert "_v_empty" in s_result and "_v_full" in s_result
    print(f"Spatial SHAP OK — {len(s_result)} keys (includes _v_empty/_v_full)")

    # ActorSHAP temporal SHAP.
    windows = build_temporal_windows(T, stride_period=8, K=4)
    t_result = compute_temporal_shap(
        model, _mock_classifier, x, y, mask, lengths,
        window_assignments=windows, n_completion_samples=2,
    )
    assert len(t_result) == 6  # 4 windows + _v_empty + _v_full
    assert "_v_empty" in t_result and "_v_full" in t_result
    print(f"Temporal SHAP OK — {list(t_result.keys())}")

    # Zero-baseline spatial SHAP.
    z_result = compute_spatial_shap_baseline(
        "zero", _mock_classifier, x, y, mask, lengths,
        n_kernel_samples=10, seed=0,
    )
    assert set(z_result.keys()) == _EXPECTED_SPATIAL
    print(f"Zero-baseline spatial SHAP OK — {len(z_result)} keys")

    # Mean-baseline spatial SHAP.
    joint_means = torch.randn(J, F)
    m_result = compute_spatial_shap_baseline(
        "mean", _mock_classifier, x, y, mask, lengths,
        joint_means=joint_means, n_kernel_samples=10, seed=0,
    )
    assert set(m_result.keys()) == _EXPECTED_SPATIAL
    print(f"Mean-baseline spatial SHAP OK")

    # Marginal-baseline spatial SHAP.
    train_pool = torch.randn(8, J, F, T)
    mg_result = compute_spatial_shap_baseline(
        "marginal", _mock_classifier, x, y, mask, lengths,
        train_pool=train_pool, n_kernel_samples=4, n_marginal_samples=2, seed=0,
    )
    assert set(mg_result.keys()) == _EXPECTED_SPATIAL
    print(f"Marginal-baseline spatial SHAP OK")

    _EXPECTED_TEMPORAL = 6  # 4 windows + _v_empty + _v_full

    # Zero-baseline temporal SHAP.
    zt_result = compute_temporal_shap_baseline(
        "zero", _mock_classifier, x, y, mask, lengths,
        window_assignments=windows,
    )
    assert len(zt_result) == _EXPECTED_TEMPORAL
    assert "_v_empty" in zt_result and "_v_full" in zt_result
    print(f"Zero-baseline temporal SHAP OK — {list(zt_result.keys())}")

    # Mean-baseline temporal SHAP.
    mt_result = compute_temporal_shap_baseline(
        "mean", _mock_classifier, x, y, mask, lengths,
        window_assignments=windows, joint_means=joint_means,
    )
    assert len(mt_result) == _EXPECTED_TEMPORAL
    print(f"Mean-baseline temporal SHAP OK")

    # Marginal-baseline temporal SHAP.
    mgt_result = compute_temporal_shap_baseline(
        "marginal", _mock_classifier, x, y, mask, lengths,
        window_assignments=windows, train_pool=train_pool,
        n_marginal_samples=2, seed=0,
    )
    assert len(mgt_result) == _EXPECTED_TEMPORAL
    print(f"Marginal-baseline temporal SHAP OK")

    # Kernel weight sanity.
    for M in [4, 17]:
        for s in range(1, M):
            assert _shapley_kernel_weight(s, M) > 0

    print("OK")
