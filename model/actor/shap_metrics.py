"""shap_metrics.py — Evaluation metrics for ActorSHAP generative quality and SHAP faithfulness.

Two tiers of metrics:

1. Generative quality — validates that VAEAC completions stay on the data manifold.
   Without manifold validity the SHAP faithfulness metrics below are meaningless.
   - masked_rmse:            imputation RMSE on masked positions.
   - compute_fid:            Fréchet distance between real and completed sequence
                             distributions in the frozen full-encoder feature space.
                             The full encoder is used as a fixed feature extractor
                             (like InceptionNet in image FID) — not because it generates
                             anything, but because it was trained on the data distribution
                             and provides a meaningful embedding.
   - compute_completion_diversity: mean pairwise distance across K stochastic completions
                             of the *same* masked input — measures the masked encoder's
                             ability to represent uncertainty about the masked region.

2. SHAP faithfulness — validates that attributions correctly identify important features.
   Based on ShapGCN (Tempel 2024, arxiv:2411.03714) and OTFlowSHAP (Zhang 2026,
   arxiv:2603.05093).
   - compute_pgi_pgu:            Prediction Gap on Important/Unimportant joints.
                                 (ShapGCN, Eq. 5-6: absolute difference, not clipped)
   - compute_pgi_pgu_random:     Same but with randomly selected joints — control
                                 condition from ShapGCN. SHAP is useful only if
                                 informed PGI > random PGI and informed PGU < random PGU.
   - compute_deletion_insertion_auc: Area under the deletion/insertion curve.
                                 (OTFlowSHAP "Faithfulness"). More robust than PGI@k
                                 because it integrates over all k simultaneously.
   - compute_shapley_completeness: Sanity check: Σφⱼ should ≈ f(x) − E[f(x_ref)].
   - compute_shap_rank_correlation: Consistency of joint importance rankings across
                                 independent KernelSHAP runs (from OTFlowSHAP).

Dependencies:
  model.actor.actor_shap  (ActorSHAP)
  model.actor.shap_compute (compute_spatial_shap)
  scipy.linalg            (sqrtm for FID)
"""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Generative quality metrics
# ---------------------------------------------------------------------------

def masked_rmse(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    coalition_mask: torch.Tensor,
) -> float:
    """RMSE between completion and ground truth over masked (imputed) positions only.

    Observed positions are pasted back unchanged, so their error is always zero and
    should be excluded. This metric measures whether the masked encoder produces
    accurate imputations, not just reconstructions of observed regions.

    Args:
        x_hat:          (B, J, F, T) model output.
        x:              (B, J, F, T) ground truth.
        coalition_mask: (B, J) spatial or (B, T) temporal — True = observed.

    Returns:
        float — RMSE over imputed positions.
    """
    B, J, F, T = x.shape
    if coalition_mask.shape[-1] == J:
        imputed = ~coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
    else:
        imputed = ~coalition_mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)

    n = imputed.float().sum().clamp(min=1.0)
    mse = ((x_hat - x).pow(2) * imputed.float()).sum() / n
    return float(mse.sqrt())


def compute_fid(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
) -> float:
    """Fréchet distance between real and generated sequence distributions.

    Features are encoder mu vectors (latent_dim,) extracted by the frozen full
    encoder — it serves as a fixed feature extractor, equivalent to InceptionNet
    in image FID. Lower FID means completions are distributionally closer to real
    sequences, validating manifold adherence.

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2})

    Args:
        real_feats: (N_r, D) encoder mu for real sequences.
        gen_feats:  (N_g, D) encoder mu for generated completions.

    Returns:
        float — FID (lower = better).
    """
    from scipy.linalg import sqrtm

    mu_r, mu_g = real_feats.mean(axis=0), gen_feats.mean(axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float((mu_r - mu_g) @ (mu_r - mu_g) + np.trace(sigma_r + sigma_g - 2.0 * covmean))


@torch.no_grad()
def compute_completion_diversity(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    coalition_mask: torch.Tensor,
    n_samples: int = 20,
) -> float:
    """Mean pairwise L2 distance across K stochastic completions of a single masked input.

    Measures the masked encoder's ability to represent uncertainty: if the masked
    region is ambiguous, multiple completions should vary; if the observed context
    strongly constrains the missing region, they should be similar. This metric
    uses the frozen full encoder as a feature extractor to project completions into
    a consistent semantic space.

    Args:
        model:          ActorSHAP (needs model.encoder and model.sample_completions).
        x:              (1, J, F, T) input sequence.
        y, mask, lengths: standard ACTOR batch fields.
        coalition_mask: (1, J) or (1, T) coalition mask.
        n_samples:      number of stochastic completions.

    Returns:
        float — mean pairwise L2 of encoder mu vectors (higher = more diverse).
    """
    completions = model.sample_completions(x, y, mask, lengths, coalition_mask,
                                            n_samples=n_samples)
    # Use a full-observation coalition_mask for the encoder (all joints observed).
    # We only want a consistent feature-space projection of each completion;
    # the conditioning mask used for sampling is irrelevant here.
    J = x.shape[1]
    full_cm = torch.ones(1, J, dtype=torch.bool, device=x.device)
    feats = []
    for x_hat in completions:
        enc = model.encoder({"x": x_hat, "y": y, "mask": mask, "coalition_mask": full_cm})
        feats.append(enc["mu_full"].cpu().numpy())
    feats = np.stack(feats, axis=0)  # (K, D)

    K = feats.shape[0]
    if K < 2:
        return 0.0
    dists = [
        float(np.linalg.norm(feats[i] - feats[j]))
        for i in range(K)
        for j in range(i + 1, K)
    ]
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# SHAP faithfulness — informed perturbation
# ---------------------------------------------------------------------------

def compute_pgi_pgu(
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    shap_values: dict[str, float],
    impute_fn: Callable,
    k_list: tuple[int, ...] = (1, 2, 3, 5, 10),
    joint_names: list[str] | None = None,
) -> dict[int, dict[str, float]]:
    """Prediction Gap on Important (PGI) and Unimportant (PGU) joints.

    PGI: mask top-k joints by |SHAP|, impute with impute_fn, measure |f(x) - f(x')|.
         High PGI confirms the explanation correctly identifies features the classifier
         relies on.
    PGU: same for bottom-k joints. Low PGU confirms unimportant features don't matter.

    Formula follows ShapGCN Eq. 5-6: absolute value (not clipped). Both PGI and
    PGU are unsigned: PGI > PGU is the success condition, not PGI > 0.

    The impute_fn argument controls which imputation strategy is used for perturbation,
    so the evaluation protocol stays consistent with how the SHAP values were computed.
    Pass impute_fn built with make_actor_impute_fn, make_zero_impute_fn, etc.

    Args:
        classifier:   callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1,n_classes).
        x:            (1, J, F, T) single sequence.
        y:            (1,) class label.
        mask:         (1, T) real-frame mask.
        lengths:      (1,) frame count.
        shap_values:  {joint_name: shap_value} from compute_spatial_shap.
        impute_fn:    callable(x, y, mask, lengths, coalition_mask) → float
                      (the mean classifier probability after imputation).
                      Built by one of the make_*_impute_fn helpers below.
        k_list:       numbers of joints to mask.
        joint_names:  joint name list (default: H36M_JOINT_NAMES).

    Returns:
        {k: {"pgi": float, "pgu": float}} for each k in k_list.
    """
    from model.actor.shap_masking import H36M_JOINT_NAMES, build_spatial_shap_mask

    if joint_names is None:
        joint_names = H36M_JOINT_NAMES

    device = x.device
    J = len(joint_names)

    with torch.no_grad():
        p_base = _class_prob(classifier, x, int(y[0].item()))

    ranked_most = sorted(range(J),
                         key=lambda j: abs(shap_values.get(joint_names[j], 0.0)),
                         reverse=True)
    ranked_least = list(reversed(ranked_most))

    results: dict[int, dict[str, float]] = {}
    for k in k_list:
        if k > J:
            continue
        for key, masked_joints in [("pgi", ranked_most[:k]), ("pgu", ranked_least[:k])]:
            observed = [j for j in range(J) if j not in masked_joints]
            cm = build_spatial_shap_mask(observed, device, n_joints=J).unsqueeze(0)
            p_perturbed = impute_fn(x, y, mask, lengths, cm)
            results.setdefault(k, {})[key] = abs(p_base - p_perturbed)

    return results


# ---------------------------------------------------------------------------
# impute_fn factory helpers — keep evaluation protocol consistent with SHAP
# ---------------------------------------------------------------------------

def make_actor_impute_fn(
    model: Any,
    classifier: Callable,
    n_samples: int = 20,
) -> Callable:
    """Return an impute_fn that uses ActorSHAP manifold-constrained completions."""
    def _fn(x, y, mask, lengths, coalition_mask):
        from model.actor.shap_compute import _batch_classify
        completions = model.sample_completions(x, y, mask, lengths, coalition_mask,
                                               n_samples=n_samples)
        class_idx = int(y[0].item())
        probs = _batch_classify(classifier, completions, class_idx)
        return float(probs.mean())
    return _fn


def make_zero_impute_fn(classifier: Callable) -> Callable:
    """Return an impute_fn that replaces masked joints with zeros."""
    from model.actor.shap_compute import value_fn_zero
    def _fn(x, y, mask, lengths, coalition_mask):
        return value_fn_zero(classifier, x, y, coalition_mask)
    return _fn


def make_mean_impute_fn(classifier: Callable, joint_means: torch.Tensor) -> Callable:
    """Return an impute_fn that replaces masked joints with per-joint training mean."""
    from model.actor.shap_compute import value_fn_mean
    def _fn(x, y, mask, lengths, coalition_mask):
        return value_fn_mean(classifier, x, y, coalition_mask, joint_means)
    return _fn


def make_marginal_impute_fn(
    classifier: Callable,
    train_pool: torch.Tensor,
    n_samples: int = 20,
    seed: int = 0,
) -> Callable:
    """Return an impute_fn that samples masked joints from the training distribution."""
    from model.actor.shap_compute import value_fn_marginal
    rng = np.random.default_rng(seed)
    def _fn(x, y, mask, lengths, coalition_mask):
        return value_fn_marginal(classifier, x, y, coalition_mask, train_pool,
                                 rng=rng, n_samples=n_samples)
    return _fn


# ---------------------------------------------------------------------------
# SHAP faithfulness — random perturbation control (ShapGCN)
# ---------------------------------------------------------------------------

def compute_pgi_pgu_random(
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    impute_fn: Callable,
    k_list: tuple[int, ...] = (1, 2, 3, 5, 10),
    n_random_repeats: int = 10,
    joint_names: list[str] | None = None,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """PGI/PGU with randomly selected joints — control condition from ShapGCN.

    SHAP is valuable only when informed PGI > random PGI and informed PGU < random PGU.
    If random selection matches SHAP-guided selection, the explanation adds nothing
    beyond knowing the perturbation method.

    The random selection is repeated n_random_repeats times and averaged to reduce
    variance, matching ShapGCN's control procedure.

    Use the same impute_fn as in compute_pgi_pgu so the evaluation protocol is consistent
    with the SHAP method being assessed.

    Args:
        classifier:          callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1,n_classes).
        x, y, mask, lengths: same as compute_pgi_pgu.
        impute_fn:           callable(x, y, mask, lengths, coalition_mask) → float.
        k_list:              numbers of joints to mask.
        n_random_repeats:    number of random joint-selection draws per k.
        joint_names:         joint name list.
        seed:                random seed for reproducibility.

    Returns:
        {k: {"pgi": float, "pgu": float}} for each k in k_list.
    """
    from model.actor.shap_masking import H36M_JOINT_NAMES, build_spatial_shap_mask

    if joint_names is None:
        joint_names = H36M_JOINT_NAMES

    rng = random.Random(seed)
    device = x.device
    J = len(joint_names)
    all_joints = list(range(J))

    with torch.no_grad():
        p_base = _class_prob(classifier, x, int(y[0].item()))

    results: dict[int, dict[str, float]] = {}
    for k in k_list:
        if k > J:
            continue
        gap_vals = []
        for _ in range(n_random_repeats):
            masked = rng.sample(all_joints, k)
            observed = [j for j in all_joints if j not in masked]
            cm = build_spatial_shap_mask(observed, device, n_joints=J).unsqueeze(0)
            p = impute_fn(x, y, mask, lengths, cm)
            gap_vals.append(abs(p_base - p))

        # PGI-rand and PGU-rand are the same quantity (random selection has no
        # concept of importance ordering); the comparison is per-k.
        avg_gap = float(np.mean(gap_vals))
        results[k] = {"pgi": avg_gap, "pgu": avg_gap}

    return results


# ---------------------------------------------------------------------------
# SHAP faithfulness — deletion / insertion AUC (OTFlowSHAP)
# ---------------------------------------------------------------------------

def compute_deletion_insertion_auc(
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    shap_values: dict[str, float],
    impute_fn: Callable,
    joint_names: list[str] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Area under the deletion and insertion curves (OTFlowSHAP "Faithfulness").

    More robust than PGI@k because it integrates over all k simultaneously,
    removing the threshold choice.

    Deletion: start with all joints observed; progressively mask joints from most
              to least important (by |SHAP|); measure classifier probability.
              Lower AUC = better (removing important features hurts more steeply).

    Insertion: start with all joints masked; progressively reveal joints from most
               to least important; measure classifier probability.
               Higher AUC = better (adding important features helps more steeply).

    Random deletion/insertion curves are also computed as internal controls.

    Use the same impute_fn as in compute_pgi_pgu so the evaluation protocol stays
    consistent with the SHAP method being assessed.

    Args:
        classifier:   callable(x_hat: Tensor(1,J,F,T)) → logits Tensor(1,n_classes).
        x, y, mask, lengths: same as compute_pgi_pgu.
        shap_values:  {joint_name: shap_value} from compute_spatial_shap.
        impute_fn:    callable(x, y, mask, lengths, coalition_mask) → float.
        joint_names:  joint name list.
        seed:         for the random-order shuffle.

    Returns:
        {"deletion_auc": float,     (lower = better)
         "insertion_auc": float,    (higher = better)
         "random_deletion_auc": float,
         "random_insertion_auc": float}
    """
    from model.actor.shap_masking import H36M_JOINT_NAMES, build_spatial_shap_mask

    if joint_names is None:
        joint_names = H36M_JOINT_NAMES

    device = x.device
    J = len(joint_names)
    class_idx = int(y[0].item())

    # Sort joints most → least important.
    ranked = sorted(range(J),
                    key=lambda j: abs(shap_values.get(joint_names[j], 0.0)),
                    reverse=True)
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(J).tolist()

    def _curve(order: list[int], deletion: bool) -> float:
        """Compute AUC for one curve type."""
        probs = []
        for step in range(J + 1):
            if deletion:
                masked_set = set(order[:step])
                observed = [j for j in range(J) if j not in masked_set]
            else:
                observed = list(order[:step])

            if len(observed) == J:
                # All joints observed — evaluate directly, no imputation needed.
                with torch.no_grad():
                    probs.append(_class_prob(classifier, x, class_idx))
                continue

            if len(observed) == 0:
                # No joints observed — fully masked coalition.
                cm = torch.zeros(1, J, dtype=torch.bool, device=device)
            else:
                cm = build_spatial_shap_mask(observed, device, n_joints=J).unsqueeze(0)

            p = impute_fn(x, y, mask, lengths, cm)
            probs.append(p)

        # Trapezoid AUC normalised to [0,1] range over J steps.
        return float(np.trapz(probs, dx=1.0 / max(J, 1)))

    return {
        "deletion_auc":        _curve(ranked, deletion=True),
        "insertion_auc":       _curve(ranked, deletion=False),
        "random_deletion_auc": _curve(random_order, deletion=True),
        "random_insertion_auc": _curve(random_order, deletion=False),
    }


# ---------------------------------------------------------------------------
# SHAP faithfulness — correctness and stability
# ---------------------------------------------------------------------------

def compute_shapley_completeness(
    shap_values: dict[str, float],
    f_x: float,
    f_baseline: float,
    joint_names: list[str] | None = None,
) -> float:
    """Check Shapley efficiency axiom: Σφⱼ ≈ f(x) − f(x_baseline).

    This is a sanity check on the KernelSHAP regression. A large completeness
    error indicates the coalition sampling or regression was poorly conditioned.

    Args:
        shap_values:  {joint_name: shap_value} from compute_spatial_shap.
        f_x:          classifier probability on unmasked x.
        f_baseline:   classifier probability on fully-masked / reference input.
        joint_names:  joint name list.

    Returns:
        float — |Σφⱼ − (f(x) − f(x_baseline))| (lower = better, ~0 is correct).
    """
    from model.actor.shap_masking import H36M_JOINT_NAMES

    if joint_names is None:
        joint_names = H36M_JOINT_NAMES

    total_attribution = sum(shap_values.get(name, 0.0) for name in joint_names)
    expected_change = f_x - f_baseline
    return abs(total_attribution - expected_change)


def compute_shap_rank_correlation(
    model: Any,
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    n_runs: int = 5,
    **kwargs,
) -> dict[str, float]:
    """Mean pairwise Spearman rank correlation of SHAP joint rankings across runs.

    Measures explanation stability (OTFlowSHAP Table 2 "Rank Corr"). A high rank
    correlation means the relative ordering of joints is consistent across different
    KernelSHAP coalition samples; low rank correlation indicates high variance due to
    insufficient coalition sampling or a poorly conditioned value function.

    Args:
        model, classifier, x, y, mask, lengths: single sequence (B=1).
        n_runs:   independent KernelSHAP runs.
        **kwargs: forwarded to compute_spatial_shap (e.g. n_kernel_samples).

    Returns:
        {"mean_rank_corr": float,   average pairwise Spearman correlation [0,1]
         "std_rank_corr":  float}
    """
    from scipy.stats import spearmanr

    from model.actor.shap_compute import compute_spatial_shap
    from model.actor.shap_masking import H36M_JOINT_NAMES

    all_vals = []
    for seed in range(n_runs):
        sv = compute_spatial_shap(
            model, classifier, x, y, mask, lengths, seed=seed, **kwargs
        )
        all_vals.append([sv[name] for name in H36M_JOINT_NAMES])

    corrs = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            r, _ = spearmanr(all_vals[i], all_vals[j])
            corrs.append(float(r))

    return {
        "mean_rank_corr": float(np.mean(corrs)),
        "std_rank_corr":  float(np.std(corrs)),
    }


def compute_shap_rsd(
    model: Any,
    classifier: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    n_runs: int = 5,
    **kwargs,
) -> dict[str, float]:
    """Per-joint Relative Standard Deviation (σ/|μ|) of Shapley values across runs.

    Args:
        model, classifier, x, y, mask, lengths: single sequence (B=1).
        n_runs:   independent KernelSHAP runs.
        **kwargs: forwarded to compute_spatial_shap.

    Returns:
        {joint_name: RSD} for all 17 joints.
    """
    from model.actor.shap_compute import compute_spatial_shap
    from model.actor.shap_masking import H36M_JOINT_NAMES

    all_runs = [
        compute_spatial_shap(model, classifier, x, y, mask, lengths, seed=s, **kwargs)
        for s in range(n_runs)
    ]
    rsd: dict[str, float] = {}
    for name in H36M_JOINT_NAMES:
        vals = np.array([r[name] for r in all_runs])
        rsd[name] = float(np.clip(vals.std() / (abs(vals.mean()) + 1e-8), 0.0, 10.0))
    return rsd


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_encoder_features(
    model: Any,
    sequences: list[torch.Tensor],
    y_list: list[torch.Tensor],
    mask_list: list[torch.Tensor],
) -> np.ndarray:
    """Extract frozen full-encoder mu features for a list of sequences.

    This is the feature extractor for compute_fid and compute_completion_diversity.
    The full encoder is used (not the masked encoder) because it provides a
    consistent, high-quality projection of any sequence into the data distribution's
    feature space regardless of what generated it.

    Args:
        model:      ActorSHAP.
        sequences:  list of (1, J, F, T) tensors.
        y_list:     list of (1,) label tensors.
        mask_list:  list of (1, T) mask tensors.

    Returns:
        (N, D) numpy array of mu vectors.
    """
    feats = []
    for xseq, y, m in zip(sequences, y_list, mask_list):
        J = xseq.shape[1]
        full_cm = torch.ones(1, J, dtype=torch.bool, device=xseq.device)
        enc = model.encoder({"x": xseq, "y": y, "mask": m, "coalition_mask": full_cm})
        feats.append(enc["mu_full"].cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _class_prob(classifier: Callable, x_hat: torch.Tensor, class_idx: int) -> float:
    logits = classifier(x_hat)
    return float(torch.softmax(logits, dim=-1)[0, class_idx].item())


# ---------------------------------------------------------------------------
# Batched faithfulness — all spatial metrics in one pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_spatial_faithfulness_batched(
    classifier_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    shap_values: dict[str, float],
    method: str,
    *,
    joint_means: torch.Tensor | None = None,
    train_pool: torch.Tensor | None = None,
    actor_shap: Any | None = None,
    n_samples: int = 20,
    k_list: tuple[int, ...] = (1, 2, 3, 5),
    n_random_repeats: int = 10,
    p_full: float | None = None,
    p_ref: float | None = None,
    seq_idx: int = 0,
    chunk_size: int = 256,
) -> dict:
    """Compute PGI/PGU, PGI/PGU-random, deletion/insertion AUC, and
    completeness error for one SHAP method in a single batched pass.

    Replaces the sequential pattern of ``compute_pgi_pgu`` +
    ``compute_pgi_pgu_random`` + ``compute_deletion_insertion_auc`` +
    ``compute_shapley_completeness``.  All coalition perturbations are built
    upfront and classified in large batched GPU calls, reducing thousands of
    B=1 forward passes to a handful of B=chunk_size passes.

    Args:
        method:     ``"zero"`` | ``"mean"`` | ``"marginal"`` | ``"actor"``.
        n_samples:  Donor draws (marginal) or completions (actor) per coalition.
        seq_idx:    Seed for random PGI/PGU joint selection and random
                    deletion/insertion order — matches ``seed=`` in the
                    individual metric functions for reproducibility.

    Returns:
        Same dict structure as the ``_faithfulness_block`` pattern in
        ``evaluate_shap*.py``.
    """
    from model.actor.shap_compute import _batch_classify, _classify_chunked
    from model.actor.shap_masking import H36M_JOINT_NAMES

    device = x.device
    J, F, T = x.shape[1], x.shape[2], x.shape[3]
    class_idx = int(y[0].item())
    joint_names = H36M_JOINT_NAMES

    if p_full is None:
        p_full = _class_prob(classifier_fn, x, class_idx)

    # --- 1. Build all observed-joint lists for every evaluation point ---
    ranked = sorted(
        range(J),
        key=lambda j: abs(shap_values.get(joint_names[j], 0.0)),
        reverse=True,
    )
    ranked_least = list(reversed(ranked))

    rng_py = random.Random(seq_idx)
    rng_np = np.random.default_rng(seq_idx)
    random_order = rng_np.permutation(J).tolist()
    all_joints = list(range(J))

    observed_lists: list[list[int]] = []
    idx_pgi: dict[int, int] = {}
    idx_pgu: dict[int, int] = {}
    idx_rand: dict[int, list[int]] = {}
    idx_del: list[int] = []
    idx_ins: list[int] = []
    idx_rdel: list[int] = []
    idx_rins: list[int] = []

    def _append(obs: list[int]) -> int:
        i = len(observed_lists)
        observed_lists.append(obs)
        return i

    for k in k_list:
        if k > J:
            continue
        idx_pgi[k] = _append([j for j in range(J) if j not in ranked[:k]])
        idx_pgu[k] = _append([j for j in range(J) if j not in ranked_least[:k]])

    for k in k_list:
        if k > J:
            continue
        idxs: list[int] = []
        for _ in range(n_random_repeats):
            masked = rng_py.sample(all_joints, k)
            idxs.append(_append([j for j in all_joints if j not in masked]))
        idx_rand[k] = idxs

    for step in range(J + 1):
        idx_del.append(_append([j for j in range(J) if j not in ranked[:step]]))
        idx_ins.append(_append(list(ranked[:step])))
        idx_rdel.append(_append([j for j in range(J) if j not in random_order[:step]]))
        idx_rins.append(_append(list(random_order[:step])))

    N = len(observed_lists)

    # --- 2. Boolean coalition mask tensor (N, J) ---
    cms_np = np.zeros((N, J), dtype=bool)
    for i, obs in enumerate(observed_lists):
        for j in obs:
            cms_np[i, j] = True
    cms_t = torch.tensor(cms_np, dtype=torch.bool, device=device)

    # --- 3. Build perturbed inputs and classify ---
    if method == "zero":
        mask_b = cms_t[:, :, None, None]
        x_batch = torch.where(
            mask_b,
            x.expand(N, -1, -1, -1),
            torch.zeros(1, 1, 1, 1, dtype=x.dtype, device=device),
        )
        all_probs = _classify_chunked(
            classifier_fn, x_batch, class_idx, chunk_size=chunk_size,
        )

    elif method == "mean":
        mean_fill = joint_means.unsqueeze(-1).expand(-1, -1, T)
        mask_b = cms_t[:, :, None, None]
        x_batch = torch.where(
            mask_b,
            x.expand(N, -1, -1, -1),
            mean_fill.unsqueeze(0).expand(N, -1, -1, -1),
        )
        all_probs = _classify_chunked(
            classifier_fn, x_batch, class_idx, chunk_size=chunk_size,
        )

    elif method == "marginal":
        rng_donors = np.random.default_rng(seq_idx + (1 << 31))
        N_pool = train_pool.shape[0]
        donor_indices = rng_donors.integers(0, N_pool, size=(N, n_samples))
        COAL_CHUNK = 32
        all_probs_parts: list[np.ndarray] = []
        x0 = x[0]

        for c0 in range(0, N, COAL_CHUNK):
            c1 = min(c0 + COAL_CHUNK, N)
            C = c1 - c0
            donors = train_pool[donor_indices[c0:c1]].to(device)
            cm_b = cms_t[c0:c1, None, :, None, None]
            x_out = torch.where(cm_b, x0[None, None], donors)
            x_flat = x_out.reshape(C * n_samples, J, F, T)
            probs_c = _classify_chunked(
                classifier_fn, x_flat, class_idx, chunk_size=chunk_size,
            )
            all_probs_parts.append(
                probs_c.reshape(C, n_samples).mean(axis=1),
            )
        all_probs = np.concatenate(all_probs_parts)

    elif method == "actor":
        if actor_shap is None:
            raise ValueError("actor_shap is required for method='actor'")
        all_completions: list[torch.Tensor] = []
        slice_ends: list[int] = []
        for i in range(N):
            comps = actor_shap.sample_completions(
                x, y, mask, lengths, cms_t[i : i + 1], n_samples=n_samples,
            )
            all_completions.extend(comps)
            slice_ends.append(len(all_completions))
        all_comp_probs = _batch_classify(
            classifier_fn, all_completions, class_idx, chunk_size=chunk_size,
        )
        starts = [0] + slice_ends[:-1]
        all_probs = np.array([
            all_comp_probs[s:e].mean() for s, e in zip(starts, slice_ends)
        ])

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # --- 4. Extract metrics ---
    pgi_pgu: dict[int, dict[str, float]] = {}
    for k in k_list:
        if k > J:
            continue
        pgi_pgu[k] = {
            "pgi": abs(p_full - float(all_probs[idx_pgi[k]])),
            "pgu": abs(p_full - float(all_probs[idx_pgu[k]])),
        }

    pgi_pgu_rand: dict[int, dict[str, float]] = {}
    for k in k_list:
        if k > J:
            continue
        gaps = [abs(p_full - float(all_probs[i])) for i in idx_rand[k]]
        avg = float(np.mean(gaps))
        pgi_pgu_rand[k] = {"pgi": avg, "pgu": avg}

    dx = 1.0 / max(J, 1)
    del_probs = [float(all_probs[i]) for i in idx_del]
    ins_probs = [float(all_probs[i]) for i in idx_ins]
    rdel_probs = [float(all_probs[i]) for i in idx_rdel]
    rins_probs = [float(all_probs[i]) for i in idx_rins]

    return {
        "pgi_pgu":              {str(k): v for k, v in pgi_pgu.items()},
        "pgi_pgu_rand":         {str(k): v for k, v in pgi_pgu_rand.items()},
        "deletion_auc":         float(np.trapz(del_probs, dx=dx)),
        "insertion_auc":        float(np.trapz(ins_probs, dx=dx)),
        "random_deletion_auc":  float(np.trapz(rdel_probs, dx=dx)),
        "random_insertion_auc": float(np.trapz(rins_probs, dx=dx)),
        "completeness_error":   compute_shapley_completeness(shap_values, p_full, p_ref),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    B, J, F, T = 1, 17, 3, 16
    x = torch.randn(B, J, F, T)
    x_hat = x + 0.1 * torch.randn_like(x)
    y = torch.zeros(B, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.bool)
    lengths = torch.full((B,), T, dtype=torch.long)

    # --- masked_rmse ---
    cm_s = torch.ones(B, J, dtype=torch.bool); cm_s[:, :3] = False
    assert masked_rmse(x_hat, x, cm_s) >= 0
    cm_t = torch.ones(B, T, dtype=torch.bool); cm_t[:, :T // 4] = False
    assert masked_rmse(x_hat, x, cm_t) >= 0
    print("masked_rmse OK")

    # --- compute_fid ---
    fid = compute_fid(np.random.randn(50, 32), np.random.randn(50, 32))
    assert isinstance(fid, float) and fid >= 0
    print(f"compute_fid OK: {fid:.3f}")

    # --- mock model / classifier for the remaining tests ---
    class _Enc:
        def __call__(self, batch):
            B2 = batch["x"].shape[0]
            d = batch["x"].device
            return {"mu_full": torch.randn(B2, 32, device=d)}

    class _MockModel:
        encoder = _Enc()
        def sample_completions(self, x, y, mask, lengths, cm, n_samples=20, **kw):
            return [torch.randn_like(x) for _ in range(n_samples)]

    def _clf(x_hat):
        return torch.randn(1, 3)

    model = _MockModel()

    # --- compute_completion_diversity ---
    div = compute_completion_diversity(model, x, y, mask, lengths, cm_s, n_samples=4)
    assert isinstance(div, float) and div >= 0
    print(f"compute_completion_diversity OK: {div:.3f}")

    # --- build impute_fn for the remaining tests ---
    from model.actor.shap_masking import H36M_JOINT_NAMES
    actor_impute_fn = make_actor_impute_fn(model, _clf, n_samples=2)
    sv = {name: float(np.random.randn()) for name in H36M_JOINT_NAMES}

    # --- compute_pgi_pgu ---
    result = compute_pgi_pgu(_clf, x, y, mask, lengths, sv,
                             impute_fn=actor_impute_fn, k_list=(1, 3))
    assert set(result.keys()) == {1, 3}
    assert result[1]["pgi"] >= 0 and result[1]["pgu"] >= 0
    print(f"compute_pgi_pgu OK: {result}")

    # --- compute_pgi_pgu_random ---
    rand_result = compute_pgi_pgu_random(_clf, x, y, mask, lengths,
                                         impute_fn=actor_impute_fn,
                                         k_list=(1, 3), n_random_repeats=3)
    assert set(rand_result.keys()) == {1, 3}
    print(f"compute_pgi_pgu_random OK: {rand_result}")

    # --- compute_shapley_completeness ---
    err = compute_shapley_completeness(sv, f_x=0.8, f_baseline=0.3)
    assert isinstance(err, float)
    print(f"compute_shapley_completeness OK: error={err:.4f}")

    # --- compute_deletion_insertion_auc ---
    auc = compute_deletion_insertion_auc(_clf, x, y, mask, lengths, sv,
                                         impute_fn=actor_impute_fn)
    assert set(auc.keys()) == {"deletion_auc", "insertion_auc",
                                "random_deletion_auc", "random_insertion_auc"}
    print(f"compute_deletion_insertion_auc OK: {auc}")

    print("OK")
