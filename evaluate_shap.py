"""evaluate_shap.py — SHAP evaluation pipeline for CARE-PD pretrained classifiers.

RECIPE — what you need before running this script
===================================================

1.  **Trained ActorSHAP checkpoint** (e.g. ``actor_shap_last.ckpt`` from
    ``train_actor_shap.py``), plus ``config.json`` in the same run directory
    (used to rebuild the Transformer architecture).  Optional override:
    ``--actor_shap_config /path/to/config.json``.

2.  **Trained CARE-PD classifier** (checkpoint + params JSON)

    The checkpoint is saved by ``run.py`` at::

        experiment_outs/<experiment_name>/<model_prefix>/<run_num>/models/<clarification>/fold<fold>/latest_epoch.pth.tr

    Note: ``run.py`` saves ``latest_epoch.pth.tr`` (not ``best_epoch.pth.tr``).
    Pass the ``latest_epoch.pth.tr`` path to ``--classifier_ckpt``.

    The params are saved at::

        experiment_outs/<experiment_name>/<model_prefix>/<run_num>/config/config_excluding_study_params.json

    ⚠️  BACKBONE COMPATIBILITY: only 3-D H36M backbones are directly compatible
    with ActorSHAP's output format (J=17, F=3, T).  Recommended backbone:

        - **potr**: uses 3-D H36M data (``data_type='h36m'``), directly
          compatible after permuting ``(1,J,F,T) → (1,T,J*F)``.

    MotionBERT, MixSTE, PoseFormerV2, and MotionAGFormer take 2-D projected
    sequences and require a camera-projection step before they can be used as
    the black-box classifier — this is not currently supported.

3.  **Dataset** (already on disk in ``assets/datasets/h36m/``).

    This script loads the same test split that was used during ``run.py``
    evaluation, using the project's standard fold structure.

WHAT THE SCRIPT DOES (per test sequence)
=========================================

For each sequence ``(x, y, mask, lengths)`` in the test fold:

A.  Compute spatial SHAP values under **4 methods** (same KernelSHAP sampling
    and WLS regression, different imputation strategy for masked joints):

    =================== ======================================================
    Method              Imputation of masked joints ``j ∈ S̄``
    =================== ======================================================
    ActorSHAP (ours)    K stochastic completions from VAEAC ``r_ψ(z|x_S, y)``
    Zero                Set to zero (off-manifold)
    Mean                Set to per-joint training-set mean (static posture)
    Marginal            Copy from a random training sequence (independent draw)
    =================== ======================================================

B.  Compute faithfulness metrics for **each method**:

    - ``PGI / PGU`` at ``k ∈ {1, 2, 3, 5}`` joints
    - ``PGI-rand / PGU-rand`` (random-selection control for same ``k``)
    - ``Deletion AUC / Insertion AUC`` (curve integrated over all ``k``)
    - ``Shapley completeness error``

C.  Compute **rank stability** (ActorSHAP only):

    - ``Shapley rank correlation`` across 5 independent KernelSHAP runs

D.  Compute **generative quality** metrics (ActorSHAP only):

    - ``Masked RMSE`` on held-out joints
    - ``Completion diversity`` (pairwise encoder distance across K completions)

E.  Compute **FID** once over the full test set (ActorSHAP vs real sequences).

Results per sequence are written to ``<output_dir>/per_sequence.jsonl``.
Aggregate statistics (mean ± std) are written to ``<output_dir>/aggregate.json``.
Keys use ``spatial/<method>/…`` and ``temporal/<method>/…`` (same layout as
``evaluate_shap_baselines.py``) so tables are directly comparable.

PARALLEL SHARDS (multi-GPU / many jobs)
=======================================

To split the test set across processes, use ``--num_shards K`` and
``--shard_id i`` with ``0 <= i < K``.  Each job writes only its block of indices
to ``<output_dir>/shards/per_sequence_shard{i:04d}_of_{K:04d}.jsonl`` plus
``shard_meta_….json``.  FID and ``aggregate.json`` are **not** produced per shard;
after all shards finish, run::

    python merge_shap_eval.py --output_dir <same_dir> [same model/data flags…]

which merges shard JSONLs, writes ``per_sequence.jsonl`` and ``aggregate.json``,
and computes FID in one pass over the full (capped) test set.

Example: four GPUs, one process each::

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python evaluate_shap.py ... --num_shards 4 --shard_id $i &
    done
    wait

Multiple shard jobs may share one GPU (different ``shard_id``, same ``num_shards``);
VRAM limits how many can run concurrently.

USAGE EXAMPLE
=============
::

    python evaluate_shap.py \\
        --actor_shap_ckpt experiment_outs/actor_shap/<run>/actor_shap_best.ckpt \\
        --backbone potr \\
        --config BMCLab.json \\
        --num_folds 23 \\
        --fold 1 \\
        --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \\
        --output_dir results/shap_actor_bmclab_fold1 \\
        --device cuda:4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch

from model.actor.actor_shap import ActorSHAP, CoalitionFullEncoder, MaskedActorEncoder
from model.actor.transformer_arch import Decoder_TRANSFORMER
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.shap_compute import (
    compute_spatial_shap,
    compute_spatial_shap_baseline,
    compute_temporal_shap,
    compute_temporal_shap_baseline,
    value_fn,
)
from model.actor.shap_masking import (
    H36M_JOINT_NAMES,
    build_temporal_shap_mask,
    build_temporal_windows,
    detect_stride_period,
)
from model.actor.shap_metrics import (
    _class_prob,
    compute_completion_diversity,
    compute_fid,
    compute_shap_rank_correlation,
    compute_shapley_completeness,
    compute_spatial_faithfulness_batched,
    extract_encoder_features,
    masked_rmse,
)
from model.backbone_loader import load_pretrained_backbone, load_pretrained_weights
from model.motion_encoder import MotionEncoder
from data.dataloaders import collate_fn
from const import const

# Shared utilities (backbone param loading, data helpers, classifier wrapper,
# temporal faithfulness, output directory naming, p_full warning).
from model.actor.shap_eval_shared import (
    _load_backbone_params,
    _raw_data_args,
    build_train_pool,
    build_zscore_stats_for_potr,
    build_classifier_fn,
    _temporal_deletion_insertion_auc_batched,
    load_motion_encoder as _load_motion_encoder_shared,
    resolve_output_dir,
    check_p_full_warning,
)


def _x_to_actor(
    x: torch.Tensor,
    merge_last_dim: bool,
    njoints: int,
    nfeats: int,
) -> torch.Tensor:
    """Convert dataloader output to ACTOR ``(B, J, F, T)`` format."""
    if merge_last_dim:
        B, T, _ = x.shape
        return x.reshape(B, T, njoints, nfeats).permute(0, 2, 3, 1).contiguous()
    return x.permute(0, 2, 3, 1).contiguous()


def _temporal_deletion_insertion_auc_actor(
    actor_shap: ActorSHAP,
    classifier_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    window_assignments: list[list[int]],
    temporal_shap_vals: dict[str, float],
    n_completion_samples: int,
) -> dict:
    """Temporal deletion / insertion AUC using ActorSHAP completions (exact windows)."""
    K = len(window_assignments)
    T = x.shape[-1]
    device = x.device
    class_idx = int(y[0].item())
    window_names = list(temporal_shap_vals.keys())
    order = sorted(range(K), key=lambda k: -temporal_shap_vals[window_names[k]])

    def _pred_full() -> float:
        return _class_prob(classifier_fn, x, class_idx)

    del_curve = [_pred_full()]
    for step in range(K):
        observed = [k for k in range(K) if k not in order[: step + 1]]
        cm_t = build_temporal_shap_mask(
            observed, window_assignments, T, device,
        ).unsqueeze(0)
        del_curve.append(
            float(
                value_fn(
                    actor_shap, classifier_fn, x, y, mask, lengths, cm_t,
                    n_samples=n_completion_samples,
                )
            )
        )

    cm_empty = build_temporal_shap_mask(
        [], window_assignments, T, device,
    ).unsqueeze(0)
    ins_curve = [
        float(
            value_fn(
                actor_shap, classifier_fn, x, y, mask, lengths, cm_empty,
                n_samples=n_completion_samples,
            )
        )
    ]
    for step in range(K):
        observed = order[: step + 1]
        cm_t = build_temporal_shap_mask(
            observed, window_assignments, T, device,
        ).unsqueeze(0)
        ins_curve.append(
            float(
                value_fn(
                    actor_shap, classifier_fn, x, y, mask, lengths, cm_t,
                    n_samples=n_completion_samples,
                )
            )
        )

    xs = np.linspace(0.0, 1.0, K + 1)
    return {
        "deletion_auc": float(np.trapz(del_curve, xs)),
        "insertion_auc": float(np.trapz(ins_curve, xs)),
        "p_empty": float(ins_curve[0]),
    }


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_actor_shap_training_config(
    ckpt_path: str,
    config_override: Optional[str] = None,
) -> dict:
    """Load ``train_actor_shap.py`` hyperparameters (same schema as ``config.json``)."""
    candidates = []
    if config_override:
        candidates.append(config_override)
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidates.append(os.path.join(ckpt_dir, "config.json"))
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return {}


def load_actor_shap(
    ckpt_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
) -> ActorSHAP:
    """Load ActorSHAP from a Lightning checkpoint (``ActorSHAPModule``).

    Checkpoints store weights under the ``model.`` prefix.  Architecture must
    match training: we rebuild ``CoalitionFullEncoder`` / ``MaskedActorEncoder`` /
    ``Decoder_TRANSFORMER`` using ``config.json`` saved next to the checkpoint
    (or ``--actor_shap_config``), with the same defaults as ``train_actor_shap.py``.
    """
    cfg = _load_actor_shap_training_config(ckpt_path, config_override=config_path)
    if not cfg:
        print(
            "[load_actor_shap] WARNING: no config.json next to checkpoint and no "
            "--actor_shap_config; using default architecture (latent_dim=256, …).",
        )

    latent_dim = int(cfg.get("latent_dim", 256))
    ff_size = int(cfg.get("ff_size", 1024))
    num_layers = int(cfg.get("num_layers", 8))
    num_heads = int(cfg.get("num_heads", 4))
    dropout = float(cfg.get("dropout", 0.1))
    num_classes = int(cfg.get("num_classes", 3))

    common = dict(
        modeltype="cvae",
        njoints=17,
        nfeats=3,
        num_frames=0,
        num_classes=num_classes,
        translation=True,
        pose_rep="xyz",
        glob=True,
        glob_rot=[3.141592653589793, 0, 0],
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        ablation=None,
        activation="gelu",
    )
    # CoalitionFullEncoder wraps Encoder_TRANSFORMER + target_marker_spatial.
    # Must match the architecture used in train_actor_shap.py.
    encoder        = CoalitionFullEncoder(**common)
    masked_encoder = MaskedActorEncoder(**common)
    decoder        = Decoder_TRANSFORMER(**common)
    model = ActorSHAP(
        encoder,
        masked_encoder,
        decoder,
        latent_dim=latent_dim,
        device=device,
        pose_rep="xyz",
        num_classes=num_classes,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    raw = ckpt.get("state_dict", ckpt)
    state_dict = {
        k[len("model."):] if k.startswith("model.") else k: v
        for k, v in raw.items()
    }
    # Lightning should only store ``model.*`` weights; drop anything else.
    state_dict = {k: v for k, v in state_dict.items() if k.split(".")[0] in (
        "encoder", "masked_encoder", "decoder",
    )}

    model.load_state_dict(state_dict, strict=True)

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Cached ActorSHAP wrapper — serves completions from pre-generated NPZ files
# ---------------------------------------------------------------------------

class CachedActorSHAP:
    """Wraps ActorSHAP to serve KernelSHAP completions from a pre-computed cache.

    Pre-generated by ``cache_actor_shap_completions.py``.  For coalitions that
    are in the cache (the KernelSHAP set), completions are loaded from disk —
    no GPU model call needed.  For cache misses (faithfulness metric coalitions,
    ~50 per sequence), falls back to the real model.

    Usage in evaluate_shap.py::

        cached = CachedActorSHAP(actor_shap, cache_dir)
        cached.load_for_seq(seq_idx)
        # Pass ``cached`` wherever ``actor_shap`` is used.
    """

    def __init__(self, real_model: ActorSHAP, cache_dir: str):
        self._model = real_model
        self._cache_dir = cache_dir
        # coalition key → (n_samp, J, F, T) float16 numpy array
        self._lookup: dict[tuple, np.ndarray] = {}
        self._n_cached = 0
        self._n_misses = 0

    def load_for_seq(self, seq_idx: int) -> None:
        """Load cached completions for the given sequence index."""
        path = os.path.join(self._cache_dir, f"seq_{seq_idx:05d}.npz")
        if not os.path.isfile(path):
            print(f"  [CachedActorSHAP] WARNING: no cache for seq {seq_idx} at {path}. "
                  "Falling back to real model for all coalitions.")
            self._lookup = {}
            return
        data = np.load(path)
        self._lookup = {}
        for cm, comps in zip(data["coalition_masks"], data["completions"]):
            key = tuple(cm.astype(bool).tolist())
            self._lookup[key] = comps   # (n_samp, J, F, T) float16
        # Also load temporal completions if present.
        if "temporal_coalition_masks" in data:
            for tm, comps in zip(data["temporal_coalition_masks"],
                                  data["temporal_completions"]):
                key = ("temporal",) + tuple(tm.astype(bool).tolist())
                self._lookup[key] = comps
        self._n_cached = len(self._lookup)
        self._n_misses = 0

    # Expose attributes the rest of evaluate_shap.py reads directly.
    @property
    def num_classes(self):
        return self._model.num_classes

    @property
    def latent_dim(self):
        return self._model.latent_dim

    @property
    def encoder(self):
        return self._model.encoder

    @property
    def masked_encoder(self):
        return self._model.masked_encoder

    @property
    def decoder(self):
        return self._model.decoder

    def sample_completions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        coalition_mask: torch.Tensor,
        n_samples: int = 20,
        paste_observed: bool = True,
    ) -> list[torch.Tensor]:
        """Serve from cache if available; else call real model."""
        # Build lookup key from coalition_mask.
        cm_np = coalition_mask[0].cpu().numpy().astype(bool)
        is_temporal = coalition_mask.shape[-1] != x.shape[1]  # T ≠ J
        if is_temporal:
            key = ("temporal",) + tuple(cm_np.tolist())
        else:
            key = tuple(cm_np.tolist())

        if key in self._lookup:
            comps_np = self._lookup[key]                # (n_samp_cached, J, F, T)
            n_avail  = comps_np.shape[0]
            # Use min(n_samples, n_avail) — fewer cached samples is still valid.
            use_n = min(n_samples, n_avail)
            device = x.device
            out = [
                torch.from_numpy(comps_np[i].astype(np.float32)).unsqueeze(0).to(device)
                for i in range(use_n)
            ]
            return out

        # Cache miss — call the real model (faithfulness metric coalitions).
        self._n_misses += 1
        return self._model.sample_completions(
            x, y, mask, lengths, coalition_mask, n_samples, paste_observed,
        )

    def __getattr__(self, name):
        # Forward any other attribute access to the real model.
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Per-sequence SHAP evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    seq_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    actor_shap: ActorSHAP,
    motion_encoder: MotionEncoder,
    backbone_name: str,
    train_pool: torch.Tensor,
    joint_means: torch.Tensor,
    zscore_mean: Optional[torch.Tensor],
    zscore_std: Optional[torch.Tensor],
    cfg: dict,
) -> dict:
    """Run all SHAP methods and metrics for a single sequence.

    Args:
        seq_idx:       index in the test set (for logging).
        x:             (1, J, F, T) sequence tensor.
        y:             (1,) UPDRS class label.
        mask:          (1, T) valid-frame boolean mask.
        lengths:       (1,) frame count.
        actor_shap:    trained ActorSHAP model.
        motion_encoder: pretrained CARE-PD classifier.
        train_pool:    (N, J, F, T) training sequences on CPU.
        joint_means:   (J, F) per-joint training mean on device.
        cfg:           evaluation configuration dict.

    Returns:
        result dict suitable for JSON serialisation.
    """
    device = x.device
    n_kernel = cfg.get("n_kernel_samples", 3000)
    n_completions = cfg.get("n_completion_samples", 20)
    n_rank_runs = cfg.get("n_rank_runs", 5)
    k_list = tuple(cfg.get("k_list", [1, 2, 3, 5]))
    fps = float(cfg.get("fps", 30.0))

    # Build classifier callable closed over this sequence's mask.
    classifier_fn = build_classifier_fn(
        motion_encoder, mask, backbone_name,
        zscore_mean=zscore_mean, zscore_std=zscore_std,
    )

    # ------------------------------------------------------------------
    # Compute SHAP values for all 4 methods.
    # ------------------------------------------------------------------
    print(f"  [seq {seq_idx}] computing SHAP values …")
    shap_actor = compute_spatial_shap(
        actor_shap, classifier_fn, x, y, mask, lengths,
        n_kernel_samples=n_kernel, n_completion_samples=n_completions, seed=seq_idx,
    )
    shap_zero = compute_spatial_shap_baseline(
        "zero", classifier_fn, x, y, mask, lengths,
        n_kernel_samples=n_kernel, seed=seq_idx,
    )
    shap_mean = compute_spatial_shap_baseline(
        "mean", classifier_fn, x, y, mask, lengths,
        joint_means=joint_means, n_kernel_samples=n_kernel, seed=seq_idx,
    )
    shap_marginal = compute_spatial_shap_baseline(
        "marginal", classifier_fn, x, y, mask, lengths,
        train_pool=train_pool, n_kernel_samples=n_kernel,
        n_marginal_samples=n_completions, seed=seq_idx,
    )

    # v_empty and v_full for each method are already computed inside the SHAP
    # functions (boundary coalition evaluations for constrained WLS).  Reuse
    # them for completeness metrics — avoids a second classifier call and
    # ensures the same value function call is used for both the WLS constraint
    # and the completeness check.
    p_full         = _class_prob(classifier_fn, x, int(y[0].item()))
    p_ref_actor    = shap_actor["_v_empty"]
    p_ref_zero     = shap_zero["_v_empty"]
    p_ref_mean     = shap_mean["_v_empty"]
    p_ref_marginal = shap_marginal["_v_empty"]

    # ------------------------------------------------------------------
    # Faithfulness metrics for all 4 methods (batched).
    # ------------------------------------------------------------------
    print(f"  [seq {seq_idx}] computing faithfulness metrics (batched) …")
    metrics_actor = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_actor, "actor",
        actor_shap=actor_shap, n_samples=n_completions,
        k_list=k_list, p_full=p_full, p_ref=p_ref_actor, seq_idx=seq_idx,
    )
    metrics_zero = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_zero, "zero",
        k_list=k_list, p_full=p_full, p_ref=p_ref_zero, seq_idx=seq_idx,
    )
    metrics_mean = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_mean, "mean",
        joint_means=joint_means,
        k_list=k_list, p_full=p_full, p_ref=p_ref_mean, seq_idx=seq_idx,
    )
    metrics_marginal = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_marginal, "marginal",
        train_pool=train_pool, n_samples=n_completions,
        k_list=k_list, p_full=p_full, p_ref=p_ref_marginal, seq_idx=seq_idx,
    )

    # ------------------------------------------------------------------
    # Temporal SHAP (K=4 stride windows, exact enumeration; shared windows).
    # ------------------------------------------------------------------
    print(f"  [seq {seq_idx}] temporal SHAP (exact, K=4) …")
    x_np = x[0].permute(2, 0, 1).cpu().numpy()
    stride_period, stride_fallback = detect_stride_period(x_np, fps=int(fps))
    window_assignments = build_temporal_windows(x.shape[-1], stride_period, K=4)

    t_shap_actor = compute_temporal_shap(
        actor_shap, classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments,
        n_completion_samples=n_completions,
        fps=int(fps),
    )
    t_shap_zero = compute_temporal_shap_baseline(
        "zero", classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments, seed=seq_idx, fps=int(fps),
    )
    t_shap_mean = compute_temporal_shap_baseline(
        "mean", classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments,
        joint_means=joint_means, seed=seq_idx, fps=int(fps),
    )
    t_shap_marginal = compute_temporal_shap_baseline(
        "marginal", classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments,
        train_pool=train_pool, n_marginal_samples=n_completions,
        seed=seq_idx, fps=int(fps),
    )

    # Use _v_empty / _v_full stored in the temporal SHAP dicts (computed once
    # during the constrained WLS boundary evaluation) for completeness metrics.
    t_names = [k for k in t_shap_actor if not k.startswith("_")]
    del_actor = _temporal_deletion_insertion_auc_actor(
        actor_shap, classifier_fn, x, y, mask, lengths,
        window_assignments, t_shap_actor, n_completions,
    )
    comp_actor = compute_shapley_completeness(
        t_shap_actor, t_shap_actor["_v_full"], t_shap_actor["_v_empty"],
        joint_names=t_names,
    )
    temporal_faithfulness: dict = {
        "actor": {
            "deletion_auc": del_actor["deletion_auc"],
            "insertion_auc": del_actor["insertion_auc"],
            "completeness_error": comp_actor,
        },
    }
    for m_name, t_shap, jm, tp in (
        ("zero", t_shap_zero, None, None),
        ("mean", t_shap_mean, joint_means, None),
        ("marginal", t_shap_marginal, None, train_pool),
    ):
        t_window_names = [k for k in t_shap if not k.startswith("_")]
        del_ins = _temporal_deletion_insertion_auc_batched(
            classifier_fn, x, y, window_assignments, t_shap, m_name,
            joint_means=jm, train_pool=tp, seed=seq_idx,
            n_marginal_samples=n_completions,
        )
        comp_err = compute_shapley_completeness(
            t_shap, t_shap["_v_full"], t_shap["_v_empty"],
            joint_names=t_window_names,
        )
        temporal_faithfulness[m_name] = {
            "deletion_auc": del_ins["deletion_auc"],
            "insertion_auc": del_ins["insertion_auc"],
            "completeness_error": comp_err,
        }

    # ------------------------------------------------------------------
    # Rank stability (ActorSHAP only — most expensive).
    # Each "run" is a *full* spatial KernelSHAP (same cost as shap_actor above),
    # so default n_rank_runs=5 ≈ 5× extra Actor KernelSHAP work per sequence.
    # Use --n_rank_runs 0 to skip when you only need SHAP values / faithfulness.
    # ------------------------------------------------------------------
    if n_rank_runs < 2:
        print(
            f"  [seq {seq_idx}] skipping rank correlation "
            f"(n_rank_runs={n_rank_runs}; need >=2 for Spearman stability) …"
        )
        rank_corr = {"mean_rank_corr": float("nan"), "std_rank_corr": float("nan")}
    else:
        print(f"  [seq {seq_idx}] computing rank correlation ({n_rank_runs} runs) …")
        rank_corr = compute_shap_rank_correlation(
            actor_shap, classifier_fn, x, y, mask, lengths, n_runs=n_rank_runs,
            n_kernel_samples=n_kernel, n_completion_samples=n_completions,
        )

    # ------------------------------------------------------------------
    # Generative quality for ActorSHAP: masked RMSE + completion diversity.
    # Use a fixed held-out coalition (mask first 4 joints) for consistency.
    # ------------------------------------------------------------------
    eval_coalition = build_eval_coalition(device)
    print(f"  [seq {seq_idx}] computing generative quality metrics …")
    rmse = masked_rmse(
        actor_shap.sample_completions(x, y, mask, lengths, eval_coalition, n_samples=1)[0],
        x,
        eval_coalition,
    )
    diversity = compute_completion_diversity(
        actor_shap, x, y, mask, lengths, eval_coalition, n_samples=n_completions
    )

    return {
        "seq_idx": seq_idx,
        "true_class": int(y[0].item()),
        "p_full": p_full,
        "stride_detected": not stride_fallback,
        "shap_values": {
            "actor": {k: float(v) for k, v in shap_actor.items()},
            "zero": {k: float(v) for k, v in shap_zero.items()},
            "mean": {k: float(v) for k, v in shap_mean.items()},
            "marginal": {k: float(v) for k, v in shap_marginal.items()},
        },
        "faithfulness": {
            "actor": metrics_actor,
            "zero": metrics_zero,
            "mean": metrics_mean,
            "marginal": metrics_marginal,
        },
        "temporal_shap_values": {
            "actor": {k: float(v) for k, v in t_shap_actor.items()},
            "zero": {k: float(v) for k, v in t_shap_zero.items()},
            "mean": {k: float(v) for k, v in t_shap_mean.items()},
            "marginal": {k: float(v) for k, v in t_shap_marginal.items()},
        },
        "temporal_faithfulness": temporal_faithfulness,
        "rank_stability": rank_corr,
        "generative": {
            "masked_rmse": float(rmse),
            "completion_diversity": float(diversity),
        },
    }


def build_eval_coalition(device: torch.device, n_held_out: int = 4) -> torch.Tensor:
    """Return a (1, 17) coalition mask with the first ``n_held_out`` joints masked.

    Used consistently across all sequences for the generative quality metrics
    so that RMSE is computed over the same anatomical region for every sample.
    By default masks joints 0–3 (Pelvis, R_Hip, L_Hip, Spine).
    """
    cm = torch.ones(1, 17, dtype=torch.bool, device=device)
    cm[0, :n_held_out] = False
    return cm


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_results(per_seq: list[dict]) -> dict:
    """Compute mean ± std of scalar metrics across sequences.

    Keys use the same ``spatial/…`` and ``temporal/…`` prefixes as
    ``evaluate_shap_baselines.py`` for direct comparison.
    """
    agg: dict[str, list[float]] = defaultdict(list)

    for r in per_seq:
        agg["p_full"].append(r["p_full"])
        agg["stride_detected"].append(float(r.get("stride_detected", True)))
        agg["masked_rmse"].append(r["generative"]["masked_rmse"])
        agg["completion_diversity"].append(r["generative"]["completion_diversity"])
        agg["rank_corr_mean"].append(r["rank_stability"]["mean_rank_corr"])
        agg["rank_corr_std"].append(r["rank_stability"]["std_rank_corr"])

        for method in ("actor", "zero", "mean", "marginal"):
            f = r["faithfulness"][method]
            prefix = f"spatial/{method}"
            agg[f"{prefix}/deletion_auc"].append(f["deletion_auc"])
            agg[f"{prefix}/insertion_auc"].append(f["insertion_auc"])
            agg[f"{prefix}/random_deletion_auc"].append(f["random_deletion_auc"])
            agg[f"{prefix}/random_insertion_auc"].append(f["random_insertion_auc"])
            agg[f"{prefix}/completeness_error"].append(f["completeness_error"])
            for k, vals in f["pgi_pgu"].items():
                agg[f"{prefix}/pgi@{k}"].append(vals["pgi"])
                agg[f"{prefix}/pgu@{k}"].append(vals["pgu"])
            for k, vals in f["pgi_pgu_rand"].items():
                agg[f"{prefix}/pgi_rand@{k}"].append(vals["pgi"])
                agg[f"{prefix}/pgu_rand@{k}"].append(vals["pgu"])

        for method in ("actor", "zero", "mean", "marginal"):
            tf = r.get("temporal_faithfulness", {}).get(method, {})
            prefix = f"temporal/{method}"
            for key in ("deletion_auc", "insertion_auc", "completeness_error"):
                if key in tf:
                    agg[f"{prefix}/{key}"].append(tf[key])

    return {
        key: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for key, vals in agg.items()
    }


# ---------------------------------------------------------------------------
# FID over full test set
# ---------------------------------------------------------------------------

def compute_test_fid(
    actor_shap: ActorSHAP,
    test_batches: list[tuple],
    device: torch.device,
    n_completions: int = 10,
) -> float:
    """Compute FID between real test sequences and ActorSHAP completions.

    Uses a fixed coalition (first 4 joints masked) to generate completions,
    then compares feature distributions in the frozen full-encoder latent space.

    Returns float FID (lower = better).
    """
    real_seqs, y_list, mask_list = [], [], []
    gen_seqs, gen_y, gen_mask = [], [], []

    eval_coalition = build_eval_coalition(device)

    for x, y, mask, lengths in test_batches:
        real_seqs.append(x)
        y_list.append(y)
        mask_list.append(mask)
        completions = actor_shap.sample_completions(
            x, y, mask, lengths, eval_coalition, n_samples=n_completions
        )
        for x_hat in completions:
            gen_seqs.append(x_hat)
            gen_y.append(y)
            gen_mask.append(mask)

    real_feats = extract_encoder_features(actor_shap, real_seqs, y_list, mask_list)
    gen_feats = extract_encoder_features(actor_shap, gen_seqs, gen_y, gen_mask)
    return float(compute_fid(real_feats, gen_feats))


def collect_test_batches_for_fid(
    backbone_params: dict,
    fold: int,
    device: torch.device,
    max_test_sequences: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build ``(x, y, mask, lengths)`` tensors for indices ``0 .. n_cap-1`` (for FID)."""
    data_args = _raw_data_args(backbone_params, fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    n_full = len(test_ds)
    n_cap = n_full if max_test_sequences is None else min(n_full, max_test_sequences)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for seq_idx, raw_batch in enumerate(loader):
        if seq_idx >= n_cap:
            break
        x_raw, labels, _, _, pad_mask = raw_batch
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params["num_classes"], device, y=labels,
        )
        out.append(
            (batch["x"], batch["y"], batch["mask"], batch["lengths"]),
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP evaluation for CARE-PD pretrained classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--actor_shap_ckpt", required=True,
        help="Path to trained ActorSHAP Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--actor_shap_config", default=None,
        help="Optional path to the run's config.json (architecture hyperparameters). "
             "Default: config.json in the same directory as --actor_shap_ckpt.",
    )
    parser.add_argument(
        "--actor_shap_cache_dir", default=None,
        help="Directory of pre-generated completion NPZ files (from "
             "cache_actor_shap_completions.py).  When provided, KernelSHAP "
             "coalitions are served from cache (no ActorSHAP GPU calls); "
             "faithfulness-metric coalitions (~50/seq) still use the real model.",
    )
    parser.add_argument(
        "--backbone", required=True,
        help="Backbone name matching the classifier, e.g. 'potr'.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Config filename inside configs/<backbone>/, e.g. 'BMCLab.json'.",
    )
    parser.add_argument(
        "--classifier_ckpt", required=True,
        help="Path to latest_epoch.pth.tr saved by run.py.",
    )
    parser.add_argument(
        "--fold", type=int, required=True,
        help="Fold index (1-indexed) matching the classifier checkpoint.",
    )
    parser.add_argument(
        "--num_folds", type=int, required=True,
        help="Total folds used during classifier training (23 for LOSOCV on BMCLab).",
    )
    parser.add_argument(
        "--results_root", default=None,
        help="Root directory for auto-structured results. Output path is derived as "
             "{results_root}/{dataset}/{backbone}/fold{fold}/. "
             "Mutually exclusive with --output_dir; one must be provided.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Explicit output directory (overrides --results_root). "
             "Single-job: per_sequence.jsonl + aggregate.json. "
             "With --num_shards>1: writes shards/ only; run merge_shap_eval.py after.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device.",
    )
    parser.add_argument(
        "--n_kernel_samples", type=int, default=3000,
        help="KernelSHAP coalition samples (higher = lower variance, slower).",
    )
    parser.add_argument(
        "--n_completion_samples", type=int, default=20,
        help="ActorSHAP stochastic completions averaged per coalition.",
    )
    parser.add_argument(
        "--n_rank_runs", type=int, default=5,
        help="Full spatial KernelSHAP re-runs for rank-stability (Spearman). "
             "Each run costs ~as much as one Actor spatial SHAP; default 5 is slow. "
             "Use 0 to skip (writes NaN for rank_corr).",
    )
    parser.add_argument(
        "--k_list", nargs="+", type=int, default=[1, 2, 3, 5],
        help="Values of k for PGI/PGU metrics.",
    )
    parser.add_argument(
        "--max_train_pool", type=int, default=2000,
        help="Max training sequences to hold in memory for the marginal baseline.",
    )
    parser.add_argument(
        "--max_test_sequences", type=int, default=None,
        help="Cap on number of test sequences (for quick debugging).",
    )
    parser.add_argument(
        "--train_pool_batch_size", type=int, default=64,
        help="DataLoader batch size when building the training pool.",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Capture frame-rate for stride detection (temporal SHAP).",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Split the (capped) test set into this many disjoint index blocks. "
             "If >1, write shards/ only; run merge_shap_eval.py afterward.",
    )
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="Which block to process: 0 .. num_shards-1.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard_id must satisfy 0 <= shard_id < num_shards")

    device = torch.device(args.device)

    cfg = {
        "n_kernel_samples": args.n_kernel_samples,
        "n_completion_samples": args.n_completion_samples,
        "n_rank_runs": args.n_rank_runs,
        "k_list": args.k_list,
        "fps": args.fps,
    }

    # ------------------------------------------------------------------
    # Load backbone params (same approach as evaluate_shap_baselines.py).
    # ------------------------------------------------------------------
    print("[1/5] Loading backbone params …")
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    backbone_name = backbone_params['backbone']

    # Resolve output directory (auto-derived or explicit).
    output_dir = resolve_output_dir(
        args.results_root, args.output_dir, backbone_params, args.fold,
    )
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("[2/5] Loading ActorSHAP and CARE-PD classifier …")
    actor_shap = load_actor_shap(
        args.actor_shap_ckpt, device, config_path=args.actor_shap_config,
    )
    if args.actor_shap_cache_dir:
        print(f"  [cache] Wrapping ActorSHAP with CachedActorSHAP "
              f"(cache dir: {args.actor_shap_cache_dir})")
        actor_shap = CachedActorSHAP(actor_shap, args.actor_shap_cache_dir)

    motion_encoder = _load_motion_encoder_shared(
        args.classifier_ckpt, backbone_params, device,
    )

    # ------------------------------------------------------------------
    # Load data (raw root-centred 3D for both train pool and test set)
    # ------------------------------------------------------------------
    print("[3/5] Loading data …")
    train_pool, joint_means = build_train_pool(
        backbone_params, args.fold, device,
        max_sequences=args.max_train_pool,
        batch_size=args.train_pool_batch_size,
    )
    # Z-score stats: only needed for POTR (other backbones ignore them in
    # project_for_backbone).  Computed over the train+test union to match
    # POTRPreprocessor which normalises before fold splitting.
    if backbone_name == 'potr':
        print("  [potr] Computing z-score stats over train+test union …")
        zscore_mean, zscore_std = build_zscore_stats_for_potr(
            backbone_params, args.fold, device,
            batch_size=args.train_pool_batch_size,
        )
    else:
        zscore_mean = zscore_std = None

    data_args = _raw_data_args(backbone_params, args.fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    n_full = len(test_ds)
    max_cap = args.max_test_sequences
    n_cap = n_full if max_cap is None else min(n_full, max_cap)
    shard_mode = args.num_shards > 1

    start = end = 0
    shard_jsonl_path = ""
    if shard_mode:
        start = (args.shard_id * n_cap) // args.num_shards
        end = ((args.shard_id + 1) * n_cap) // args.num_shards
        shards_dir = os.path.join(output_dir, "shards")
        os.makedirs(shards_dir, exist_ok=True)
        shard_jsonl_path = os.path.join(
            shards_dir,
            f"per_sequence_shard{args.shard_id:04d}_of_{args.num_shards:04d}.jsonl",
        )
        shard_meta_path = os.path.join(
            shards_dir,
            f"shard_meta_shard{args.shard_id:04d}_of_{args.num_shards:04d}.json",
        )
        meta = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "index_range": [start, end],
            "n_test_effective": n_cap,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": {
                "actor_shap_ckpt": args.actor_shap_ckpt,
                "actor_shap_config": args.actor_shap_config,
                "backbone": args.backbone,
                "config": args.config,
                "classifier_ckpt": args.classifier_ckpt,
                "fold": args.fold,
                "num_folds": args.num_folds,
                "n_kernel_samples": args.n_kernel_samples,
                "n_completion_samples": args.n_completion_samples,
                "n_rank_runs": args.n_rank_runs,
                "k_list": args.k_list,
                "max_test_sequences": args.max_test_sequences,
                "max_train_pool": args.max_train_pool,
                "train_pool_batch_size": args.train_pool_batch_size,
                "fps": args.fps,
                "device": args.device,
            },
        }
        with open(shard_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if start >= end:
            print(
                f"[4/5] Shard {args.shard_id}: empty index range [{start}, {end}) — nothing to do.",
            )
            print(
                "When all shards finish, run: python merge_shap_eval.py --output_dir "
                f"{output_dir} …",
            )
            return

        eval_ds = torch.utils.data.Subset(test_ds, list(range(start, end)))
    else:
        eval_ds = test_ds

    test_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Per-sequence evaluation
    # ------------------------------------------------------------------
    print("[4/5] Running per-sequence SHAP evaluation …")
    if shard_mode:
        print(
            f"  shard {args.shard_id}/{args.num_shards}: global indices [{start}, {end}) "
            f"(n_cap={n_cap})",
        )
    per_seq_results: list[dict] = []
    test_batches_for_fid: list[tuple] = []

    for local_i, raw_batch in enumerate(test_loader):
        if shard_mode:
            seq_idx = start + local_i
        else:
            seq_idx = local_i
            if max_cap is not None and seq_idx >= max_cap:
                break

        x_raw, labels, _, _, pad_mask = raw_batch
        batch   = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params['num_classes'], device, y=labels,
        )
        x       = batch['x']        # (1, J, F, T) root-centred
        y       = batch['y']        # (1,)
        mask    = batch['mask']     # (1, T)
        lengths = batch['lengths']  # (1,)

        # Load cached completions for this sequence (if cache is active).
        if isinstance(actor_shap, CachedActorSHAP):
            actor_shap.load_for_seq(seq_idx)

        if not shard_mode:
            test_batches_for_fid.append((x, y, mask, lengths))

        result = evaluate_sequence(
            seq_idx, x, y, mask, lengths,
            actor_shap, motion_encoder, backbone_name, train_pool, joint_means,
            zscore_mean, zscore_std, cfg,
        )
        per_seq_results.append(result)

        out_path = (
            shard_jsonl_path
            if shard_mode
            else os.path.join(output_dir, "per_sequence.jsonl")
        )
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(
            f"  seq {seq_idx}: class={result['true_class']} "
            f"p_full={result['p_full']:.3f} "
            f"rmse={result['generative']['masked_rmse']:.4f} "
            f"sp_del(actor)={result['faithfulness']['actor']['deletion_auc']:.3f} "
            f"t_del(actor)={result['temporal_faithfulness']['actor']['deletion_auc']:.3f}"
        )

    if shard_mode:
        print(
            f"\nShard {args.shard_id} done: {len(per_seq_results)} sequences written to "
            f"{shard_jsonl_path}",
        )
        print(
            "FID and aggregate.json are skipped in shard mode. After all shards finish, run:\n"
            f"  python merge_shap_eval.py --output_dir {output_dir} "
            "--actor_shap_ckpt … --classifier_ckpt … (same flags as evaluate_shap.py)\n",
        )
        return

    # ------------------------------------------------------------------
    # FID over full test set
    # ------------------------------------------------------------------
    print("[5/5] Computing FID over test set …")
    fid = compute_test_fid(
        actor_shap, test_batches_for_fid, device,
        n_completions=args.n_completion_samples,
    )
    print(f"  FID = {fid:.4f}")

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    p_full_warning = check_p_full_warning(
        per_seq_results, backbone_params['num_classes'], backbone_name,
    )

    agg = aggregate_results(per_seq_results)
    agg["fid"] = {"mean": fid, "std": 0.0}
    agg["_meta"] = {
        "backbone": args.backbone,
        "dataset": backbone_params['dataset'],
        "config": args.config,
        "num_folds": args.num_folds,
        "fold": args.fold,
        "n_test_sequences": len(per_seq_results),
        "num_shards": 1,
        "shard_id": 0,
        "n_kernel_samples": args.n_kernel_samples,
        "n_completion_samples": args.n_completion_samples,
        "n_rank_runs": args.n_rank_runs,
        "k_list": args.k_list,
        "fps": args.fps,
        "actor_shap_ckpt": args.actor_shap_ckpt,
        "actor_shap_config": args.actor_shap_config,
        "classifier_ckpt": args.classifier_ckpt,
        "output_dir": output_dir,
        "p_full_warning": p_full_warning,
    }
    agg_path = os.path.join(output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\nDone. Results written to {output_dir}/")
    print(f"  per_sequence.jsonl  — {len(per_seq_results)} sequences")
    print(f"  aggregate.json      — means & stds for all metrics")
    print(f"  backbone={args.backbone}  dataset={backbone_params['dataset']}  fold={args.fold}")

    def _get(k: str) -> tuple[float, float]:
        e = agg.get(k, {})
        return e.get("mean", float("nan")), e.get("std", float("nan"))

    stride_rate = agg.get("stride_detected", {}).get("mean", float("nan"))
    print(f"\n--- Aggregate summary ({len(per_seq_results)} seqs) ---")
    print(f"  Stride auto-detected in {stride_rate * 100:.1f}% of sequences\n")

    print("  Spatial SHAP (KernelSHAP, 17 joints):")
    print(
        f'  {"Method":<10}  {"Del-AUC":>10}  {"Ins-AUC":>10}  {"PGI@1":>8}  '
        f'{"PGU@1":>8}  {"Comp.Err":>10}'
    )
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*10}')
    for method in ("actor", "zero", "mean", "marginal"):
        dm, ds = _get(f"spatial/{method}/deletion_auc")
        im, is_ = _get(f"spatial/{method}/insertion_auc")
        pm, _ = _get(f"spatial/{method}/pgi@1")
        um, _ = _get(f"spatial/{method}/pgu@1")
        cm, cs = _get(f"spatial/{method}/completeness_error")
        print(
            f"  {method:<10}  {dm:6.3f}±{ds:.3f}  {im:6.3f}±{is_:.3f}"
            f"  {pm:8.3f}  {um:8.3f}  {cm:6.3f}±{cs:.3f}",
        )

    print("\n  Temporal SHAP (exact, K=4 stride windows):")
    print(f'  {"Method":<10}  {"Del-AUC":>10}  {"Ins-AUC":>10}  {"Comp.Err":>10}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')
    for method in ("actor", "zero", "mean", "marginal"):
        dm, ds = _get(f"temporal/{method}/deletion_auc")
        im, is_ = _get(f"temporal/{method}/insertion_auc")
        cm, cs = _get(f"temporal/{method}/completeness_error")
        print(
            f"  {method:<10}  {dm:6.3f}±{ds:.3f}  {im:6.3f}±{is_:.3f}"
            f"  {cm:6.3f}±{cs:.3f}",
        )

    print(f"\n  FID = {fid:.4f}")
    print(f"  Rank corr = {agg.get('rank_corr_mean', {}).get('mean', float('nan')):.3f}")


if __name__ == "__main__":
    main()
