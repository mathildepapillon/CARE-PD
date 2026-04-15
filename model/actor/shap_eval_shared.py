"""shap_eval_shared.py — Shared utilities for evaluate_shap.py and evaluate_shap_baselines.py.

All functions here were previously duplicated verbatim in both evaluation
scripts.  Import from this module instead of defining them locally.
"""

from __future__ import annotations

import importlib
import os
from argparse import Namespace
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from const import const
from data.dataloaders import collate_fn
from model.actor.backbone_projection import compute_zscore_stats, project_for_backbone
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.motion_utils import unroot_to_global
from model.actor.shap_metrics import _class_prob
from model.backbone_loader import load_pretrained_backbone, load_pretrained_weights
from model.motion_encoder import MotionEncoder


# ---------------------------------------------------------------------------
# Backbone config helpers
# ---------------------------------------------------------------------------

_PARAM_SEED_DEFAULTS: dict = {
    'train_mode':         'classifier_only',
    'seed':               0,
    'tune_fresh':         1,
    'ntrials':            1,
    'this_run_num':       '0',
    'readstudyfrom':      None,
    'hypertune':          0,
    'just_gen_dataset':   0,
    'cross_dataset_test': 0,
    'pretrained':         0,
    'overwrite_results':  0,
    'force_LODO':         0,
    'AID':                0,
    'combine_views_preds': 0,
    'views_path':         None,
    'exp_name_rigid':     None,
    'prefer_right':       0,
    'medication':         0,
    'metadata':           [],
    'tuned_model_config': None,
}

_BACKBONE_CONFIG_MODULE = {
    'potr':           'configs.generate_config_potr',
    'motionbert':     'configs.generate_config_motionbert',
    'motionagformer': 'configs.generate_config_motionagformer',
    'poseformerv2':   'configs.generate_config_poseformerv2',
    'mixste':         'configs.generate_config_mixste',
    'momask':         'configs.generate_config_momask',
    'motionclip':     'configs.generate_config_motionclip',
}


def _load_backbone_params(backbone: str, config_file: str, num_folds: int) -> dict:
    """Regenerate full param dict for *backbone* from its config JSON."""
    mod = importlib.import_module(_BACKBONE_CONFIG_MODULE[backbone])
    seed = {**_PARAM_SEED_DEFAULTS, 'backbone': backbone, 'config': config_file}
    params, _ = mod.generate_config(seed, config_file)
    params['num_folds']   = num_folds
    params['num_classes'] = const.NUM_CLASSES_PER_DATASET[params['dataset']]
    params['LODO']        = False
    params.setdefault('classifier_hidden_dims', [])
    params.setdefault('classifier_dropout', 0.0)
    return params


# ---------------------------------------------------------------------------
# Raw H36M data helpers
# ---------------------------------------------------------------------------

def _raw_data_args(backbone_params: dict, fold: int, batch_size: int = 1) -> Namespace:
    """Build a Namespace for get_carepd_datasets from backbone params.

    Loads raw (un-normalised) H36M world-space 3D data for ALL backbones.
    The sequence length matches the backbone's own ``source_seq_len`` so that
    SHAP sequences are temporally identical to the sequences the classifier was
    trained on.  A T-specific experiment_name is used so that
    ``dataset_factory`` creates (or reuses) a separate raw-3D pkl per
    sequence length:

        motionclip_processing/ShapRaw3D_T80/BMCLab/23fold/   ← POTR
        motionclip_processing/ShapRaw3D_T90/BMCLab/23fold/   ← MotionBERT
        motionclip_processing/ShapRaw3D_T81/BMCLab/23fold/   ← MixSTE/etc.

    If the pkl does not yet exist, ``dataset_factory`` creates it automatically
    by running MotionCLIPPreprocessor on the preprocessed world-space 3D npz —
    selecting only clips whose raw length ≥ source_seq_len and taking the first
    source_seq_len frames.  This matches how the backbone-specific dataloaders
    filter clips (e.g. the motionbert pkl has 2 810 clips of 90 frames while
    the potr pkl has 2 857 clips of 80 frames).

    Root-centring is applied by actor_batch_from_carepd; per-backbone
    normalisation is applied inside build_classifier_fn via project_for_backbone.
    """
    seq_len = backbone_params.get('source_seq_len', 80)
    exp_name = f"ShapRaw3D_T{seq_len}"
    return Namespace(
        dataset=backbone_params['dataset'],
        num_folds=backbone_params['num_folds'],
        batch_size=batch_size,
        experiment_name=exp_name,
        fold=fold,
        source_seq_len=seq_len,
        carepd_pose_npz=None,
        carepd_labels_pkl=None,
    )


def raw_seq_len(backbone_params: dict) -> int:
    """Return the raw 3D sequence length used by SHAP for a given backbone.

    Convenience accessor so callers (shell scripts, caching scripts) can query
    the expected T without re-implementing the logic in _raw_data_args.
    """
    return backbone_params.get('source_seq_len', 80)


def build_train_pool(
    backbone_params: dict,
    fold: int,
    device: torch.device,
    max_sequences: int = 2000,
    batch_size: int = 64,
    root_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract training sequences in raw 3D for baselines.

    Args:
        root_centered: if True, subtract joint-0 (pelvis) so sequences are
                       root-centred; otherwise keep absolute world coordinates.

    Returns:
        train_pool:  ``(N, J=17, F=3, T)`` float32 on CPU.
        joint_means: ``(J=17, F=3)`` per-joint training mean on *device*.
    """
    data_args = _raw_data_args(backbone_params, fold, batch_size=batch_size)
    train_ds, _ = get_carepd_datasets(data_args)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    seqs: list[torch.Tensor] = []
    for x_raw, labels, _, _, pad_mask in loader:
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params['num_classes'],
            device=torch.device('cpu'), y=labels, root_centered=root_centered,
        )
        seqs.append(batch['x'])   # (B, J, F, T)
        if sum(s.shape[0] for s in seqs) >= max_sequences:
            break
    train_pool  = torch.cat(seqs, dim=0)[:max_sequences]    # (N, J, F, T)
    joint_means = train_pool.mean(dim=[0, 3]).to(device)    # (J, F)
    return train_pool, joint_means


def build_zscore_stats_for_potr(
    backbone_params: dict,
    fold: int,
    device: torch.device,
    batch_size: int = 64,
    root_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute POTR z-score stats from the train+test union for a given fold.

    ``POTRPreprocessor`` computes normalisation statistics over **all**
    sequences in the dataset before fold splitting (train + test combined).
    This function replicates that behaviour by loading both splits, concatenating
    them, and calling ``compute_zscore_stats`` on the union.

    This is only meaningful for POTR (the stats are ignored for other backbones).

    Args:
        backbone_params: Full param dict from ``_load_backbone_params``.
        fold:            Fold index (1-indexed).
        device:          Target device for the returned tensors.
        batch_size:      DataLoader batch size for loading sequences.

    Returns:
        zscore_mean: ``(J, F)`` per-joint mean on *device*.
        zscore_std:  ``(J, F)`` per-joint std  on *device*.
    """
    data_args = _raw_data_args(backbone_params, fold, batch_size=batch_size)
    train_ds, test_ds = get_carepd_datasets(data_args)

    seqs: list[torch.Tensor] = []
    for ds in (train_ds, test_ds):
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn,
        )
        for x_raw, labels, _, _, pad_mask in loader:
            batch = actor_batch_from_carepd(
                x_raw.float(), pad_mask, backbone_params['num_classes'],
                device=torch.device('cpu'), y=labels, root_centered=root_centered,
            )
            seqs.append(batch['x'])   # (B, J, F, T)

    all_seqs = torch.cat(seqs, dim=0)   # (N_total, J, F, T)
    mean, std = compute_zscore_stats(all_seqs)
    return mean.to(device), std.to(device)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_motion_encoder(
    classifier_ckpt: str,
    backbone_params: dict,
    device: torch.device,
) -> MotionEncoder:
    """Load a CARE-PD MotionEncoder from a checkpoint saved by run.py."""
    backbone = load_pretrained_backbone(backbone_params, backbone_params['backbone'])
    model = MotionEncoder(
        backbone=backbone,
        params=backbone_params,
        num_classes=backbone_params['num_classes'],
        train_mode=backbone_params.get('train_mode', 'classifier_only'),
    )
    ckpt = torch.load(classifier_ckpt, map_location=device)
    weights = ckpt['model'] if 'model' in ckpt else ckpt
    load_pretrained_weights(model, checkpoint=weights)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Classifier wrapper (projection-aware)
# ---------------------------------------------------------------------------

def build_classifier_fn(
    motion_encoder: MotionEncoder,
    mask: torch.Tensor,
    backbone_name: str,
    zscore_mean: Optional[torch.Tensor] = None,
    zscore_std: Optional[torch.Tensor] = None,
    x_orig: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap MotionEncoder as ``f(x: (1,J,F,T)) → logits (1,C)``.

    Converts global-pelvis format ActorSHAP output to full global 3D via
    ``unroot_to_global``, then projects into the backbone's input space via
    ``project_for_backbone``, ensuring the backbone always receives
    in-distribution, correctly normalised input.

    Args:
        motion_encoder: Pretrained CARE-PD classifier.
        mask:           ``(1, T)`` valid-frame bool mask (closed over per sequence).
        backbone_name:  Backbone identifier for projection dispatch.
        zscore_mean:    ``(J, F)`` z-score mean — required for POTR.
        zscore_std:     ``(J, F)`` z-score std  — required for POTR.
        x_orig:         ``(1, J, F, T)`` original (unmasked) sequence in
                        global-pelvis format.  When provided, joint 0 (the
                        global pelvis trajectory) is restored from this
                        reference before calling ``unroot_to_global``.  This
                        ensures that SHAP coalition masking affects only
                        relative joint poses, not the global walking
                        translation.  Without this, zeroed / mean-imputed
                        frames land at world origin which after perspective
                        projection maps to screen coordinates the classifier
                        never saw during training, causing near-chance
                        ``p_full`` for 2-D backbones (MixSTE, PoseFormerV2)
                        and distorted crop-scale bounding boxes for
                        MotionBERT / MotionAGFormer.  POTR is unaffected
                        because its projection subtracts joint-0 internally.
    """
    @torch.no_grad()
    def _fn(x_actor: torch.Tensor) -> torch.Tensor:
        B = x_actor.shape[0]
        # Expand (1, T) mask to (B, T) so batched classifier calls work correctly.
        pad_mask_b = mask.expand(B, -1) if B > 1 else mask

        # Restore the original pelvis trajectory so coalition masking only
        # perturbs relative joint poses, not the global walking path.
        if x_orig is not None:
            x_actor = x_actor.clone()
            pelvis_orig = x_orig[:, 0:1, :, :]                    # (1, 1, F, T)
            x_actor[:, 0:1, :, :] = pelvis_orig.expand(B, -1, -1, -1)

        # Recover full global 3D from global-pelvis ACTOR format before projection.
        # x_actor: (B, J, F, T) → permute to (B, T, J, F) → unroot → permute back.
        x_btjf = x_actor.permute(0, 3, 1, 2)          # (B, T, J, F)
        x_global = unroot_to_global(x_btjf)            # (B, T, J, F) all world-space
        x_global_actor = x_global.permute(0, 2, 3, 1)  # (B, J, F, T)
        x_proj = project_for_backbone(
            x_global_actor, backbone_name,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std,
            pad_mask=pad_mask_b,
        )
        metadata = torch.zeros(B, 0, device=x_proj.device)
        return motion_encoder(x_proj, metadata, valid_mask=pad_mask_b)

    return _fn


# ---------------------------------------------------------------------------
# Temporal faithfulness helpers
# ---------------------------------------------------------------------------

def _apply_temporal_mask(
    x: torch.Tensor,
    masked_window_indices: list[int],
    window_assignments: list[list[int]],
    method: str,
    joint_means: Optional[torch.Tensor] = None,
    train_pool: Optional[torch.Tensor] = None,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Return a copy of x with listed windows replaced by baseline values."""
    device = x.device
    x_m = x.clone()
    if not masked_window_indices:
        return x_m
    frames = torch.tensor(
        [t for k in masked_window_indices for t in window_assignments[k]],
        dtype=torch.long,
        device=device,
    )
    if method == "zero":
        x_m[0, :, :, frames] = 0.0
    elif method == "mean":
        x_m[0, :, :, frames] = (
            joint_means.unsqueeze(-1).expand(-1, -1, len(frames)).to(device)
        )
    else:  # marginal
        if rng is None:
            rng = np.random.default_rng(0)
        d = int(rng.integers(0, train_pool.shape[0]))
        x_m[0, :, :, frames] = train_pool[d, :, :, frames.cpu()].to(device)
    return x_m


def _temporal_deletion_insertion_auc(
    classifier_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    window_assignments: list[list[int]],
    temporal_shap_vals: dict,
    method: str,
    joint_means: Optional[torch.Tensor] = None,
    train_pool: Optional[torch.Tensor] = None,
    seed: int = 0,
    n_marginal_samples: int = 20,
) -> dict:
    """Temporal deletion / insertion AUC for zero / mean / marginal baselines.

    Deletion: start with full sequence, progressively remove windows in
    decreasing SHAP-value order.  AUC = area under the (K+1)-point curve.
    Insertion: start with all windows masked, progressively reveal in same order.

    For method="marginal", each masked configuration averages over
    n_marginal_samples donor draws, matching how compute_temporal_shap_baseline
    evaluates value functions (avoids high-variance single-sample estimates).
    """
    K = len(window_assignments)
    class_idx = int(y[0].item())
    rng = np.random.default_rng(seed)

    # Filter out metadata keys (_v_empty, _v_full) when sorting windows.
    window_names = [k for k in temporal_shap_vals if not k.startswith("_")]
    order = sorted(range(K), key=lambda k: -temporal_shap_vals[window_names[k]])

    def _eval(masked_idxs: list[int]) -> float:
        """Classifier probability with given windows masked, marginal-averaged if needed."""
        if not masked_idxs:
            return float(_class_prob(classifier_fn, x, class_idx))
        if method != "marginal":
            x_m = _apply_temporal_mask(
                x, masked_idxs, window_assignments, method,
                joint_means=joint_means, train_pool=train_pool, rng=rng,
            )
            return float(_class_prob(classifier_fn, x_m, class_idx))
        # Marginal: average predictions over n_marginal_samples independent donor draws
        # to match the multi-sample averaging in compute_temporal_shap_baseline.
        probs = []
        for _ in range(n_marginal_samples):
            x_m = _apply_temporal_mask(
                x, masked_idxs, window_assignments, "marginal",
                joint_means=joint_means, train_pool=train_pool, rng=rng,
            )
            probs.append(_class_prob(classifier_fn, x_m, class_idx))
        return float(np.mean(probs))

    del_curve = [float(_class_prob(classifier_fn, x, class_idx))]
    for step in range(K):
        del_curve.append(_eval(order[: step + 1]))

    all_masked = list(range(K))
    ins_curve = [_eval(all_masked)]
    for step in range(K):
        still_masked = order[step + 1:]
        ins_curve.append(_eval(still_masked))

    xs = np.linspace(0., 1., K + 1)
    return {
        "deletion_auc":  float(np.trapz(del_curve, xs)),
        "insertion_auc": float(np.trapz(ins_curve, xs)),
        "p_empty":       float(ins_curve[0]),
    }


# ---------------------------------------------------------------------------
# Batched temporal faithfulness
# ---------------------------------------------------------------------------

@torch.no_grad()
def _temporal_deletion_insertion_auc_batched(
    classifier_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    window_assignments: list[list[int]],
    temporal_shap_vals: dict,
    method: str,
    joint_means: Optional[torch.Tensor] = None,
    train_pool: Optional[torch.Tensor] = None,
    physics_completer: Optional[Any] = None,
    seed: int = 0,
    n_marginal_samples: int = 20,
    chunk_size: int = 256,
) -> dict:
    """Batched temporal deletion/insertion AUC for zero/mean/marginal/physics.

    Replaces ``_temporal_deletion_insertion_auc`` by building all
    perturbations upfront and classifying them in a single batched GPU call.
    Marginal method is ~20x faster since all donor-perturbed inputs are
    classified together instead of one at a time.
    """
    from model.actor.shap_compute import _classify_chunked
    from model.actor.shap_masking import build_temporal_shap_mask

    K = len(window_assignments)
    T = x.shape[-1]
    device = x.device
    class_idx = int(y[0].item())
    rng = np.random.default_rng(seed)

    window_names = [k for k in temporal_shap_vals if not k.startswith("_")]
    order = sorted(range(K), key=lambda k: -temporal_shap_vals[window_names[k]])

    # Deletion: step i masks the top-i windows.
    # Insertion: step i reveals the top-i windows (masks the rest).
    configs: list[list[int]] = []
    for step in range(K + 1):
        configs.append(order[:step])
    for step in range(K + 1):
        configs.append(order[step:])

    if method in ("zero", "mean"):
        perturbed: list[torch.Tensor] = []
        for masked_idxs in configs:
            x_p = x.clone()
            if masked_idxs:
                frames = torch.tensor(
                    [t for ki in masked_idxs for t in window_assignments[ki]],
                    dtype=torch.long, device=device,
                )
                if method == "zero":
                    x_p[0, :, :, frames] = 0.0
                else:
                    x_p[0, :, :, frames] = joint_means.unsqueeze(-1).expand(
                        -1, -1, len(frames),
                    ).to(device)
            perturbed.append(x_p)
        x_stacked = torch.cat(perturbed, dim=0)
        all_probs = _classify_chunked(
            classifier_fn, x_stacked, class_idx, chunk_size=chunk_size,
        )

    elif method == "marginal":
        N_pool = train_pool.shape[0]
        input_parts: list[torch.Tensor] = []
        meta: list[tuple[int, int]] = []

        for ci, masked_idxs in enumerate(configs):
            if not masked_idxs:
                input_parts.append(x)
                meta.append((ci, 1))
            else:
                frames = torch.tensor(
                    [t for ki in masked_idxs for t in window_assignments[ki]],
                    dtype=torch.long, device=device,
                )
                donor_idx = rng.integers(0, N_pool, size=n_marginal_samples)
                donors = train_pool[donor_idx].to(device)
                x_rep = x[0].unsqueeze(0).expand(
                    n_marginal_samples, -1, -1, -1,
                ).clone()
                x_rep[:, :, :, frames] = donors[:, :, :, frames]
                input_parts.append(x_rep)
                meta.append((ci, n_marginal_samples))

        x_all = torch.cat(input_parts, dim=0)
        raw_probs = _classify_chunked(
            classifier_fn, x_all, class_idx, chunk_size=chunk_size,
        )
        all_probs = np.zeros(len(configs))
        pos = 0
        for ci, n in meta:
            all_probs[ci] = raw_probs[pos : pos + n].mean()
            pos += n

    elif method == "physics":
        if physics_completer is None:
            raise ValueError("physics_completer is required for method='physics'")
        input_parts: list[torch.Tensor] = []
        meta_ph: list[tuple[int, int]] = []
        for ci, masked_idxs in enumerate(configs):
            if not masked_idxs:
                input_parts.append(x)
                meta_ph.append((ci, 1))
            else:
                observed_windows = [k for k in range(K) if k not in masked_idxs]
                cm = build_temporal_shap_mask(
                    observed_windows, window_assignments, T, device,
                ).unsqueeze(0)
                comps = physics_completer.sample_completions(
                    x, y, mask, lengths, cm, n_samples=n_marginal_samples,
                )
                x_rep = torch.cat(comps, dim=0)
                input_parts.append(x_rep)
                meta_ph.append((ci, n_marginal_samples))
        x_all = torch.cat(input_parts, dim=0)
        raw_probs = _classify_chunked(
            classifier_fn, x_all, class_idx, chunk_size=chunk_size,
        )
        all_probs = np.zeros(len(configs))
        pos = 0
        for ci, n in meta_ph:
            all_probs[ci] = raw_probs[pos : pos + n].mean()
            pos += n

    else:
        raise ValueError(
            f"Temporal batched method must be 'zero', 'mean', 'marginal', or "
            f"'physics'; got {method!r}"
        )

    del_curve = all_probs[: K + 1].tolist()
    ins_curve = all_probs[K + 1 :].tolist()
    xs_arr = np.linspace(0.0, 1.0, K + 1)
    return {
        "deletion_auc": float(np.trapz(del_curve, xs_arr)),
        "insertion_auc": float(np.trapz(ins_curve, xs_arr)),
        "p_empty": float(ins_curve[0]),
    }


# ---------------------------------------------------------------------------
# Output directory helper
# ---------------------------------------------------------------------------

def resolve_output_dir(
    results_root: Optional[str],
    output_dir: Optional[str],
    backbone_params: dict,
    fold: int,
) -> str:
    """Return the directory to write results into.

    If ``output_dir`` is given it is returned as-is (explicit override).
    Otherwise ``results_root`` is required and the path is auto-derived as::

        {results_root}/{dataset}/{backbone}/fold{fold}/

    Raises ``ValueError`` if neither argument is provided.
    """
    if output_dir:
        return output_dir
    if not results_root:
        raise ValueError(
            "Either --output_dir or --results_root must be provided."
        )
    dataset  = backbone_params['dataset']
    backbone = backbone_params['backbone']
    return os.path.join(results_root, dataset, backbone, f"fold{fold}")


# ---------------------------------------------------------------------------
# p_full sanity check
# ---------------------------------------------------------------------------

def check_p_full_warning(
    per_seq_results: list[dict],
    num_classes: int,
    backbone_name: str,
) -> bool:
    """Warn if mean p_full is near chance, suggesting the classifier is not
    handling root-centred Actor sequences well.

    Returns True if the warning was triggered, False otherwise.
    """
    if not per_seq_results:
        return False
    mean_p_full = float(np.mean([r["p_full"] for r in per_seq_results]))
    chance = 1.0 / num_classes
    triggered = mean_p_full < 2.0 * chance
    if triggered:
        print(
            f"\n*** WARNING: mean p_full={mean_p_full:.3f} is near chance "
            f"({chance:.3f}) for backbone '{backbone_name}'.  "
            "This suggests the classifier is not handling root-centred "
            "ActorSHAP sequences well (MotionBERT/MotionAGFormer/MixSTE/"
            "PoseFormerV2 were trained on non-root-centred data; the "
            "crop_scale bbox covers only local joint oscillation here).  "
            "Consider retraining ActorSHAP on absolute-motion data by using "
            "vel/rr loss as the primary signal in train_actor_cvae.py. ***\n",
            flush=True,
        )
    return triggered


