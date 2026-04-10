"""evaluate_shap_baselines.py — SHAP evaluation with zero / mean / marginal baselines.

No ActorSHAP checkpoint required.  Results are saved in a format compatible
with evaluate_shap.py (same per_sequence.jsonl / aggregate.json structure)
so they can be compared directly once ActorSHAP training completes.

DATA PIPELINE
=============
Both scripts (this one and evaluate_shap.py) load raw root-centred 3D H36M
sequences via the ActorCVAE data pipeline (``get_carepd_datasets``), then
apply a per-backbone projection inside ``build_classifier_fn`` using
``model.actor.backbone_projection.project_for_backbone``.

This guarantees that:
  - Zero / mean / marginal baselines and ActorSHAP completions all live in
    the same raw 3D space.
  - The coalition mask is applied in the same space for every method.
  - The classifier always receives in-distribution, correctly normalised
    input regardless of the backbone (POTR z-score, MotionBERT crop_scale,
    MixSTE screen-normalise, etc.).

COALITION REPRODUCIBILITY
=========================
Every method uses ``seed=seq_idx`` for KernelSHAP coalition sampling — the
same convention as evaluate_shap.py.  Running both scripts on the same fold
therefore generates identical coalition sets, enabling direct comparison of
SHAP values and faithfulness metrics across all methods.

USAGE
=====
    python evaluate_shap_baselines.py \\
        --backbone potr \\
        --config BMCLab.json \\
        --num_folds 23 \\
        --fold 1 \\
        --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \\
        --output_dir results/shap_baselines_bmclab_fold1 \\
        --device cuda:4

    # Quick smoke-test on 5 sequences:
    python evaluate_shap_baselines.py ... --max_test_sequences 5 --n_kernel_samples 200
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argparse import Namespace

from const import const
from data.dataloaders import collate_fn
from model.actor.backbone_projection import compute_zscore_stats, project_for_backbone
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.shap_compute import (
    compute_spatial_shap_baseline,
    compute_temporal_shap_baseline,
)
from model.actor.shap_masking import build_temporal_windows, detect_stride_period
from model.actor.shap_metrics import (
    _class_prob,
    compute_deletion_insertion_auc,
    compute_pgi_pgu,
    compute_pgi_pgu_random,
    compute_shapley_completeness,
    make_marginal_impute_fn,
    make_mean_impute_fn,
    make_zero_impute_fn,
)
from model.backbone_loader import load_pretrained_backbone, load_pretrained_weights
from model.motion_encoder import MotionEncoder


# ---------------------------------------------------------------------------
# Backbone param loading (for classifier instantiation only)
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
    """Regenerate full param dict for *backbone* — used to instantiate MotionEncoder."""
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
# Raw H36M data loading (shared with evaluate_shap.py)
# ---------------------------------------------------------------------------

def _raw_data_args(backbone_params: dict, fold: int, batch_size: int = 1) -> Namespace:
    """Build an argparse.Namespace for get_carepd_datasets from backbone params.

    We load raw (un-normalised, un-centred) H36M data via the ActorCVAE
    pipeline for ALL backbones.  Root-centering is applied by
    actor_batch_from_carepd; per-backbone normalisation is applied inside
    build_classifier_fn via project_for_backbone.

    The source_seq_len is taken from the backbone's own config so that clips
    have the same length as those the classifier was trained on.
    """
    return Namespace(
        dataset=backbone_params['dataset'],
        num_folds=backbone_params['num_folds'],
        batch_size=batch_size,
        experiment_name=backbone_params.get('experiment_name', 'Hypertune'),
        fold=fold,
        source_seq_len=backbone_params.get('source_seq_len', 80),
        carepd_pose_npz=None,
        carepd_labels_pkl=None,
    )


def build_train_pool(
    backbone_params: dict,
    fold: int,
    device: torch.device,
    max_sequences: int = 2000,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract training sequences in raw root-centred 3D for baselines.

    Returns:
        train_pool:  ``(N, J=17, F=3, T)`` float32, root-centred, on CPU.
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
            device=torch.device('cpu'), y=labels,
        )
        seqs.append(batch['x'])   # (B, J, F, T) root-centred
        if sum(s.shape[0] for s in seqs) >= max_sequences:
            break

    train_pool  = torch.cat(seqs, dim=0)[:max_sequences]    # (N, J, F, T)
    joint_means = train_pool.mean(dim=[0, 3]).to(device)    # (J, F)
    return train_pool, joint_means


# ---------------------------------------------------------------------------
# Classifier loading
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
# Classifier wrapper with per-backbone projection
# ---------------------------------------------------------------------------

def build_classifier_fn(
    motion_encoder: MotionEncoder,
    mask: torch.Tensor,
    backbone_name: str,
    zscore_mean: Optional[torch.Tensor] = None,
    zscore_std: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap MotionEncoder as ``f(x: (1,J,F,T)) → logits (1,C)``.

    Applies ``project_for_backbone`` to convert raw root-centred 3D input
    into the correct normalised format for *backbone_name* before passing
    to the MotionEncoder.

    Args:
        motion_encoder: Pretrained CARE-PD classifier.
        mask:           ``(1, T)`` valid-frame bool mask (closed over per sequence).
        backbone_name:  Backbone identifier for projection dispatch.
        zscore_mean:    ``(J, F)`` z-score mean — required for POTR.
        zscore_std:     ``(J, F)`` z-score std  — required for POTR.
    """
    @torch.no_grad()
    def _fn(x_actor: torch.Tensor) -> torch.Tensor:
        x_proj = project_for_backbone(
            x_actor, backbone_name,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std,
            pad_mask=mask,
        )
        metadata = torch.zeros(x_proj.shape[0], 0, device=x_proj.device)
        return motion_encoder(x_proj, metadata, valid_mask=mask)

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
    rng: "np.random.Generator | None" = None,
) -> torch.Tensor:
    """Return a copy of x with the specified windows replaced by baseline values."""
    device = x.device
    x_m = x.clone()
    if not masked_window_indices:
        return x_m
    frames = torch.tensor(
        [t for k in masked_window_indices for t in window_assignments[k]],
        dtype=torch.long, device=device,
    )
    if method == "zero":
        x_m[0, :, :, frames] = 0.0
    elif method == "mean":
        x_m[0, :, :, frames] = joint_means.unsqueeze(-1).expand(-1, -1, len(frames)).to(device)
    else:  # marginal
        if rng is None:
            rng = np.random.default_rng(0)
        d = int(rng.integers(0, train_pool.shape[0]))
        # train_pool stays on CPU; index it with CPU indices (frames may be on CUDA for x_m).
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
) -> dict:
    """Temporal deletion and insertion AUC over K=4 windows.

    Deletion: start with full sequence, progressively remove windows in
    decreasing SHAP-value order.  AUC = area under the 5-point curve.
    Insertion: start with all windows masked, progressively reveal in same order.
    """
    K = len(window_assignments)
    class_idx = int(y[0].item())
    rng = np.random.default_rng(seed)

    # Sort windows by decreasing SHAP value (most important first).
    window_names = list(temporal_shap_vals.keys())
    order = sorted(range(K), key=lambda k: -temporal_shap_vals[window_names[k]])

    def _pred(x_in):
        return _class_prob(classifier_fn, x_in, class_idx)

    # Deletion curve: remove windows one by one.
    del_curve = [_pred(x)]
    for step in range(K):
        x_m = _apply_temporal_mask(
            x, order[: step + 1], window_assignments, method,
            joint_means=joint_means, train_pool=train_pool, rng=rng,
        )
        del_curve.append(_pred(x_m))

    # Insertion curve: start all-masked, reveal windows one by one.
    x_empty = _apply_temporal_mask(
        x, list(range(K)), window_assignments, method,
        joint_means=joint_means, train_pool=train_pool, rng=rng,
    )
    ins_curve = [_pred(x_empty)]
    for step in range(K):
        still_masked = order[step + 1 :]
        x_m = _apply_temporal_mask(
            x, still_masked, window_assignments, method,
            joint_means=joint_means, train_pool=train_pool, rng=rng,
        )
        ins_curve.append(_pred(x_m))

    xs = np.linspace(0., 1., K + 1)
    return {
        "deletion_auc":  float(np.trapz(del_curve, xs)),
        "insertion_auc": float(np.trapz(ins_curve, xs)),
        "p_empty":       float(ins_curve[0]),
    }


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def _faithfulness_block(
    classifier_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    shap_vals: dict,
    impute_fn: Callable,
    p_full: float,
    p_ref: float,
    k_list: tuple[int, ...],
    seq_idx: int,
) -> dict:
    pgi_pgu = compute_pgi_pgu(
        classifier_fn, x, y, mask, lengths, shap_vals, impute_fn, k_list=k_list,
    )
    pgi_pgu_rand = compute_pgi_pgu_random(
        classifier_fn, x, y, mask, lengths, impute_fn, k_list=k_list, seed=seq_idx,
    )
    del_ins = compute_deletion_insertion_auc(
        classifier_fn, x, y, mask, lengths, shap_vals, impute_fn, seed=seq_idx,
    )
    comp_err = compute_shapley_completeness(shap_vals, p_full, p_ref)
    return {
        'pgi_pgu':              {str(k): v for k, v in pgi_pgu.items()},
        'pgi_pgu_rand':         {str(k): v for k, v in pgi_pgu_rand.items()},
        'deletion_auc':         del_ins['deletion_auc'],
        'insertion_auc':        del_ins['insertion_auc'],
        'random_deletion_auc':  del_ins['random_deletion_auc'],
        'random_insertion_auc': del_ins['random_insertion_auc'],
        'completeness_error':   comp_err,
    }


def evaluate_sequence(
    seq_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    motion_encoder: MotionEncoder,
    backbone_name: str,
    train_pool: torch.Tensor,
    joint_means: torch.Tensor,
    zscore_mean: Optional[torch.Tensor],
    zscore_std: Optional[torch.Tensor],
    cfg: dict,
) -> dict:
    """Run spatial and temporal SHAP (all three baselines) for one sequence."""
    device        = x.device
    n_kernel      = cfg.get('n_kernel_samples', 3000)
    n_completions = cfg.get('n_completion_samples', 20)
    k_list        = tuple(cfg.get('k_list', [1, 2, 3, 5]))
    fps           = cfg.get('fps', 30)

    classifier_fn = build_classifier_fn(
        motion_encoder, mask, backbone_name,
        zscore_mean=zscore_mean, zscore_std=zscore_std,
    )

    impute_zero     = make_zero_impute_fn(classifier_fn)
    impute_mean     = make_mean_impute_fn(classifier_fn, joint_means)
    impute_marginal = make_marginal_impute_fn(
        classifier_fn, train_pool, n_samples=n_completions, seed=seq_idx,
    )

    # ------------------------------------------------------------------
    # Spatial SHAP (KernelSHAP, ~3000 coalitions over 17 joints)
    # ------------------------------------------------------------------
    print(f'  [seq {seq_idx}] spatial SHAP (zero / mean / marginal) …')
    shap_zero = compute_spatial_shap_baseline(
        'zero', classifier_fn, x, y, mask, lengths,
        n_kernel_samples=n_kernel, seed=seq_idx,
    )
    shap_mean = compute_spatial_shap_baseline(
        'mean', classifier_fn, x, y, mask, lengths,
        joint_means=joint_means, n_kernel_samples=n_kernel, seed=seq_idx,
    )
    shap_marginal = compute_spatial_shap_baseline(
        'marginal', classifier_fn, x, y, mask, lengths,
        train_pool=train_pool, n_kernel_samples=n_kernel,
        n_marginal_samples=n_completions, seed=seq_idx,
    )

    zero_cm    = torch.zeros(1, 17, dtype=torch.bool, device=device)
    p_full     = _class_prob(classifier_fn, x, int(y[0].item()))
    p_ref_zero = impute_zero(x, y, mask, lengths, zero_cm)
    p_ref_mean = impute_mean(x, y, mask, lengths, zero_cm)
    p_ref_marg = impute_marginal(x, y, mask, lengths, zero_cm)

    print(f'  [seq {seq_idx}] spatial faithfulness metrics …')
    metrics_zero = _faithfulness_block(
        classifier_fn, x, y, mask, lengths,
        shap_zero, impute_zero, p_full, p_ref_zero, k_list, seq_idx,
    )
    metrics_mean = _faithfulness_block(
        classifier_fn, x, y, mask, lengths,
        shap_mean, impute_mean, p_full, p_ref_mean, k_list, seq_idx,
    )
    metrics_marginal = _faithfulness_block(
        classifier_fn, x, y, mask, lengths,
        shap_marginal, impute_marginal, p_full, p_ref_marg, k_list, seq_idx,
    )

    # ------------------------------------------------------------------
    # Temporal SHAP (exact enumeration, K=4 stride-aligned windows)
    # Stride period auto-detected from the foot's forward displacement.
    # All methods share the same window boundaries for direct comparability.
    # ------------------------------------------------------------------
    print(f'  [seq {seq_idx}] temporal SHAP (K=4 stride windows, exact) …')
    x_np = x[0].permute(2, 0, 1).cpu().numpy()   # (T, J, F)
    stride_period, fallback = detect_stride_period(x_np, fps=fps)
    window_assignments = build_temporal_windows(x.shape[-1], stride_period, K=4)

    t_shap_zero = compute_temporal_shap_baseline(
        'zero', classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments, seed=seq_idx,
    )
    t_shap_mean = compute_temporal_shap_baseline(
        'mean', classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments,
        joint_means=joint_means, seed=seq_idx,
    )
    t_shap_marginal = compute_temporal_shap_baseline(
        'marginal', classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments,
        train_pool=train_pool, seed=seq_idx,
    )

    print(f'  [seq {seq_idx}] temporal faithfulness metrics …')
    t_methods = {
        'zero':     (t_shap_zero,     None,        None),
        'mean':     (t_shap_mean,     joint_means, None),
        'marginal': (t_shap_marginal, None,        train_pool),
    }
    t_faithfulness = {}
    for m_name, (t_shap, jm, tp) in t_methods.items():
        del_ins = _temporal_deletion_insertion_auc(
            classifier_fn, x, y, window_assignments, t_shap, m_name,
            joint_means=jm, train_pool=tp, seed=seq_idx,
        )
        comp_err = compute_shapley_completeness(
            t_shap, p_full, del_ins['p_empty'], joint_names=list(t_shap.keys()),
        )
        t_faithfulness[m_name] = {
            'deletion_auc':  del_ins['deletion_auc'],
            'insertion_auc': del_ins['insertion_auc'],
            'completeness_error': comp_err,
        }

    return {
        'seq_idx':    seq_idx,
        'true_class': int(y[0].item()),
        'p_full':     p_full,
        'stride_detected': not fallback,
        # Spatial
        'shap_values': {
            'zero':     {k: float(v) for k, v in shap_zero.items()},
            'mean':     {k: float(v) for k, v in shap_mean.items()},
            'marginal': {k: float(v) for k, v in shap_marginal.items()},
        },
        'faithfulness': {
            'zero':     metrics_zero,
            'mean':     metrics_mean,
            'marginal': metrics_marginal,
        },
        # Temporal
        'temporal_shap_values': {
            'zero':     {k: float(v) for k, v in t_shap_zero.items()},
            'mean':     {k: float(v) for k, v in t_shap_mean.items()},
            'marginal': {k: float(v) for k, v in t_shap_marginal.items()},
        },
        'temporal_faithfulness': t_faithfulness,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

_BASELINE_METHODS = ('zero', 'mean', 'marginal')


def aggregate_results(per_seq: list[dict]) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)
    for r in per_seq:
        agg['p_full'].append(r['p_full'])
        agg['stride_detected'].append(float(r.get('stride_detected', True)))

        # Spatial faithfulness
        for method in _BASELINE_METHODS:
            f      = r['faithfulness'][method]
            prefix = f'spatial/{method}'
            agg[f'{prefix}/deletion_auc'].append(f['deletion_auc'])
            agg[f'{prefix}/insertion_auc'].append(f['insertion_auc'])
            agg[f'{prefix}/random_deletion_auc'].append(f['random_deletion_auc'])
            agg[f'{prefix}/random_insertion_auc'].append(f['random_insertion_auc'])
            agg[f'{prefix}/completeness_error'].append(f['completeness_error'])
            for k, vals in f['pgi_pgu'].items():
                agg[f'{prefix}/pgi@{k}'].append(vals['pgi'])
                agg[f'{prefix}/pgu@{k}'].append(vals['pgu'])
            for k, vals in f['pgi_pgu_rand'].items():
                agg[f'{prefix}/pgi_rand@{k}'].append(vals['pgi'])
                agg[f'{prefix}/pgu_rand@{k}'].append(vals['pgu'])

        # Temporal faithfulness
        for method in _BASELINE_METHODS:
            tf     = r.get('temporal_faithfulness', {}).get(method, {})
            prefix = f'temporal/{method}'
            for key in ('deletion_auc', 'insertion_auc', 'completeness_error'):
                if key in tf:
                    agg[f'{prefix}/{key}'].append(tf[key])

    return {
        key: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        for key, vals in agg.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='SHAP baseline evaluation (zero / mean / marginal) for '
                    'CARE-PD pretrained classifiers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--backbone', required=True,
        help="Backbone name, e.g. 'potr', 'motionbert', 'mixste'.",
    )
    parser.add_argument(
        '--config', required=True,
        help="Config filename inside configs/<backbone>/, e.g. 'BMCLab.json'.",
    )
    parser.add_argument(
        '--num_folds', type=int, required=True,
        help='Total folds used during classifier training (23 for LOSOCV on BMCLab).',
    )
    parser.add_argument(
        '--fold', type=int, required=True,
        help='Fold index (1-indexed) whose test split to evaluate.',
    )
    parser.add_argument(
        '--classifier_ckpt', required=True,
        help='Path to latest_epoch.pth.tr saved by run.py.',
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Directory to write per_sequence.jsonl and aggregate.json.',
    )
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument('--n_kernel_samples', type=int, default=3000)
    parser.add_argument('--n_completion_samples', type=int, default=20,
                        help='Sequences averaged per coalition for the marginal baseline.')
    parser.add_argument('--k_list', nargs='+', type=int, default=[1, 2, 3, 5])
    parser.add_argument('--max_train_pool', type=int, default=2000)
    parser.add_argument('--max_test_sequences', type=int, default=None,
                        help='Cap on test sequences (None = all; small values for debugging).')
    parser.add_argument('--train_pool_batch_size', type=int, default=64)
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Capture frame-rate for stride-period detection (temporal SHAP).')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = {
        'n_kernel_samples':    args.n_kernel_samples,
        'n_completion_samples': args.n_completion_samples,
        'k_list':              args.k_list,
        'fps':                 args.fps,
    }

    # ------------------------------------------------------------------
    # Load backbone params (for classifier instantiation only).
    # ------------------------------------------------------------------
    print('[1/4] Loading backbone params and classifier …')
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    motion_encoder  = load_motion_encoder(args.classifier_ckpt, backbone_params, device)

    # ------------------------------------------------------------------
    # Build training pool (raw root-centred 3D).
    # ------------------------------------------------------------------
    print('[2/4] Building raw training pool …')
    train_pool, joint_means = build_train_pool(
        backbone_params, args.fold, device,
        max_sequences=args.max_train_pool,
        batch_size=args.train_pool_batch_size,
    )
    print(f'  train_pool: {tuple(train_pool.shape)}  '
          f'(J={train_pool.shape[1]}, F={train_pool.shape[2]}, T={train_pool.shape[3]})')

    # Compute z-score stats from training pool (used only for POTR).
    zscore_mean, zscore_std = compute_zscore_stats(train_pool)
    zscore_mean = zscore_mean.to(device)
    zscore_std  = zscore_std.to(device)

    # ------------------------------------------------------------------
    # Load test data (same raw pipeline).
    # ------------------------------------------------------------------
    print('[3/4] Loading test data …')
    data_args = _raw_data_args(backbone_params, args.fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Per-sequence evaluation.
    # ------------------------------------------------------------------
    print('[4/4] Running per-sequence SHAP evaluation …')
    per_seq_results: list[dict] = []
    jsonl_path = os.path.join(args.output_dir, 'per_sequence.jsonl')

    for seq_idx, raw_batch in enumerate(test_loader):
        if args.max_test_sequences is not None and seq_idx >= args.max_test_sequences:
            break

        x_raw, labels, _, _, pad_mask = raw_batch
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params['num_classes'], device, y=labels,
        )
        x       = batch['x']        # (1, J, F, T) root-centred
        y       = batch['y']        # (1,)
        mask    = batch['mask']     # (1, T)
        lengths = batch['lengths']  # (1,)

        result = evaluate_sequence(
            seq_idx, x, y, mask, lengths,
            motion_encoder, args.backbone, train_pool, joint_means,
            zscore_mean, zscore_std, cfg,
        )
        per_seq_results.append(result)

        with open(jsonl_path, 'a') as fh:
            fh.write(json.dumps(result) + '\n')

        print(
            f'  seq {seq_idx}: class={result["true_class"]} '
            f'p_full={result["p_full"]:.3f}  '
            f'del_auc  zero={result["faithfulness"]["zero"]["deletion_auc"]:.3f} '
            f'mean={result["faithfulness"]["mean"]["deletion_auc"]:.3f} '
            f'marg={result["faithfulness"]["marginal"]["deletion_auc"]:.3f}',
            flush=True,
        )

    # ------------------------------------------------------------------
    # Aggregate and save.
    # ------------------------------------------------------------------
    agg = aggregate_results(per_seq_results)
    agg['_meta'] = {
        'backbone':            args.backbone,
        'config':              args.config,
        'num_folds':           args.num_folds,
        'fold':                args.fold,
        'n_test_sequences':    len(per_seq_results),
        'n_kernel_samples':    args.n_kernel_samples,
        'n_completion_samples': args.n_completion_samples,
        'k_list':              args.k_list,
        'fps':                 args.fps,
        'classifier_ckpt':     args.classifier_ckpt,
    }
    with open(os.path.join(args.output_dir, 'aggregate.json'), 'w') as fh:
        json.dump(agg, fh, indent=2)

    print(f'\nDone. Results written to {args.output_dir}/')
    print(f'  per_sequence.jsonl — {len(per_seq_results)} sequences')
    print(f'  aggregate.json     — means & stds for all metrics')
    stride_rate = agg.get('stride_detected', {}).get('mean', float('nan'))
    print(f'\n--- Aggregate summary (mean ± std, {len(per_seq_results)} seqs) ---')
    print(f'  Stride auto-detected in {stride_rate * 100:.1f}% of sequences\n')

    def _get(k):
        e = agg.get(k, {})
        return e.get('mean', float('nan')), e.get('std', float('nan'))

    # --- Spatial SHAP ---
    print('  Spatial SHAP (KernelSHAP, 17 joints):')
    print(f'  {"Method":<10}  {"Del-AUC":>10}  {"Ins-AUC":>10}  {"PGI@1":>8}  {"PGU@1":>8}  {"Comp.Err":>10}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*10}')
    for method in _BASELINE_METHODS:
        dm, ds = _get(f'spatial/{method}/deletion_auc')
        im, is_ = _get(f'spatial/{method}/insertion_auc')
        pm, _ = _get(f'spatial/{method}/pgi@1')
        um, _ = _get(f'spatial/{method}/pgu@1')
        cm, cs = _get(f'spatial/{method}/completeness_error')
        print(
            f'  {method:<10}  {dm:6.3f}±{ds:.3f}  {im:6.3f}±{is_:.3f}'
            f'  {pm:8.3f}  {um:8.3f}  {cm:6.3f}±{cs:.3f}',
        )

    # --- Temporal SHAP ---
    print('\n  Temporal SHAP (exact, K=4 stride windows):')
    print(f'  {"Method":<10}  {"Del-AUC":>10}  {"Ins-AUC":>10}  {"Comp.Err":>10}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')
    for method in _BASELINE_METHODS:
        dm, ds = _get(f'temporal/{method}/deletion_auc')
        im, is_ = _get(f'temporal/{method}/insertion_auc')
        cm, cs = _get(f'temporal/{method}/completeness_error')
        print(
            f'  {method:<10}  {dm:6.3f}±{ds:.3f}  {im:6.3f}±{is_:.3f}'
            f'  {cm:6.3f}±{cs:.3f}',
        )


if __name__ == '__main__':
    main()
