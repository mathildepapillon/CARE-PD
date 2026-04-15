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
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from const import const
from data.dataloaders import collate_fn
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.shap_compute import (
    compute_spatial_shap_baseline,
    compute_temporal_shap_baseline,
)
from model.actor.shap_masking import build_temporal_windows, detect_stride_period
from model.actor.shap_metrics import (
    _class_prob,
    compute_shapley_completeness,
    compute_spatial_faithfulness_batched,
)
from model.actor.physics_completer import PhysicsInformedCompleter, load_physics_completer
from model.motion_encoder import MotionEncoder

# Shared utilities (backbone param loading, data helpers, classifier wrapper,
# temporal faithfulness, output directory naming, p_full warning).
from model.actor.shap_eval_shared import (
    _load_backbone_params,
    _raw_data_args,
    build_train_pool,
    build_zscore_stats_for_potr,
    load_motion_encoder,
    build_classifier_fn,
    _temporal_deletion_insertion_auc_batched,
    resolve_output_dir,
    check_p_full_warning,
)


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

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
    physics_completer: Optional[PhysicsInformedCompleter] = None,
    subject_id: Optional[str] = None,
    physics_cache_record: Optional[dict] = None,
) -> dict:
    """Run spatial and temporal SHAP (all baselines) for one sequence."""
    device        = x.device
    n_kernel      = cfg.get('n_kernel_samples', 3000)
    n_completions = cfg.get('n_completion_samples', 20)
    k_list        = tuple(cfg.get('k_list', [1, 2, 3, 5]))
    fps           = cfg.get('fps', 30)
    physics_n_kernel   = cfg.get('physics_n_kernel_samples', 500)
    physics_n_samples  = cfg.get('physics_n_samples', 5)

    classifier_fn = build_classifier_fn(
        motion_encoder, mask, backbone_name,
        zscore_mean=zscore_mean, zscore_std=zscore_std,
        x_orig=x,
    )

    if physics_completer is not None and subject_id is not None:
        physics_completer.set_subject(subject_id)

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

    p_full     = _class_prob(classifier_fn, x, int(y[0].item()))
    p_ref_zero = shap_zero["_v_empty"]
    p_ref_mean = shap_mean["_v_empty"]
    p_ref_marg = shap_marginal["_v_empty"]

    print(f'  [seq {seq_idx}] spatial faithfulness metrics (batched) …')
    metrics_zero = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_zero, 'zero',
        k_list=k_list, p_full=p_full, p_ref=p_ref_zero, seq_idx=seq_idx,
    )
    metrics_mean = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_mean, 'mean',
        joint_means=joint_means,
        k_list=k_list, p_full=p_full, p_ref=p_ref_mean, seq_idx=seq_idx,
    )
    metrics_marginal = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_marginal, 'marginal',
        train_pool=train_pool, n_samples=n_completions,
        k_list=k_list, p_full=p_full, p_ref=p_ref_marg, seq_idx=seq_idx,
    )

    if physics_cache_record is not None:
        # Fast path: pre-computed results loaded from disk.
        shap_physics    = physics_cache_record['shap_values']
        metrics_physics = physics_cache_record['faithfulness']
    elif physics_completer is not None:
        # Live path: run KernelSHAP + faithfulness now (slow).
        print(f'  [seq {seq_idx}] spatial SHAP (physics) …')
        shap_physics = compute_spatial_shap_baseline(
            'physics', classifier_fn, x, y, mask, lengths,
            physics_completer=physics_completer,
            n_kernel_samples=physics_n_kernel,
            n_marginal_samples=physics_n_samples,
            seed=seq_idx,
        )
        p_ref_phys = shap_physics['_v_empty']
        metrics_physics = compute_spatial_faithfulness_batched(
            classifier_fn, x, y, mask, lengths, shap_physics, 'physics',
            physics_completer=physics_completer, n_samples=physics_n_samples,
            k_list=k_list, p_full=p_full, p_ref=p_ref_phys, seq_idx=seq_idx,
        )
    else:
        shap_physics    = None
        metrics_physics = None

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
        train_pool=train_pool, n_marginal_samples=n_completions,
        seed=seq_idx,
    )

    t_shap_physics = None
    if physics_completer is not None:
        print(f'  [seq {seq_idx}] temporal SHAP (physics, K=4 windows, exact) …')
        t_shap_physics = compute_temporal_shap_baseline(
            'physics', classifier_fn, x, y, mask, lengths,
            window_assignments=window_assignments,
            physics_completer=physics_completer,
            n_marginal_samples=physics_n_samples,
            seed=seq_idx,
        )

    print(f'  [seq {seq_idx}] temporal faithfulness metrics (batched) …')
    t_methods = {
        'zero':     (t_shap_zero,     None,        None,        None),
        'mean':     (t_shap_mean,     joint_means, None,        None),
        'marginal': (t_shap_marginal, None,        train_pool,  None),
    }
    if t_shap_physics is not None:
        t_methods['physics'] = (t_shap_physics, None, None, physics_completer)

    t_faithfulness = {}
    for m_name, (t_shap, jm, tp, pcomp) in t_methods.items():
        t_window_names = [k for k in t_shap if not k.startswith('_')]
        del_ins = _temporal_deletion_insertion_auc_batched(
            classifier_fn, x, y, mask, lengths, window_assignments, t_shap, m_name,
            joint_means=jm, train_pool=tp, physics_completer=pcomp,
            seed=seq_idx,
            n_marginal_samples=n_completions if m_name != 'physics' else physics_n_samples,
        )
        comp_err = compute_shapley_completeness(
            t_shap, t_shap['_v_full'], t_shap['_v_empty'],
            joint_names=t_window_names,
        )
        t_faithfulness[m_name] = {
            'deletion_auc':  del_ins['deletion_auc'],
            'insertion_auc': del_ins['insertion_auc'],
            'completeness_error': comp_err,
        }

    shap_vals: dict[str, dict] = {
        'zero':     {k: float(v) for k, v in shap_zero.items()},
        'mean':     {k: float(v) for k, v in shap_mean.items()},
        'marginal': {k: float(v) for k, v in shap_marginal.items()},
    }
    faithfulness: dict[str, dict] = {
        'zero':     metrics_zero,
        'mean':     metrics_mean,
        'marginal': metrics_marginal,
    }
    if shap_physics is not None:
        shap_vals['physics']    = {k: float(v) for k, v in shap_physics.items()}
        faithfulness['physics'] = metrics_physics

    return {
        'seq_idx':    seq_idx,
        'true_class': int(y[0].item()),
        'p_full':     p_full,
        'stride_detected': not fallback,
        # Spatial
        'shap_values':   shap_vals,
        'faithfulness':  faithfulness,
        # Temporal
        'temporal_shap_values': {
            'zero':     {k: float(v) for k, v in t_shap_zero.items()},
            'mean':     {k: float(v) for k, v in t_shap_mean.items()},
            'marginal': {k: float(v) for k, v in t_shap_marginal.items()},
            **(
                {'physics': {k: float(v) for k, v in t_shap_physics.items()}}
                if t_shap_physics is not None else {}
            ),
        },
        'temporal_faithfulness': t_faithfulness,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

_BASELINE_METHODS = ('zero', 'mean', 'marginal')


def aggregate_results(per_seq: list[dict]) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)

    # Detect which spatial methods are present (always includes zero/mean/marginal,
    # optionally includes physics when --physics_stats was provided).
    spatial_methods = list(per_seq[0]['faithfulness'].keys()) if per_seq else list(_BASELINE_METHODS)
    temporal_methods = (
        list(per_seq[0]['temporal_faithfulness'].keys()) if per_seq
        else list(_BASELINE_METHODS)
    )

    for r in per_seq:
        agg['p_full'].append(r['p_full'])
        agg['stride_detected'].append(float(r.get('stride_detected', True)))

        # Spatial faithfulness
        for method in spatial_methods:
            f = r['faithfulness'].get(method)
            if f is None:
                continue
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

        # Temporal faithfulness (zero/mean/marginal only)
        for method in temporal_methods:
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
        '--results_root', default=None,
        help='Root directory for auto-structured results. Output path is derived as '
             '{results_root}/{dataset}/{backbone}/fold{fold}/. '
             'Mutually exclusive with --output_dir; one must be provided.',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Explicit output directory (overrides --results_root). '
             'Writes per_sequence.jsonl and aggregate.json.',
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
    parser.add_argument('--start_seq', type=int, default=0,
                        help='Skip the first N test sequences (0-indexed). '
                             'Use to resume a partially-completed run.')
    parser.add_argument('--train_pool_batch_size', type=int, default=64)
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Capture frame-rate for stride-period detection (temporal SHAP).')
    parser.add_argument('--root_centered', action='store_true', default=False,
                        help='Subtract joint-0 (pelvis) so sequences are root-centred. '
                             'Default: absolute world coordinates (no root-centering).')
    # Physics-informed baseline (optional — two mutually exclusive modes)
    parser.add_argument('--physics_stats', default=None,
                        help='Path to motion_stats_fold*.pkl from compute_motion_stats.py. '
                             'Loads PhysicsInformedCompleter for physics temporal SHAP and/or '
                             'live physics spatial SHAP. May be combined with --physics_cache_dir '
                             '(cache supplies pre-computed spatial physics JSONs).')
    parser.add_argument('--physics_cache_dir', default=None,
                        help='Directory of pre-computed JSON files from '
                             'scripts/precompute_physics_shap.py '
                             '(spatial SHAP + faithfulness). For temporal physics, also pass '
                             '--physics_stats.')
    parser.add_argument('--physics_n_kernel_samples', type=int, default=500,
                        help='KernelSHAP coalition pairs for the physics method '
                             '(only used with --physics_stats, not --physics_cache_dir).')
    parser.add_argument('--physics_n_samples', type=int, default=5,
                        help='Physics completions averaged per coalition '
                             '(only used with --physics_stats).')
    args = parser.parse_args()

    # Both may be set: cache supplies pre-computed *spatial* physics; stats file
    # instantiates PhysicsInformedCompleter for temporal physics (and live spatial).

    device = torch.device(args.device)

    cfg = {
        'n_kernel_samples':        args.n_kernel_samples,
        'n_completion_samples':    args.n_completion_samples,
        'k_list':                  args.k_list,
        'fps':                     args.fps,
        'physics_n_kernel_samples': args.physics_n_kernel_samples,
        'physics_n_samples':        args.physics_n_samples,
    }

    # ------------------------------------------------------------------
    # Load backbone params (for classifier instantiation only).
    # ------------------------------------------------------------------
    print('[1/4] Loading backbone params and classifier …')
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    backbone_name   = backbone_params['backbone']

    # Resolve output directory (auto-derived or explicit).
    output_dir = resolve_output_dir(
        args.results_root, args.output_dir, backbone_params, args.fold,
    )
    os.makedirs(output_dir, exist_ok=True)

    motion_encoder  = load_motion_encoder(args.classifier_ckpt, backbone_params, device)

    # ------------------------------------------------------------------
    # Build training pool (raw root-centred 3D).
    # ------------------------------------------------------------------
    print('[2/4] Building raw training pool …')
    train_pool, joint_means = build_train_pool(
        backbone_params, args.fold, device,
        max_sequences=args.max_train_pool,
        batch_size=args.train_pool_batch_size,
        root_centered=args.root_centered,
    )
    print(f'  train_pool: {tuple(train_pool.shape)}  '
          f'(J={train_pool.shape[1]}, F={train_pool.shape[2]}, T={train_pool.shape[3]})')

    # Z-score stats: only needed for POTR (other backbones ignore them in
    # project_for_backbone).  Computed over the train+test union to match
    # POTRPreprocessor which normalises before fold splitting.
    if backbone_name == 'potr':
        print('  [potr] Computing z-score stats over train+test union …')
        zscore_mean, zscore_std = build_zscore_stats_for_potr(
            backbone_params, args.fold, device,
            batch_size=args.train_pool_batch_size,
            root_centered=args.root_centered,
        )
    else:
        zscore_mean = zscore_std = None

    # Physics completer (--physics_stats) and/or spatial cache (--physics_cache_dir).
    physics_completer = None
    physics_cache: dict[int, dict] = {}   # seq_idx → pre-loaded record

    if args.physics_stats:
        print(f'  Loading PhysicsInformedCompleter from {args.physics_stats} …')
        physics_completer = load_physics_completer(args.physics_stats, device)
        print(f'  physics baseline: n_kernel={args.physics_n_kernel_samples}  '
              f'n_samples={args.physics_n_samples}')
    if args.physics_cache_dir:
        import glob as _glob, json as _json
        cache_files = sorted(_glob.glob(os.path.join(args.physics_cache_dir, 'seq_*.json')))
        for cf in cache_files:
            with open(cf) as fh:
                rec = _json.load(fh)
            physics_cache[rec['seq_idx']] = rec
        print(f'  Loaded {len(physics_cache)} pre-computed physics records '
              f'from {args.physics_cache_dir}')

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
    jsonl_path = os.path.join(output_dir, 'per_sequence.jsonl')

    for seq_idx, raw_batch in enumerate(test_loader):
        if seq_idx < args.start_seq:
            continue
        if args.max_test_sequences is not None and seq_idx >= args.start_seq + args.max_test_sequences:
            break

        x_raw, labels, _, _, pad_mask = raw_batch
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params['num_classes'], device,
            y=labels, root_centered=args.root_centered,
        )
        x       = batch['x']        # (1, J, F, T)
        y       = batch['y']        # (1,)
        mask    = batch['mask']     # (1, T)
        lengths = batch['lengths']  # (1,)

        # Physics: either live completer or pre-loaded cache record.
        subject_id: Optional[str] = None
        if physics_completer is not None:
            video_name = test_ds.video_names[seq_idx]
            subject_id = video_name.split('__')[0]

        result = evaluate_sequence(
            seq_idx, x, y, mask, lengths,
            motion_encoder, backbone_name, train_pool, joint_means,
            zscore_mean, zscore_std, cfg,
            physics_completer=physics_completer,
            subject_id=subject_id,
            physics_cache_record=physics_cache.get(seq_idx),
        )
        per_seq_results.append(result)

        with open(jsonl_path, 'a') as fh:
            fh.write(json.dumps(result) + '\n')

        phys_str = ''
        if 'physics' in result['faithfulness']:
            phys_str = f' phys={result["faithfulness"]["physics"]["deletion_auc"]:.3f}'
        print(
            f'  seq {seq_idx}: class={result["true_class"]} '
            f'p_full={result["p_full"]:.3f}  '
            f'del_auc  zero={result["faithfulness"]["zero"]["deletion_auc"]:.3f} '
            f'mean={result["faithfulness"]["mean"]["deletion_auc"]:.3f} '
            f'marg={result["faithfulness"]["marginal"]["deletion_auc"]:.3f}'
            f'{phys_str}',
            flush=True,
        )

    # ------------------------------------------------------------------
    # Aggregate and save.
    # ------------------------------------------------------------------
    p_full_warning = check_p_full_warning(
        per_seq_results, backbone_params['num_classes'], backbone_name,
    )

    agg = aggregate_results(per_seq_results)
    agg['_meta'] = {
        'backbone':                 args.backbone,
        'dataset':                  backbone_params['dataset'],
        'config':                   args.config,
        'num_folds':                args.num_folds,
        'fold':                     args.fold,
        'n_test_sequences':         len(per_seq_results),
        'n_kernel_samples':         args.n_kernel_samples,
        'n_completion_samples':     args.n_completion_samples,
        'k_list':                   args.k_list,
        'fps':                      args.fps,
        'classifier_ckpt':          args.classifier_ckpt,
        'output_dir':               output_dir,
        'p_full_warning':           p_full_warning,
        'physics_stats':            args.physics_stats,
        'physics_cache_dir':        args.physics_cache_dir,
        'physics_n_kernel_samples': args.physics_n_kernel_samples,
        'physics_n_samples':        args.physics_n_samples,
        'physics_sequences_cached': len(physics_cache),
    }
    with open(os.path.join(output_dir, 'aggregate.json'), 'w') as fh:
        json.dump(agg, fh, indent=2)

    print(f'\nDone. Results written to {output_dir}/')
    print(f'  per_sequence.jsonl — {len(per_seq_results)} sequences')
    print(f'  aggregate.json     — means & stds for all metrics')
    print(f'  backbone={args.backbone}  dataset={backbone_params["dataset"]}  fold={args.fold}')
    stride_rate = agg.get('stride_detected', {}).get('mean', float('nan'))
    print(f'\n--- Aggregate summary (mean ± std, {len(per_seq_results)} seqs) ---')
    print(f'  Stride auto-detected in {stride_rate * 100:.1f}% of sequences\n')

    def _get(k):
        e = agg.get(k, {})
        return e.get('mean', float('nan')), e.get('std', float('nan'))

    # --- Spatial SHAP ---
    _has_phys_spatial = (
        physics_completer is not None
        or (per_seq_results and 'physics' in per_seq_results[0].get('faithfulness', {}))
    )
    spatial_methods_present = list(
        dict.fromkeys(list(_BASELINE_METHODS) + (['physics'] if _has_phys_spatial else []))
    )
    print('  Spatial SHAP (KernelSHAP, 17 joints):')
    print(f'  {"Method":<10}  {"Del-AUC":>10}  {"Ins-AUC":>10}  {"PGI@1":>8}  {"PGU@1":>8}  {"Comp.Err":>10}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*10}')
    for method in spatial_methods_present:
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
    for method in ('zero', 'mean', 'marginal', 'physics'):
        if f'temporal/{method}/deletion_auc' not in agg:
            continue
        dm, ds = _get(f'temporal/{method}/deletion_auc')
        im, is_ = _get(f'temporal/{method}/insertion_auc')
        cm, cs = _get(f'temporal/{method}/completeness_error')
        print(
            f'  {method:<10}  {dm:6.3f}±{ds:.3f}  {im:6.3f}±{is_:.3f}'
            f'  {cm:6.3f}±{cs:.3f}',
        )


if __name__ == '__main__':
    main()
