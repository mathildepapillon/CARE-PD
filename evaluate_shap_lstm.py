"""evaluate_shap_lstm.py — SHAP evaluation comparing LSTM VAE completions against
zero / mean / marginal baselines.

Loads a trained LSTM VAE checkpoint (with masked encoder) and uses it as a
generative completion model for KernelSHAP (spatial, 17 joints) and exact
temporal SHAP (K=4 stride-aligned windows).  Results are directly comparable
to evaluate_shap_baselines.py and evaluate_shap.py (ActorSHAP).

DATA PIPELINE
=============
Sequences are loaded in **global-pelvis** format (joint 0 retains the global
world position, joints 1-16 are relative to pelvis) via
``actor_batch_from_carepd``.  The LSTM VAE wrapper internally root-centres
before encoding and restores the original pelvis trajectory after decoding,
so the classifier always receives properly positioned sequences.

The classifier wrapper (``build_classifier_fn``) additionally restores the
original pelvis before ``project_for_backbone``, so imputed sequences keep
the real global walking path regardless of the method.

COALITION REPRODUCIBILITY
=========================
``seed=seq_idx`` is used for all methods — same coalitions as
evaluate_shap_baselines.py and evaluate_shap.py.

USAGE
=====
    python evaluate_shap_lstm.py \\
        --backbone potr \\
        --config BMCLab.json \\
        --num_folds 23 \\
        --fold 1 \\
        --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \\
        --lstm_vae_ckpt experiment_outs/lstm_vae/BMCLab_fold1/checkpoints/last.ckpt \\
        --output_dir results/shap_lstm_bmclab_fold1 \\
        --device cuda:0

    # Quick smoke-test on 2 sequences:
    python evaluate_shap_lstm.py ... --max_test_sequences 2 --n_kernel_samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Callable, Optional

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
    value_fn,
)
from model.actor.shap_masking import (
    build_temporal_shap_mask,
    build_temporal_windows,
    detect_stride_period,
)
from model.actor.shap_metrics import (
    _class_prob,
    compute_shapley_completeness,
    compute_spatial_faithfulness_batched,
)
from model.motion_encoder import MotionEncoder
from model.lstm_vae.shap_wrapper import LstmVaeShapWrapper

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
# Temporal faithfulness for the LSTM-VAE (completion-based, same pattern as
# _temporal_deletion_insertion_auc_actor in evaluate_shap.py).
# ---------------------------------------------------------------------------

def _temporal_del_ins_auc_lstm_vae(
    lstm_wrapper: LstmVaeShapWrapper,
    classifier_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    window_assignments: list[list[int]],
    temporal_shap_vals: dict[str, float],
    n_completion_samples: int,
) -> dict:
    """Temporal deletion / insertion AUC using LSTM VAE completions."""
    K = len(window_assignments)
    T = x.shape[-1]
    device = x.device
    class_idx = int(y[0].item())

    window_names = [k for k in temporal_shap_vals if not k.startswith("_")]
    order = sorted(range(K), key=lambda k: -temporal_shap_vals[window_names[k]])

    del_curve = [_class_prob(classifier_fn, x, class_idx)]
    for step in range(K):
        observed = [k for k in range(K) if k not in order[: step + 1]]
        cm_t = build_temporal_shap_mask(
            observed, window_assignments, T, device,
        ).unsqueeze(0)
        del_curve.append(
            float(value_fn(
                lstm_wrapper, classifier_fn, x, y, mask, lengths, cm_t,
                n_samples=n_completion_samples,
            ))
        )

    cm_empty = build_temporal_shap_mask(
        [], window_assignments, T, device,
    ).unsqueeze(0)
    ins_curve = [
        float(value_fn(
            lstm_wrapper, classifier_fn, x, y, mask, lengths, cm_empty,
            n_samples=n_completion_samples,
        ))
    ]
    for step in range(K):
        observed = order[: step + 1]
        cm_t = build_temporal_shap_mask(
            observed, window_assignments, T, device,
        ).unsqueeze(0)
        ins_curve.append(
            float(value_fn(
                lstm_wrapper, classifier_fn, x, y, mask, lengths, cm_t,
                n_samples=n_completion_samples,
            ))
        )

    xs = np.linspace(0.0, 1.0, K + 1)
    return {
        "deletion_auc": float(np.trapz(del_curve, xs)),
        "insertion_auc": float(np.trapz(ins_curve, xs)),
        "p_empty": float(ins_curve[0]),
    }


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
    lstm_wrapper: LstmVaeShapWrapper,
) -> dict:
    """Run spatial and temporal SHAP (baselines + lstm_vae) for one sequence."""
    device        = x.device
    n_kernel      = cfg.get('n_kernel_samples', 3000)
    n_completions = cfg.get('n_completion_samples', 20)
    n_vae_samples = cfg.get('n_vae_samples', 20)
    k_list        = tuple(cfg.get('k_list', [1, 2, 3, 5]))
    fps           = cfg.get('fps', 30)

    classifier_fn = build_classifier_fn(
        motion_encoder, mask, backbone_name,
        zscore_mean=zscore_mean, zscore_std=zscore_std,
        x_orig=x,
    )

    # ------------------------------------------------------------------
    # Spatial SHAP — baselines (zero / mean / marginal)
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

    # ------------------------------------------------------------------
    # Spatial SHAP — LSTM VAE (batched: all coalitions in GPU chunks)
    # ------------------------------------------------------------------
    print(f'  [seq {seq_idx}] spatial SHAP (lstm_vae, batched) …')
    shap_lstm = lstm_wrapper.compute_spatial_shap_batched(
        classifier_fn, x, y, mask, lengths,
        n_kernel_samples=n_kernel, n_completion_samples=n_vae_samples,
        seed=seq_idx,
    )

    # ------------------------------------------------------------------
    # Spatial faithfulness
    # ------------------------------------------------------------------
    p_full     = _class_prob(classifier_fn, x, int(y[0].item()))
    p_ref_zero = shap_zero["_v_empty"]
    p_ref_mean = shap_mean["_v_empty"]
    p_ref_marg = shap_marginal["_v_empty"]
    p_ref_lstm = shap_lstm["_v_empty"]

    print(f'  [seq {seq_idx}] spatial faithfulness metrics …')
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
    metrics_lstm = compute_spatial_faithfulness_batched(
        classifier_fn, x, y, mask, lengths, shap_lstm, 'actor',
        actor_shap=lstm_wrapper, n_samples=n_vae_samples,
        k_list=k_list, p_full=p_full, p_ref=p_ref_lstm, seq_idx=seq_idx,
    )

    # ------------------------------------------------------------------
    # Temporal SHAP (exact enumeration, K=4 stride-aligned windows)
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

    print(f'  [seq {seq_idx}] temporal SHAP (lstm_vae, batched) …')
    t_shap_lstm = lstm_wrapper.compute_temporal_shap_batched(
        classifier_fn, x, y, mask, lengths,
        window_assignments=window_assignments,
        n_completion_samples=n_vae_samples,
    )

    # ------------------------------------------------------------------
    # Temporal faithfulness
    # ------------------------------------------------------------------
    print(f'  [seq {seq_idx}] temporal faithfulness metrics …')
    t_methods_baseline: dict[str, tuple] = {
        'zero':     (t_shap_zero,     None,        None),
        'mean':     (t_shap_mean,     joint_means, None),
        'marginal': (t_shap_marginal, None,        train_pool),
    }

    t_faithfulness: dict[str, dict] = {}
    for m_name, (t_shap, jm, tp) in t_methods_baseline.items():
        t_window_names = [k for k in t_shap if not k.startswith('_')]
        del_ins = _temporal_deletion_insertion_auc_batched(
            classifier_fn, x, y, mask, lengths, window_assignments, t_shap, m_name,
            joint_means=jm, train_pool=tp,
            seed=seq_idx,
            n_marginal_samples=n_completions,
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

    # LSTM VAE temporal faithfulness (completion-based)
    t_lstm_window_names = [k for k in t_shap_lstm if not k.startswith('_')]
    t_lstm_del_ins = _temporal_del_ins_auc_lstm_vae(
        lstm_wrapper, classifier_fn, x, y, mask, lengths,
        window_assignments, t_shap_lstm, n_vae_samples,
    )
    t_lstm_comp_err = compute_shapley_completeness(
        t_shap_lstm, t_shap_lstm['_v_full'], t_shap_lstm['_v_empty'],
        joint_names=t_lstm_window_names,
    )
    t_faithfulness['lstm_vae'] = {
        'deletion_auc':  t_lstm_del_ins['deletion_auc'],
        'insertion_auc': t_lstm_del_ins['insertion_auc'],
        'completeness_error': t_lstm_comp_err,
    }

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    shap_vals: dict[str, dict] = {
        'zero':     {k: float(v) for k, v in shap_zero.items()},
        'mean':     {k: float(v) for k, v in shap_mean.items()},
        'marginal': {k: float(v) for k, v in shap_marginal.items()},
        'lstm_vae': {k: float(v) for k, v in shap_lstm.items()},
    }
    faithfulness: dict[str, dict] = {
        'zero':     metrics_zero,
        'mean':     metrics_mean,
        'marginal': metrics_marginal,
        'lstm_vae': metrics_lstm,
    }

    return {
        'seq_idx':    seq_idx,
        'true_class': int(y[0].item()),
        'p_full':     p_full,
        'stride_detected': not fallback,
        'shap_values':   shap_vals,
        'faithfulness':  faithfulness,
        'temporal_shap_values': {
            'zero':     {k: float(v) for k, v in t_shap_zero.items()},
            'mean':     {k: float(v) for k, v in t_shap_mean.items()},
            'marginal': {k: float(v) for k, v in t_shap_marginal.items()},
            'lstm_vae': {k: float(v) for k, v in t_shap_lstm.items()},
        },
        'temporal_faithfulness': t_faithfulness,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

_ALL_METHODS = ('zero', 'mean', 'marginal', 'lstm_vae')


def aggregate_results(per_seq: list[dict]) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)

    spatial_methods = (
        list(per_seq[0]['faithfulness'].keys()) if per_seq
        else list(_ALL_METHODS)
    )
    temporal_methods = (
        list(per_seq[0]['temporal_faithfulness'].keys()) if per_seq
        else list(_ALL_METHODS)
    )

    for r in per_seq:
        agg['p_full'].append(r['p_full'])
        agg['stride_detected'].append(float(r.get('stride_detected', True)))

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
        description='SHAP evaluation comparing LSTM VAE completions against '
                    'zero / mean / marginal baselines.',
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
        '--lstm_vae_ckpt', required=True,
        help='Path to LSTM VAE Lightning checkpoint (.ckpt) with masked encoder.',
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
    parser.add_argument('--n_vae_samples', type=int, default=20,
                        help='LSTM VAE completions averaged per coalition.')
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
    args = parser.parse_args()
    args.root_centered = False

    device = torch.device(args.device)

    cfg = {
        'n_kernel_samples':     args.n_kernel_samples,
        'n_completion_samples': args.n_completion_samples,
        'n_vae_samples':        args.n_vae_samples,
        'k_list':               args.k_list,
        'fps':                  args.fps,
    }

    # ------------------------------------------------------------------
    # 1. Load backbone params and classifier
    # ------------------------------------------------------------------
    print('[1/5] Loading backbone params and classifier …')
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    backbone_name   = backbone_params['backbone']

    output_dir = resolve_output_dir(
        args.results_root, args.output_dir, backbone_params, args.fold,
    )
    os.makedirs(output_dir, exist_ok=True)

    motion_encoder = load_motion_encoder(args.classifier_ckpt, backbone_params, device)

    # ------------------------------------------------------------------
    # 2. Load LSTM VAE
    # ------------------------------------------------------------------
    print(f'[2/5] Loading LSTM VAE from {args.lstm_vae_ckpt} …')
    lstm_wrapper = LstmVaeShapWrapper.from_checkpoint(args.lstm_vae_ckpt, device)
    print(f'  model latent_dim={lstm_wrapper.model.latent_dim}  '
          f'n_mix={lstm_wrapper.model.masked_encoder.n_mix}  '
          f'seq_len={lstm_wrapper.model.seq_len}')

    # ------------------------------------------------------------------
    # 3. Build training pool (raw root-centred 3D)
    # ------------------------------------------------------------------
    print('[3/5] Building raw training pool …')
    train_pool, joint_means = build_train_pool(
        backbone_params, args.fold, device,
        max_sequences=args.max_train_pool,
        batch_size=args.train_pool_batch_size,
        root_centered=args.root_centered,
    )
    print(f'  train_pool: {tuple(train_pool.shape)}  '
          f'(J={train_pool.shape[1]}, F={train_pool.shape[2]}, T={train_pool.shape[3]})')

    if backbone_name == 'potr':
        print('  [potr] Computing z-score stats over train+test union …')
        zscore_mean, zscore_std = build_zscore_stats_for_potr(
            backbone_params, args.fold, device,
            batch_size=args.train_pool_batch_size,
            root_centered=args.root_centered,
        )
    else:
        zscore_mean = zscore_std = None

    # ------------------------------------------------------------------
    # 4. Load test data
    # ------------------------------------------------------------------
    print('[4/5] Loading test data …')
    data_args = _raw_data_args(backbone_params, args.fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # 5. Per-sequence evaluation
    # ------------------------------------------------------------------
    print('[5/5] Running per-sequence SHAP evaluation …')
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
        mask_b  = batch['mask']     # (1, T)
        lengths = batch['lengths']  # (1,)

        result = evaluate_sequence(
            seq_idx, x, y, mask_b, lengths,
            motion_encoder, backbone_name, train_pool, joint_means,
            zscore_mean, zscore_std, cfg,
            lstm_wrapper=lstm_wrapper,
        )
        per_seq_results.append(result)

        with open(jsonl_path, 'a') as fh:
            fh.write(json.dumps(result) + '\n')

        print(
            f'  seq {seq_idx}: class={result["true_class"]} '
            f'p_full={result["p_full"]:.3f}  '
            f'del_auc  zero={result["faithfulness"]["zero"]["deletion_auc"]:.3f} '
            f'mean={result["faithfulness"]["mean"]["deletion_auc"]:.3f} '
            f'marg={result["faithfulness"]["marginal"]["deletion_auc"]:.3f} '
            f'lstm={result["faithfulness"]["lstm_vae"]["deletion_auc"]:.3f}',
            flush=True,
        )

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    p_full_warning = check_p_full_warning(
        per_seq_results, backbone_params['num_classes'], backbone_name,
    )

    agg = aggregate_results(per_seq_results)
    agg['_meta'] = {
        'backbone':             args.backbone,
        'dataset':              backbone_params['dataset'],
        'config':               args.config,
        'num_folds':            args.num_folds,
        'fold':                 args.fold,
        'n_test_sequences':     len(per_seq_results),
        'n_kernel_samples':     args.n_kernel_samples,
        'n_completion_samples': args.n_completion_samples,
        'n_vae_samples':        args.n_vae_samples,
        'k_list':               args.k_list,
        'fps':                  args.fps,
        'classifier_ckpt':      args.classifier_ckpt,
        'lstm_vae_ckpt':        args.lstm_vae_ckpt,
        'output_dir':           output_dir,
        'root_centered':        args.root_centered,
        'p_full_warning':       p_full_warning,
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
    print('  Spatial SHAP (KernelSHAP, 17 joints):')
    print(f'  {"Method":<10}  {"Del-AUC":>10}  {"Ins-AUC":>10}  '
          f'{"PGI@1":>8}  {"PGU@1":>8}  {"Comp.Err":>10}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*10}')
    for method in _ALL_METHODS:
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
    for method in _ALL_METHODS:
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
