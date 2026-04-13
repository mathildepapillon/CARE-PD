"""
cache_backbone_features.py
==========================
Pre-computes frozen backbone features for every clip in every fold split and
saves them to disk.  Subsequent classifier training loads these cached vectors
instead of running the expensive backbone + preprocessing on every epoch.

Each saved .npz contains per-clip pooled backbone features (post-backbone
projection, *pre*-FC layers), so classifier training only needs to run the
tiny classifier head on each batch (~1 ms/epoch instead of ~500 s/epoch).

Usage (BMCLab, LOSO, MotionBERT backbone, GPU 4):

    python cache_backbone_features.py \\
        --backbone motionbert \\
        --config BMCLab.json \\
        --num_folds -1 \\
        --device cuda:4

Output files are written to:
    assets/cached_features/<backbone>/<experiment_name>/<dataset>[_<views>]_<num_folds>fold/
        fold<N>_train.npz
        fold<N>_eval.npz

For view-specific backbones (motionbert, mixste, motionagformer, poseformerv2)
the view string is included in the directory name to avoid collisions between
backright and sideright caches, e.g.:
    motionbert/Hypertune/BMCLab_backright_23fold/

Single-view backbones (potr, motionclip, momask) have no view suffix:
    potr/Hypertune/BMCLab_23fold/

Each .npz has keys:
    features   float32  (N, feature_dim)  — pooled backbone output per clip
    labels     int64    (N,)
    video_idxs int64    (N,)
    metadata   float32  (N, M)  — M=0 when no metadata
    pad_masks  float32  (N, T)
"""

from __future__ import annotations

import argparse
import importlib
import json as _json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Resolve project imports (handles running from repo root or a subdirectory)
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataloaders import dataset_factory, collate_fn
from model.backbone_loader import load_pretrained_backbone
from model.motion_encoder import MotionEncoder
from const import const

# Map backbone name → config generator module
_BACKBONE_CONFIG_MODULE = {
    'potr':           'configs.generate_config_potr',
    'motionbert':     'configs.generate_config_motionbert',
    'motionagformer': 'configs.generate_config_motionagformer',
    'poseformerv2':   'configs.generate_config_poseformerv2',
    'mixste':         'configs.generate_config_mixste',
    'momask':         'configs.generate_config_momask',
    'motionclip':     'configs.generate_config_motionclip',
}


# ---------------------------------------------------------------------------
# Path helpers (shared by run.py via import)
# ---------------------------------------------------------------------------

def _views_tag(params: dict) -> str:
    """Return a hyphen-joined view tag, or '' for backbones without views."""
    views = params.get('views') or []
    return '-'.join(sorted(views)) if views else ''


def make_cache_dir(out_root: str, backbone_name: str, params: dict) -> str:
    """Canonical cache directory path for a given backbone + params config.

    Mirrors _cached_features_dir in run.py — both functions must stay in sync.
    The view tag is included for view-specific backbones to avoid cache
    collisions between backright and sideright configs.
    """
    views = _views_tag(params)
    folder = (
        f"{params['dataset']}_{views}_{params['num_folds']}fold"
        if views else
        f"{params['dataset']}_{params['num_folds']}fold"
    )
    return os.path.join(out_root, backbone_name, params['experiment_name'], folder)


# ---------------------------------------------------------------------------
# Params loader
# ---------------------------------------------------------------------------

def _load_params(backbone: str, config_file: str, num_folds: int) -> dict:
    """Load and resolve params for the given backbone + config.

    num_folds=-1 is resolved to NUM_OF_PATIENTS_PER_DATASET[dataset], matching
    the same logic run.py applies at the start of its config loop.
    """
    if backbone not in _BACKBONE_CONFIG_MODULE:
        raise ValueError(f"Unsupported backbone '{backbone}'. "
                         f"Supported: {list(_BACKBONE_CONFIG_MODULE)}")
    mod = importlib.import_module(_BACKBONE_CONFIG_MODULE[backbone])
    param_seed = {
        'backbone':  backbone,
        'config':    config_file,
        'train_mode': 'classifier_only',
        'num_folds':  num_folds,
        'seed':       0,
        'tune_fresh': 1,
        'ntrials':    1,
        'this_run_num': '0',
        'readstudyfrom': None,
        'hypertune':  0,
        'just_gen_dataset': 0,
        'cross_dataset_test': 0,
        'pretrained': 0,
        'overwrite_results': 0,
        'force_LODO': 0,
        'AID': 0,
        'combine_views_preds': 0,
        'views_path': None,
        'exp_name_rigid': None,
        'prefer_right': 0,
        'medication': 0,
        'metadata': [],
        'tuned_model_config': None,
    }
    params, _ = mod.generate_config(param_seed, config_file)

    # Resolve LOSO fold count (-1 → patient count), same as run.py line 472.
    if num_folds == -1:
        num_folds = const.NUM_OF_PATIENTS_PER_DATASET[params['dataset']]
    params['num_folds']   = num_folds
    params['num_classes'] = const.NUM_CLASSES_PER_DATASET[params['dataset']]
    params['LODO']        = False

    # ClassifierHead.__init__ expects these keys; set safe defaults since FC
    # layers are instantiated but never called during caching.
    params.setdefault('classifier_hidden_dims', [])
    params.setdefault('classifier_dropout', 0.0)

    return params


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(model: MotionEncoder,
                      loader: DataLoader,
                      device: torch.device) -> dict:
    """Run backbone + pooling (no FC) over all clips and collect results."""
    model.eval()
    all_feat, all_lbl, all_vidx, all_meta, all_pad = [], [], [], [], []

    for x, labels, video_idxs, metadata, pad_mask in tqdm(loader, leave=False):
        x        = x.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        # 1. Backbone forward (frozen)
        raw = model.backbone(x)

        # 2. Pooling only – ClassifierHead.forward with forward_classifier=False
        #    This applies backbone-specific temporal / joint pooling, returning
        #    a flat (B, feature_dim) vector ready for FC classification.
        feat = model.head(raw, valid_frame_mask=pad_mask.bool(),
                          forward_classifier=False)

        all_feat.append(feat.cpu().float().numpy())
        all_lbl.append(labels.numpy())
        all_vidx.append(video_idxs.numpy())
        all_meta.append(metadata.numpy().astype(np.float32))
        all_pad.append(pad_mask.cpu().numpy().astype(np.float32))

    return {
        'features':   np.concatenate(all_feat,  axis=0),
        'labels':     np.concatenate(all_lbl,   axis=0),
        'video_idxs': np.concatenate(all_vidx,  axis=0),
        'metadata':   np.concatenate(all_meta,  axis=0),
        'pad_masks':  np.concatenate(all_pad,   axis=0),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_all_folds(
    params: dict,
    backbone_name: str,
    folds,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
    out_root: str = 'assets/cached_features',
) -> None:
    """Extract and save frozen backbone features for every fold split.

    This is the callable entry-point used both by the CLI (main()) and by
    run.py's auto-cache logic in _build_splits_with_cache().

    Args:
        params:        Config dict as returned by _load_params (or run.py's
                       generate_config).  Must have 'num_folds' already
                       resolved to a positive integer (not -1).
        backbone_name: Backbone identifier, e.g. 'motionbert'.
        folds:         Iterable of fold indices to cache (1-based).
        device:        Torch device for the frozen backbone forward pass.
        batch_size:    Batch size for feature extraction (no grad, so large
                       values like 256 are safe).
        num_workers:   DataLoader worker count.
        out_root:      Root directory for cached .npz files.
    """
    out_dir = make_cache_dir(out_root, backbone_name, params)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cache] Features will be written to: {out_dir}")

    print(f"[cache] Loading {backbone_name} backbone ...")
    backbone = load_pretrained_backbone(params, backbone_name)
    model    = MotionEncoder(backbone=backbone, params=params,
                             num_classes=params['num_classes'],
                             train_mode='classifier_only')
    model    = model.to(device)
    model.eval()

    num_folds = params['num_folds']
    for fold in folds:
        names_path = os.path.join(out_dir, f"fold{fold}_video_names.json")

        for split in ('train', 'eval'):
            out_path     = os.path.join(out_dir, f"fold{fold}_{split}.npz")
            npz_exists   = os.path.exists(out_path)
            names_exists = os.path.exists(names_path)

            if npz_exists and names_exists:
                print(f"[cache][SKIP] {out_path} already exists.")
                continue

            print(f"[cache] Fold {fold}/{num_folds}  split={split} ...")
            train_ds, eval_ds = dataset_factory(params, backbone_name, fold)
            ds = train_ds if split == 'train' else eval_ds

            # Save per-clip video names sidecar (once per fold).
            if not names_exists and hasattr(ds, 'video_names'):
                with open(names_path, 'w') as f:
                    _json.dump(list(ds.video_names), f)
                print(f"  -> {len(ds.video_names)} video names saved to {names_path}")

            if npz_exists:
                print(f"[cache][SKIP] {out_path} already exists – skipping extraction.")
                continue

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
            )

            data = _extract_features(model, loader, device)
            np.savez_compressed(out_path, **data)
            feat_dim = data['features'].shape[1]
            print(f"  -> {data['features'].shape[0]} clips, "
                  f"feat_dim={feat_dim}  saved to {out_path}")

    print("[cache] Feature caching complete.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute frozen backbone features for all fold splits.",
    )
    parser.add_argument('--backbone',    required=True,
                        help='Backbone name, e.g. potr or motionbert')
    parser.add_argument('--config',      required=True,
                        help='Config JSON filename under configs/<backbone>/, '
                             'e.g. BMCLab.json')
    parser.add_argument('--num_folds',   type=int, default=-1,
                        help='Number of CV folds. -1 = LOSO (auto-detect from dataset).')
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--batch_size',  type=int, default=256,
                        help='Batch size for feature extraction (no grad)')
    parser.add_argument('--out_root',    default='assets/cached_features',
                        help='Root directory for cached .npz files')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[cache] Using device: {device}")

    params = _load_params(args.backbone, args.config, args.num_folds)
    folds  = range(1, params['num_folds'] + 1)

    cache_all_folds(
        params,
        args.backbone,
        folds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_root=args.out_root,
    )


if __name__ == '__main__':
    main()
