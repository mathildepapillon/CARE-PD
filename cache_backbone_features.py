"""
cache_backbone_features.py
==========================
Pre-computes frozen backbone features for every clip in every fold split and
saves them to disk.  Subsequent classifier training loads these cached vectors
instead of running the expensive backbone + preprocessing on every epoch.

Each saved .npz contains per-clip pooled backbone features (post-backbone
projection, *pre*-FC layers), so classifier training only needs to run the
tiny classifier head on each batch (~1 ms/epoch instead of ~500 s/epoch).

Usage (BMCLab, 6-fold, POTR backbone, GPU 4):

    python cache_backbone_features.py \
        --backbone potr \
        --config train_BMCLab_test_BMCLab_6fold.json \
        --num_folds 6 \
        --device cuda:4

Output files are written to:
    assets/cached_features/<backbone>/<experiment_name>/<dataset>_<num_folds>fold/
        fold<N>_train.npz
        fold<N>_eval.npz

Each .npz has keys:
    features   float32  (N, feature_dim)  — pooled backbone output per clip
    labels     int64    (N,)
    video_idxs int64    (N,)
    metadata   float32  (N, M)  — M=0 when no metadata
    pad_masks  float32  (N, T)
"""

import argparse
import importlib
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


def _load_params(backbone: str, config_file: str, num_folds: int) -> dict:
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
    params['num_folds']   = num_folds
    params['num_classes'] = const.NUM_CLASSES_PER_DATASET[params['dataset']]
    params['LODO']        = False

    # `ClassifierHead.__init__` expects these keys.  The normal run.py path
    # injects them from the Optuna best-params JSON; we set safe defaults here
    # because the FC layers are instantiated but never called during caching.
    params.setdefault('classifier_hidden_dims', [])
    params.setdefault('classifier_dropout', 0.0)

    return params


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',   required=True,
                        help='Backbone name, e.g. potr')
    parser.add_argument('--config',     required=True,
                        help='Config JSON filename, e.g. train_BMCLab_test_BMCLab_6fold.json')
    parser.add_argument('--num_folds',  type=int, default=6)
    parser.add_argument('--device',     default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for feature extraction (no grad, so large is fine)')
    parser.add_argument('--out_root',   default='assets/cached_features',
                        help='Root directory for cached .npz files')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    params = _load_params(args.backbone, args.config, args.num_folds)

    out_dir = os.path.join(
        args.out_root,
        args.backbone,
        params['experiment_name'],
        f"{params['dataset']}_{args.num_folds}fold",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Features will be cached to: {out_dir}")

    # Build model once (backbone is frozen)
    print(f"[INFO] Loading {args.backbone} backbone ...")
    backbone = load_pretrained_backbone(params, args.backbone)
    model    = MotionEncoder(backbone=backbone, params=params,
                             num_classes=params['num_classes'],
                             train_mode='classifier_only')
    model    = model.to(device)
    model.eval()

    for fold in range(1, args.num_folds + 1):
        # video_names sidecar: one JSON per fold (same for train and eval because
        # video_name_to_index covers all videos in the fold, not just one split).
        names_path = os.path.join(out_dir, f"fold{fold}_video_names.json")

        for split in ('train', 'eval'):
            out_path = os.path.join(out_dir, f"fold{fold}_{split}.npz")
            npz_exists   = os.path.exists(out_path)
            names_exists = os.path.exists(names_path)

            if npz_exists and names_exists:
                print(f"[SKIP] {out_path} + video_names already exist.")
                continue

            print(f"[INFO] Fold {fold}/{args.num_folds}  split={split} ...")
            train_ds, eval_ds = dataset_factory(params, args.backbone, fold)
            ds = train_ds if split == 'train' else eval_ds

            # --- Save video_names sidecar (once per fold, both splits share it) ---
            # train.py accesses dataset.video_names[video_idx] where video_idx is
            # the value from video_name_to_index — which is a CLIP index (the last
            # clip position for each unique video name).  So we save the full
            # per-clip names list so that names[clip_idx] returns the right name.
            if not names_exists and hasattr(ds, 'video_names'):
                import json as _json
                with open(names_path, 'w') as f:
                    _json.dump(list(ds.video_names), f)
                print(f"  → {len(ds.video_names)} per-clip video names saved to {names_path}")

            if npz_exists:
                print(f"[SKIP] {out_path} already exists – skipping feature extraction.")
                continue

            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            data = _extract_features(model, loader, device)
            np.savez_compressed(out_path, **data)
            feat_dim = data['features'].shape[1]
            print(f"  → {data['features'].shape[0]} clips, feat_dim={feat_dim}  saved to {out_path}")

    print("[INFO] Feature caching complete.")


if __name__ == '__main__':
    main()
