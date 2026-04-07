"""
train_gaitvae_phase1_sanity.py — Phase-1-only VAE sanity check (Lightning)
=========================================================================

Purpose
-------
Isolate the **Phase 1** training objective (standard VAE on the full sequence:
`full_encoder` + `decoder`, KL to N(0,I), β annealing) without Phase 2 / VAEAC.
Use this to verify that **MotionCLIP-initialized full encoder + decoder** can
achieve reasonable reconstruction **before** debugging masked-encoder / VAEAC
issues.

This uses the same `GaitVAEACModule(..., phase="phase1")` path as
`train_gaitvae_lightning.py` Phase 1, so metrics and checkpoints are comparable.

Usage
-----
# Defaults (BMCLab, MotionCLIP init, Phase-1-only):
python train_gaitvae_phase1_sanity.py

# Same config JSON as the two-phase script (phase2_* keys are ignored):
python train_gaitvae_phase1_sanity.py --config configs/gaitvae/BMCLab.json

# Multi-GPU:
python train_gaitvae_phase1_sanity.py --config configs/gaitvae/BMCLab.json \\
    --devices 0,1
"""

import argparse
import datetime
import json
import os

import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from const import path
from data.dataloaders import collate_fn
from model.gaitvae.vaeac import GaitVAEAC
from train_gaitvae_lightning import (
    GaitVAEACModule,
    get_train_val_datasets,
    make_trainer,
)
from utility.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase-1-only GaitVAE sanity run (full_encoder + decoder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config; same keys as BMCLab.json (phase2_* ignored).")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated GPU indices.")
    parser.add_argument("--dataset", type=str, default="BMCLab")
    parser.add_argument("--num_folds", type=int, default=6)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--motionclip_ckpt", type=str,
        default=os.path.join(
            path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH,
            "motionclip", "motionclip_encoder_checkpoint_0100.pth.tar",
        ),
    )
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60,
                        help="Phase-1 training epochs (maps to phase1_epochs in module).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument(
        "--lambda_vel", type=float, default=0.05,
        help="Weight on MSE of per-frame velocity (delta x) vs ground truth.",
    )
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="./experiment_outs/gaitvae_phase1_sanity",
    )
    parser.add_argument("--experiment_name", type=str, default="GaitVAE_P1_sanity")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--vis_every_n_epochs", type=int, default=10)
    parser.add_argument("--vis_n_examples", type=int, default=3)

    pre_args, _ = parser.parse_known_args()
    if pre_args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pre_args.devices
    if pre_args.config is not None:
        with open(pre_args.config) as f:
            cfg = {k: v for k, v in json.load(f).items()
                   if not k.startswith("_")}
        # Map two-phase script key to this script's --epochs
        if "phase1_epochs" in cfg and "epochs" not in cfg:
            cfg["epochs"] = cfg["phase1_epochs"]
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    # GaitVAEACModule reads phase1_epochs for documentation hooks; mirror --epochs
    args.phase1_epochs = args.epochs
    # Unused but may exist if config was written for two-phase trainer
    if not hasattr(args, "lambda_kl"):
        args.lambda_kl = 1.0
    if not hasattr(args, "lr_full_encoder"):
        args.lr_full_encoder = 1e-5
    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    run_tag = (
        f"p1_sanity_{args.dataset}_fold{args.fold}of{args.num_folds}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ckpt_dir = os.path.join(args.checkpoint_dir, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print("\n" + "=" * 60)
    print("GaitVAE Phase-1 sanity (full_encoder + decoder only)")
    print(run_tag)
    print("=" * 60)
    print(json.dumps(vars(args), indent=4))
    print("=" * 60 + "\n")

    logger = False
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_tag,
            config=vars(args),
            save_dir=ckpt_dir,
        )

    print("[INFO] Loading datasets...")
    train_dataset, eval_dataset = get_train_val_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        pin_memory=True, num_workers=4,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True, num_workers=4,
    )
    print(f"[INFO] Train: {len(train_dataset)}  Val: {len(eval_dataset)}")

    model = GaitVAEAC(
        njoints=25, nfeats=6,
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, lambda_kl=args.lambda_kl,
    )
    print("[INFO] Loading MotionCLIP weights into full_encoder...")
    model.load_full_encoder_from_motionclip(args.motionclip_ckpt)

    module = GaitVAEACModule(model, args, phase="phase1")
    periodic_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=10,
        filename="phase1_epoch{epoch:04d}",
        save_top_k=-1,
    )
    trainer = make_trainer(
        n_epochs=args.epochs,
        n_gpus=n_gpus,
        ckpt_dir=ckpt_dir,
        logger=logger,
        monitor_key="val/P1/total",
        phase_tag="phase1_sanity",
        extra_callbacks=[periodic_ckpt],
        find_unused_parameters=True,
    )
    trainer.fit(module, train_loader, eval_loader)
    print(f"\n[INFO] Done. Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
