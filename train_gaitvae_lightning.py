"""
train_gaitvae_lightning.py — Two-Phase GaitVAEAC Training with PyTorch Lightning
=================================================================================

Identical training logic to train_gaitvae.py but all distributed training,
checkpointing, logging, and epoch loops are delegated to Lightning — removing
~800 lines of boilerplate.

Usage
-----
# Single GPU:
python train_gaitvae_lightning.py --config configs/gaitvae/BMCLab.json

# Multi-GPU (runs on physical GPUs 4,5,6,7):
python train_gaitvae_lightning.py --config configs/gaitvae/BMCLab.json --devices 4,5,6,7

# Skip Phase 1, resume from saved weights:
python train_gaitvae_lightning.py --config configs/gaitvae/BMCLab.json \\
    --phase2_resume path/to/phase1_best.ckpt
"""

import argparse
import datetime
import json
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from const import path, const
from data.dataloaders import dataset_factory, collate_fn
from model.gaitvae.vaeac import GaitVAEAC
from model.gaitvae.visualize import make_recon_images
from utility.utils import set_random_seed

_SUPPORTED_DATASETS = ["BMCLab", "T-SDU-PD", "PD-GaM", "3DGait"]


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def masked_mse(x_hat: torch.Tensor, x: torch.Tensor,
               pad_mask: torch.Tensor) -> torch.Tensor:
    """MSE over real (non-padded) frames only."""
    mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x).float()
    return ((x_hat - x) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def masked_velocity_mse(x_hat: torch.Tensor, x: torch.Tensor,
                        pad_mask: torch.Tensor) -> torch.Tensor:
    """MSE on first temporal differences; only where both frames are valid."""
    if x.shape[1] < 2:
        return x_hat.new_zeros(())
    dx = x[:, 1:] - x[:, :-1]
    dx_hat = x_hat[:, 1:] - x_hat[:, :-1]
    pair = pad_mask[:, :-1] & pad_mask[:, 1:]
    m = pair.unsqueeze(-1).unsqueeze(-1).float().expand_as(dx)
    return ((dx_hat - dx) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def annealed_lambda_kl(epoch: int, warmup_epochs: int,
                       target: float) -> float:
    """Linear ramp from 0 → target over warmup_epochs (1-indexed epoch)."""
    if warmup_epochs <= 0:
        return target
    return target * min(epoch / warmup_epochs, 1.0)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_params(dataset_name, num_folds, batch_size,
                 motionclip_ckpt, experiment_name):
    return {
        "backbone": "motionclip", "dataset": dataset_name,
        "data_type": "6DSMPL", "experiment_name": experiment_name,
        "in_data_dim": 6, "source_seq_len": 60, "num_folds": num_folds,
        "data_centered": False, "merge_last_dim": False,
        "simulate_confidence_score": False, "data_norm": False,
        "select_middle": False, "views": [""],
        "LODO": False, "hypertune": False,
        "cross_dataset_test": False, "AID": False,
        "medication": False, "metadata": [],
        "mirror_prob": 0.0, "rotation_prob": 0.0,
        "noise_prob": 0.0, "axis_mask_prob": 0.0,
        "data_path": [path.POSE_AND_LABEL[dataset_name]["6DSMPL"]["PATH_POSES"]],
        "labels_path": path.POSE_AND_LABEL[dataset_name]["6DSMPL"]["PATH_LABELS"],
        "batch_size": batch_size, "dim_rep": 512,
        "model_checkpoint_path": motionclip_ckpt,
    }


def get_train_val_datasets(args):
    if args.dataset == "all":
        train_list, eval_list = [], []
        for ds_name in _SUPPORTED_DATASETS:
            p = build_params(ds_name, args.num_folds, args.batch_size,
                             args.motionclip_ckpt, args.experiment_name)
            tr, ev = dataset_factory(p, "motionclip", args.fold)
            train_list.append(tr)
            eval_list.append(ev)
        return (torch.utils.data.ConcatDataset(train_list),
                torch.utils.data.ConcatDataset(eval_list))
    else:
        assert args.dataset in _SUPPORTED_DATASETS
        p = build_params(args.dataset, args.num_folds, args.batch_size,
                         args.motionclip_ckpt, args.experiment_name)
        return dataset_factory(p, "motionclip", args.fold)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class GaitVAEACModule(pl.LightningModule):
    """
    One LightningModule covers both training phases.  Pass phase="phase1"
    for the standard VAE warmup and phase="phase2" for joint VAEAC training.

    Because Lightning wraps the LightningModule (not self.model) with DDP,
    self.model is always the unwrapped GaitVAEAC — no .module gymnastics.
    """

    def __init__(self, model: GaitVAEAC, args, phase: str = "phase1"):
        super().__init__()
        self.model = model
        self.args = args
        self.phase = phase
        self.effective_lambda_kl = 0.0
        self.effective_beta_kl = 0.0   # annealed in Phase 1 (mirrors Phase 2)
        self._vis_batch = None

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _unpack(self, batch):
        x, _labels, _vidx, _meta, pad_mask = batch
        pad_mask = pad_mask.bool()
        lengths = pad_mask.sum(dim=-1).long()
        return x, lengths, pad_mask

    def _phase1_loss(self, x, lengths, pad_mask):
        mu_f, logvar_f = self.model._full_encode(x, pad_mask)
        z = self.model.reparameterize(mu_f, logvar_f)
        x_hat = self.model.decoder(z, lengths, pad_mask)
        rc = masked_mse(x_hat, x, pad_mask)
        kl = -0.5 * (1 + logvar_f - mu_f.pow(2) - logvar_f.exp()).mean()
        loss = rc + self.effective_beta_kl * kl
        lam_v = float(getattr(self.args, "lambda_vel", 0.0) or 0.0)
        vel = x_hat.new_zeros(())
        if lam_v > 0.0:
            vel = masked_velocity_mse(x_hat, x, pad_mask)
            loss = loss + lam_v * vel
        return loss, {
            "rc": rc, "kl": kl, "vel": vel,
            "beta_kl": self.effective_beta_kl,
            "lambda_vel": lam_v,
            "total": loss,
        }

    def _phase2_loss(self, x, lengths, pad_mask, detach_full_encoder):
        batch_out = self.model(x, lengths, padding_mask=pad_mask)
        loss, ld = self.model.compute_loss(
            batch_out,
            detach_full_encoder=detach_full_encoder,
            lambda_kl=self.effective_lambda_kl,
        )
        return loss, ld

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, lengths, pad_mask = self._unpack(batch)
        if self.phase == "phase1":
            loss, ld = self._phase1_loss(x, lengths, pad_mask)
            prefix = "train/P1"
        else:
            loss, ld = self._phase2_loss(x, lengths, pad_mask,
                                         detach_full_encoder=False)
            prefix = "train/P2"
        self.log_dict({f"{prefix}/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, pad_mask = self._unpack(batch)

        # Cache first batch on rank-0 for visualisation
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._vis_batch = (x.detach().cpu(), pad_mask.detach().cpu())

        if self.phase == "phase1":
            loss, ld = self._phase1_loss(x, lengths, pad_mask)
            prefix = "val/P1"
        else:
            loss, ld = self._phase2_loss(x, lengths, pad_mask,
                                         detach_full_encoder=True)
            prefix = "val/P2"
        self.log_dict({f"{prefix}/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        if self.phase == "phase1":
            # Linear ramp from 0 → beta_kl over kl_warmup_epochs.
            # Holds beta=0 for the first few epochs so the decoder learns
            # temporal dynamics via pure reconstruction before the KL term
            # compresses z toward N(0,I) (which causes the static-pose
            # / posterior-collapse failure mode).
            self.effective_beta_kl = annealed_lambda_kl(
                self.current_epoch + 1,
                self.args.kl_warmup_epochs,
                self.args.beta_kl,
            )
        elif self.phase == "phase2":
            self.effective_lambda_kl = annealed_lambda_kl(
                self.current_epoch + 1,    # Lightning is 0-indexed
                self.args.kl_warmup_epochs,
                self.args.lambda_kl,
            )

    def on_validation_epoch_end(self):
        """Log skeleton reconstruction images to W&B periodically."""
        if not self.trainer.is_global_zero:
            return
        if self._vis_batch is None or not self.logger:
            return
        n = self.args.vis_every_n_epochs
        if n <= 0 or (self.current_epoch + 1) % n != 0:
            return
        x, pad_mask = self._vis_batch
        imgs = make_recon_images(
            self.model, x, pad_mask, self.device,
            n_examples=self.args.vis_n_examples,
        )
        if imgs:
            phase_tag = self.phase.replace("phase", "P")
            # Do not pass an explicit `step=` here — W&B auto-increments,
            # which avoids a non-monotonic step error when Phase 2 resets
            # current_epoch to 0 while the run's step counter is already
            # past Phase 1's total.
            self.logger.experiment.log(
                {f"val/{phase_tag}/recon_viz": imgs},
            )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        if self.phase == "phase1":
            params = (list(self.model.full_encoder.parameters())
                      + list(self.model.decoder.parameters()))
            opt = torch.optim.AdamW(params, lr=self.args.lr,
                                    weight_decay=1e-4)
        else:
            opt = torch.optim.AdamW(
                [
                    {
                        "params": list(self.model.full_encoder.parameters()),
                        "lr": self.args.lr_full_encoder,
                        "weight_decay": 1e-4,
                    },
                    {
                        "params": (
                            list(self.model.masked_encoder.parameters())
                            + list(self.model.decoder.parameters())
                        ),
                        "lr": self.args.lr,
                        "weight_decay": 1e-4,
                    },
                ]
            )
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def make_trainer(n_epochs, n_gpus, ckpt_dir, logger,
                 monitor_key, phase_tag, extra_callbacks=None,
                 find_unused_parameters=False):
    """Build a pl.Trainer with DDP (if >1 GPU), checkpointing, and logging."""
    strategy = (DDPStrategy(find_unused_parameters=find_unused_parameters)
                if n_gpus > 1 else "auto")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=monitor_key,
            save_top_k=1,
            filename=f"{phase_tag}_best",
            mode="min",
        ),
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    return pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu" if n_gpus > 0 else "cpu",
        devices=max(n_gpus, 1),
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )


# ---------------------------------------------------------------------------
# Argument parsing (identical surface to train_gaitvae.py)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GaitVAEAC (Lightning) on CARE-PD 6D_SMPL gait data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file; keys map to args below.")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated GPU indices, e.g. '4,5,6,7'.")
    parser.add_argument("--dataset", type=str, default="BMCLab",
                        help=f"One of {_SUPPORTED_DATASETS} or 'all'.")
    parser.add_argument("--num_folds",  type=int, default=6)
    parser.add_argument("--fold",       type=int, default=1)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument(
        "--motionclip_ckpt", type=str,
        default=os.path.join(
            path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH,
            "motionclip", "motionclip_encoder_checkpoint_0100.pth.tar",
        ),
    )
    parser.add_argument("--phase2_resume", type=str, default=None,
                        help="Lightning .ckpt to load before Phase 2 "
                             "(skips Phase 1).")
    parser.add_argument("--latent_dim",  type=int,   default=512)
    parser.add_argument("--num_layers",  type=int,   default=8)
    parser.add_argument("--num_heads",   type=int,   default=4)
    parser.add_argument("--ff_size",     type=int,   default=1024)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--phase1_epochs",    type=int,   default=50)
    parser.add_argument("--phase2_epochs",    type=int,   default=100)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--lr_full_encoder",  type=float, default=1e-5)
    parser.add_argument("--beta_kl",          type=float, default=1.0)
    parser.add_argument("--lambda_kl",        type=float, default=1.0)
    parser.add_argument("--kl_warmup_epochs", type=int,   default=20)
    parser.add_argument(
        "--lambda_vel", type=float, default=0.05,
        help="Phase 1 only: weight on MSE between consecutive-frame deltas "
             "(x_hat vs x). Reduces static 'mean pose' reconstructions.",
    )
    parser.add_argument("--checkpoint_dir",   type=str,
                        default="./experiment_outs/gaitvae")
    parser.add_argument("--experiment_name",  type=str, default="GaitVAE")
    parser.add_argument("--wandb_project",    type=str, default=None)
    parser.add_argument("--wandb_entity",     type=str, default=None)
    parser.add_argument("--wandb_run_name",   type=str, default=None)
    parser.add_argument("--vis_every_n_epochs", type=int, default=10)
    parser.add_argument("--vis_n_examples",     type=int, default=3)

    # Two-pass parse: apply --devices and --config before the full parse.
    pre_args, _ = parser.parse_known_args()
    if pre_args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pre_args.devices
    if pre_args.config is not None:
        with open(pre_args.config) as f:
            cfg = {k: v for k, v in json.load(f).items()
                   if not k.startswith("_")}
        parser.set_defaults(**cfg)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_random_seed(args.seed)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    run_tag = (
        f"{args.dataset}_fold{args.fold}of{args.num_folds}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ckpt_dir = os.path.join(args.checkpoint_dir, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print("\n" + "=" * 60)
    print(f"GaitVAEAC Lightning  |  {run_tag}")
    print("=" * 60)
    print(json.dumps(vars(args), indent=4))
    print("=" * 60 + "\n")

    # W&B logger (False = disable Lightning's default CSV logger too)
    logger = False
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_tag,
            config=vars(args),
            save_dir=ckpt_dir,
        )

    # Datasets and loaders
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

    # Model
    model = GaitVAEAC(
        njoints=25, nfeats=6,
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, lambda_kl=args.lambda_kl,
    )

    # ------------------------------------------------------------------
    # Phase 1 — standard VAE warmup
    # ------------------------------------------------------------------
    if args.phase2_resume is not None:
        print(f"[INFO] Skipping Phase 1 — loading: {args.phase2_resume}")
        # Load raw state dict saved by either script
        ckpt = torch.load(args.phase2_resume, map_location="cpu",
                          weights_only=False)
        sd = (ckpt.get("state_dict")
              or ckpt.get("model_state_dict")
              or ckpt)
        # Strip Lightning's "model." prefix if present
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

    elif args.phase1_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 1 — Standard VAE warmup  ({args.phase1_epochs} epochs)")
        print(f"{'='*60}")
        model.load_full_encoder_from_motionclip(args.motionclip_ckpt)
        module1 = GaitVAEACModule(model, args, phase="phase1")
        periodic_ckpt1 = ModelCheckpoint(
            dirpath=ckpt_dir,
            every_n_epochs=10,
            filename="phase1_epoch{epoch:04d}",
            save_top_k=-1,   # keep all periodic saves
        )
        trainer1 = make_trainer(
            n_epochs=args.phase1_epochs,
            n_gpus=n_gpus,
            ckpt_dir=ckpt_dir,
            logger=logger,
            monitor_key="val/P1/total",
            phase_tag="phase1",
            extra_callbacks=[periodic_ckpt1],
            find_unused_parameters=True,   # masked_encoder unused in Phase 1
        )
        trainer1.fit(module1, train_loader, eval_loader)
        # model weights are updated in-place; no checkpoint handoff needed

    else:
        print("[INFO] Phase 1 skipped — loading MotionCLIP weights.")
        model.load_full_encoder_from_motionclip(args.motionclip_ckpt)

    # ------------------------------------------------------------------
    # Phase 2 — joint VAEAC training
    # ------------------------------------------------------------------
    if args.phase2_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 2 — Joint VAEAC training  ({args.phase2_epochs} epochs)")
        print(f"{'='*60}")
        module2 = GaitVAEACModule(model, args, phase="phase2")
        periodic_ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            every_n_epochs=10,
            filename="phase2_epoch{epoch:04d}",
            save_top_k=-1,   # keep all periodic saves
        )
        trainer2 = make_trainer(
            n_epochs=args.phase2_epochs,
            n_gpus=n_gpus,
            ckpt_dir=ckpt_dir,
            logger=logger,
            monitor_key="val/P2/total",
            phase_tag="phase2",
            extra_callbacks=[periodic_ckpt],
        )
        trainer2.fit(module2, train_loader, eval_loader)

    print(f"\n[INFO] Done. Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
