"""
train_actor_cvae.py — ACTOR-aligned CVAE for motion sequences
==============================================================

Data modes
----------
``carepd``  CARE-PD clinical H36M-style folds (17 joints × 3 XYZ).
``6dsmpl``  CARE-PD 6D_SMPL rotations — ACTOR-parity mode (24 joints × 6D).

ACTOR parity (mode = ``6dsmpl``)
---------------------------------
This mode replicates the exact Mathux/ACTOR training objective:

  Representation : 6D rotation matrices, 24 SMPL joints × 6 features/joint.
  Loss           : rc + rcxyz + kl  (weights: 1, 1, 1e-5)
                    rc     = MSE on raw 6D rotations (matches ACTOR's rc)
                    rcxyz  = MSE on SMPL FK joint positions (matches ACTOR's rcxyz)
                    kl     = -0.5 * sum(1 + logvar - μ² - exp(logvar))  [sum, not mean]
  Optimiser      : AdamW, lr=1e-4, weight_decay=1e-4, no scheduler
  Architecture   : Transformer encoder + decoder, num_layers=8, latent_dim=256

The SMPL FK is performed by ``model/actor/rotation2xyz.py`` using the
SMPL neutral body model at
``data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl``.

Metrics (each epoch)
---------------------
``train/*`` and ``val/*`` include:
  mpjpe             — Mean per-joint L2 error on valid frames.
                      For rot6d: computed on FK XYZ output.
  recon_joint_motion — Mean frame-delta speed of reconstruction.
  gt_joint_motion    — Mean frame-delta speed of ground truth.
  (individual losses: rc, rcxyz, kl, mixed)

Recon GIFs
----------
With ``--recon_gif_every_n_epochs`` (default 10), saves one train + one val GIF
per interval under ``<run>/recon_gifs/``.  For rot6d mode the stick figures are
drawn from FK XYZ positions so the animation reflects true spatial articulation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from data.dataloaders import collate_fn
from model.actor.cvae import ActorCVAE
from model.actor.cvae_data import (
    actor_batch_from_carepd,
    actor_batch_from_6dsmpl,
    get_carepd_datasets,
    get_6dsmpl_datasets,
)
from model.actor.transformer_arch import Decoder_TRANSFORMER, Encoder_TRANSFORMER
from train_gaitvae_lightning import make_trainer
from utility.utils import set_random_seed

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from visualize_actor_cvae import save_actor_recon_gifs  # noqa: E402


def masked_mpjpe(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-joint L2 error (MPJPE) over valid frames. Shapes: x (B,J,F,T), mask (B,T)."""
    err = torch.linalg.norm(x_hat - x, dim=2)
    m = mask.unsqueeze(1).expand_as(err)
    return err.masked_select(m).mean()


def masked_mean_joint_motion(
    x_bjft: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean over joints of ||x[t]-x[t-1]||_2, averaged over batch and valid consecutive pairs."""
    if x_bjft.shape[-1] < 2:
        return x_bjft.new_zeros(())
    d = x_bjft[..., 1:] - x_bjft[..., :-1]
    vel = torch.linalg.norm(d, dim=2)
    pair = mask[:, :-1] & mask[:, 1:]
    m = pair.unsqueeze(1).expand_as(vel)
    sel = vel.masked_select(m)
    if sel.numel() == 0:
        return x_bjft.new_zeros(())
    return sel.mean()


class ActorCVAEModule(pl.LightningModule):
    """PyTorch Lightning module wrapping ActorCVAE.

    Args:
        model:        Constructed ``ActorCVAE`` instance.
        lr:           AdamW learning rate.
        data_mode:    ``"carepd"`` or ``"6dsmpl"``.  Controls how the raw
                      dataloader batch is converted to an ACTOR batch dict.
        lr_scheduler: ``"none"`` (ACTOR default) or ``"step"``.
    """

    def __init__(
        self,
        model: ActorCVAE,
        lr: float,
        *,
        data_mode: str = "carepd",
        lr_scheduler: str = "none",
        lr_step_size: int = 10,
        lr_gamma: float = 0.9,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.data_mode = data_mode
        self.lr_scheduler_name = lr_scheduler
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

    def _build_actor_batch(
        self,
        x: torch.Tensor,
        lab: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> dict:
        """Convert a raw dataloader batch to the ACTOR batch dict format."""
        device = x.device
        pad_mask = pad_mask.bool()
        if self.data_mode == "6dsmpl":
            b = actor_batch_from_6dsmpl(x, pad_mask, device)
        else:
            b = actor_batch_from_carepd(x, pad_mask, self.model.num_classes, device)
        # Always use real clinical class labels (0/1/2 = UPDRS score).
        b["y"] = lab.long().to(device)
        return b

    def _step(self, batch, train: bool):
        x, lab, _vidx, _meta, pad_mask = batch
        b = self._build_actor_batch(x, lab, pad_mask)

        out = self.model(b)
        loss, ld = self.model.compute_loss(out)
        mask = out["mask"]

        with torch.no_grad():
            # For rot6d mode, MPJPE and motion metrics operate on FK XYZ output
            # so the numbers are in metres (same units as the ground-truth XYZ).
            # For XYZ mode, x_xyz == x, so this is identical to the old behaviour.
            x_for_metric   = out.get("x_xyz",      out["x"])
            out_for_metric = out.get("output_xyz",  out["output"])
            mpjpe        = masked_mpjpe(out_for_metric, x_for_metric, mask)
            recon_motion = masked_mean_joint_motion(out_for_metric, mask)
            gt_motion    = masked_mean_joint_motion(x_for_metric,   mask)

        prefix = "train" if train else "val"
        self.log(f"{prefix}/mpjpe", mpjpe, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            f"{prefix}/recon_joint_motion",
            recon_motion,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/gt_joint_motion",
            gt_motion,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss, ld

    def training_step(self, batch, batch_idx):
        loss, ld = self._step(batch, train=True)
        self.log_dict({f"train/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ld = self._step(batch, train=False)
        self.log_dict({f"val/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        cm = self.trainer.callback_metrics
        keys = ("val/mpjpe", "val/recon_joint_motion", "val/gt_joint_motion")
        parts = []
        for k in keys:
            if k not in cm:
                continue
            v = cm[k]
            parts.append(f"{k}={float(v):.6f}")
        if parts:
            print(
                f"[ActorCVAE epoch {self.current_epoch}] " + "  ".join(parts),
                flush=True,
            )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        if self.lr_scheduler_name == "none":
            return opt
        if self.lr_scheduler_name == "step":
            sch = torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.lr_step_size, gamma=self.lr_gamma,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_name}")


class ActorCvaeReconGifCallback(pl.Callback):
    """Save one train + one val reconstruction GIF every N epochs.

    For ``data_mode='6dsmpl'`` the GIF stick figures are drawn from the
    FK-derived XYZ positions (``batch['output_xyz']`` / ``batch['x_xyz']``)
    so the animation reflects true spatial articulation, not raw rotation
    values.
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        out_dir: str,
        *,
        every_n_epochs: int = 10,
        fps: int = 12,
    ):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = out_dir
        self.every_n_epochs = every_n_epochs
        self.fps = fps

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: ActorCVAEModule) -> None:
        if self.every_n_epochs <= 0:
            return
        if getattr(trainer, "global_rank", 0) != 0:
            return
        ep = trainer.current_epoch + 1
        if ep % self.every_n_epochs != 0:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        device = next(pl_module.model.parameters()).device
        x_t, lab_t, _, _, pm_t = next(iter(self.train_loader))
        x_v, lab_v, _, _, pm_v = next(iter(self.val_loader))
        tag = f"ep{ep:04d}"
        paths: list[str] = []
        paths.extend(
            save_actor_recon_gifs(
                pl_module.model,
                x_t,
                pm_t,
                device,
                self.out_dir,
                batch_idx=0,
                n_examples=1,
                fps=self.fps,
                output_filenames=[f"{tag}_train.gif"],
                verbose=False,
                data_mode=pl_module.data_mode,
                labels=lab_t,
            )
        )
        paths.extend(
            save_actor_recon_gifs(
                pl_module.model,
                x_v,
                pm_v,
                device,
                self.out_dir,
                batch_idx=0,
                n_examples=1,
                fps=self.fps,
                output_filenames=[f"{tag}_val.gif"],
                verbose=False,
                data_mode=pl_module.data_mode,
                labels=lab_v,
            )
        )
        print(
            f"[ActorCVAE recon GIFs] epoch {ep} -> {paths}",
            flush=True,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="ACTOR-style CVAE for motion sequences (H36M XYZ or 6D_SMPL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file; keys override argument defaults.")
    p.add_argument("--devices", type=str, default=None,
                   help="CUDA_VISIBLE_DEVICES string (e.g. '0,1').")
    p.add_argument(
        "--data_mode", type=str, default="carepd",
        choices=("carepd", "6dsmpl"),
        help=(
            "carepd:  CARE-PD H36M-style clinical folds (17×3 XYZ). "
            "6dsmpl:  CARE-PD 6D_SMPL rotations — ACTOR-parity mode "
            "(24 joints × 6D, rc+rcxyz+kl loss with SMPL FK)."
        ),
    )
    p.add_argument("--dataset", type=str, default="BMCLab",
                   choices=["BMCLab", "T-SDU-PD", "PD-GaM", "3DGait"],
                   help="CARE-PD dataset name (carepd / 6dsmpl modes).")
    p.add_argument(
        "--carepd_pose_npz", type=str, default=None,
        help="Override pose NPZ path (default: from const/path.py for --dataset).",
    )
    p.add_argument(
        "--carepd_labels_pkl", type=str, default=None,
        help="Override labels PKL path.",
    )
    p.add_argument("--num_folds", type=int, default=6)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source_seq_len", type=int, default=81,
                   help="Clip length in frames fed to the model. Must match the "
                        "target classifier's source_seq_len (81 for PoseFormerV2 / "
                        "MixSTE / MotionAGFormer, 80 for POTR, 90 for MotionBERT).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="AdamW learning rate (ACTOR uses 1e-4).")
    p.add_argument(
        "--lr_scheduler", type=str, default="none", choices=("none", "step"),
        help="LR schedule: 'none' = plain AdamW (ACTOR default); 'step' = StepLR.",
    )
    p.add_argument("--lr_step_size", type=int, default=10)
    p.add_argument("--lr_gamma", type=float, default=0.9)
    p.add_argument("--latent_dim", type=int, default=256,
                   help="Latent space dimensionality.")
    p.add_argument("--ff_size", type=int, default=1024,
                   help="Transformer feed-forward hidden size.")
    p.add_argument(
        "--num_layers", type=int, default=8,
        help="Transformer depth (ACTOR README uses 8).",
    )
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--num_classes", type=int, default=3,
        help=(
            "Action/class count for class conditioning.  "
            "For 6dsmpl / carepd modes, set to the number of UPDRS classes (3). "
            "Set to 1 for unconditional (y=0 always)."
        ),
    )
    # njoints / nfeats / pose_rep: auto-set per data_mode in main().
    # Users can still override for non-standard configs.
    p.add_argument("--njoints", type=int, default=None,
                   help="Override joint count (auto-set from data_mode if not given).")
    p.add_argument("--nfeats", type=int, default=None,
                   help="Override feature dim (auto-set from data_mode if not given).")
    p.add_argument("--pose_rep", type=str, default=None,
                   help="Override pose_rep (auto-set from data_mode if not given).")

    # --- Loss weights ---
    p.add_argument(
        "--lambda_rc", type=float, default=1.0,
        help="MSE on raw representation (rotations or XYZ).",
    )
    p.add_argument(
        "--lambda_rcxyz", type=float, default=None,
        help=(
            "MSE on FK-derived XYZ positions.  "
            "Default: 1.0 for 6dsmpl, 0.0 for other modes."
        ),
    )
    p.add_argument(
        "--lambda_kl", type=float, default=1e-5,
        help="KL weight (sum reduction, matching ACTOR).",
    )
    p.add_argument(
        "--lambda_rr", type=float, default=None,
        help=(
            "Root-relative MSE.  Default: 0.0 for both modes "
            "(rcxyz covers articulation in 6dsmpl; vel covers it in carepd)."
        ),
    )
    p.add_argument(
        "--lambda_vel", type=float, default=None,
        help=(
            "Frame-delta MSE on raw representation.  Default: 1.0 for 6dsmpl; "
            "5.0 for carepd."
        ),
    )
    p.add_argument(
        "--lambda_velxyz", type=float, default=None,
        help=(
            "Frame-delta MSE on FK XYZ positions.  Prevents mean-pose collapse "
            "on gait-only data.  Default: 10.0 for 6dsmpl, 0.0 otherwise."
        ),
    )

    # --- Output / logging ---
    p.add_argument("--checkpoint_dir", type=str, default="./experiment_outs/actor_cvae")
    p.add_argument("--experiment_name", type=str, default="ActorCVAE")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--recon_gif_every_n_epochs", type=int, default=10,
        help="Save recon GIFs every N epochs (0 to disable).",
    )
    p.add_argument("--recon_gif_fps", type=int, default=12)
    p.add_argument(
        "--smpl_path", type=str, default=None,
        help="Override SMPL_NEUTRAL.pkl path (6dsmpl mode only).",
    )

    # Two-pass parse: load config first so its values can be overridden by CLI.
    pre, _ = p.parse_known_args()
    if pre.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pre.devices
    if pre.config is not None:
        with open(pre.config) as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        p.set_defaults(**cfg)

    return p.parse_args()


def _resolve_mode_defaults(args):
    """Fill in data-mode-dependent argument defaults that cannot be expressed
    with argparse defaults (because they depend on another argument's value).

    Mutates ``args`` in place.
    """
    if args.data_mode == "6dsmpl":
        # ACTOR-parity: 24 SMPL joints × 6D rotations
        if args.njoints is None:
            args.njoints = 24
        if args.nfeats is None:
            args.nfeats = 6
        if args.pose_rep is None:
            args.pose_rep = "rot6d"
        # Losses: rc + rcxyz + vel + velxyz + kl
        if args.lambda_rcxyz is None:
            args.lambda_rcxyz = 1.0
        if args.lambda_rr is None:
            args.lambda_rr = 0.0
        if args.lambda_vel is None:
            args.lambda_vel = 1.0
        if not hasattr(args, "lambda_velxyz") or args.lambda_velxyz is None:
            args.lambda_velxyz = 10.0
    else:
        # carepd mode: 17 joints × 3 XYZ
        if args.njoints is None:
            args.njoints = 17
        if args.nfeats is None:
            args.nfeats = 3
        if args.pose_rep is None:
            args.pose_rep = "xyz"
        if args.lambda_rcxyz is None:
            args.lambda_rcxyz = 0.0
        if args.lambda_rr is None:
            args.lambda_rr = 0.0
        if args.lambda_vel is None:
            args.lambda_vel = 5.0
        if not hasattr(args, "lambda_velxyz") or args.lambda_velxyz is None:
            args.lambda_velxyz = 0.0


def main():
    args = parse_args()
    _resolve_mode_defaults(args)
    set_random_seed(args.seed)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    tag_ds = f"{args.dataset}_fold{args.fold}"
    run_tag = f"actor_{args.data_mode}_{tag_ds}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build loss-weight dict (drop zero-weight terms) -----------------
    lambdas = {}
    if args.lambda_rc     > 0: lambdas["rc"]     = args.lambda_rc
    if args.lambda_kl     > 0: lambdas["kl"]     = args.lambda_kl
    if args.lambda_rcxyz  > 0: lambdas["rcxyz"]  = args.lambda_rcxyz
    if args.lambda_rr     > 0: lambdas["rr"]     = args.lambda_rr
    if args.lambda_vel    > 0: lambdas["vel"]    = args.lambda_vel
    if getattr(args, "lambda_velxyz", 0.0) > 0:
        lambdas["velxyz"] = args.lambda_velxyz

    # ---- Build Transformer encoder + decoder (shared kwargs) -------------
    common = dict(
        modeltype="cvae",
        njoints=args.njoints,
        nfeats=args.nfeats,
        num_frames=0,
        num_classes=args.num_classes,
        translation=True,
        pose_rep=args.pose_rep,
        glob=True,
        glob_rot=[3.141592653589793, 0, 0],
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ablation=None,
        activation="gelu",
    )
    enc = Encoder_TRANSFORMER(**common)
    dec = Decoder_TRANSFORMER(**common)

    # ---- Build rotation2xyz (6dsmpl only) --------------------------------
    rotation2xyz = None
    if args.data_mode == "6dsmpl":
        from model.actor.rotation2xyz import Rotation2xyz
        r2xyz_kwargs = {}
        if args.smpl_path:
            r2xyz_kwargs["smpl_path"] = args.smpl_path
        rotation2xyz = Rotation2xyz(device, **r2xyz_kwargs)

    model = ActorCVAE(
        enc, dec,
        lambdas=lambdas,
        latent_dim=args.latent_dim,
        device=device,
        pose_rep=args.pose_rep,
        num_classes=args.num_classes,
        rotation2xyz=rotation2xyz,
    ).to(device)

    # ---- Load datasets ---------------------------------------------------
    if args.data_mode == "6dsmpl":
        train_ds, val_ds = get_6dsmpl_datasets(args)
    else:
        train_ds, val_ds = get_carepd_datasets(args)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # ---- Logging ---------------------------------------------------------
    logger = False
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_tag,
            config=vars(args),
            save_dir=ckpt_dir,
        )

    # ---- Lightning module + callbacks ------------------------------------
    module = ActorCVAEModule(
        model, lr=args.lr,
        data_mode=args.data_mode,
        lr_scheduler=args.lr_scheduler,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
    )
    extra_cb: list[pl.Callback] = []
    if args.recon_gif_every_n_epochs > 0:
        extra_cb.append(
            ActorCvaeReconGifCallback(
                train_loader,
                val_loader,
                os.path.join(ckpt_dir, "recon_gifs"),
                every_n_epochs=args.recon_gif_every_n_epochs,
                fps=args.recon_gif_fps,
            )
        )
    trainer = make_trainer(
        n_epochs=args.epochs,
        n_gpus=n_gpus,
        ckpt_dir=ckpt_dir,
        logger=logger,
        monitor_key="val/mixed",
        phase_tag="actor_cvae",
        find_unused_parameters=False,
        extra_callbacks=extra_cb or None,
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"Done. Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
