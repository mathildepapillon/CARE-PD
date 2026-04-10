"""
train_actor_shap.py — ActorSHAP proper VAEAC training
======================================================

Implements the corrected VAEAC ELBO (Ivanov 2019 / Olsen et al. JMLR 2022):

  L = E_{z~q_ϕ(z|x,S)} [log p_θ(x_S|z,...)]
      − λ_kl · KL( q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S) )
      − λ_reg · prior_reg(μ_ψ, σ_ψ)

All three components are trained jointly (previous implementation froze q_ϕ):
  q_ϕ (CoalitionFullEncoder):   receives reconstruction + KL gradients.
  r_ψ (MaskedActorEncoder):     receives KL + prior-reg gradients only.
  p_θ (Decoder):                receives reconstruction gradient only.

Weights initialised from a trained ActorCVAE checkpoint (--actor_cvae_ckpt):
  - CoalitionFullEncoder._encoder → ActorCVAE encoder weights, target_marker=0
  - MaskedActorEncoder._encoder  → ActorCVAE encoder weights, mask_token=0
  - Decoder                      → ActorCVAE decoder weights

KL annealing: λ_kl ramps linearly 0 → --lambda_kl over --kl_warmup_epochs.

Key diagnostic metrics (logged to wandb and stdout):
  val/mu_div_across_S      — pairwise distance of μ_ψ for 8 random coalitions.
                             Rising value confirms r_ψ is becoming coalition-specific.
  val/mu_div_full_across_S — same for q_ϕ. Must rise BEFORE r_ψ can track it.
  val/completion_diversity — pose RMSE of masked joints across n completions.

Typical command:
  python train_actor_shap.py \\
      --actor_cvae_ckpt experiment_outs/actor_cvae/<run>/actor_cvae_best.ckpt \\
      --dataset BMCLab --fold 1 --epochs 150 --lr 1e-4 \\
      --lambda_kl 1.0 --kl_warmup_epochs 20 --lambda_reg 1e-6

Mirrors train_actor_cvae.py in structure (argparse → JSON → CLI pattern,
make_trainer, WandB logger, LightningModule _step pattern).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.dataloaders import collate_fn
from model.actor.actor_shap import ActorSHAP, CoalitionFullEncoder, MaskedActorEncoder
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.shap_masking import (
    sample_spatial_training_mask,
    sample_temporal_training_mask,
)
from model.actor.transformer_arch import Decoder_TRANSFORMER
from train_actor_cvae import masked_mean_joint_motion, masked_mpjpe
from train_gaitvae_lightning import make_trainer
from utility.utils import set_random_seed


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class ActorSHAPModule(pl.LightningModule):
    """PyTorch Lightning module for ActorSHAP proper VAEAC training.

    Trains all three VAEAC components jointly:
      q_ϕ (CoalitionFullEncoder):   lr           — reconstruction + KL gradients.
      r_ψ (MaskedActorEncoder):     lr           — KL + prior-reg gradients only.
      p_θ (Decoder):                lr_decoder   — reconstruction gradient only.

    Args:
        model:             ActorSHAP instance.
        lr:                learning rate for q_ϕ and r_ψ.
        lr_decoder:        learning rate for p_θ (typically lr/10; 0 = frozen).
        lambda_kl:         final KL weight (forward KL(q_ϕ‖r_ψ)) after annealing.
        kl_warmup_epochs:  epochs over which λ_kl ramps linearly 0 → lambda_kl.
        lambda_reg:        weight for r_ψ prior regularization (Olsen 2022, Sec 3.3.1).
        lambda_kl_full:    weight for KL(q_ϕ ‖ N(0,I)) — prevents σ_ϕ collapse.
        data_mode:         only "carepd" supported (H36M xyz).
        mask_axis:         "spatial" or "temporal" — coalition axis used during training.
        diversity_n_samples: completions drawn per val epoch for the diversity probe.
    """

    def __init__(
        self,
        model: ActorSHAP,
        lr: float,
        lr_decoder: float,
        *,
        lambda_kl: float = 1.0,
        kl_warmup_epochs: int = 20,
        lambda_reg: float = 1e-6,
        lambda_kl_full: float = 1e-4,
        data_mode: str = "carepd",
        mask_axis: str = "spatial",
        diversity_n_samples: int = 10,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_decoder = lr_decoder
        self.lambda_kl = lambda_kl
        self.kl_warmup_epochs = kl_warmup_epochs
        self.lambda_reg = lambda_reg
        self.lambda_kl_full = lambda_kl_full
        self.data_mode = data_mode
        self.mask_axis = mask_axis
        self.diversity_n_samples = diversity_n_samples
        # Fixed single-sequence batch for diversity tracking — populated on first
        # validation step (rank 0 only) and reused every epoch thereafter.
        self._div_batch: dict | None = None

    def _build_actor_batch(
        self,
        x: torch.Tensor,
        lab: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> dict:
        """Convert raw dataloader batch to ACTOR batch dict."""
        device = x.device
        pad_mask = pad_mask.bool()
        b = actor_batch_from_carepd(x, pad_mask, self.model.num_classes, device)
        b["y"] = lab.long().to(device)
        return b

    def _effective_lambda_kl(self, epoch: int) -> float:
        """Linear ramp from 0 → lambda_kl over kl_warmup_epochs."""
        if self.kl_warmup_epochs <= 0:
            return self.lambda_kl
        frac = min(1.0, epoch / max(1, self.kl_warmup_epochs))
        return frac * self.lambda_kl

    def _step(self, batch, train: bool):
        x, lab, _vidx, _meta, pad_mask = batch
        b = self._build_actor_batch(x, lab, pad_mask)

        # Sample coalition mask for this batch.
        B = x.shape[0]
        T = b["mask"].shape[1]
        device = x.device
        if self.mask_axis == "temporal":
            b["coalition_mask"] = sample_temporal_training_mask(B, T, device)
        else:
            b["coalition_mask"] = sample_spatial_training_mask(B, device)

        out = self.model(b, phase=2)
        lam = self._effective_lambda_kl(self.current_epoch)
        loss, ld = self.model.compute_loss(
            out, lambda_kl=lam, lambda_reg=self.lambda_reg,
            lambda_kl_full=self.lambda_kl_full,
        )

        mask = out["mask"]
        with torch.no_grad():
            x_m = out.get("x_xyz", out["x"])
            o_m = out.get("output_xyz", out["output"])
            mpjpe = masked_mpjpe(o_m, x_m, mask)
            recon_motion = masked_mean_joint_motion(o_m, mask)
            gt_motion = masked_mean_joint_motion(x_m, mask)

        prefix = "train" if train else "val"
        self.log(f"{prefix}/mpjpe", mpjpe, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}/recon_joint_motion", recon_motion,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}/gt_joint_motion", gt_motion,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss, ld

    def training_step(self, batch, batch_idx):
        loss, ld = self._step(batch, train=True)
        self.log_dict({f"train/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Store the very first validation batch (single sequence) once on rank 0
        # to use as a fixed probe for completion diversity across all epochs.
        if (
            batch_idx == 0
            and self._div_batch is None
            and getattr(self.trainer, "global_rank", 0) == 0
        ):
            x, lab, _vidx, _meta, pad_mask = batch
            b = self._build_actor_batch(x[:1], lab[:1], pad_mask[:1])
            # Move to CPU so it survives device changes / DDP teardown.
            self._div_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                               for k, v in b.items()}

        loss, ld = self._step(batch, train=False)
        self.log_dict({f"val/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        cm = self.trainer.callback_metrics
        # _best_diversity_ckpt_dir is set by main() after ckpt_dir is known.
        _best_div_dir = getattr(self, "_best_diversity_ckpt_dir", None)
        keys = ("val/mpjpe", "val/recon_joint_motion", "val/gt_joint_motion")
        parts = [f"{k}={float(cm[k]):.6f}" for k in keys if k in cm]

        # Completion diversity — fixed coalition (first 4 joints masked) on the
        # stored probe sequence.  Measures pose-space spread of masked joints
        # across multiple stochastic completions.  Higher = less collapsed.
        #
        # NOTE: we measure pairwise RMSE of the masked-joint region directly,
        # NOT via re-encoding through the full encoder.  Re-encoding after
        # paste_observed=True gives artificially near-zero values when few
        # joints are masked (the 13 observed joints dominate the full encoder).
        if self._div_batch is not None:
            device = next(self.model.parameters()).device
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in self._div_batch.items()}
            # Fixed spatial coalition: Pelvis, R_Hip, L_Hip, Spine masked.
            # coalition_mask: True = observed
            cm_div = torch.ones(1, 17, dtype=torch.bool, device=device)
            cm_div[0, :4] = False          # first 4 joints are held-out
            masked_joint_idx = (~cm_div[0]).nonzero(as_tuple=True)[0]  # [0,1,2,3]

            with torch.no_grad():
                completions = self.model.sample_completions(
                    b["x"], b["y"], b["mask"], b["lengths"],
                    coalition_mask=cm_div,
                    n_samples=self.diversity_n_samples,
                    paste_observed=True,
                )
            # Completions: list of (1, J, F, T)
            # Extract only masked-joint region for diversity measurement.
            masked_poses = [
                xh[0, masked_joint_idx]  # (4, F, T)
                for xh in completions
            ]
            K = len(masked_poses)
            div = 0.0
            if K >= 2:
                stacked = torch.stack(masked_poses, dim=0)  # (K, 4, F, T)
                pairs = 0
                for i in range(K):
                    for j in range(i + 1, K):
                        diff = (stacked[i] - stacked[j]).pow(2).mean().sqrt()
                        div += float(diff.item())
                        pairs += 1
                div /= max(pairs, 1)

            # μ diversity across N random coalitions for BOTH encoders.
            #
            # val/mu_div_full_across_S — q_ϕ (full encoder):
            #   Must rise FIRST. q_ϕ's target_marker_spatial learns to perturb
            #   unobserved joints differently per coalition → coalition-specific μ_ϕ.
            #   If this stays near zero, q_ϕ is not becoming coalition-aware and
            #   cannot provide a useful KL teaching signal to r_ψ.
            #
            # val/mu_div_across_S — r_ψ (masked encoder):
            #   Follows q_ϕ via the forward KL.  If q_ϕ is diverse but r_ψ is not,
            #   the KL weight or warmup schedule may need adjustment.
            #   This is the primary SHAP quality indicator: diverse μ_ψ across S →
            #   coalition-dependent completions → non-flat value function v(S).
            with torch.no_grad():
                n_probe = 8
                mu_masked_list = []
                mu_full_list   = []
                for _ in range(n_probe):
                    cm_rand = sample_spatial_training_mask(1, device)
                    out_m = self.model.masked_encoder({
                        "x": b["x"], "y": b["y"], "mask": b["mask"],
                        "coalition_mask": cm_rand,
                    })
                    mu_masked_list.append(out_m["mu_masked"].squeeze(0))
                    out_f = self.model.encoder({
                        "x": b["x"], "y": b["y"], "mask": b["mask"],
                        "coalition_mask": cm_rand,
                    })
                    mu_full_list.append(out_f["mu_full"].squeeze(0))

                mu_masked_t = torch.stack(mu_masked_list, dim=0)  # (n_probe, D)
                mu_full_t   = torch.stack(mu_full_list,   dim=0)
                mu_div_S      = float(torch.cdist(mu_masked_t, mu_masked_t).mean().item())
                mu_div_full_S = float(torch.cdist(mu_full_t,   mu_full_t).mean().item())

            # σ diagnostics — watch these to catch σ_ϕ/σ_ψ collapse early.
            # Both should stay near 1.0.  If they drop below ~0.1, the KL
            # regularization (lambda_kl_full / lambda_reg) is too weak.
            with torch.no_grad():
                fixed_cm = torch.ones(1, 17, dtype=torch.bool, device=device)
                fixed_cm[0, :4] = False
                out_m2 = self.model.masked_encoder({
                    "x": b["x"], "y": b["y"], "mask": b["mask"],
                    "coalition_mask": fixed_cm,
                })
                sigma_psi = float((0.5 * out_m2["logvar_masked"]).exp().mean().item())
                out_f2 = self.model.encoder({
                    "x": b["x"], "y": b["y"], "mask": b["mask"],
                    "coalition_mask": fixed_cm,
                })
                sigma_phi = float((0.5 * out_f2["logvar_full"]).exp().mean().item())

            self.log("val/completion_diversity",    div,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/mu_div_across_S",         mu_div_S,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/mu_div_full_across_S",    mu_div_full_S,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/sigma_phi",               sigma_phi,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/sigma_psi",               sigma_psi,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            parts.append(f"val/completion_diversity={div:.6f}")
            parts.append(f"val/mu_div_across_S={mu_div_S:.6f}")
            parts.append(f"val/mu_div_full_across_S={mu_div_full_S:.6f}")
            parts.append(f"val/sigma_phi={sigma_phi:.4f}  val/sigma_psi={sigma_psi:.4f}")

            # Manually save best-diversity checkpoint on rank 0.
            # We use torch.save directly (not trainer.save_checkpoint) because
            # trainer.save_checkpoint is a collective DDP operation — calling it
            # inside rank-0-only code causes a deadlock on the other ranks.
            if _best_div_dir is not None:
                best_so_far = getattr(self, "_best_diversity_val", -1.0)
                if div > best_so_far:
                    self._best_diversity_val = div
                    save_path = os.path.join(_best_div_dir, "actor_shap_best_diversity.ckpt")
                    torch.save({"state_dict": self.state_dict()}, save_path)
                    parts.append(f"[saved best-diversity ckpt @ {div:.6f}]")

        if parts:
            print(
                f"[ActorSHAP epoch {self.current_epoch}] " + "  ".join(parts),
                flush=True,
            )

    def configure_optimizers(self):
        # q_ϕ (CoalitionFullEncoder): trained — receives reconstruction + KL gradients.
        # r_ψ (MaskedActorEncoder):   trained — receives KL + prior-reg gradients.
        # p_θ (Decoder):              trained at lr_decoder (frozen if lr_decoder==0).
        param_groups = [
            {"params": self.model.encoder.parameters(),        "lr": self.lr},
            {"params": self.model.masked_encoder.parameters(), "lr": self.lr},
        ]
        if self.lr_decoder > 0:
            param_groups.append(
                {"params": self.model.decoder.parameters(), "lr": self.lr_decoder}
            )
        return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# Argument parsing — mirrors train_actor_cvae.py
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ActorSHAP (VAEAC) training for manifold-constrained SHAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file; keys override argument defaults.")
    p.add_argument("--devices", type=str, default=None,
                   help="CUDA_VISIBLE_DEVICES string (e.g. '0,1').")
    p.add_argument("--dataset", type=str, default="BMCLab",
                   choices=["BMCLab", "T-SDU-PD", "PD-GaM", "3DGait"])
    p.add_argument("--data_mode", type=str, default="carepd",
                   choices=("carepd",),
                   help="Only 'carepd' (H36M xyz) is supported for ActorSHAP.")
    p.add_argument("--carepd_pose_npz", type=str, default=None)
    p.add_argument("--carepd_labels_pkl", type=str, default=None)
    p.add_argument("--num_folds", type=int, default=6)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source_seq_len", type=int, default=81,
                   help="Clip length in frames. Must match the ActorCVAE checkpoint "
                        "this run initialises from, and the target classifier's "
                        "source_seq_len (81 for PoseFormerV2/MixSTE/MotionAGFormer).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_decoder", type=float, default=None,
                   help="Decoder LR in Phase 2 (default: lr/10).")

    # Architecture (must match the ActorCVAE checkpoint).
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--ff_size", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=3)

    # ActorSHAP-specific arguments.
    p.add_argument("--actor_cvae_ckpt", type=str, required=False, default=None,
                   help="Path to a trained ActorCVAE Lightning checkpoint (required).")
    p.add_argument("--mask_axis", type=str, default="spatial",
                   choices=("spatial", "temporal"),
                   help="Coalition axis used during training.")
    p.add_argument("--lambda_kl", type=float, default=1.0,
                   help="Final weight for forward KL(q_ϕ‖r_ψ) after annealing.")
    p.add_argument("--kl_warmup_epochs", type=int, default=20,
                   help="Epochs over which λ_kl ramps linearly from 0 to --lambda_kl.")
    p.add_argument("--lambda_reg", type=float, default=1e-6,
                   help="Weight for r_ψ prior regularization (Olsen 2022 Sec. 3.3.1). "
                        "Normal prior on μ_ψ (σ_μ=100) + Gamma prior on σ_ψ (σ_σ=100). "
                        "Prevents μ_ψ/σ_ψ from diverging when q_ϕ target is moving.")
    p.add_argument("--lambda_kl_full", type=float, default=1e-4,
                   help="Weight for KL(q_ϕ ‖ N(0,I)) regularization on the full encoder. "
                        "Prevents σ_ϕ from collapsing to zero (which would cascade to "
                        "σ_ψ → 0 via KL, killing completion diversity). "
                        "Set between 1e-5 and 1e-3; too large will hurt coalition specificity.")
    p.add_argument("--diversity_n_samples", type=int, default=10,
                   help="Completions drawn per validation epoch for the "
                        "val/completion_diversity probe (lower = cheaper).")
    p.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=100,
        help="Also save actor_shap_epoch{epoch:04d}.ckpt every N epochs (0 = disabled). "
        "Final epoch is always saved as actor_shap_last.ckpt.",
    )

    # Output / logging.
    p.add_argument("--checkpoint_dir", type=str, default="./experiment_outs/actor_shap")
    p.add_argument("--experiment_name", type=str, default="ActorSHAP")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    # Two-pass parse: load config file first, then let CLI override.
    pre, _ = p.parse_known_args()
    if pre.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pre.devices
    if pre.config is not None:
        with open(pre.config) as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        p.set_defaults(**cfg)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_random_seed(args.seed)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.lr_decoder is None:
        args.lr_decoder = args.lr / 10.0

    tag_ds = f"{args.dataset}_fold{args.fold}"
    run_tag = f"actor_shap_{tag_ds}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build encoder / decoder (same architecture kwargs as ActorCVAE) ----
    common = dict(
        modeltype="cvae",
        njoints=17, nfeats=3,
        num_frames=0, num_classes=args.num_classes,
        translation=True, pose_rep="xyz",
        glob=True, glob_rot=[3.141592653589793, 0, 0],
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, ablation=None, activation="gelu",
    )
    # CoalitionFullEncoder: wraps Encoder_TRANSFORMER + learnable target_marker_spatial.
    # Replaces the old frozen Encoder_TRANSFORMER — now trained jointly.
    encoder        = CoalitionFullEncoder(**common)
    masked_encoder = MaskedActorEncoder(**common)
    decoder        = Decoder_TRANSFORMER(**common)

    model = ActorSHAP(
        encoder, masked_encoder, decoder,
        latent_dim=args.latent_dim,
        device=device,
        pose_rep="xyz",
        num_classes=args.num_classes,
    ).to(device)

    # ---- Phase 0: load weights from ActorCVAE checkpoint ------------------
    if args.actor_cvae_ckpt:
        model.load_from_actor_cvae(args.actor_cvae_ckpt)
        print(f"Loaded ActorCVAE checkpoint: {args.actor_cvae_ckpt}")
    else:
        print("WARNING: --actor_cvae_ckpt not provided. Starting from random weights.")

    # ---- Datasets -----------------------------------------------------------
    train_ds, val_ds = get_carepd_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # ---- Logging ------------------------------------------------------------
    logger = False
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_tag,
            config=vars(args),
            save_dir=ckpt_dir,
        )

    # ---- Lightning module + trainer -----------------------------------------
    module = ActorSHAPModule(
        model,
        lr=args.lr,
        lr_decoder=args.lr_decoder,
        lambda_kl=args.lambda_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        lambda_reg=args.lambda_reg,
        lambda_kl_full=args.lambda_kl_full,
        data_mode=args.data_mode,
        mask_axis=args.mask_axis,
        diversity_n_samples=args.diversity_n_samples,
    )
    # Tell the module where to manually save the best-diversity checkpoint.
    # We do this manually (not via ModelCheckpoint) because val/completion_diversity
    # is rank_zero_only and DDP ModelCheckpoint would crash on non-zero ranks.
    module._best_diversity_ckpt_dir = ckpt_dir

    extra_callbacks = []
    if args.checkpoint_every_n_epochs and args.checkpoint_every_n_epochs > 0:
        extra_callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                every_n_epochs=args.checkpoint_every_n_epochs,
                filename="actor_shap_epoch{epoch:04d}",
                save_top_k=-1,
            ),
        )
    trainer = make_trainer(
        n_epochs=args.epochs,
        n_gpus=n_gpus,
        ckpt_dir=ckpt_dir,
        logger=logger,
        monitor_key="val/mixed",
        phase_tag="actor_shap",
        extra_callbacks=extra_callbacks if extra_callbacks else None,
        find_unused_parameters=True,
        save_last_only=True,
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"Done. Last epoch: {ckpt_dir}/actor_shap_last.ckpt")
    if extra_callbacks:
        print(
            f"      Periodic (every {args.checkpoint_every_n_epochs} epochs): "
            f"{ckpt_dir}/actor_shap_epoch*.ckpt",
        )


if __name__ == "__main__":
    main()
