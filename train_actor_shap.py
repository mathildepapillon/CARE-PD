"""
train_actor_shap.py — ActorSHAP proper VAEAC training
======================================================

Training strategy
-----------------

Optional Phase 0  (epochs 0 … phase0_epochs-1, default: SKIPPED) — r_ψ warm-up
  Set --phase0_epochs > 0 to enable.  The decoder is kept at its CVAE
  initialisation.  Only r_ψ (MaskedActorEncoder) is trained using a simple
  reconstruction objective on *observed* joints (z = μ_ψ, deterministic).
  Recommended when cold-start observed-joint degradation is severe (>2× CVAE
  baseline).  With a z-only CVAE the degradation is mild (~17%) and the VAEAC
  KL term closes this gap on its own, so Phase 0 can be skipped (default=0).

VAEAC training  (epochs phase0_epochs … end) — full VAEAC ELBO, decoder UNFROZEN
  Implements the corrected VAEAC ELBO (Ivanov 2019 / Olsen et al. JMLR 2022):

      L = E_{z~q_ϕ(z|x,S)} [log p_θ(x_S|z,...)]
          − λ_kl · KL( q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S) )
          − λ_reg · prior_reg(μ_ψ, σ_ψ)
          + λ_rc_psi · MSE(dec(z~r_ψ), x)_{all joints}
          + λ_rc_obs · MSE(dec(z~q_ϕ), x)_{observed joints}

  KL annealing: λ_kl ramps linearly 0 → --lambda_kl over --kl_warmup_epochs,
  starting from epoch phase0_epochs (epoch 0 when Phase 0 is skipped).

Weights initialised from a trained ActorCVAE checkpoint (--actor_cvae_ckpt):
  - CoalitionFullEncoder._encoder → ActorCVAE encoder weights, target_marker=0
  - MaskedActorEncoder._encoder  → ActorCVAE encoder weights, mask_token=0
  - Decoder                      → ActorCVAE decoder weights

Key diagnostic metrics (logged to wandb and stdout):
  val/sigma_psi            — r_ψ posterior std.  If it collapses to ~1.0
                             (N(0,I)), reduce --lambda_kl to 5e-3.
  val/mu_div_across_S      — pairwise distance of μ_ψ for 8 random coalitions.
                             Rising value confirms r_ψ is becoming coalition-specific.
  val/completion_diversity — pose RMSE of masked joints across n completions.

Typical command (z-only CVAE, Phase 0 skipped):
  python train_actor_shap.py \\
      --actor_cvae_ckpt experiment_outs/actor_cvae/<run>/actor_cvae_best.ckpt \\
      --dataset BMCLab --fold 1 --epochs 500 --phase0_epochs 0 --lr 1e-4 \\
      --lambda_kl 0.01 --kl_warmup_epochs 20

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
from model.actor.motion_utils import unroot_to_global
from model.actor.shap_masking import (
    H36M_GROUPS,
    sample_spatial_training_mask,
    sample_temporal_training_mask,
)
from model.actor.transformer_arch import Decoder_TRANSFORMER
from train_actor_cvae import masked_mean_joint_motion, masked_mpjpe
from train_utils import make_trainer
from utility.utils import set_random_seed


# ---------------------------------------------------------------------------
# Periodic GIF callback
# ---------------------------------------------------------------------------

class PeriodicGifCallback(pl.Callback):
    """Save GT-vs-completion GIFs every N validation epochs using a fixed val batch.

    A small fixed set of validation sequences is captured on the first
    validation pass and reused every ``every_n_epochs`` epochs.  Three
    representative anatomical groups are masked in turn so progress across
    spine, lower limb, and upper limb can be tracked visually without
    requiring a separate evaluation run.

    Args:
        out_dir:          Root directory; per-epoch sub-dirs are created inside.
        every_n_epochs:   Render GIFs after epochs N, 2N, 3N, …
        n_sequences:      Number of val sequences to render per group.
        fps:              Frame rate of saved GIFs.
    """

    # Body-part groups to mask — one GIF per group per sequence.
    VIZ_GROUPS = ("spine", "right_leg", "left_arm")

    def __init__(
        self,
        out_dir: str,
        every_n_epochs: int,
        n_sequences: int = 3,
        fps: int = 12,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_epochs = every_n_epochs
        self.n_sequences = n_sequences
        self.fps = fps
        self._fixed_batch: dict | None = None

    # ------------------------------------------------------------------
    # Capture a small fixed val batch on the first validation pass.
    # ------------------------------------------------------------------

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        if batch_idx != 0 or self._fixed_batch is not None:
            return
        if getattr(trainer, "global_rank", 0) != 0:
            return
        x, lab, _vidx, _meta, pad_mask = batch
        device = next(pl_module.model.parameters()).device
        n = min(self.n_sequences, x.shape[0])
        b = actor_batch_from_carepd(
            x[:n].float().to(device), pad_mask[:n].to(device),
            pl_module.model.num_classes, device,
        )
        b["y"] = lab[:n].long().to(device)
        self._fixed_batch = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in b.items()
        }

    # ------------------------------------------------------------------
    # Render GIFs every N epochs on rank 0.
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._fixed_batch is None:
            return
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # Lazy import of viz_utils from the scripts/ sub-directory.
        import sys as _sys
        _scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        if _scripts_dir not in _sys.path:
            _sys.path.insert(0, _scripts_dir)
        import viz_utils  # noqa: PLC0415

        device = next(pl_module.model.parameters()).device
        b = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self._fixed_batch.items()
        }

        epoch_dir = os.path.join(self.out_dir, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)

        was_training = pl_module.model.training
        pl_module.model.eval()
        n_saved = 0
        try:
            with torch.no_grad():
                for group_name in self.VIZ_GROUPS:
                    joint_ids = H36M_GROUPS[group_name]
                    cm_j = torch.ones(17, dtype=torch.bool, device=device)
                    cm_j[joint_ids] = False
                    if cm_j.sum() == 0:
                        cm_j[0] = True  # guard: keep at least pelvis observed

                    for bi in range(b["x"].shape[0]):
                        x1       = b["x"][bi : bi + 1]
                        y1       = b["y"][bi : bi + 1]
                        mask1    = b["mask"][bi : bi + 1]
                        lengths1 = b["lengths"][bi : bi + 1]

                        comps = pl_module.model.sample_completions(
                            x1, y1, mask1, lengths1,
                            coalition_mask=cm_j.unsqueeze(0),
                            n_samples=1,
                            paste_observed=False,
                        )
                        x_hat = comps[0]
                        real_len = int(mask1[0].sum().item())

                        # (B, J, F, T) → (B, T, J, F) → unroot → numpy
                        gt_btj3 = unroot_to_global(
                            x1[:, :, :, :real_len].permute(0, 3, 1, 2)
                        ).cpu().numpy()
                        out_btj3 = unroot_to_global(
                            x_hat[:, :, :, :real_len].permute(0, 3, 1, 2)
                        ).cpu().numpy()

                        edges = viz_utils.edges_for_njoints(gt_btj3.shape[2])
                        out_path = os.path.join(
                            epoch_dir,
                            f"ep{epoch:04d}_s{bi:02d}_mask_{group_name}.gif",
                        )
                        viz_utils.save_motion_comparison_gif(
                            gt_btj3[0], out_btj3[0], edges, out_path, self.fps,
                            title_prefix=f"ep{epoch} mask={group_name}",
                            legend_pred_label="completion",
                            verbose=False,
                        )
                        n_saved += 1
        finally:
            if was_training:
                pl_module.model.train()

        print(
            f"[GifCallback] epoch {epoch}: saved {n_saved} GIF(s) → {epoch_dir}/",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class ActorSHAPModule(pl.LightningModule):
    """PyTorch Lightning module for ActorSHAP two-phase VAEAC training.

    Phase 0 (epochs 0 … phase0_epochs-1) — r_ψ warm-up:
      Decoder and q_ϕ are frozen.  Only r_ψ (MaskedActorEncoder) is trained
      via deterministic reconstruction of observed joints.  This closes the
      cold-start attention-contamination gap before VAEAC training begins.

    Phase 1+ (epochs phase0_epochs … end) — full VAEAC ELBO:
      q_ϕ (CoalitionFullEncoder):   lr           — reconstruction + KL gradients.
      r_ψ (MaskedActorEncoder):     lr           — KL + prior-reg + rc_psi gradients.
      p_θ (Decoder):                lr_decoder   — reconstruction gradient only.

    Args:
        model:             ActorSHAP instance.
        lr:                learning rate for r_ψ (and q_ϕ in Phase 1+).
        lr_decoder:        learning rate for p_θ in Phase 1+ (0 = keep frozen).
        phase0_epochs:     epochs to spend in Phase 0 warm-up (default 50).
        lambda_kl:         final KL weight (forward KL(q_ϕ‖r_ψ)) after annealing.
        kl_warmup_epochs:  epochs over which λ_kl ramps linearly 0 → lambda_kl,
                           counted from the START of Phase 1 (not epoch 0).
        lambda_reg:        weight for r_ψ prior regularization (Olsen 2022, Sec 3.3.1).
        lambda_kl_full:    weight for KL(q_ϕ ‖ N(0,I)) — prevents σ_ϕ collapse.
        lambda_vel:        weight for velocity MSE (default 5.0).
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
        phase0_epochs: int = 50,
        lambda_kl: float = 0.01,
        kl_warmup_epochs: int = 20,
        lambda_reg: float = 1e-6,
        lambda_kl_full: float = 1e-4,
        lambda_vel: float = 5.0,
        lambda_rc_obs: float = 0.1,
        lambda_rc_psi: float = 1.0,
        data_mode: str = "carepd",
        mask_axis: str = "spatial",
        diversity_n_samples: int = 10,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_decoder = lr_decoder
        self.phase0_epochs = phase0_epochs
        self.lambda_kl = lambda_kl
        self.kl_warmup_epochs = kl_warmup_epochs
        self.lambda_reg = lambda_reg
        self.lambda_kl_full = lambda_kl_full
        self.lambda_vel = lambda_vel
        self.lambda_rc_obs = lambda_rc_obs
        self.lambda_rc_psi = lambda_rc_psi
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

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        """Freeze/unfreeze decoder and q_ϕ based on current training phase."""
        in_phase0 = self.current_epoch < self.phase0_epochs
        # Decoder: frozen in Phase 0, trainable (at lr_decoder) in Phase 1+.
        for p in self.model.decoder.parameters():
            p.requires_grad_(not in_phase0)
        # q_ϕ base encoder: always frozen (set in configure_optimizers).
        # q_ϕ target_marker_spatial: only meaningful in Phase 1+ VAEAC training.
        self.model.encoder.target_marker_spatial.requires_grad_(not in_phase0)

        if self.current_epoch == 0:
            if self.phase0_epochs == 0:
                print(
                    "[ActorSHAP] Phase 0 skipped (phase0_epochs=0). "
                    "Starting VAEAC training from epoch 0.",
                    flush=True,
                )
            else:
                print(
                    f"[ActorSHAP] Phase 0 warm-up: training r_ψ only for "
                    f"{self.phase0_epochs} epochs (decoder + q_ϕ frozen).",
                    flush=True,
                )
        elif self.current_epoch == self.phase0_epochs and self.phase0_epochs > 0:
            print(
                f"[ActorSHAP] Phase 1 VAEAC: decoder unfrozen "
                f"(lr_decoder={self.lr_decoder}), KL annealing starts now.",
                flush=True,
            )

    def _effective_lambda_kl(self, epoch: int) -> float:
        """Zero during Phase 0; linear ramp 0 → lambda_kl over kl_warmup_epochs
        starting from the first Phase 1 epoch."""
        if epoch < self.phase0_epochs:
            return 0.0
        elapsed = epoch - self.phase0_epochs
        if self.kl_warmup_epochs <= 0:
            return self.lambda_kl
        frac = min(1.0, elapsed / max(1, self.kl_warmup_epochs))
        return frac * self.lambda_kl

    # ------------------------------------------------------------------
    # Phase 0 step: r_ψ warm-up (observed joints, deterministic decoding)
    # ------------------------------------------------------------------

    def _step_phase0(self, b: dict) -> tuple[torch.Tensor, dict]:
        """Phase 0 loss: MSE on observed joints decoded deterministically from r_ψ.

        z = μ_ψ (no reparameterisation).  Decoder is frozen so r_ψ must
        produce latents that the pre-trained CVAE decoder already decodes well.

        The objective is purely to teach r_ψ to handle mask tokens without
        contaminating observed-joint representations — NOT to reconstruct
        masked joints (that comes in Phase 1 via rc_psi).
        """
        B, J, F, T = b["x"].shape
        device = b["x"].device

        # Encode with masked encoder (r_ψ) — decoder stays frozen, no grad flows through it.
        out_m = self.model.masked_encoder(b)
        z  = out_m["mu_masked"]                     # deterministic
        ft = out_m.get("frame_tokens_masked")

        dec_batch = {**b, "z": z}
        if ft is not None:
            dec_batch["frame_tokens"] = ft
        dec_batch.update(self.model.decoder(dec_batch))
        output = dec_batch["output"]                # (B, J, F, T)

        # Mask over observed joints × real frames.
        obs_mask    = b["coalition_mask"].unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
        real_frames = b["mask"].unsqueeze(1).unsqueeze(1).expand(B, J, F, T)
        active      = obs_mask & real_frames

        rc = ((output - b["x"]) ** 2)[active].mean()

        # Velocity loss on observed joints for temporal coherence.
        vel_loss = torch.tensor(0.0, device=device)
        if self.lambda_vel > 0 and T > 1:
            vel_p = output[:, :, :, 1:] - output[:, :, :, :-1]
            vel_g = b["x"][:, :, :, 1:]  - b["x"][:, :, :, :-1]
            real_vel = b["mask"][:, 1:].unsqueeze(1).unsqueeze(1).expand(B, J, F, T - 1)
            obs_vel  = b["coalition_mask"].unsqueeze(2).unsqueeze(3).expand(B, J, F, T - 1)
            active_vel = obs_vel & real_vel
            if active_vel.any():
                vel_loss = self.lambda_vel * ((vel_p - vel_g) ** 2)[active_vel].mean()

        loss = rc + vel_loss

        # Log overall MPJPE (all joints) for a comparable val/mpjpe across phases.
        with torch.no_grad():
            mpjpe = masked_mpjpe(output, b["x"], b["mask"])
            recon_motion = masked_mean_joint_motion(output, b["mask"])
            gt_motion    = masked_mean_joint_motion(b["x"],  b["mask"])

        ld = {
            "rc_obs_p0":   float(rc.item()),
            "vel_p0":      float(vel_loss.item()),
            "mixed":       float(loss.item()),
            "_mpjpe":      float(mpjpe.item()),
            "_recon_mot":  float(recon_motion.item()),
            "_gt_mot":     float(gt_motion.item()),
        }
        return loss, ld

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

        if self.current_epoch < self.phase0_epochs:
            # ---- Phase 0: r_ψ warm-up, decoder frozen -------------------------
            loss, ld = self._step_phase0(b)
            mpjpe        = ld.pop("_mpjpe")
            recon_motion = ld.pop("_recon_mot")
            gt_motion    = ld.pop("_gt_mot")
        else:
            # ---- Phase 1+: full VAEAC ELBO ------------------------------------
            out = self.model(b, phase=2)
            lam = self._effective_lambda_kl(self.current_epoch)
            loss, ld = self.model.compute_loss(
                out, lambda_kl=lam, lambda_reg=self.lambda_reg,
                lambda_kl_full=self.lambda_kl_full,
                lambda_vel=self.lambda_vel,
                lambda_rc_obs=self.lambda_rc_obs,
                lambda_rc_psi=self.lambda_rc_psi,
            )
            mask = out["mask"]
            with torch.no_grad():
                x_m = out.get("x_xyz", out["x"])
                o_m = out.get("output_xyz", out["output"])
                mpjpe        = masked_mpjpe(o_m, x_m, mask)
                recon_motion = masked_mean_joint_motion(o_m, mask)
                gt_motion    = masked_mean_joint_motion(x_m, mask)

                # r_ψ temporal quality: run phase=3 (r_ψ inference) on the same
                # batch and measure recon_joint_motion from r_ψ's output.
                # This is what the GIFs show — the logged metric above uses q_ϕ's
                # z (frozen CVAE encoder) and is therefore blind to r_ψ's quality.
                if not train:
                    b3 = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in b.items()}
                    out3   = self.model(b3, phase=3)
                    o3     = out3.get("output_xyz", out3["output"])
                    rpsi_motion = masked_mean_joint_motion(o3, mask)
                    self.log("val/rpsi_recon_motion", rpsi_motion,
                             on_step=False, on_epoch=True, sync_dist=True)

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
        keys = ("val/mpjpe", "val/recon_joint_motion", "val/rpsi_recon_motion",
                "val/gt_joint_motion")
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
        # Gradient routing (static, always frozen):
        #   q_ϕ base encoder (_encoder weights): ALWAYS frozen — keeps z distribution
        #     at CVAE initialisation so r_ψ has a stable, fixed KL target.
        #
        # Dynamic freeze/unfreeze (via on_train_epoch_start):
        #   Phase 0: q_ϕ.target_marker_spatial and decoder are frozen.
        #   Phase 1+: both are unfrozen and receive gradients.
        #
        # All trainable param groups are registered here; on_train_epoch_start
        # toggles requires_grad so the optimizer simply sees zero gradients for
        # frozen parameters (AdamW skips state updates when grad is None).
        for p in self.model.encoder._encoder.parameters():
            p.requires_grad_(False)
        param_groups = [
            {"params": [self.model.encoder.target_marker_spatial], "lr": self.lr},
            {"params": self.model.masked_encoder.parameters(),     "lr": self.lr},
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
    p.add_argument("--num_folds", type=int, default=23,
                   help="Number of CV folds. BMCLab canonical = 23 (LOSO by patient).")
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
    p.add_argument("--phase0_epochs", type=int, default=0,
                   help="Epochs to spend in Phase 0 (r_ψ warm-up with frozen decoder). "
                        "During Phase 0 only r_ψ is trained using deterministic "
                        "reconstruction of observed joints. Set to 0 (default) to skip "
                        "Phase 0 entirely — recommended when initialising from a z-only "
                        "CVAE checkpoint where init observed-joint degradation is mild "
                        "(~17%%). The VAEAC KL term handles this gap on its own.")
    p.add_argument("--mask_axis", type=str, default="spatial",
                   choices=("spatial", "temporal"),
                   help="Coalition axis used during training.")
    p.add_argument("--lambda_kl", type=float, default=0.01,
                   help="Final weight for forward KL(q_ϕ‖r_ψ) after annealing. "
                        "VAEAC theory requires KL>0 for the training/inference bound "
                        "to hold (Ivanov 2019, Eq. 6). However with frozen q_ϕ, q_ϕ "
                        "is barely coalition-aware (mu_div_full≈0.008), so a large "
                        "lambda_kl(=1.0) suppresses r_ψ's useful coalition diversity. "
                        "Use a small value (0.01-0.05) so rc_psi dominates r_ψ training "
                        "while KL still provides theoretical regularization. "
                        "At epoch 149 KL≈0.031 and rc_psi≈0.004, so lambda_kl=0.01 "
                        "gives KL:rc_psi ratio of ~0.08, letting rc_psi dominate.")
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
    p.add_argument("--lambda_vel", type=float, default=5.0,
                   help="Weight for velocity (frame-delta) MSE on held-out positions. "
                        "Matches ActorCVAE pre-training (lambda_vel=5) to preserve "
                        "temporal coherence. Set to 0 to disable.")
    p.add_argument("--lambda_rc_psi", type=float, default=1.0,
                   help="Weight for auxiliary reconstruction via z~r_ψ. "
                        "Runs the decoder a second time per step with z sampled "
                        "from r_ψ (masked encoder) and adds held-out MSE loss. "
                        "This gives r_ψ a direct reconstruction gradient, closing "
                        "the training/inference gap (default 1.0). Set to 0 to "
                        "use strict VAEAC ELBO (KL-only supervision for r_ψ).")
    p.add_argument("--lambda_rc_obs", type=float, default=0.1,
                   help="Weight for reconstruction MSE on *observed* joints. "
                        "Prevents the decoder from drifting on joints it never "
                        "receives held-out gradients for, keeping full-sequence "
                        "reconstruction intact when paste_observed=False. "
                        "Set to 0.0 for the strict VAEAC ELBO.")
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
    p.add_argument(
        "--gif_every_n_epochs",
        type=int,
        default=50,
        help="Render GT-vs-completion GIFs every N epochs into "
             "<checkpoint_dir>/train_gifs/epoch_NNNN/ (0 = disabled).",
    )
    p.add_argument(
        "--gif_n_sequences",
        type=int,
        default=3,
        help="Number of validation sequences to render per GIF epoch.",
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
        phase0_epochs=args.phase0_epochs,
        lambda_kl=args.lambda_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        lambda_reg=args.lambda_reg,
        lambda_kl_full=args.lambda_kl_full,
        lambda_vel=args.lambda_vel,
        lambda_rc_obs=args.lambda_rc_obs,
        lambda_rc_psi=args.lambda_rc_psi,
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
    if args.gif_every_n_epochs and args.gif_every_n_epochs > 0:
        gif_dir = os.path.join(ckpt_dir, "train_gifs")
        extra_callbacks.append(
            PeriodicGifCallback(
                out_dir=gif_dir,
                every_n_epochs=args.gif_every_n_epochs,
                n_sequences=args.gif_n_sequences,
            )
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
