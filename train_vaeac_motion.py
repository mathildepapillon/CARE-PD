"""
train_vaeac_motion.py — Train VaeacMotion for manifold-constrained SHAP
=======================================================================

Implements proper VAEAC training (Ivanov ICLR 2019; Olsen JMLR 2022) on the
ACTOR Transformer backbone.

Architecture
------------
q_ϕ(z|x,S)     VaeacFullEncoder   — sees full x + coalition marker.
r_ψ(z|x_S,S)   VaeacMaskedEncoder — sees only observed joints (mask tokens).
p_θ(x̂|z,x_S,S) Decoder_TRANSFORMER conditioned on z AND observed-feature
                embeddings.  This is the key difference from previous attempts:
                the decoder cross-attends to the raw observed input, not to
                encoder representations.  Training and inference memory are
                identical, eliminating the mismatch that caused mean-pose
                collapse.

VAEAC ELBO
----------
L = E_{z~q_ϕ} [MSE_{held-out}(x, x̂)]
    − λ_kl · KL(q_ϕ(z|x,S) ‖ r_ψ(z|x_S,S))
    − λ_reg · prior_reg(r_ψ)
    + λ_vel · velocity_loss(held-out)
    + λ_rc_obs · MSE_{observed}(x, x̂)

Gradient routing (per Ivanov 2019):
    q_ϕ: reconstruction + velocity (via reparameterised z).
          DETACHED from KL so KL only trains r_ψ.
    r_ψ: KL + prior regularisation only.
    p_θ: reconstruction + velocity.
    observed_projection: reconstruction + velocity (decoder memory).

Weights initialised from a pre-trained ActorCVAE checkpoint:
    q_ϕ._encoder → CVAE encoder,  target_marker = 0.
    r_ψ._encoder → CVAE encoder,  mask tokens = 0.
    p_θ          → CVAE decoder.
    observed_projection → random (new parameter).

Typical command::

    python train_vaeac_motion.py \\
        --cvae_ckpt experiment_outs/actor_cvae/<run>/actor_cvae_best.ckpt \\
        --dataset BMCLab --fold 1 --epochs 300 --lr 1e-4 \\
        --lambda_kl 0.01 --kl_warmup_epochs 30 \\
        --wandb_project CARE-PD
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.dataloaders import collate_fn
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.motion_utils import unroot_to_global
from model.actor.shap_masking import (
    H36M_GROUPS,
    build_temporal_shap_mask,
    build_temporal_windows,
    sample_spatial_training_mask,
    sample_temporal_training_mask,
)
from model.actor.transformer_arch import Decoder_TRANSFORMER, Encoder_TRANSFORMER
from model.actor.vaeac_motion import VaeacFullEncoder, VaeacMaskedEncoder, VaeacMotion
from train_actor_cvae import masked_mean_joint_motion, masked_mpjpe
from train_utils import make_trainer
from utility.utils import set_random_seed

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


# ---------------------------------------------------------------------------
# Periodic GIF callback
# ---------------------------------------------------------------------------

class ImputationGifCallback(pl.Callback):
    """Save GT-vs-completion GIFs every N epochs using a fixed val batch."""

    VIZ_GROUPS = ("spine", "right_leg", "left_arm")

    def __init__(self, out_dir: str, every_n_epochs: int,
                 n_sequences: int = 3, fps: int = 12):
        super().__init__()
        self.out_dir = out_dir
        self.every_n_epochs = every_n_epochs
        self.n_sequences = n_sequences
        self.fps = fps
        self._fixed_batch: dict | None = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, *args):
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

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._fixed_batch is None:
            return
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        import viz_utils  # noqa: PLC0415

        device = next(pl_module.model.parameters()).device
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in self._fixed_batch.items()}

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
                        cm_j[0] = True

                    for bi in range(b["x"].shape[0]):
                        x1 = b["x"][bi:bi + 1]
                        y1 = b["y"][bi:bi + 1]
                        mask1 = b["mask"][bi:bi + 1]
                        lengths1 = b["lengths"][bi:bi + 1]

                        comps = pl_module.model.sample_completions(
                            x1, y1, mask1, lengths1,
                            coalition_mask=cm_j.unsqueeze(0),
                            n_samples=1, paste_observed=False,
                        )
                        x_hat = comps[0]
                        real_len = int(mask1[0].sum().item())

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
                            gt_btj3[0], out_btj3[0], edges, out_path,
                            self.fps,
                            title_prefix=f"ep{epoch} mask={group_name}",
                            legend_pred_label="completion",
                            verbose=False,
                        )
                        n_saved += 1
        finally:
            if was_training:
                pl_module.model.train()
        print(f"[GifCallback] epoch {epoch}: saved {n_saved} GIF(s) → {epoch_dir}/",
              flush=True)


# ---------------------------------------------------------------------------
# Temporal mask sampler — mixes contiguous quarters and stride-simulated windows
# ---------------------------------------------------------------------------

def sample_temporal_training_mask_mixed(
    B: int,
    T: int,
    device: torch.device,
    stride_sim_prob: float = 0.5,
) -> torch.Tensor:
    """Sample temporal coalition masks that cover both training and inference patterns.

    At inference (evaluate_shap.py) temporal coalitions are stride-phase-aligned
    non-contiguous windows (e.g. W0 = {0-8, 27-35, 54-62} across strides).
    Training with only contiguous quarter-blocks means the observed_projection W
    has never seen such alternating indicator patterns.

    This function mixes two distributions per batch element:
      - p = 1 - stride_sim_prob: contiguous quarter-blocks (original behaviour).
      - p = stride_sim_prob:     stride-simulated non-contiguous windows, using
                                 a random stride period uniform in [15, 45] frames
                                 and randomly held-out gait-phase windows.

    Args:
        B:               Batch size.
        T:               Number of frames.
        device:          Target device.
        stride_sim_prob: Fraction of batch elements that use stride-simulated masks.

    Returns:
        BoolTensor (B, T) — True = frame observed.
    """
    masks = torch.ones(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        if torch.rand(1).item() < stride_sim_prob:
            # Stride-simulated: pick a random plausible gait stride period.
            stride_period = int(torch.randint(15, 46, (1,)).item())
            windows = build_temporal_windows(T, stride_period, K=4)
            # Randomly hold out each of the 4 gait-phase windows with p=0.5.
            observed = [k for k in range(4) if torch.rand(1).item() > 0.5]
            if len(observed) == 0:
                observed = [0]  # guarantee at least one observed window
            masks[i] = build_temporal_shap_mask(observed, windows, T, device)
        else:
            # Contiguous quarter-blocks (original sample_temporal_training_mask).
            quarter = T // 4
            window_frames = [
                list(range(k * quarter, (k + 1) * quarter if k < 3 else T))
                for k in range(4)
            ]
            for frames in window_frames:
                if torch.rand(1).item() < 0.5:
                    masks[i, frames] = False
            if masks[i].sum() == 0:
                masks[i, window_frames[0]] = True
    return masks


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class VaeacMotionModule(pl.LightningModule):
    """PyTorch Lightning wrapper for VaeacMotion training.

    Args:
        model:              VaeacMotion instance.
        lr:                 Learning rate for r_ψ, decoder, observed_projection.
        lr_full_enc:        Learning rate for q_ϕ._encoder (usually lower
                            to preserve pre-trained quality).
        lambda_kl:          Final KL weight (after warmup).
        kl_warmup_epochs:   Linear ramp 0 → lambda_kl over this many epochs.
        kl_n_cycles:        Cyclical KL annealing (Fu et al. 2019).  Each cycle
                            ramps 0→lambda_kl over half the cycle and holds for
                            the other half.  0 = plain linear warmup (legacy).
        total_epochs:       Total epochs (needed for cyclical schedule).
        lambda_reg:         Weight for r_ψ prior regularization.
        lambda_vel:         Weight for velocity loss on held-out joints.
        lambda_rc_obs:      Weight for reconstruction on observed joints.
        prior_sigma_mu:     Normal prior width on μ_ψ (Ivanov Eq.8).
        prior_sigma_sigma:  Gamma prior tightness on σ_ψ (Ivanov Eq.8).
        mask_axis:          "spatial", "temporal", or "both".
        diversity_n_samples: Completions per val epoch for diversity probe.
    """

    def __init__(
        self,
        model: VaeacMotion,
        lr: float,
        lr_full_enc: float | None = None,
        *,
        lambda_kl: float = 0.01,
        kl_warmup_epochs: int = 30,
        kl_n_cycles: int = 0,
        total_epochs: int = 300,
        lambda_reg: float = 1e-6,
        lambda_vel: float = 5.0,
        lambda_rc_obs: float = 0.1,
        prior_sigma_mu: float = 1e4,
        prior_sigma_sigma: float = 1e-4,
        mask_axis: str = "spatial",
        diversity_n_samples: int = 10,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_full_enc = lr_full_enc if lr_full_enc is not None else lr / 10.0
        self.lambda_kl = lambda_kl
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_n_cycles = kl_n_cycles
        self.total_epochs = total_epochs
        self.lambda_reg = lambda_reg
        self.lambda_vel = lambda_vel
        self.lambda_rc_obs = lambda_rc_obs
        self.prior_sigma_mu = prior_sigma_mu
        self.prior_sigma_sigma = prior_sigma_sigma
        self.mask_axis = mask_axis
        self.diversity_n_samples = diversity_n_samples
        self._div_batch: dict | None = None

    def _build_actor_batch(self, x, lab, pad_mask):
        device = x.device
        pad_mask = pad_mask.bool()
        b = actor_batch_from_carepd(x, pad_mask, self.model.num_classes, device)
        b["y"] = lab.long().to(device)
        return b

    def _effective_lambda_kl(self) -> float:
        """KL weight with optional cyclical annealing (Fu et al. 2019).

        kl_n_cycles > 0:  Divide training into N equal cycles.  Within each
            cycle, linearly ramp 0 → lambda_kl over the first half, then hold
            lambda_kl for the second half.  This gives the decoder repeated
            opportunities to learn z-dependence before KL is raised again.
        kl_n_cycles == 0: Legacy linear warmup over kl_warmup_epochs.
        """
        if self.kl_n_cycles > 0:
            cycle_len = max(1, self.total_epochs // self.kl_n_cycles)
            pos_in_cycle = self.current_epoch % cycle_len
            ramp_half = cycle_len // 2
            if ramp_half <= 0:
                frac = 1.0
            elif pos_in_cycle < ramp_half:
                frac = pos_in_cycle / ramp_half
            else:
                frac = 1.0
            return frac * self.lambda_kl

        if self.kl_warmup_epochs <= 0:
            return self.lambda_kl
        frac = min(1.0, self.current_epoch / max(1, self.kl_warmup_epochs))
        return frac * self.lambda_kl

    def _step(self, batch, train: bool):
        x, lab, _vidx, _meta, pad_mask = batch
        b = self._build_actor_batch(x, lab, pad_mask)

        B = x.shape[0]
        T = b["mask"].shape[1]
        device = x.device

        if self.mask_axis == "temporal":
            b["coalition_mask"] = sample_temporal_training_mask_mixed(B, T, device)
        elif self.mask_axis == "both":
            if torch.rand(1).item() < 0.5:
                b["coalition_mask"] = sample_spatial_training_mask(B, device)
            else:
                b["coalition_mask"] = sample_temporal_training_mask_mixed(B, T, device)
        else:
            b["coalition_mask"] = sample_spatial_training_mask(B, device)

        out = self.model(dict(b), phase="train")
        lam_kl = self._effective_lambda_kl()
        loss, ld = self.model.compute_loss(
            out,
            lambda_kl=lam_kl,
            lambda_reg=self.lambda_reg,
            lambda_vel=self.lambda_vel,
            lambda_rc_obs=self.lambda_rc_obs,
            prior_sigma_mu=self.prior_sigma_mu,
            prior_sigma_sigma=self.prior_sigma_sigma,
        )

        mask = out["mask"]
        with torch.no_grad():
            x_m = out.get("x_xyz", out["x"])
            o_m = out.get("output_xyz", out["output"])
            mpjpe = masked_mpjpe(o_m, x_m, mask)
            recon_motion = masked_mean_joint_motion(o_m, mask)
            gt_motion = masked_mean_joint_motion(x_m, mask)

        prefix = "train" if train else "val"
        self.log(f"{prefix}/mpjpe", mpjpe,
                 on_step=False, on_epoch=True, sync_dist=True)
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
        if (
            batch_idx == 0
            and self._div_batch is None
            and getattr(self.trainer, "global_rank", 0) == 0
        ):
            x, lab, _vidx, _meta, pad_mask = batch
            b = self._build_actor_batch(x[:1], lab[:1], pad_mask[:1])
            self._div_batch = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }

        loss, ld = self._step(batch, train=False)
        self.log_dict({f"val/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if getattr(self.trainer, "global_rank", 0) != 0:
            return

        cm = self.trainer.callback_metrics
        keys = ("val/mpjpe", "val/recon_joint_motion", "val/gt_joint_motion")
        parts = [f"{k}={float(cm[k]):.6f}" for k in keys if k in cm]

        if self._div_batch is not None:
            device = next(self.model.parameters()).device
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in self._div_batch.items()}

            cm_div = torch.ones(1, 17, dtype=torch.bool, device=device)
            cm_div[0, :4] = False
            masked_idx = (~cm_div[0]).nonzero(as_tuple=True)[0]

            with torch.no_grad():
                completions = self.model.sample_completions(
                    b["x"], b["y"], b["mask"], b["lengths"],
                    coalition_mask=cm_div,
                    n_samples=self.diversity_n_samples,
                    paste_observed=True,
                )

            masked_poses = [xh[0, masked_idx] for xh in completions]
            K = len(masked_poses)
            div = 0.0
            if K >= 2:
                stacked = torch.stack(masked_poses, dim=0)
                pairs = 0
                for i in range(K):
                    for j in range(i + 1, K):
                        diff = (stacked[i] - stacked[j]).pow(2).mean().sqrt()
                        div += float(diff.item())
                        pairs += 1
                div /= max(pairs, 1)

            # r_ψ inference-mode imputation: how dynamic is the output?
            with torch.no_grad():
                b_infer = {**b, "coalition_mask": cm_div}
                out_inf = self.model(dict(b_infer), phase="infer")
                imp_motion = masked_mean_joint_motion(
                    out_inf["output"], out_inf["mask"],
                )

            # μ diversity across random coalitions.
            with torch.no_grad():
                n_probe = 8
                mu_masked_list, mu_full_list = [], []
                for _ in range(n_probe):
                    cm_rand = sample_spatial_training_mask(1, device)
                    out_m = self.model.masked_encoder({
                        "x": b["x"], "y": b["y"], "mask": b["mask"],
                        "coalition_mask": cm_rand,
                    })
                    mu_masked_list.append(out_m["mu_masked"].squeeze(0))
                    out_f = self.model.full_encoder({
                        "x": b["x"], "y": b["y"], "mask": b["mask"],
                        "coalition_mask": cm_rand,
                    })
                    mu_full_list.append(out_f["mu_full"].squeeze(0))

                mu_masked_t = torch.stack(mu_masked_list, dim=0)
                mu_full_t = torch.stack(mu_full_list, dim=0)
                mu_div_psi = float(torch.cdist(mu_masked_t, mu_masked_t).mean())
                mu_div_phi = float(torch.cdist(mu_full_t, mu_full_t).mean())

            # σ diagnostics.
            with torch.no_grad():
                fixed_cm = torch.ones(1, 17, dtype=torch.bool, device=device)
                fixed_cm[0, :4] = False
                out_m2 = self.model.masked_encoder({
                    "x": b["x"], "y": b["y"], "mask": b["mask"],
                    "coalition_mask": fixed_cm,
                })
                sigma_psi = float((0.5 * out_m2["logvar_masked"]).exp().mean())

            self.log("val/completion_diversity", div,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/imputation_motion", float(imp_motion),
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/mu_div_psi", mu_div_psi,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/mu_div_phi", mu_div_phi,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            self.log("val/sigma_psi", sigma_psi,
                     on_step=False, on_epoch=True, rank_zero_only=True)
            parts.append(f"div={div:.6f}")
            parts.append(f"imp_mot={float(imp_motion):.6f}")
            parts.append(f"μ_div_ψ={mu_div_psi:.4f}  μ_div_ϕ={mu_div_phi:.4f}")
            parts.append(f"σ_ψ={sigma_psi:.4f}")

            # Save best-diversity checkpoint.
            _best_dir = getattr(self, "_best_diversity_ckpt_dir", None)
            if _best_dir is not None:
                best_so_far = getattr(self, "_best_diversity_val", -1.0)
                if div > best_so_far:
                    self._best_diversity_val = div
                    save_path = os.path.join(_best_dir, "vaeac_motion_best_diversity.ckpt")
                    torch.save({"state_dict": self.state_dict()}, save_path)
                    parts.append(f"[saved best-div ckpt @ {div:.6f}]")

        if parts:
            print(
                f"[VaeacMotion epoch {self.current_epoch}] " + "  ".join(parts),
                flush=True,
            )

    def configure_optimizers(self):
        # q_ϕ base encoder: slow LR to preserve pre-trained weights.
        # q_ϕ target_marker: full LR — must learn coalition awareness.
        # r_ψ: full LR — must learn mask-token handling.
        # Decoder + observed_projection: full LR.
        full_enc_backbone = list(self.model.full_encoder._encoder.parameters())
        full_enc_marker = [self.model.full_encoder.target_marker]
        full_enc_backbone_ids = {id(p) for p in full_enc_backbone}
        full_enc_marker_ids = {id(p) for p in full_enc_marker}

        other_params = [
            p for p in self.model.parameters()
            if id(p) not in full_enc_backbone_ids
            and id(p) not in full_enc_marker_ids
        ]

        param_groups = [
            {"params": full_enc_backbone, "lr": self.lr_full_enc},
            {"params": full_enc_marker, "lr": self.lr},
            {"params": other_params, "lr": self.lr},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="VaeacMotion training for manifold-constrained temporal SHAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--devices", type=str, default="4,5,6,7")
    p.add_argument("--dataset", type=str, default="BMCLab",
                   choices=["BMCLab", "T-SDU-PD", "PD-GaM", "3DGait", "H36M"])
    p.add_argument("--carepd_pose_npz", type=str, default=None)
    p.add_argument("--carepd_labels_pkl", type=str, default=None)
    p.add_argument("--num_folds", type=int, default=23)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source_seq_len", type=int, default=81)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Per-GPU batch size. Default 16 × 4 GPUs = 64 effective.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_full_enc", type=float, default=None,
                   help="q_ϕ backbone LR (default: lr/10).")

    # Architecture.
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--ff_size", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=3)

    # Checkpoint.
    p.add_argument("--cvae_ckpt", type=str, default=None,
                   help="Path to a trained ActorCVAE checkpoint.")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to a VaeacMotion Lightning checkpoint to resume from. "
                        "Loads model weights only (not optimizer/epoch state). "
                        "Compatible with legacy checkpoints (auto-detects "
                        "use_obs_indicator). Mutually exclusive with --cvae_ckpt.")

    # VAEAC loss hyperparameters.
    p.add_argument("--mask_axis", type=str, default="spatial",
                   choices=("spatial", "temporal", "both"))
    p.add_argument("--lambda_kl", type=float, default=0.01)
    p.add_argument("--kl_warmup_epochs", type=int, default=30)
    p.add_argument("--kl_n_cycles", type=int, default=0,
                   help="Cyclical KL annealing (Fu et al. 2019). Number of "
                        "ramp-up/hold cycles over the full training run. "
                        "0 = legacy linear warmup over --kl_warmup_epochs.")
    p.add_argument("--lambda_reg", type=float, default=1e-6)
    p.add_argument("--lambda_vel", type=float, default=5.0)
    p.add_argument("--lambda_rc_obs", type=float, default=0.1)
    p.add_argument("--obs_emb_drop", type=float, default=0.0,
                   help="Probability of dropping entire obs_emb frame-tokens "
                        "during training. Forces decoder to rely on z for "
                        "information about unobserved joints.")
    p.add_argument("--obs_emb_bottleneck", type=int, default=0,
                   help="[Mod A] Bottleneck dim for the observed_projection MLP "
                        "(0 = plain Linear; 16/32/64 = 2-layer MLP with that "
                        "bottleneck). Forces lossy compression of per-frame obs "
                        "signal so decoder must use z for lost information. "
                        "Incompatible with --resume_ckpt (projection reinitialised).")
    p.add_argument("--use_masked_enc_memory", action="store_true",
                   help="[Mod B] Use masked encoder (r_psi) per-frame "
                        "representations as decoder memory instead of the raw "
                        "observed_projection path. Aligns with original VAEAC "
                        "design: same memory at train and inference, natural "
                        "information bottleneck through the encoder transformer.")
    p.add_argument("--no_obs_emb", action="store_true",
                   help="Standard VAEAC / z-only decoder. Do not pass any "
                        "observed-feature embeddings to the decoder. The decoder "
                        "receives only z, as in the original ACTOR and standard "
                        "VAEAC formulation. x_S conditioning is implicit through "
                        "r_psi encoding x_S into z. Removes the per-frame shortcut "
                        "that causes diversity collapse. Recommended for H36M "
                        "pretraining where r_psi has enough data to learn a rich "
                        "conditional prior.")
    p.add_argument("--prior_sigma_mu", type=float, default=1e4,
                   help="Ivanov Eq.8: normal prior width on μ_ψ.")
    p.add_argument("--prior_sigma_sigma", type=float, default=1e-4,
                   help="Ivanov Eq.8: gamma prior tightness on σ_ψ. "
                        "Larger = more freedom for σ_ψ to deviate from 1.0.")
    p.add_argument("--diversity_n_samples", type=int, default=10)

    # Callbacks.
    p.add_argument("--checkpoint_every_n_epochs", type=int, default=100)
    p.add_argument("--gif_every_n_epochs", type=int, default=50)
    p.add_argument("--gif_n_sequences", type=int, default=3)

    # Output.
    p.add_argument("--checkpoint_dir", type=str,
                   default="./experiment_outs/vaeac_motion")
    p.add_argument("--experiment_name", type=str, default="VaeacMotion")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

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

    if args.lr_full_enc is None:
        args.lr_full_enc = args.lr / 10.0

    tag_ds = f"{args.dataset}_fold{args.fold}"
    run_tag = f"vaeac_motion_{tag_ds}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    full_enc = VaeacFullEncoder(**common)
    masked_enc = VaeacMaskedEncoder(**common)
    decoder = Decoder_TRANSFORMER(**common)

    model = VaeacMotion(
        full_enc, masked_enc, decoder,
        latent_dim=args.latent_dim,
        njoints=17, nfeats=3,
        device=device,
        pose_rep="xyz",
        num_classes=args.num_classes,
        obs_emb_drop=args.obs_emb_drop,
        obs_emb_bottleneck=args.obs_emb_bottleneck,
        use_masked_enc_memory=args.use_masked_enc_memory,
        use_obs_emb=not args.no_obs_emb,
    ).to(device)

    if args.resume_ckpt and args.cvae_ckpt:
        raise SystemExit("--resume_ckpt and --cvae_ckpt are mutually exclusive.")
    if args.resume_ckpt:
        raw = torch.load(args.resume_ckpt, map_location=device)
        sd = raw.get("state_dict", raw)
        sd = {(k[len("model."):] if k.startswith("model.") else k): v
              for k, v in sd.items()}

        # Auto-detect legacy / z-only checkpoints.
        ckpt_proj_key = "observed_projection.weight"  # present in plain-Linear ckpts
        ckpt_has_plain_linear = ckpt_proj_key in sd
        if ckpt_has_plain_linear:
            proj_shape = sd[ckpt_proj_key].shape
            proj_in = proj_shape[1]
            # (1, 1) dummy weight → z-only checkpoint; model must match.
            if proj_shape == (1, 1):
                if model.use_obs_emb:
                    raise RuntimeError(
                        "[resume] Checkpoint is z-only (use_obs_emb=False) but "
                        "model was built without --no_obs_emb. Add --no_obs_emb."
                    )
            elif not model.use_obs_emb:
                raise RuntimeError(
                    "[resume] Checkpoint has obs_emb projection but model was "
                    "built with --no_obs_emb. Remove --no_obs_emb."
                )
            elif proj_in == 17 * 3 and model.use_obs_indicator:
                print("[resume] Legacy checkpoint (obs_proj input=51) detected — "
                      "setting use_obs_indicator=False and rebuilding projection.")
                model.use_obs_indicator = False
                if args.obs_emb_bottleneck > 0:
                    model.observed_projection = torch.nn.Sequential(
                        torch.nn.Linear(17 * 3, args.obs_emb_bottleneck),
                        torch.nn.GELU(),
                        torch.nn.Linear(args.obs_emb_bottleneck, args.latent_dim),
                    ).to(device)
                else:
                    model.observed_projection = torch.nn.Linear(
                        17 * 3, args.latent_dim).to(device)

        # Mod A: checkpoint has a plain Linear but model has a bottleneck MLP.
        # Drop the projection keys and reinitialise from scratch.
        drop_proj = args.obs_emb_bottleneck > 0 and ckpt_has_plain_linear
        if drop_proj:
            n_dropped = sum(1 for k in sd if k.startswith("observed_projection."))
            sd = {k: v for k, v in sd.items() if not k.startswith("observed_projection.")}
            print(f"[resume] Mod A: dropping {n_dropped} observed_projection.* keys "
                  f"from checkpoint (bottleneck={args.obs_emb_bottleneck}, will reinit).")

        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Any mismatch outside the projection layer is a real error.
        real_missing = [k for k in missing if not k.startswith("observed_projection.")]
        real_unexpected = [k for k in unexpected if not k.startswith("observed_projection.")]
        if real_missing or real_unexpected:
            raise RuntimeError(
                f"Checkpoint load error — missing: {real_missing}, "
                f"unexpected: {real_unexpected}"
            )
        if missing or unexpected:
            print(f"[resume] Skipped projection keys — missing: {missing}, "
                  f"unexpected: {unexpected}")
        print(f"Resumed VaeacMotion weights from: {args.resume_ckpt}")
    elif args.cvae_ckpt:
        model.load_from_cvae_checkpoint(args.cvae_ckpt)
        print(f"Loaded ActorCVAE checkpoint: {args.cvae_ckpt}")
    else:
        print("WARNING: no --cvae_ckpt or --resume_ckpt provided. "
              "Starting from random weights.")

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
    module = VaeacMotionModule(
        model,
        lr=args.lr,
        lr_full_enc=args.lr_full_enc,
        lambda_kl=args.lambda_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_n_cycles=args.kl_n_cycles,
        total_epochs=args.epochs,
        lambda_reg=args.lambda_reg,
        lambda_vel=args.lambda_vel,
        lambda_rc_obs=args.lambda_rc_obs,
        prior_sigma_mu=args.prior_sigma_mu,
        prior_sigma_sigma=args.prior_sigma_sigma,
        mask_axis=args.mask_axis,
        diversity_n_samples=args.diversity_n_samples,
    )
    module._best_diversity_ckpt_dir = ckpt_dir

    extra_callbacks = []
    if args.checkpoint_every_n_epochs and args.checkpoint_every_n_epochs > 0:
        extra_callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                every_n_epochs=args.checkpoint_every_n_epochs,
                filename="vaeac_motion_epoch{epoch:04d}",
                save_top_k=-1,
            ),
        )
    if args.gif_every_n_epochs and args.gif_every_n_epochs > 0:
        gif_dir = os.path.join(ckpt_dir, "train_gifs")
        extra_callbacks.append(
            ImputationGifCallback(
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
        phase_tag="vaeac_motion",
        extra_callbacks=extra_callbacks if extra_callbacks else None,
        find_unused_parameters=True,
        save_last_only=True,
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"Done. Last: {ckpt_dir}/vaeac_motion_last.ckpt")


if __name__ == "__main__":
    main()
