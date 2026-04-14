"""
train_vaeac_actor_motion.py — Train VaeacActorMotion for manifold-constrained SHAP
====================================================================================

Trains a sequence-latent VAEAC where the encoder latent is a full T×D sequence
rather than a single D-vector.  See model/actor/vaeac_actor_motion.py for the
full architectural description.

Architecture
------------
q_φ(z | x)          VaeacActorFullEncoder — sees complete sequence, outputs
                     per-frame (T, B, D) mu / logvar.
r_ψ(z | x_S, S)     VaeacActorMaskedEncoder — replaces unobserved positions
                     with learned mask tokens, outputs (T, B, D) mu / logvar.
p_θ(x̂ | z)          Decoder_TRANSFORMER — cross-attends to z_seq as T memory
                     tokens (plus one class-conditioning token).

Objective
---------
L = MSE(p_θ(z_φ), x)          full-sequence reconstruction (like ACTOR)
    + λ_vel · velocity_loss    full-sequence velocity
    − λ_kl  · KL(q_φ ‖ r_ψ)  per-frame regularisation aligning r_ψ to q_φ
    − λ_reg · prior_reg(r_ψ)  Ivanov prior stabiliser

Gradient routing:
    q_φ: reconstruction + velocity only (DETACHED from KL).
    r_ψ: KL + prior regularisation only.
    p_θ: reconstruction + velocity.

Typical command::

    python train_vaeac_actor_motion.py \\
        --dataset H36M --fold 1 --epochs 300 --lr 1e-4 \\
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
)
from model.actor.transformer_arch import Decoder_TRANSFORMER
from model.actor.vaeac_actor_motion import (
    VaeacActorFullEncoder,
    VaeacActorMaskedEncoder,
    VaeacActorMotion,
)
from train_actor_cvae import masked_mean_joint_motion, masked_mpjpe
from train_utils import make_trainer
from utility.utils import set_random_seed

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


# ---------------------------------------------------------------------------
# Temporal mask sampler — same as train_vaeac_motion.py
# ---------------------------------------------------------------------------

def sample_temporal_training_mask_mixed(
    B: int,
    T: int,
    device: torch.device,
    stride_sim_prob: float = 0.5,
) -> torch.Tensor:
    """Mix contiguous quarter-blocks and stride-simulated temporal coalition masks."""
    masks = torch.ones(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        if torch.rand(1).item() < stride_sim_prob:
            stride_period = int(torch.randint(15, 46, (1,)).item())
            windows = build_temporal_windows(T, stride_period, K=4)
            observed = [k for k in range(4) if torch.rand(1).item() > 0.5]
            if len(observed) == 0:
                observed = [0]
            masks[i] = build_temporal_shap_mask(observed, windows, T, device)
        else:
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
# GIF callback (identical to train_vaeac_motion.py)
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
# Lightning module
# ---------------------------------------------------------------------------

class VaeacActorMotionModule(pl.LightningModule):
    """PyTorch Lightning wrapper for VaeacActorMotion training.

    Gradient routing:
        q_φ backbone: slow LR (preserves representation quality).
        r_ψ + decoder: full LR (must learn mask-token encoding fast).

    Diversity probe (on_validation_epoch_end):
        mu/logvar from encoders are (T, B, D).  We mean-pool over T before
        computing pairwise latent distances, giving a D-dim per-sequence
        summary comparable to VaeacMotion's (B, D) latent diversity.
    """

    def __init__(
        self,
        model: VaeacActorMotion,
        lr: float,
        lr_full_enc: float | None = None,
        *,
        lambda_kl: float = 0.01,
        kl_warmup_epochs: int = 30,
        kl_n_cycles: int = 0,
        total_epochs: int = 300,
        lambda_reg: float = 1e-6,
        lambda_vel: float = 5.0,
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
        self.prior_sigma_mu = prior_sigma_mu
        self.prior_sigma_sigma = prior_sigma_sigma
        self.mask_axis = mask_axis
        self.diversity_n_samples = diversity_n_samples
        self._div_batch: dict | None = None

    def _build_actor_batch(self, x, lab, pad_mask):
        device = x.device
        b = actor_batch_from_carepd(x, pad_mask.bool(), self.model.num_classes, device)
        b["y"] = lab.long().to(device)
        return b

    def _effective_lambda_kl(self) -> float:
        if self.kl_n_cycles > 0:
            cycle_len = max(1, self.total_epochs // self.kl_n_cycles)
            pos_in_cycle = self.current_epoch % cycle_len
            ramp_half = cycle_len // 2
            frac = pos_in_cycle / ramp_half if (ramp_half > 0 and pos_in_cycle < ramp_half) else 1.0
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

            # ---- completion diversity ----
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
                        div += float((stacked[i] - stacked[j]).pow(2).mean().sqrt().item())
                        pairs += 1
                div /= max(pairs, 1)

            # ---- inference-mode imputation motion ----
            with torch.no_grad():
                b_infer = {**b, "coalition_mask": cm_div}
                out_inf = self.model(dict(b_infer), phase="infer")
                imp_motion = masked_mean_joint_motion(
                    out_inf["output"], out_inf["mask"],
                )

            # ---- μ diversity across random coalitions ----
            # mu_masked / mu_full are (T, B, D); mean-pool over T for pairwise L2.
            with torch.no_grad():
                n_probe = 8
                mu_masked_list, mu_full_list = [], []
                for _ in range(n_probe):
                    cm_rand = sample_spatial_training_mask(1, device)
                    out_m = self.model.masked_encoder({
                        "x": b["x"], "y": b["y"], "mask": b["mask"],
                        "coalition_mask": cm_rand,
                    })
                    # (T, 1, D) → mean over T → (1, D) → squeeze → (D,)
                    mu_masked_list.append(out_m["mu_masked"].mean(0).squeeze(0))

                    out_f = self.model.full_encoder({
                        "x": b["x"], "y": b["y"], "mask": b["mask"],
                    })
                    mu_full_list.append(out_f["mu_full"].mean(0).squeeze(0))

                mu_masked_t = torch.stack(mu_masked_list, dim=0)  # (n_probe, D)
                mu_full_t = torch.stack(mu_full_list, dim=0)
                mu_div_psi = float(torch.cdist(mu_masked_t, mu_masked_t).mean())
                mu_div_phi = float(torch.cdist(mu_full_t, mu_full_t).mean())

            # ---- σ diagnostics ----
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
                    save_path = os.path.join(_best_dir, "vaeac_actor_motion_best_diversity.ckpt")
                    torch.save({"state_dict": self.state_dict()}, save_path)
                    parts.append(f"[saved best-div ckpt @ {div:.6f}]")

        if parts:
            print(
                f"[VaeacActorMotion epoch {self.current_epoch}] " + "  ".join(parts),
                flush=True,
            )

    def configure_optimizers(self):
        # q_φ backbone: slow LR — already encodes high-quality motion.
        # r_ψ + decoder: full LR — must learn to work with mask tokens.
        full_enc_params = list(self.model.full_encoder._encoder.parameters())
        full_enc_ids = {id(p) for p in full_enc_params}
        other_params = [
            p for p in self.model.parameters()
            if id(p) not in full_enc_ids
        ]
        param_groups = [
            {"params": full_enc_params, "lr": self.lr_full_enc},
            {"params": other_params,    "lr": self.lr},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="VaeacActorMotion training — per-frame latent VAEAC on ACTOR backbone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--devices", type=str, default="4,5,6,7")
    p.add_argument("--dataset", type=str, default="H36M",
                   choices=["BMCLab", "T-SDU-PD", "PD-GaM", "3DGait", "H36M"])
    p.add_argument("--carepd_pose_npz", type=str, default=None)
    p.add_argument("--carepd_labels_pkl", type=str, default=None)
    p.add_argument("--num_folds", type=int, default=7,
                   help="Number of LOSO folds (7 for H36M, 23 for BMCLab).")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source_seq_len", type=int, default=81)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Per-GPU batch size. Default 16 × 4 GPUs = 64 effective.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_full_enc", type=float, default=None,
                   help="q_φ backbone LR (default: lr/10).")

    # Architecture.
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--ff_size", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=3)

    # Checkpoint resume (VaeacActorMotion only — no CVAE warm-start).
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to a VaeacActorMotion Lightning checkpoint to resume from. "
                        "Loads model weights only (not optimizer/epoch state).")

    # VAEAC loss hyperparameters.
    p.add_argument("--mask_axis", type=str, default="spatial",
                   choices=("spatial", "temporal", "both"))
    p.add_argument("--lambda_kl", type=float, default=0.01,
                   help="KL weight. /T normalisation in compute_loss keeps this "
                        "at the same scale as VaeacMotion's λ_kl.")
    p.add_argument("--kl_warmup_epochs", type=int, default=30)
    p.add_argument("--kl_n_cycles", type=int, default=0,
                   help="Cyclical KL annealing (Fu et al. 2019). 0 = linear warmup.")
    p.add_argument("--lambda_reg", type=float, default=1e-6)
    p.add_argument("--lambda_vel", type=float, default=5.0)
    p.add_argument("--prior_sigma_mu", type=float, default=1e4,
                   help="Ivanov Eq.8: normal prior width on μ_ψ.")
    p.add_argument("--prior_sigma_sigma", type=float, default=1e-4,
                   help="Ivanov Eq.8: gamma prior tightness on σ_ψ.")
    p.add_argument("--diversity_n_samples", type=int, default=10)

    # Callbacks.
    p.add_argument("--checkpoint_every_n_epochs", type=int, default=100)
    p.add_argument("--gif_every_n_epochs", type=int, default=50)
    p.add_argument("--gif_n_sequences", type=int, default=3)

    # Output.
    p.add_argument("--checkpoint_dir", type=str,
                   default="./experiment_outs/vaeac_actor_motion")
    p.add_argument("--experiment_name", type=str, default="VaeacActorMotion")
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
    run_tag = f"vaeac_actor_motion_{tag_ds}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
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

    full_enc = VaeacActorFullEncoder(**common)
    masked_enc = VaeacActorMaskedEncoder(**common)
    decoder = Decoder_TRANSFORMER(**common)

    model = VaeacActorMotion(
        full_enc, masked_enc, decoder,
        latent_dim=args.latent_dim,
        njoints=17, nfeats=3,
        device=device,
        pose_rep="xyz",
        num_classes=args.num_classes,
    ).to(device)

    if args.resume_ckpt:
        raw = torch.load(args.resume_ckpt, map_location=device)
        sd = raw.get("state_dict", raw)
        sd = {(k[len("model."):] if k.startswith("model.") else k): v
              for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load error — missing: {missing}, unexpected: {unexpected}"
            )
        print(f"Resumed VaeacActorMotion weights from: {args.resume_ckpt}")
    else:
        print("Starting VaeacActorMotion from random weights (no --resume_ckpt provided).")

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
    module = VaeacActorMotionModule(
        model,
        lr=args.lr,
        lr_full_enc=args.lr_full_enc,
        lambda_kl=args.lambda_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_n_cycles=args.kl_n_cycles,
        total_epochs=args.epochs,
        lambda_reg=args.lambda_reg,
        lambda_vel=args.lambda_vel,
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
                filename="vaeac_actor_motion_epoch{epoch:04d}",
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
        phase_tag="vaeac_actor_motion",
        extra_callbacks=extra_callbacks if extra_callbacks else None,
        find_unused_parameters=True,
        save_last_only=True,
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"Done. Last: {ckpt_dir}/vaeac_actor_motion_last.ckpt")


if __name__ == "__main__":
    main()
