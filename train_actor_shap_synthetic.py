"""train_actor_shap_synthetic.py — Train ActorSHAP on synthetic benchmark data.

Self-contained — does NOT import from train_actor_shap.py or train_actor_cvae.py
(which cascade to data.dataloaders and require torch_dct).

Two data modes:
  --data_mode synthetic_gaussian   — Gaussian equicorrelation × AR(1) motion data
  --data_mode synthetic_diagnostic — Fourier-series gait with diagnostic joints

On training completion, saves:
  <ckpt_dir>/actor_shap_synthetic_last.ckpt   — model checkpoint
  <ckpt_dir>/synthetic_test.pt                — test tensors for evaluate script
  <ckpt_dir>/synthetic_benchmark.pkl          — GaussianMotionBenchmark instance
  <ckpt_dir>/synthetic_clf.pt                 — classifier state_dict
  <ckpt_dir>/x_train_jft.npy                 — (N_train, J, F, T) training sequences
  <ckpt_dir>/config.json                      — all CLI arguments

TYPICAL USAGE
-------------
Gaussian benchmark (fast iteration):

    python train_actor_shap_synthetic.py \\
        --data_mode synthetic_gaussian \\
        --rho 0.5 --alpha 0.8 \\
        --n_train 2000 --n_val 500 --n_test 100 \\
        --epochs 200 --phase0_epochs 0 \\
        --latent_dim 128 --num_layers 6 --num_heads 4 \\
        --lambda_kl 0.01 --lambda_rc_psi 1.0 \\
        --checkpoint_dir experiment_outs/actor_shap_synthetic

Diagnostic benchmark:

    python train_actor_shap_synthetic.py \\
        --data_mode synthetic_diagnostic \\
        --n_train 800 --n_val 100 --n_test 100 \\
        --mask_axis spatial \\
        --epochs 200 --phase0_epochs 0 \\
        --checkpoint_dir experiment_outs/actor_shap_diagnostic
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

# These import cleanly (no data.dataloaders / torch_dct cascade).
from model.actor.actor_shap import ActorSHAP, CoalitionFullEncoder, MaskedActorEncoder
from model.actor.shap_masking import sample_spatial_training_mask, sample_temporal_training_mask
from model.actor.transformer_arch import Decoder_TRANSFORMER
from train_utils import make_trainer


def set_random_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

from synthetic.gaussian_motion import (
    GaussianMotionBenchmark,
    SyntheticMLPClassifier,
    build_gaussian_benchmark_and_classifier,
)
from synthetic.diagnostic_motion import build_diagnostic_dataset


# ---------------------------------------------------------------------------
# Inlined helpers (avoid cascade import from train_actor_cvae.py)
# ---------------------------------------------------------------------------

def _masked_mpjpe(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean per-joint L2 error over valid frames.  x: (B,J,F,T), mask: (B,T)."""
    err = torch.linalg.norm(x_hat - x, dim=2)
    m = mask.unsqueeze(1).expand_as(err)
    return err.masked_select(m).mean()


def _masked_mean_joint_motion(x_bjft: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean ||x[t]-x[t-1]||_2 over valid consecutive pairs."""
    if x_bjft.shape[-1] < 2:
        return x_bjft.new_zeros(())
    d = x_bjft[..., 1:] - x_bjft[..., :-1]
    vel = torch.linalg.norm(d, dim=2)
    pair = mask[:, :-1] & mask[:, 1:]
    m = pair.unsqueeze(1).expand_as(vel)
    sel = vel.masked_select(m)
    return sel.mean() if sel.numel() > 0 else x_bjft.new_zeros(())


def _make_actor_batch(
    x_bttf: torch.Tensor,     # (B, T, J, F)  ACTOR format
    lab: torch.Tensor,         # (B,) long
    pad_mask: torch.Tensor,    # (B, T) bool
    device: torch.device,
) -> dict:
    """Minimal ACTOR batch dict for synthetic (Gaussian-mean, no root-centering)."""
    x_bttf   = x_bttf.float().to(device)
    pad_mask = pad_mask.bool().to(device)
    x_bjft   = x_bttf.permute(0, 2, 3, 1).contiguous()  # (B, J, F, T)
    lengths  = pad_mask.sum(dim=-1).long()
    return {
        "x":       x_bjft,
        "y":       lab.long().to(device),
        "mask":    pad_mask,
        "lengths": lengths,
    }


# ---------------------------------------------------------------------------
# Self-contained Lightning module
# ---------------------------------------------------------------------------

class SyntheticActorSHAPModule(pl.LightningModule):
    """ActorSHAP VAEAC training module for synthetic benchmark data.

    Implements the same two-phase VAEAC ELBO as ActorSHAPModule
    (from train_actor_shap.py) but without data-loading infrastructure
    imports (data.dataloaders, torch_dct, etc.).

    Batches are plain 3-tuples: (x_bttf: Tensor, y: Tensor, pad_mask: Tensor).
    """

    def __init__(
        self,
        model: ActorSHAP,
        lr: float,
        lr_decoder: float,
        *,
        phase0_epochs: int = 0,
        lambda_kl: float = 0.01,
        kl_warmup_epochs: int = 20,
        lambda_reg: float = 1e-6,
        lambda_kl_full: float = 1e-4,
        lambda_vel: float = 5.0,
        lambda_rc_obs: float = 0.1,
        lambda_rc_psi: float = 1.0,
        mask_axis: str = "temporal",
        diversity_n_samples: int = 10,
    ):
        super().__init__()
        self.model              = model
        self.lr                 = lr
        self.lr_decoder         = lr_decoder
        self.phase0_epochs      = phase0_epochs
        self.lambda_kl          = lambda_kl
        self.kl_warmup_epochs   = kl_warmup_epochs
        self.lambda_reg         = lambda_reg
        self.lambda_kl_full     = lambda_kl_full
        self.lambda_vel         = lambda_vel
        self.lambda_rc_obs      = lambda_rc_obs
        self.lambda_rc_psi      = lambda_rc_psi
        self.mask_axis          = mask_axis
        self.diversity_n_samples = diversity_n_samples
        self._div_batch: dict | None = None

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        in_phase0 = self.current_epoch < self.phase0_epochs
        for p in self.model.decoder.parameters():
            p.requires_grad_(not in_phase0)
        self.model.encoder.target_marker_spatial.requires_grad_(not in_phase0)
        if self.current_epoch == 0:
            if self.phase0_epochs == 0:
                print("[SyntheticActorSHAP] Phase 0 skipped. VAEAC training from epoch 0.", flush=True)
            else:
                print(f"[SyntheticActorSHAP] Phase 0 warm-up for {self.phase0_epochs} epochs.", flush=True)
        elif self.current_epoch == self.phase0_epochs and self.phase0_epochs > 0:
            print(f"[SyntheticActorSHAP] Phase 1 VAEAC: decoder unfrozen.", flush=True)

    def _effective_lambda_kl(self, epoch: int) -> float:
        if epoch < self.phase0_epochs:
            return 0.0
        elapsed = epoch - self.phase0_epochs
        if self.kl_warmup_epochs <= 0:
            return self.lambda_kl
        return min(1.0, elapsed / max(1, self.kl_warmup_epochs)) * self.lambda_kl

    # ------------------------------------------------------------------
    # Phase 0 step
    # ------------------------------------------------------------------

    def _step_phase0(self, b: dict) -> tuple[torch.Tensor, dict]:
        B, J, F, T = b["x"].shape
        device = b["x"].device

        out_m = self.model.masked_encoder(b)
        z  = out_m["mu_masked"]
        ft = out_m.get("frame_tokens_masked")

        dec_batch = {**b, "z": z}
        if ft is not None:
            dec_batch["frame_tokens"] = ft
        dec_batch.update(self.model.decoder(dec_batch))
        output = dec_batch["output"]

        obs_mask    = b["coalition_mask"].unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
        real_frames = b["mask"].unsqueeze(1).unsqueeze(1).expand(B, J, F, T)
        active      = obs_mask & real_frames

        rc = ((output - b["x"]) ** 2)[active].mean()

        vel_loss = torch.tensor(0.0, device=device)
        if self.lambda_vel > 0 and T > 1:
            vel_p = output[:, :, :, 1:] - output[:, :, :, :-1]
            vel_g = b["x"][:, :, :, 1:]  - b["x"][:, :, :, :-1]
            real_vel = b["mask"][:, 1:].unsqueeze(1).unsqueeze(1).expand(B, J, F, T - 1)
            obs_vel  = b["coalition_mask"].unsqueeze(2).unsqueeze(3).expand(B, J, F, T - 1)
            if (obs_vel & real_vel).any():
                vel_loss = self.lambda_vel * ((vel_p - vel_g) ** 2)[obs_vel & real_vel].mean()

        loss = rc + vel_loss
        with torch.no_grad():
            mpjpe        = _masked_mpjpe(output, b["x"], b["mask"])
            recon_motion = _masked_mean_joint_motion(output, b["mask"])
            gt_motion    = _masked_mean_joint_motion(b["x"],  b["mask"])

        ld = {
            "rc_obs_p0": float(rc.item()),
            "vel_p0":    float(vel_loss.item()),
            "mixed":     float(loss.item()),
            "_mpjpe":    float(mpjpe.item()),
            "_recon_mot": float(recon_motion.item()),
            "_gt_mot":   float(gt_motion.item()),
        }
        return loss, ld

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def _step(self, batch, train: bool):
        x, lab, pad_mask = batch
        device = next(self.model.parameters()).device
        b = _make_actor_batch(x, lab, pad_mask, device)

        B = x.shape[0]
        T = b["mask"].shape[1]
        if self.mask_axis == "temporal":
            b["coalition_mask"] = sample_temporal_training_mask(B, T, device)
        else:
            b["coalition_mask"] = sample_spatial_training_mask(B, device)

        if self.current_epoch < self.phase0_epochs:
            loss, ld = self._step_phase0(b)
            mpjpe        = ld.pop("_mpjpe")
            recon_motion = ld.pop("_recon_mot")
            gt_motion    = ld.pop("_gt_mot")
        else:
            out = self.model(b, phase=2)
            lam = self._effective_lambda_kl(self.current_epoch)
            loss, ld = self.model.compute_loss(
                out,
                lambda_kl=lam,
                lambda_reg=self.lambda_reg,
                lambda_kl_full=self.lambda_kl_full,
                lambda_vel=self.lambda_vel,
                lambda_rc_obs=self.lambda_rc_obs,
                lambda_rc_psi=self.lambda_rc_psi,
            )
            mask = out["mask"]
            with torch.no_grad():
                x_m = out.get("x_xyz", out["x"])
                o_m = out.get("output_xyz", out["output"])
                mpjpe        = _masked_mpjpe(o_m, x_m, mask)
                recon_motion = _masked_mean_joint_motion(o_m, mask)
                gt_motion    = _masked_mean_joint_motion(x_m, mask)

                if not train:
                    b3 = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in b.items()}
                    out3   = self.model(b3, phase=3)
                    o3     = out3.get("output_xyz", out3["output"])
                    rpsi   = _masked_mean_joint_motion(o3, mask)
                    self.log("val/rpsi_recon_motion", rpsi,
                             on_step=False, on_epoch=True, sync_dist=True)

        prefix = "train" if train else "val"
        self.log(f"{prefix}/mpjpe",             mpjpe,        on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}/recon_joint_motion", recon_motion, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}/gt_joint_motion",    gt_motion,    on_step=False, on_epoch=True, sync_dist=True)
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
            x, lab, pad_mask = batch
            dev = next(self.model.parameters()).device
            b = _make_actor_batch(x[:1], lab[:1], pad_mask[:1], dev)
            self._div_batch = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
        loss, ld = self._step(batch, train=False)
        self.log_dict({f"val/{k}": v for k, v in ld.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        """Log completion diversity for monitoring."""
        if (
            self._div_batch is None
            or getattr(self.trainer, "global_rank", 0) != 0
            or self.current_epoch < self.phase0_epochs
        ):
            return
        try:
            dev = next(self.model.parameters()).device
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                 for k, v in self._div_batch.items()}
            n = self.diversity_n_samples
            # Build a simple coalition mask: observe all joints/temporal windows.
            B, J, F, T = b["x"].shape
            if self.mask_axis == "temporal":
                # Mask all windows: observe first 2, hide last 2.
                cm = sample_temporal_training_mask(1, T, dev)
                # Override: observe first half, hide second half.
                cm[0, T // 2:] = False
            else:
                cm = sample_spatial_training_mask(1, dev)
            b_probe = {**b, "coalition_mask": cm}
            comps = self.model.sample_completions(
                b["x"], b["y"], b["mask"], b["lengths"], cm, n_samples=n
            )
            if len(comps) >= 2:
                stacked = torch.cat(comps, dim=0)  # (n, J, F, T)
                # Diversity = mean pairwise RMSE over masked frames.
                hid = ~cm[0]  # (T,) hidden frames
                x_hid = stacked[:, :, :, hid]  # (n, J, F, n_hid)
                diffs = x_hid[:, None] - x_hid[None, :]  # (n, n, J, F, n_hid)
                rmse = diffs.norm(dim=3).mean(dim=(3, 4))  # (n, n, J)
                diversity = rmse.mean()
                self.log("val/completion_diversity", diversity,
                         on_step=False, on_epoch=True, rank_zero_only=True)
        except Exception:
            pass  # diversity metric is non-critical

    def configure_optimizers(self):
        # Freeze q_ϕ base encoder (always).
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
        return optim.AdamW(param_groups, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ActorSHAP on synthetic benchmark data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_mode", default="synthetic_gaussian",
                   choices=("synthetic_gaussian", "synthetic_diagnostic"))
    # Gaussian-specific.
    p.add_argument("--rho",   type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val",   type=int, default=500)
    p.add_argument("--n_test",  type=int, default=100)
    # Diagnostic-specific.
    p.add_argument("--signal_scale", type=float, default=0.12)
    # Architecture.
    p.add_argument("--J",           type=int,   default=17)
    p.add_argument("--F",           type=int,   default=3)
    p.add_argument("--T",           type=int,   default=81)
    p.add_argument("--latent_dim",  type=int,   default=128)
    p.add_argument("--ff_size",     type=int,   default=512)
    p.add_argument("--num_layers",  type=int,   default=6)
    p.add_argument("--num_heads",   type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--num_classes", type=int,   default=3)
    # Training hyperparameters.
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--lr_decoder",       type=float, default=None)
    p.add_argument("--phase0_epochs",    type=int,   default=0)
    p.add_argument("--mask_axis",        type=str,   default="temporal",
                   choices=("spatial", "temporal"))
    p.add_argument("--lambda_kl",        type=float, default=0.01)
    p.add_argument("--kl_warmup_epochs", type=int,   default=20)
    p.add_argument("--lambda_reg",       type=float, default=1e-6)
    p.add_argument("--lambda_kl_full",   type=float, default=1e-4)
    p.add_argument("--lambda_vel",       type=float, default=5.0)
    p.add_argument("--lambda_rc_obs",    type=float, default=0.1)
    p.add_argument("--lambda_rc_psi",    type=float, default=1.0)
    p.add_argument("--diversity_n_samples", type=int, default=10)
    p.add_argument("--actor_cvae_ckpt",  type=str,   default=None)
    # Output.
    p.add_argument("--checkpoint_dir",   type=str,   default="experiment_outs/actor_shap_synthetic")
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--devices",          type=str,   default=None,
                   help="CUDA_VISIBLE_DEVICES value (e.g. '0,1').")
    p.add_argument("--checkpoint_every_n_epochs", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    if args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    if args.lr_decoder is None:
        args.lr_decoder = args.lr / 10.0

    torch.set_float32_matmul_precision("high")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = f"actor_shap_synthetic_{args.data_mode}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Build ActorSHAP model ----------------------------------------
    common = dict(
        modeltype="cvae",
        njoints=args.J, nfeats=args.F,
        num_frames=0, num_classes=args.num_classes,
        translation=True, pose_rep="xyz",
        glob=True, glob_rot=[3.141592653589793, 0, 0],
        latent_dim=args.latent_dim, ff_size=args.ff_size,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, ablation=None, activation="gelu",
    )
    encoder        = CoalitionFullEncoder(**common)
    masked_encoder = MaskedActorEncoder(**common)
    decoder        = Decoder_TRANSFORMER(**common)
    model = ActorSHAP(
        encoder, masked_encoder, decoder,
        latent_dim=args.latent_dim, device=device,
        pose_rep="xyz", num_classes=args.num_classes,
    ).to(device)

    if args.actor_cvae_ckpt:
        model.load_from_actor_cvae(args.actor_cvae_ckpt)
        print(f"[SyntheticTrain] Loaded ActorCVAE checkpoint: {args.actor_cvae_ckpt}")
    else:
        print("[SyntheticTrain] Starting from random weights.")

    # ---- Build datasets -----------------------------------------------
    print(f"[SyntheticTrain] Building {args.data_mode} data …")

    if args.data_mode == "synthetic_gaussian":
        bench, clf, train_ds, val_ds, test_ds = build_gaussian_benchmark_and_classifier(
            rho=args.rho, alpha=args.alpha,
            J=args.J, F=args.F, T=args.T,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            clf_epochs=80, seed=args.seed, device=device,
        )
        # Save benchmark.
        bench.save(os.path.join(ckpt_dir, "synthetic_benchmark.pkl"))
        torch.save(clf.state_dict(), os.path.join(ckpt_dir, "synthetic_clf.pt"))
        clf_meta = {"type": "SyntheticMLPClassifier",
                    "J": args.J, "F": args.F, "T": args.T, "K": 4,
                    "num_classes": args.num_classes}
        with open(os.path.join(ckpt_dir, "synthetic_clf_meta.json"), "w") as fp:
            json.dump(clf_meta, fp)
        print(f"  Gaussian: rho={args.rho}, alpha={args.alpha}")

    elif args.data_mode == "synthetic_diagnostic":
        clf_diag, x_test_np, phi_true_test, train_ds, val_ds, test_ds = build_diagnostic_dataset(
            N=args.n_train + args.n_val + args.n_test,
            T=args.T, signal_scale=args.signal_scale,
            seed=args.seed, num_classes=args.num_classes,
        )
        bench = None
        torch.save(clf_diag.state_dict(), os.path.join(ckpt_dir, "synthetic_clf.pt"))
        np.save(os.path.join(ckpt_dir, "x_test.npy"),        x_test_np)
        np.save(os.path.join(ckpt_dir, "phi_true_test.npy"), phi_true_test)
        np.save(os.path.join(ckpt_dir, "w_true.npy"),        clf_diag.w.cpu().numpy())
        if hasattr(clf_diag, "mu_j"):
            np.save(os.path.join(ckpt_dir, "mu_j.npy"), clf_diag.mu_j.cpu().numpy())
        print(f"  Diagnostic: signal_scale={args.signal_scale}")

    else:
        raise ValueError(f"Unknown data_mode: {args.data_mode}")

    # Save raw test tensors.
    x_test_t, y_test_t, pm_test = test_ds.tensors
    torch.save({"x": x_test_t, "y": y_test_t, "pad_mask": pm_test},
               os.path.join(ckpt_dir, "synthetic_test.pt"))

    # Save training sequences in (N, J, F, T) format for baselines.
    x_tr_t, _, _ = train_ds.tensors
    x_tr_jft = x_tr_t.permute(0, 2, 3, 1).contiguous().numpy()  # (N, J, F, T)
    np.save(os.path.join(ckpt_dir, "x_train_jft.npy"), x_tr_jft)

    print(f"  {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test sequences.")

    # ---- DataLoaders --------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=(n_gpus > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=(n_gpus > 0))

    # ---- Lightning module + trainer -----------------------------------
    module = SyntheticActorSHAPModule(
        model,
        lr=args.lr, lr_decoder=args.lr_decoder,
        phase0_epochs=args.phase0_epochs,
        lambda_kl=args.lambda_kl, kl_warmup_epochs=args.kl_warmup_epochs,
        lambda_reg=args.lambda_reg, lambda_kl_full=args.lambda_kl_full,
        lambda_vel=args.lambda_vel, lambda_rc_obs=args.lambda_rc_obs,
        lambda_rc_psi=args.lambda_rc_psi,
        mask_axis=args.mask_axis,
        diversity_n_samples=args.diversity_n_samples,
    )

    extra_callbacks = []
    if args.checkpoint_every_n_epochs > 0:
        extra_callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                every_n_epochs=args.checkpoint_every_n_epochs,
                filename="actor_shap_epoch{epoch:04d}",
                save_top_k=-1,
            )
        )

    trainer = make_trainer(
        n_epochs=args.epochs, n_gpus=n_gpus,
        ckpt_dir=ckpt_dir, logger=False,
        monitor_key="val/mixed",
        phase_tag="actor_shap_synthetic",
        extra_callbacks=extra_callbacks or None,
        find_unused_parameters=True,
        save_last_only=True,
    )
    trainer.fit(module, train_loader, val_loader)

    # Explicitly save a final checkpoint so evaluate_shap_synthetic.py always
    # finds one, regardless of make_trainer's every_n_epochs trigger.
    final_ckpt = os.path.join(ckpt_dir, "actor_shap_synthetic_last.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"\n[SyntheticTrain] Saved final checkpoint: {final_ckpt}")

    print(f"\n[SyntheticTrain] Done.")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  Evaluate with:")
    mode = "gaussian" if args.data_mode == "synthetic_gaussian" else "diagnostic"
    print(f"    python evaluate_shap_synthetic.py {mode} --ckpt_dir {ckpt_dir}")


if __name__ == "__main__":
    main()
