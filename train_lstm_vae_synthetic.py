"""train_lstm_vae_synthetic.py — Train LstmVAE on the same synthetic benchmark
used by train_actor_shap_synthetic.py, enabling side-by-side comparison in
evaluate_shap_synthetic.py.

Usage
-----
# Reuse an existing ActorSHAP data directory (recommended — same data, fair comparison):
python train_lstm_vae_synthetic.py \\
    --actor_data_dir experiment_outs/actor_shap_synthetic/actor_shap_synthetic_synthetic_gaussian_20260415_120510 \\
    --epochs 200 --n_mix 5

# Generate fresh synthetic data independently:
python train_lstm_vae_synthetic.py \\
    --data_mode synthetic_gaussian \\
    --rho 0.5 --alpha 0.8 \\
    --n_train 2000 --n_val 500 --n_test 100 \\
    --epochs 200 --n_mix 5 \\
    --checkpoint_dir experiment_outs/lstm_vae_synthetic

On completion saves into the output directory:
  lstm_vae_synthetic_last.ckpt    — Lightning checkpoint (restorable with SyntheticLstmVAEModule)
  lstm_vae_model.pt               — raw model state_dict (for evaluate_shap_synthetic.py)
  lstm_vae_config.json            — model + data hyperparameters

The output directory is:
  - <actor_data_dir>/  (when --actor_data_dir is used)
  - <checkpoint_dir>/lstm_vae_synthetic_<timestamp>/  (otherwise)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model.lstm_vae.model import LstmVAE, kl_full_vs_mixture


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Mask sampling (inlined from model.actor.shap_masking to avoid cascade)
# ---------------------------------------------------------------------------

def _sample_temporal_training_mask(B: int, T: int, device: torch.device, K: int = 4) -> Tensor:
    """Random temporal coalition masks: K equal windows, each masked with p=0.5."""
    quarter = T // K
    windows = []
    for k in range(K):
        start = k * quarter
        end = (k + 1) * quarter if k < K - 1 else T
        windows.append(list(range(start, end)))
    masks = torch.ones(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        for frames in windows:
            if torch.rand(1).item() < 0.5:
                masks[i, frames] = False
        if masks[i].sum() == 0:
            masks[i, windows[0]] = True
    return masks


def _sample_spatial_training_mask(B: int, J: int, device: torch.device) -> Tensor:
    """Random spatial coalition masks: per-joint Bernoulli."""
    masks = torch.ones(B, J, dtype=torch.bool, device=device)
    for i in range(B):
        p_mask = 0.1 + 0.8 * torch.rand(1).item()
        keep = torch.bernoulli(torch.full((J,), 1.0 - p_mask)).bool()
        masks[i] = keep
        if masks[i].sum() == 0:
            masks[i, 0] = True
    return masks


# ---------------------------------------------------------------------------
# Data format helpers
# ---------------------------------------------------------------------------

def _bttf_to_bti(x: Tensor) -> Tensor:
    """(B, T, J, F) → (B, T, J*F)."""
    B, T, J, F = x.shape
    return x.reshape(B, T, J * F)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class SyntheticLstmVAEModule(pl.LightningModule):
    """VAEAC-style LstmVAE for synthetic motion data.

    Loss
    ----
    L = L1_recon + vel_w * vel_loss + std_w * tstd_loss
        + beta * kl_free_bits
        + gamma * KL(q_full ‖ r_masked)   [once epoch ≥ mask_warmup_epochs]
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        encoder_hidden_dim: int = 256,
        encoder_num_layers: int = 2,
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 4,
        latent_dim: int = 128,
        dropout: float = 0.1,
        n_mix: int = 5,
        beta: float = 0.5,
        kl_anneal_epochs: int = 10,
        free_nats: float = 0.1,
        vel_w: float = 2.0,
        std_w: float = 10.0,
        gamma: float = 1.0,
        mask_warmup_epochs: int = 20,
        mask_axis: str = "temporal",
        lr: float = 3e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = LstmVAE(
            input_dim=input_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_num_layers=encoder_num_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
            n_mix=n_mix,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _kl_free_bits(mu: Tensor, log_var: Tensor, free_nats: float) -> Tensor:
        kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        return torch.clamp(kl, min=free_nats).mean()

    def _beta(self) -> float:
        anneal = self.hparams.kl_anneal_epochs
        if anneal <= 0:
            return self.hparams.beta
        return self.hparams.beta * min(1.0, self.current_epoch / max(anneal, 1))

    @staticmethod
    def _vel_loss(recon: Tensor, x: Tensor) -> Tensor:
        return F.mse_loss(recon[:, 1:] - recon[:, :-1], x[:, 1:] - x[:, :-1])

    @staticmethod
    def _tstd_loss(recon: Tensor, x: Tensor) -> Tensor:
        return F.mse_loss(recon.std(dim=1), x.std(dim=1))

    def _coalition_mask(self, x: Tensor) -> Tensor | None:
        if self.model.masked_encoder is None:
            return None
        if self.current_epoch < self.hparams.mask_warmup_epochs:
            return None
        B, T, D = x.shape
        J = D // 3
        axis = self.hparams.mask_axis
        if axis == "both":
            use_temporal = torch.rand(1).item() < 0.5
        else:
            use_temporal = axis == "temporal"
        if use_temporal:
            return _sample_temporal_training_mask(B, T, x.device)
        return _sample_spatial_training_mask(B, J, x.device)

    def _step(self, batch: Tensor, stage: str) -> Tensor:
        x = batch  # (B, T, D)

        recon, mu, log_var = self.model(x)
        recon_loss = F.l1_loss(recon, x)
        vel_loss   = self._vel_loss(recon, x)
        std_loss   = self._tstd_loss(recon, x)
        kl_loss    = self._kl_free_bits(mu, log_var, self.hparams.free_nats)
        beta       = self._beta()

        loss = (recon_loss
                + self.hparams.vel_w * vel_loss
                + self.hparams.std_w * std_loss
                + beta * kl_loss)

        kl_mask_val = torch.zeros(1, device=x.device)
        cm = self._coalition_mask(x)
        if cm is not None:
            log_pi, mu_mix, logvar_mix = self.model.forward_masked(x, cm)
            kl_mask_val = kl_full_vs_mixture(
                mu.detach(), log_var.detach(), log_pi, mu_mix, logvar_mix,
            )
            loss = loss + self.hparams.gamma * kl_mask_val

        self.log(f"{stage}/recon", recon_loss, prog_bar=(stage == "train"), on_epoch=True, on_step=False)
        self.log(f"{stage}/kl",    kl_loss,    prog_bar=False,              on_epoch=True, on_step=False)
        self.log(f"{stage}/kl_m",  kl_mask_val,prog_bar=False,              on_epoch=True, on_step=False)
        self.log(f"{stage}/loss",  loss,        prog_bar=True,              on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# Module-level collate so it's picklable by DataLoader workers.
def _lstm_collate(batch):
    """(B, T, J, F) TensorDataset batch → (B, T, J*F) float tensor."""
    xs, _, _ = zip(*batch)
    x = torch.stack(xs)
    return x.reshape(x.shape[0], x.shape[1], -1).float()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_datasets(
    args: argparse.Namespace,
    actor_data_dir: str | None,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Return (train_loader, val_loader, input_dim, seq_len).

    If *actor_data_dir* is given, loads synthetic_benchmark.pkl from that dir
    to reuse the exact same distribution as the ActorSHAP training run.
    """
    from synthetic.gaussian_motion import GaussianMotionBenchmark
    from synthetic.diagnostic_motion import generate_diagnostic_gait

    J, F, T = args.J, args.F, args.T

    if actor_data_dir is not None:
        bench_path = os.path.join(actor_data_dir, "synthetic_benchmark.pkl")
        if not os.path.exists(bench_path):
            raise FileNotFoundError(
                f"No synthetic_benchmark.pkl in {actor_data_dir}.\n"
                "Run train_actor_shap_synthetic.py first."
            )
        with open(bench_path, "rb") as fh:
            bench = pickle.load(fh)
        J, F, T = bench.J, bench.F, bench.T
        print(f"[LstmVAE] Reusing benchmark from {actor_data_dir}  (J={J}, F={F}, T={T})")

    data_mode = args.data_mode
    n_train, n_val = args.n_train, args.n_val

    if data_mode == "synthetic_gaussian":
        if actor_data_dir is None:
            bench = GaussianMotionBenchmark(
                J=J, F=F, T=T, rho=args.rho, alpha=args.alpha,
            )
        ds_tr = bench.build_pytorch_dataset(n_train, seed=1)
        ds_va = bench.build_pytorch_dataset(n_val,   seed=2)
    elif data_mode == "synthetic_diagnostic":
        x_tr, y_tr = generate_diagnostic_gait(n_train, J=J, F=F, T=T,
                                               signal_scale=args.signal_scale, seed=1)
        x_va, y_va = generate_diagnostic_gait(n_val,   J=J, F=F, T=T,
                                               signal_scale=args.signal_scale, seed=2)
        # generate_diagnostic_gait returns (N, J, F, T) → convert to (N, T, J, F)
        x_tr_t = torch.tensor(x_tr).permute(0, 3, 1, 2).contiguous()
        x_va_t = torch.tensor(x_va).permute(0, 3, 1, 2).contiguous()
        pm_tr = torch.ones(n_train, T, dtype=torch.bool)
        pm_va = torch.ones(n_val,   T, dtype=torch.bool)
        ds_tr = TensorDataset(x_tr_t, torch.zeros(n_train, dtype=torch.long), pm_tr)
        ds_va = TensorDataset(x_va_t, torch.zeros(n_val,   dtype=torch.long), pm_va)
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

    input_dim = J * F
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_lstm_collate, num_workers=0)
    val_loader   = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                              collate_fn=_lstm_collate, num_workers=0)
    return train_loader, val_loader, input_dim, T


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LstmVAE on synthetic motion data for ActorSHAP comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ------------------------------------------------------------------
    # Data source
    # ------------------------------------------------------------------
    p.add_argument(
        "--actor_data_dir", type=str, default=None,
        help="Existing train_actor_shap_synthetic.py output dir. "
             "When set, reuses the benchmark and test data for a fair comparison. "
             "LstmVAE checkpoint is saved into this same directory.",
    )
    p.add_argument("--data_mode", default="synthetic_gaussian",
                   choices=("synthetic_gaussian", "synthetic_diagnostic"))
    # Gaussian-specific
    p.add_argument("--rho",   type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val",   type=int, default=500)
    p.add_argument("--n_test",  type=int, default=100)
    # Diagnostic-specific
    p.add_argument("--signal_scale", type=float, default=0.12)
    # Skeleton dimensions (ignored when --actor_data_dir is given)
    p.add_argument("--J", type=int, default=17)
    p.add_argument("--F", type=int, default=3)
    p.add_argument("--T", type=int, default=81)
    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    p.add_argument("--latent_dim",          type=int,   default=128)
    p.add_argument("--encoder_hidden_dim",  type=int,   default=256)
    p.add_argument("--encoder_num_layers",  type=int,   default=2)
    p.add_argument("--decoder_hidden_dim",  type=int,   default=256)
    p.add_argument("--decoder_num_layers",  type=int,   default=4)
    p.add_argument("--dropout",             type=float, default=0.1)
    p.add_argument("--n_mix",               type=int,   default=5,
                   help="Mixture components in MaskedLstmEncoder (must be > 0).")
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--batch_size",         type=int,   default=64)
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--beta",               type=float, default=0.5,
                   help="KL weight for full encoder.")
    p.add_argument("--kl_anneal_epochs",   type=int,   default=10)
    p.add_argument("--free_nats",          type=float, default=0.1)
    p.add_argument("--vel_w",              type=float, default=2.0)
    p.add_argument("--std_w",              type=float, default=10.0)
    p.add_argument("--gamma",              type=float, default=1.0,
                   help="KL(q‖r) weight for masked encoder regularisation.")
    p.add_argument("--mask_warmup_epochs", type=int,   default=20,
                   help="Epochs before masked encoder training starts.")
    p.add_argument("--mask_axis",          type=str,   default="temporal",
                   choices=("spatial", "temporal", "both"))
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    p.add_argument("--checkpoint_dir", type=str,
                   default="experiment_outs/lstm_vae_synthetic",
                   help="Root for new runs (ignored when --actor_data_dir is set).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--devices", type=str, default=None,
                   help="CUDA_VISIBLE_DEVICES value (e.g. '0').")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    if args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # Determine output directory.
    if args.actor_data_dir is not None:
        out_dir = args.actor_data_dir
        # Read data_mode from the actor run's config if not overridden.
        actor_cfg_path = os.path.join(args.actor_data_dir, "config.json")
        if os.path.exists(actor_cfg_path):
            with open(actor_cfg_path) as fh:
                actor_cfg = json.load(fh)
            # Use actor's data_mode unless explicitly overridden.
            if args.data_mode == "synthetic_gaussian":  # default value
                args.data_mode = actor_cfg.get("data_mode", args.data_mode)
    else:
        out_dir = os.path.join(args.checkpoint_dir, f"lstm_vae_synthetic_{args.data_mode}")
        os.makedirs(out_dir, exist_ok=True)

    print(f"[LstmVAE] Output directory: {out_dir}")

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_loader, val_loader, input_dim, seq_len = _build_datasets(
        args, actor_data_dir=args.actor_data_dir,
    )
    print(f"[LstmVAE] input_dim={input_dim}  seq_len={seq_len}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if args.n_mix <= 0:
        raise ValueError("--n_mix must be > 0; the masked encoder is required for SHAP.")

    module = SyntheticLstmVAEModule(
        input_dim=input_dim,
        seq_len=seq_len,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        n_mix=args.n_mix,
        beta=args.beta,
        kl_anneal_epochs=args.kl_anneal_epochs,
        free_nats=args.free_nats,
        vel_w=args.vel_w,
        std_w=args.std_w,
        gamma=args.gamma,
        mask_warmup_epochs=args.mask_warmup_epochs,
        mask_axis=args.mask_axis,
        lr=args.lr,
    )

    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"[LstmVAE] Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Use at most 1 GPU by default; masked encoder params go unused before warmup
    # ends which DDP flags as an error unless find_unused_parameters is set.
    use_gpus = min(n_gpus, 1)

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="lstm_vae_synthetic_best",
        monitor="val/recon",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpus > 0 else "cpu",
        devices=use_gpus if use_gpus > 0 else 1,
        callbacks=[ckpt_cb, lr_cb],
        enable_checkpointing=True,
        log_every_n_steps=max(1, len(train_loader) // 2),
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader)

    # ------------------------------------------------------------------
    # Save final checkpoint + raw state dict + config
    # ------------------------------------------------------------------
    final_ckpt = os.path.join(out_dir, "lstm_vae_synthetic_last.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"[LstmVAE] Saved Lightning checkpoint: {final_ckpt}")

    model_pt = os.path.join(out_dir, "lstm_vae_model.pt")
    torch.save(module.model.state_dict(), model_pt)
    print(f"[LstmVAE] Saved raw state dict: {model_pt}")

    cfg = {
        "input_dim": input_dim,
        "seq_len": seq_len,
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "encoder_num_layers": args.encoder_num_layers,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "decoder_num_layers": args.decoder_num_layers,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "n_mix": args.n_mix,
        "data_mode": args.data_mode,
        "J": input_dim // 3,
        "F": 3,
        "T": seq_len,
    }
    with open(os.path.join(out_dir, "lstm_vae_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[LstmVAE] Saved config: {os.path.join(out_dir, 'lstm_vae_config.json')}")
    print("\n[LstmVAE] Training complete.")


if __name__ == "__main__":
    main()
