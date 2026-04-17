"""
train_flow_matching.py — PyTorch Lightning training for the flow-matching
velocity predictor on CARE-PD skeletal sequences.

Pipeline
--------
1.  Reads a JSON config (see ``configs/flow_matching/*.json``).
2.  Loads ``cache.npz`` + ``sanity.npz`` produced by
    ``scripts/generate_velocity.py``.
3.  Samples ``t ~ U(0, 1)`` and ``x_0 ~ N(0, I)`` per training step and uses
    ``flow_matching.path.AffineProbPath(CondOTScheduler())`` to obtain
    ``(x_t, u_t)``. The network predicts ``v_theta(x_t, t)`` and is trained
    with masked MSE against ``u_t``.
4.  Keeps an EMA copy of the weights for validation / sampling.
5.  Every ``val_sample_every_n_epochs`` epochs, integrates a few trajectories
    from noise to ``t=1`` with ``ODESolver`` and logs them to wandb.
6.  Uses ``make_trainer`` from ``train_utils.py`` for DDP / checkpointing.

Run
---
::

    python scripts/generate_velocity.py --dataset BMCLab --fold 1
    python train_flow_matching.py \
        --config configs/flow_matching/bmclab_h36m3d_fold1.json

Smoke mode (shape test + 8-clip overfit + sanity-tuple loss)::

    python train_flow_matching.py \
        --config configs/flow_matching/bmclab_h36m3d_fold1.json --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from model.flow_matching import VelocityNet, H36M_17J_MIRROR_PERM
from train_utils import make_trainer


PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dataset + DataModule
# ---------------------------------------------------------------------------

class FlowClipDataset(Dataset):
    """Wraps the ``x1_train`` / ``x1_val`` arrays from ``cache.npz``.

    If ``mirror_augment=True`` (train only), each item is mirrored with
    probability 0.5 using the H36M 17-joint left/right permutation and
    negating the lateral axis (default X, index 0).
    """

    def __init__(
        self,
        x1: np.ndarray,
        mask: np.ndarray,
        mirror_augment: bool = False,
        mirror_axis: int = 0,
    ):
        assert x1.ndim == 4 and x1.shape[-1] == 3, f"expected (N, T, 17, 3), got {x1.shape}"
        assert mask.shape == x1.shape[:2], f"mask shape {mask.shape} vs x1 {x1.shape}"
        self.x1 = x1.astype(np.float32, copy=False)
        self.mask = mask.astype(bool, copy=False)
        self.mirror_augment = mirror_augment
        self.mirror_axis = mirror_axis
        self.mirror_perm = np.array(H36M_17J_MIRROR_PERM, dtype=np.int64)

    def __len__(self) -> int:
        return self.x1.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.x1[idx]          # (T, 17, 3)
        m = self.mask[idx]        # (T,)
        if self.mirror_augment and random.random() < 0.5:
            x = x[:, self.mirror_perm, :].copy()
            x[..., self.mirror_axis] *= -1.0
        return {
            "x1": torch.from_numpy(x),
            "mask": torch.from_numpy(m),
        }


class FlowClipDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.cache_path = PROJECT_ROOT / cfg["cache_dir"] / "cache.npz"

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"{self.cache_path} not found. "
                f"Run `python scripts/generate_velocity.py --dataset {self.cfg['dataset']} "
                f"--fold {self.cfg['fold']} ...` first."
            )
        d = np.load(self.cache_path, allow_pickle=True)
        self.train_dataset = FlowClipDataset(
            x1=d["x1_train"], mask=d["mask_train"],
            mirror_augment=bool(self.cfg.get("mirror_augment", True)),
            mirror_axis=int(self.cfg.get("mirror_axis", 0)),
        )
        self.val_dataset = FlowClipDataset(
            x1=d["x1_val"], mask=d["mask_val"],
            mirror_augment=False,  # val never mirrors
        )
        print(f"[Data] train clips: {len(self.train_dataset)}, val clips: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg["batch_size"]),
            shuffle=True,
            num_workers=int(self.cfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=len(self.train_dataset) >= int(self.cfg["batch_size"]),
            persistent_workers=int(self.cfg.get("num_workers", 4)) > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg["batch_size"]),
            shuffle=False,
            num_workers=int(self.cfg.get("num_workers", 4)),
            pin_memory=True,
            persistent_workers=int(self.cfg.get("num_workers", 4)) > 0,
        )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Minimal exponential moving average of parameter values.

    Updates on every optimizer step; maintains a separate ``shadow`` state
    dict. ``apply_to`` swaps in the EMA weights into ``target_model`` and
    returns a handle to restore the original weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone() for n, p in model.state_dict().items()
            if p.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for n, p in model.state_dict().items():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_into(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA params into ``model``. Returns the original params for restore()."""
        sd = model.state_dict()
        original = {n: sd[n].detach().clone() for n in self.shadow.keys()}
        for n, p in self.shadow.items():
            sd[n].copy_(p)
        return original

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, original: dict[str, torch.Tensor]) -> None:
        sd = model.state_dict()
        for n, p in original.items():
            sd[n].copy_(p)


# ---------------------------------------------------------------------------
# ModelWrapper for ODE sampling
# ---------------------------------------------------------------------------

class VelocityModelWrapper(ModelWrapper):
    """Wraps VelocityNet so ODESolver can call it with scalar t.

    ``ODESolver.sample`` calls ``model(x, t, **extras)`` where ``x`` is the
    batch tensor and ``t`` may be a 0-d scalar tensor; our VelocityNet needs
    ``t`` of shape ``(B,)``, so we broadcast.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim > 1:
            t = t.reshape(-1)
        return self.model(x, t)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over real frames only. ``mask`` is ``(B, T)`` bool."""
    m = mask.unsqueeze(-1).unsqueeze(-1).to(pred.dtype)  # (B, T, 1, 1)
    sq = (pred - target) ** 2
    sq_masked = sq * m
    denom = m.sum().clamp(min=1.0) * pred.shape[-1] * pred.shape[-2]
    return sq_masked.sum() / denom


class FlowMatchingLit(pl.LightningModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters({k: v for k, v in cfg.items() if not isinstance(v, dict)})
        self.cfg = cfg

        self.model = VelocityNet(
            n_joints=17, n_coords=3,
            d_model=int(cfg["d_model"]),
            nhead=int(cfg["nhead"]),
            num_layers=int(cfg["num_layers"]),
            ff_dim=int(cfg["ff_dim"]),
            dropout=float(cfg["dropout"]),
            time_emb_dim=int(cfg["time_emb_dim"]),
            max_len=max(int(cfg["seq_len"]) + 16, 256),
        )
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.ema: EMA | None = None  # created lazily in on_fit_start
        self._val_sample_cached: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def _step_loss(self, x1: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1)
        sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        v_pred = self.model(sample.x_t, sample.t, mask=mask)
        loss = masked_mse(v_pred, sample.dx_t, mask)
        logs = {"loss": loss.detach()}
        return loss, logs

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x1, mask = batch["x1"], batch["mask"]
        loss, logs = self._step_loss(x1, mask)
        self.log("train/loss", logs["loss"], prog_bar=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Maintain EMA on each optimizer step (after weights change).
        pass  # moved to on_train_batch_end to run after the optimizer step

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update(self.model)

    def on_fit_start(self) -> None:
        if self.ema is None:
            self.ema = EMA(self.model, decay=float(self.cfg.get("ema_decay", 0.999)))

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x1, mask = batch["x1"], batch["mask"]
        # Deterministic val loss: same t per epoch per batch for stable curves.
        g = torch.Generator(device=x1.device).manual_seed(
            int(self.current_epoch) * 1000 + batch_idx
        )
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device, generator=g, dtype=x1.dtype)
        x0 = torch.randn(x1.shape, device=x1.device, generator=g, dtype=x1.dtype)
        sample = self.path.sample(t=t, x_0=x0, x_1=x1)

        # Live-weight loss
        v_live = self.model(sample.x_t, sample.t, mask=mask)
        live_loss = masked_mse(v_live, sample.dx_t, mask)
        self.log("val/loss", live_loss, prog_bar=True, sync_dist=True)

        # EMA loss (more meaningful for downstream ODE sampling)
        if self.ema is not None:
            original = self.ema.copy_into(self.model)
            self.model.eval()
            v_ema = self.model(sample.x_t, sample.t, mask=mask)
            ema_loss = masked_mse(v_ema, sample.dx_t, mask)
            self.ema.restore(self.model, original)
            self.log("val/loss_ema", ema_loss, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        every = int(self.cfg.get("val_sample_every_n_epochs", 10))
        if every <= 0 or (self.current_epoch + 1) % every != 0:
            return
        # Only rank-0 does the ODE sampling / wandb logging.
        if self.trainer.is_global_zero:
            self._log_ode_samples()

    # ------------------------------------------------------------------
    # ODE sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_ode_samples(self) -> None:
        if self.ema is None:
            return
        device = next(self.model.parameters()).device
        n = int(self.cfg.get("val_sample_n", 4))
        T = int(self.cfg["seq_len"])
        steps = int(self.cfg.get("val_sample_steps", 50))
        x_init = torch.randn(n, T, 17, 3, device=device)

        # Use EMA weights for the sample.
        original = self.ema.copy_into(self.model)
        self.model.eval()
        wrapper = VelocityModelWrapper(self.model)
        solver = ODESolver(velocity_model=wrapper)
        time_grid = torch.linspace(0.0, 1.0, steps, device=device)
        x_end = solver.sample(
            x_init=x_init,
            step_size=1.0 / max(steps - 1, 1),
            method="midpoint",
            time_grid=time_grid,
        )
        self.ema.restore(self.model, original)

        # Log summary statistics rather than full tensors for wandb simplicity.
        x_end = x_end.detach()
        stats = {
            "val_sample/x_end_mean": x_end.mean().item(),
            "val_sample/x_end_std": x_end.std().item(),
            "val_sample/x_end_absmax": x_end.abs().max().item(),
            "val_sample/per_frame_vel_norm_mean": (
                (x_end[:, 1:] - x_end[:, :-1]).pow(2).sum(dim=(-1, -2)).sqrt().mean().item()
            ),
        }
        for k, v in stats.items():
            self.log(k, v, sync_dist=False, rank_zero_only=True)

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg["lr"]),
            weight_decay=float(self.cfg.get("weight_decay", 0.0)),
        )
        warmup_steps = int(self.cfg.get("warmup_steps", 1000))
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(warmup_steps, 1))
            # Cosine decay from 1.0 to 0.1 over the remaining steps.
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cos

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke(cfg: dict[str, Any]) -> None:
    """Light sanity run: forward shape, 8-clip overfit, sanity-tuple loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    # 1. Forward shape
    net = VelocityNet(
        n_joints=17, n_coords=3,
        d_model=int(cfg["d_model"]), nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]), ff_dim=int(cfg["ff_dim"]),
        dropout=0.0, time_emb_dim=int(cfg["time_emb_dim"]),
        max_len=int(cfg["seq_len"]) + 16,
    ).to(device)
    B, T = 2, int(cfg["seq_len"])
    x = torch.randn(B, T, 17, 3, device=device)
    t = torch.rand(B, device=device)
    y = net(x, t)
    assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
    print(f"[smoke] forward shape OK: {tuple(y.shape)}  params={net.count_parameters():,}")

    # 2. Overfit 8 clips
    cache = np.load(PROJECT_ROOT / cfg["cache_dir"] / "cache.npz", allow_pickle=True)
    x1_np = cache["x1_train"][:8]
    mask_np = cache["mask_train"][:8]
    x1 = torch.from_numpy(x1_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = []
    steps = 200
    for step in range(steps):
        tt = torch.rand(x1.shape[0], device=device)
        x0 = torch.randn_like(x1)
        sample = path.sample(t=tt, x_0=x0, x_1=x1)
        v = net(sample.x_t, sample.t, mask=mask)
        loss = masked_mse(v, sample.dx_t, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 40 == 0:
            print(f"[smoke] step {step:3d}  loss={loss.item():.4f}")
    initial = float(np.mean(losses[:5]))
    final = float(np.mean(losses[-10:]))
    ratio = initial / max(final, 1e-8)
    print(f"[smoke] overfit loss: initial={initial:.4f}  final={final:.4f}  ratio={ratio:.1f}x")
    assert ratio >= 5.0, f"overfit check failed: loss did not drop enough (ratio={ratio:.2f})"

    # 3. Sanity-tuple loss
    sanity = np.load(PROJECT_ROOT / cfg["cache_dir"] / "sanity.npz")
    xt = torch.from_numpy(sanity["x_t"]).to(device)
    tv = torch.from_numpy(sanity["t"]).to(device)
    ut = torch.from_numpy(sanity["u_t"]).to(device)
    net.eval()
    with torch.no_grad():
        v = net(xt, tv)
        sanity_loss = F.mse_loss(v, ut).item()
    print(f"[smoke] sanity-tuple MSE against cached u_t: {sanity_loss:.4f}")
    print("[smoke] PASS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke test only (no Lightning, no wandb).")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging (useful for local debugging).")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    for k in ["fold", "devices", "max_epochs", "wandb_project", "wandb_run_name", "seed"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    _seed_everything(int(cfg.get("seed", 42)))
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    if args.smoke:
        run_smoke(cfg)
        return

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = False
    if not args.no_wandb and cfg.get("wandb_project"):
        logger = WandbLogger(
            project=str(cfg["wandb_project"]),
            name=str(cfg.get("wandb_run_name", f"{cfg['dataset']}_fold{cfg['fold']}")),
            save_dir=str(PROJECT_ROOT / "wandb"),
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Data + Model
    # ------------------------------------------------------------------
    dm = FlowClipDataModule(cfg)
    model = FlowMatchingLit(cfg)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    ckpt_dir = PROJECT_ROOT / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_gpus = int(cfg.get("devices", 1)) if str(cfg.get("accelerator", "gpu")) == "gpu" else 0
    trainer = _build_trainer(cfg, n_gpus=n_gpus, ckpt_dir=ckpt_dir, logger=logger)

    trainer.fit(model, datamodule=dm)


def _build_trainer(cfg: dict[str, Any], *, n_gpus: int, ckpt_dir: Path, logger) -> pl.Trainer:
    """Wrap ``train_utils.make_trainer`` so we can inject ``precision``.

    ``make_trainer`` doesn't expose a precision arg. We call it to obtain the
    correctly-configured callbacks (ModelCheckpoint + LearningRateMonitor)
    and DDP strategy, then build a fresh Trainer that also accepts precision.
    """
    base = make_trainer(
        n_epochs=int(cfg["max_epochs"]),
        n_gpus=n_gpus,
        ckpt_dir=str(ckpt_dir),
        logger=logger,
        monitor_key="val/loss_ema",
        phase_tag=f"flow_matching_{cfg['dataset']}_fold{cfg['fold']}",
        gradient_clip_val=float(cfg.get("grad_clip", 1.0)),
    )
    # make_trainer only saves the best-val checkpoint; also persist the
    # most recent weights so training can be resumed or the final epoch
    # inspected even if val/loss_ema never improved at the very end.
    last_ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"flow_matching_{cfg['dataset']}_fold{cfg['fold']}_last",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )
    callbacks = list(base.callbacks) + [last_ckpt]
    # Build a fresh strategy instance; reusing ``base.strategy`` collides
    # with the ``accelerator`` flag because it was already attached to
    # ``make_trainer``'s Trainer.
    strategy: Any = (
        DDPStrategy(find_unused_parameters=False) if n_gpus > 1 else "auto"
    )
    kwargs = dict(
        max_epochs=int(cfg["max_epochs"]),
        accelerator="gpu" if n_gpus > 0 else "cpu",
        devices=max(n_gpus, 1),
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=float(cfg.get("grad_clip", 1.0)),
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    precision = cfg.get("precision", None)
    if precision is not None and n_gpus > 0:
        kwargs["precision"] = str(precision)
    return pl.Trainer(**kwargs)


if __name__ == "__main__":
    main()
