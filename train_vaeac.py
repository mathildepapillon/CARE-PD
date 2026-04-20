"""train_vaeac.py — PyTorch-Lightning training for the transformer-backbone
VAEAC baseline on CARE-PD skeletal sequences.

Pipeline
--------
1. Read a JSON config (see ``configs/vaeac/*.json``).
2. Load ``cache.npz`` + ``sanity.npz`` produced by
   ``scripts/generate_velocity.py`` / ``scripts/build_synthetic_gaussian_data.py``
   — the **exact same cache** the flow-matching model is trained on.
3. For each training batch, sample a random coalition mask (mixture of
   temporal / spatial / element-wise) and minimise the ELBO.
4. Keep an EMA copy of the weights for validation / deployment.
5. Checkpoint best-val-ELBO and last epoch.

Usage
-----
::

    python train_vaeac.py --config configs/vaeac/synthetic_gaussian.json

Smoke test
----------
::

    python train_vaeac.py --config configs/vaeac/synthetic_gaussian.json --smoke
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
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from model.vaeac import VAEAC
from model.vaeac.vaeac import (
    sample_training_mask,
    sample_window_mask_uniform,
    sample_joint_mask_uniform,
    build_window_to_frame,
)
from train_utils import make_trainer


PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VAEACClipDataset(Dataset):
    """Same shape contract as ``FlowClipDataset`` — reads ``cache.npz``."""

    def __init__(self, x1: np.ndarray, mask: np.ndarray):
        assert x1.ndim == 4 and x1.shape[-1] == 3, f"expected (N, T, 17, 3), got {x1.shape}"
        assert mask.shape == x1.shape[:2], f"mask shape {mask.shape} vs x1 {x1.shape}"
        self.x1 = x1.astype(np.float32, copy=False)
        self.mask = mask.astype(bool, copy=False)

    def __len__(self) -> int:
        return self.x1.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x1":   torch.from_numpy(self.x1[idx]),              # (T, 17, 3)
            "mask": torch.from_numpy(self.mask[idx]),            # (T,)
        }


class VAEACClipDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.cache_path = PROJECT_ROOT / cfg["cache_dir"] / "cache.npz"

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"{self.cache_path} not found. Build the cache first "
                f"(same cache as the flow-matching model)."
            )
        d = np.load(self.cache_path, allow_pickle=True)
        self.train_dataset = VAEACClipDataset(d["x1_train"], d["mask_train"])
        self.val_dataset   = VAEACClipDataset(d["x1_val"],   d["mask_val"])
        print(f"[Data] train clips: {len(self.train_dataset)}, val clips: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg["batch_size"]),
            shuffle=True,
            num_workers=int(self.cfg.get("num_workers", 2)),
            pin_memory=True,
            drop_last=len(self.train_dataset) >= int(self.cfg["batch_size"]),
            persistent_workers=int(self.cfg.get("num_workers", 2)) > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg["batch_size"]),
            shuffle=False,
            num_workers=int(self.cfg.get("num_workers", 2)),
            pin_memory=True,
            persistent_workers=int(self.cfg.get("num_workers", 2)) > 0,
        )


# ---------------------------------------------------------------------------
# EMA (copied from train_flow_matching.py — keep in sync)
# ---------------------------------------------------------------------------

class EMA:
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
# Lightning module
# ---------------------------------------------------------------------------

class VAEACLit(pl.LightningModule):
    """Lightning wrapper: ELBO, KL annealing, free-bits, EMA."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters({k: v for k, v in cfg.items() if not isinstance(v, dict)})
        self.cfg = cfg

        self.model = VAEAC(
            n_joints=17, n_coords=3,
            d_model=int(cfg["d_model"]),
            nhead=int(cfg["nhead"]),
            num_layers=int(cfg["num_layers"]),
            ff_dim=int(cfg["ff_dim"]),
            dropout=float(cfg["dropout"]),
            d_latent=int(cfg["d_latent"]),
            max_len=max(int(cfg["seq_len"]) + 16, 256),
            decoder_head=str(cfg.get("decoder_head", "gaussian_scalar")),
            ivanov_min_sigma=float(cfg.get("ivanov_min_sigma", 1e-2)),
            use_prior_memory=bool(cfg.get("use_prior_memory", False)),
            prior_reg_sigma_mu=float(cfg.get("prior_reg_sigma_mu", 1e4)),
            prior_reg_sigma_sigma=float(cfg.get("prior_reg_sigma_sigma", 1e-4)),
        )
        self.ema: EMA | None = None

        self._mask_p_temporal = float(cfg.get("mask_p_temporal", 0.40))
        self._mask_p_spatial  = float(cfg.get("mask_p_spatial",  0.40))
        self._mask_p_element  = float(cfg.get("mask_p_element",  0.20))

        # --------------------------------------------------------------
        # Mask distribution — either the legacy mixture ("mixture") or
        # one matched to the EC coalition support ("benchmark").  In the
        # latter case we need the benchmark's K + window_assignments
        # (temporal) or the player_mode (spatial); these come from the
        # pickled ``GaussianMotionBenchmark`` pointed at by ``bench_path``.
        # --------------------------------------------------------------
        self._mask_mode = str(cfg.get("mask_mode", "mixture"))
        self._bench_player_mode: Optional[str] = None
        self._bench_K:           Optional[int] = None
        self._bench_w2f:         Optional[torch.Tensor] = None
        if self._mask_mode == "benchmark":
            bench_path = cfg.get("bench_path")
            if not bench_path:
                raise ValueError(
                    "cfg['mask_mode'] == 'benchmark' requires cfg['bench_path'] "
                    "pointing at a synthetic_benchmark.pkl."
                )
            from synthetic.gaussian_motion import GaussianMotionBenchmark
            bench = GaussianMotionBenchmark.load(str(bench_path))
            self._bench_player_mode = str(bench.player_mode)
            self._bench_K           = int(bench.K) if self._bench_player_mode == "temporal" else None
            if self._bench_player_mode == "temporal":
                w2f_cpu = build_window_to_frame(
                    K=int(bench.K), T=int(bench.T),
                    window_assignments=bench.window_assignments,
                    device=torch.device("cpu"),
                )
                # Registered as a buffer so Lightning moves it with .to(device).
                self.register_buffer("_window_to_frame", w2f_cpu, persistent=False)
                self._bench_w2f = self._window_to_frame
            print(
                f"[VAEAC] mask_mode=benchmark  player_mode={self._bench_player_mode}  "
                f"K={self._bench_K}  bench_path={bench_path}"
            )
        else:
            print(
                f"[VAEAC] mask_mode=mixture  "
                f"p_temporal={self._mask_p_temporal}  p_spatial={self._mask_p_spatial}  "
                f"p_element={self._mask_p_element}"
            )

        self._kl_anneal_start = int(cfg.get("kl_anneal_start_epoch", 0))
        self._kl_anneal_end   = int(cfg.get("kl_anneal_end_epoch",   20))
        self._kl_weight_final = float(cfg.get("kl_weight_final",     1.0))

        self._free_bits       = float(cfg.get("free_bits",           0.0))

        bd = self.model.count_parameters_breakdown()
        print(
            f"[VAEAC] head={self.model.decoder_head_name}  "
            f"use_prior_memory={self.model.use_prior_memory}"
        )
        print(
            f"[VAEAC] params — full_enc={bd['full_encoder']:,} "
            f"prior_enc={bd['prior_encoder']:,} dec={bd['decoder']:,} "
            f"head={bd.get('output_head', 0):,} total={bd['total']:,}"
        )

    # ------------------------------------------------------------------
    # KL schedule / free-bits
    # ------------------------------------------------------------------

    def _kl_weight(self) -> float:
        e = int(self.current_epoch)
        if e <= self._kl_anneal_start:
            return 0.0
        if e >= self._kl_anneal_end:
            return self._kl_weight_final
        frac = (e - self._kl_anneal_start) / max(1, self._kl_anneal_end - self._kl_anneal_start)
        return self._kl_weight_final * frac

    def _apply_free_bits(self, kl: torch.Tensor) -> torch.Tensor:
        """Per-latent-dim free-bits (Kingma et al. 2016).

        Our :meth:`VAEAC.elbo` averages the KL over (frames × latent-dims)
        already.  Free-bits here simply clamps the mean below ``free_bits``
        nats per dim; effective as a guard against posterior collapse.
        """
        if self._free_bits <= 0.0:
            return kl
        return kl.clamp(min=self._free_bits)

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def _sample_mask(
        self,
        B: int, T: int, J: int, C: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Dispatch mask sampling according to ``self._mask_mode``."""
        if self._mask_mode == "benchmark":
            if self._bench_player_mode == "temporal":
                return sample_window_mask_uniform(
                    B=B, T=T, J=J, C=C,
                    window_to_frame=self._window_to_frame,
                    device=device, generator=generator,
                )
            if self._bench_player_mode == "spatial":
                return sample_joint_mask_uniform(
                    B=B, T=T, J=J, C=C,
                    device=device, generator=generator,
                )
            raise RuntimeError(
                f"Unknown bench player_mode: {self._bench_player_mode!r}"
            )
        return sample_training_mask(
            B=B, T=T, J=J, C=C, device=device,
            p_temporal=self._mask_p_temporal,
            p_spatial=self._mask_p_spatial,
            p_element=self._mask_p_element,
            generator=generator,
        )

    def _step_loss(
        self,
        x1: torch.Tensor,
        pad: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, J, C = x1.shape
        obs = self._sample_mask(B, T, J, C, x1.device, generator=generator)
        kl_w = self._kl_weight()
        loss_uncomposed, parts = self.model.elbo(x1, obs, pad_mask=pad, kl_weight=1.0)
        kl_fb = self._apply_free_bits(parts["kl"])
        # prior_reg is small and weight-independent; always include.
        loss = parts["recon_nll"] + kl_w * kl_fb + parts["prior_reg"]
        return loss, {**parts, "kl_weight": torch.tensor(kl_w), "kl_fb": kl_fb.detach()}

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x1, pad = batch["x1"], batch["mask"]
        loss, logs = self._step_loss(x1, pad)
        self.log("train/loss",      loss.detach(),                prog_bar=True,  sync_dist=True)
        self.log("train/recon_nll", logs["recon_nll"].detach(),   prog_bar=True,  sync_dist=True)
        self.log("train/kl",        logs["kl"].detach(),          prog_bar=True,  sync_dist=True)
        self.log("train/kl_weight", logs["kl_weight"],            prog_bar=False, sync_dist=True)
        if "mean_sigma" in logs:
            self.log("train/mean_sigma", logs["mean_sigma"].detach(), prog_bar=False, sync_dist=True)
        if "prior_reg" in logs:
            self.log("train/prior_reg", logs["prior_reg"].detach(), prog_bar=False, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update(self.model)

    def on_fit_start(self) -> None:
        if self.ema is None:
            self.ema = EMA(self.model, decay=float(self.cfg.get("ema_decay", 0.999)))

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x1, pad = batch["x1"], batch["mask"]
        # Deterministic val: same mask per epoch per batch.
        g = torch.Generator(device=x1.device).manual_seed(
            int(self.current_epoch) * 1000 + batch_idx
        )
        B, T, J, C = x1.shape
        obs = self._sample_mask(B, T, J, C, x1.device, generator=g)
        # Live weights
        _, parts = self.model.elbo(x1, obs, pad_mask=pad, kl_weight=1.0)
        kl_w = self._kl_weight()
        with torch.no_grad():
            live_loss = (
                parts["recon_nll_detached"]
                + kl_w * parts["kl_detached"]
                + parts["prior_reg_detached"]
            )
        self.log("val/loss",      live_loss,                   prog_bar=True,  sync_dist=True)
        self.log("val/recon_nll", parts["recon_nll_detached"], prog_bar=False, sync_dist=True)
        self.log("val/kl",        parts["kl_detached"],        prog_bar=False, sync_dist=True)

        # EMA weights
        if self.ema is not None:
            original = self.ema.copy_into(self.model)
            self.model.eval()
            _, ema_parts = self.model.elbo(x1, obs, pad_mask=pad, kl_weight=1.0)
            with torch.no_grad():
                ema_loss = (
                    ema_parts["recon_nll_detached"]
                    + kl_w * ema_parts["kl_detached"]
                    + ema_parts["prior_reg_detached"]
                )
            self.ema.restore(self.model, original)
            self.log("val/loss_ema",      ema_loss,                       prog_bar=False, sync_dist=True)
            self.log("val/recon_nll_ema", ema_parts["recon_nll_detached"], prog_bar=False, sync_dist=True)
            self.log("val/kl_ema",        ema_parts["kl_detached"],        prog_bar=False, sync_dist=True)

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
# Smoke
# ---------------------------------------------------------------------------

def run_smoke(cfg: dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    model = VAEAC(
        n_joints=17, n_coords=3,
        d_model=int(cfg["d_model"]), nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]), ff_dim=int(cfg["ff_dim"]),
        dropout=0.0, d_latent=int(cfg["d_latent"]),
        max_len=int(cfg["seq_len"]) + 16,
        decoder_head=str(cfg.get("decoder_head", "gaussian_scalar")),
        ivanov_min_sigma=float(cfg.get("ivanov_min_sigma", 1e-2)),
        use_prior_memory=bool(cfg.get("use_prior_memory", False)),
        prior_reg_sigma_mu=float(cfg.get("prior_reg_sigma_mu", 1e4)),
        prior_reg_sigma_sigma=float(cfg.get("prior_reg_sigma_sigma", 1e-4)),
    ).to(device)
    bd = model.count_parameters_breakdown()
    print(
        f"[smoke] head={model.decoder_head_name} "
        f"use_prior_memory={model.use_prior_memory}"
    )
    print(
        f"[smoke] params — full_enc={bd['full_encoder']:,} "
        f"prior_enc={bd['prior_encoder']:,} dec={bd['decoder']:,} "
        f"head={bd.get('output_head', 0):,} total={bd['total']:,}"
    )

    # Forward shape check
    B, T = 4, int(cfg["seq_len"])
    x = torch.randn(B, T, 17, 3, device=device)
    pad = torch.ones(B, T, dtype=torch.bool, device=device)
    obs = sample_training_mask(B=B, T=T, J=17, C=3, device=device)
    loss, parts = model.elbo(x, obs, pad_mask=pad, kl_weight=1.0)
    print(
        f"[smoke] forward OK — loss={loss.item():.4f} "
        f"recon={parts['recon_nll_detached'].item():.4f} "
        f"kl={parts['kl_detached'].item():.4f}"
    )

    # Overfit 8 clips
    cache = np.load(PROJECT_ROOT / cfg["cache_dir"] / "cache.npz", allow_pickle=True)
    x1_np   = cache["x1_train"][:8]
    mask_np = cache["mask_train"][:8]
    x1   = torch.from_numpy(x1_np).to(device)
    pad  = torch.from_numpy(mask_np).to(device).bool()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    steps = 200
    for step in range(steps):
        obs = sample_training_mask(
            B=x1.shape[0], T=x1.shape[1], J=x1.shape[2], C=x1.shape[3],
            device=device,
            p_temporal=float(cfg.get("mask_p_temporal", 0.40)),
            p_spatial=float(cfg.get("mask_p_spatial",  0.40)),
            p_element=float(cfg.get("mask_p_element",  0.20)),
        )
        loss, parts = model.elbo(x1, obs, pad_mask=pad, kl_weight=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 40 == 0:
            print(
                f"[smoke] step {step:3d}  loss={loss.item():.4f} "
                f"recon={parts['recon_nll_detached'].item():.4f} "
                f"kl={parts['kl_detached'].item():.4f}"
            )
    initial = float(np.mean(losses[:5]))
    final   = float(np.mean(losses[-10:]))
    print(f"[smoke] overfit: initial={initial:.4f}  final={final:.4f}  drop={initial - final:.4f}")
    assert final < initial - 0.1, "overfit check failed: ELBO did not decrease."

    # Inference shape
    model.eval()
    with torch.no_grad():
        comp = model.sample_completions(x1, obs, pad_mask=pad, n_samples=3, temperature=1.0)
    assert comp.shape == (x1.shape[0] * 3, *x1.shape[1:]), f"bad comp shape {comp.shape}"
    # Observed entries preserved?
    rep = x1.repeat_interleave(3, dim=0)
    obs_rep = obs.repeat_interleave(3, dim=0)
    diff = ((comp - rep)[obs_rep]).abs().max().item()
    print(f"[smoke] completion shape OK {tuple(comp.shape)}; obs-preservation maxabs={diff:.2e}")
    assert diff < 1e-5, "Observed entries not preserved bit-exact!"
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
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--ckpt_dir_override", type=str, default=None,
                   help="If set, overrides cfg['checkpoint_dir'].")
    p.add_argument("--bench_path", type=str, default=None,
                   help="Path to synthetic_benchmark.pkl (required when "
                        "mask_mode=='benchmark'; overrides cfg).")
    p.add_argument("--mask_mode", type=str, default=None,
                   choices=["mixture", "benchmark"],
                   help="Override cfg['mask_mode'].")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    for k in ["fold", "devices", "max_epochs", "wandb_project", "wandb_run_name",
              "seed", "bench_path", "mask_mode"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.ckpt_dir_override is not None:
        cfg["checkpoint_dir"] = args.ckpt_dir_override

    _seed_everything(int(cfg.get("seed", 42)))
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    if args.smoke:
        run_smoke(cfg)
        return

    logger = False
    if not args.no_wandb and cfg.get("wandb_project"):
        logger = WandbLogger(
            project=str(cfg["wandb_project"]),
            name=str(cfg.get("wandb_run_name", f"vaeac_{cfg['dataset']}_fold{cfg['fold']}")),
            save_dir=str(PROJECT_ROOT / "wandb"),
            config=cfg,
        )

    dm = VAEACClipDataModule(cfg)
    model = VAEACLit(cfg)

    ckpt_dir = PROJECT_ROOT / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_gpus = int(cfg.get("devices", 1)) if str(cfg.get("accelerator", "gpu")) == "gpu" else 0
    trainer = _build_trainer(cfg, n_gpus=n_gpus, ckpt_dir=ckpt_dir, logger=logger)
    trainer.fit(model, datamodule=dm)


def _build_trainer(cfg: dict[str, Any], *, n_gpus: int, ckpt_dir: Path, logger) -> pl.Trainer:
    base = make_trainer(
        n_epochs=int(cfg["max_epochs"]),
        n_gpus=n_gpus,
        ckpt_dir=str(ckpt_dir),
        logger=logger,
        monitor_key="val/loss_ema",
        phase_tag=f"vaeac_{cfg['dataset']}_fold{cfg['fold']}",
        gradient_clip_val=float(cfg.get("grad_clip", 1.0)),
    )
    last_ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"vaeac_{cfg['dataset']}_fold{cfg['fold']}_last",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )
    callbacks = list(base.callbacks) + [last_ckpt]
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
