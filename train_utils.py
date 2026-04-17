"""train_utils.py — Shared training utilities for CARE-PD Lightning trainers."""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


class EpochProgressPrinter(Callback):
    """Prints a plain-text epoch summary line that survives `tee` and log files."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        parts = [f"epoch {trainer.current_epoch:>4d}/{trainer.max_epochs}"]
        for key in ("train/loss", "val/loss", "val/mixed", "train/mixed"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        # Fall back to any logged val_ or train_ scalars not already captured.
        for k, v in metrics.items():
            label = k.replace("train/", "").replace("val/", "")
            if not any(label in p for p in parts):
                try:
                    parts.append(f"{k}={float(v):.4f}")
                except (TypeError, ValueError):
                    pass
        print("  ".join(parts), flush=True)


def make_trainer(
    n_epochs: int,
    n_gpus: int,
    ckpt_dir: str,
    logger,
    monitor_key: str,
    phase_tag: str,
    extra_callbacks: list | None = None,
    find_unused_parameters: bool = False,
    save_last_only: bool = False,
    gradient_clip_val: float = 1.0,
) -> pl.Trainer:
    """Build a pl.Trainer with DDP (if >1 GPU), checkpointing, and logging.

    Args:
        n_epochs:               Total training epochs.
        n_gpus:                 Number of GPUs (0 = CPU).
        ckpt_dir:               Directory where checkpoints are written.
        logger:                 Lightning logger instance, or False for none.
        monitor_key:            Metric to monitor for best-checkpoint saving.
        phase_tag:              Prefix for checkpoint filenames.
        extra_callbacks:        Additional Lightning callbacks to attach.
        find_unused_parameters: Passed to DDPStrategy (set True when the
                                model has parameters not used in every forward,
                                e.g. VaeacMotion in infer phase).
        save_last_only:         When True, skip the best-val checkpoint and
                                instead save only the final epoch as
                                ``{phase_tag}_last.ckpt``.  Use for models
                                that are always loaded from their last epoch
                                rather than selected by validation loss
                                (e.g. VaeacMotion, ActorSHAP).
    """
    strategy = (
        DDPStrategy(find_unused_parameters=find_unused_parameters)
        if n_gpus > 1
        else "auto"
    )

    if save_last_only:
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"{phase_tag}_last",
                save_last=False,
                save_top_k=1,
                every_n_epochs=n_epochs,
            ),
        ]
    else:
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=monitor_key,
                save_top_k=1,
                filename=f"{phase_tag}_best",
                mode="min",
            ),
        ]

    callbacks.append(EpochProgressPrinter())
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
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
