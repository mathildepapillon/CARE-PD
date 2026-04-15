"""train_utils.py — Shared training utilities for CARE-PD Lightning trainers."""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


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
