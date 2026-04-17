"""
train_lstm_vae.py — Standalone training script for the LSTM VAE.

Usage
-----
python train_lstm_vae.py \\
    --dataset BMCLab \\
    --fold 1 \\
    --seq_len 80 \\
    --latent_dim 256 \\
    --encoder_hidden_dim 256 --encoder_num_layers 2 \\
    --decoder_hidden_dim 256 --decoder_num_layers 4 \\
    --dropout 0.1 \\
    --batch_size 64 \\
    --lr 3e-4 \\
    --max_epochs 150 \\
    --beta 0.5 --free_nats 0.1 --vel_w 2.0 --std_w 50.0

Data
----
Loads h36m 17-joint XYZ (T, 17, 3) directly from an NPZ file.
Fold-based train / eval split is resolved from the project fold pickles.
Participant ID is inferred from NPZ key prefix (e.g. "SUB01" from "SUB01__...").

No code from model/ or other train_ scripts is reused here except the model
itself (model.lstm_vae.LstmVAE).
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from model.lstm_vae import LstmVAE, kl_full_vs_mixture
from model.actor.shap_masking import (
    sample_spatial_training_mask,
    sample_temporal_training_mask,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_NPZ = {
    "BMCLab":   "assets/datasets/h36m/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
    "T-SDU-PD": "assets/datasets/h36m/T-SDU-PD/h36m_3d_world_floorXZZplus_30f_or_longer_slopeCorrected.npz",
    "PD-GaM":   "assets/datasets/h36m/PD-GaM/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
    "3DGait":   "assets/datasets/h36m/3DGait/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
    "H36M":     "assets/datasets/h36m/H36M/h36m_3d_world_floorXZZplus_30f_or_longer.npz",
}

FOLD_PICKLE = {
    ("BMCLab",   6):  "assets/datasets/folds/UPDRS_Datasets/BMCLab_6fold_participants.pkl",
    ("BMCLab",  23):  "assets/datasets/folds/UPDRS_Datasets/BMCLab_23fold_participants.pkl",
    ("T-SDU-PD", 14): "assets/datasets/folds/UPDRS_Datasets/T-SDU-PD_14fold_participants.pkl",
    ("3DGait",   6):  "assets/datasets/folds/UPDRS_Datasets/3DGait_6fold_participants.pkl",
    ("3DGait",  43):  "assets/datasets/folds/UPDRS_Datasets/3DGait_43fold_participants.pkl",
}

N_JOINTS = 17
INPUT_DIM = N_JOINTS * 3  # 51


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Minimal dataset that loads h36m XYZ clips from an NPZ file.

    Each NPZ key encodes the participant ID as the prefix before ``__``
    (e.g. ``"SUB01__SUB01_off_walk_1_down0"`` → participant ``"SUB01"``).

    Preprocessing applied in ``__init__``:
    1. Root-centre: subtract hip (joint 0) at every frame.
    2. Clip sequences into non-overlapping windows of length ``seq_len``;
       the last window is zero-padded if shorter than ``seq_len``.
    3. Flatten spatial dims: ``(T, 17, 3)`` → ``(T, 51)``.

    Parameters
    ----------
    npz_path:
        Path to the ``.npz`` file containing ``{seq_name: (T, 17, 3)}`` arrays.
    participant_ids:
        Set of participant IDs whose sequences are included in this split.
    seq_len:
        Fixed clip length in frames.
    """

    def __init__(
        self,
        npz_path: str | Path,
        participant_ids: set[str],
        seq_len: int = 80,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.clips: list[np.ndarray] = []

        npz = np.load(str(npz_path), allow_pickle=False)
        for key in npz.files:
            pid = key.split("__")[0]
            if pid not in participant_ids:
                continue
            seq = npz[key].astype(np.float32)  # (T_raw, 17, 3)
            seq = self._root_centre(seq)
            for clip in self._make_clips(seq, seq_len):
                self.clips.append(clip)  # each clip: (seq_len, 51)

    # ------------------------------------------------------------------

    @staticmethod
    def _root_centre(seq: np.ndarray) -> np.ndarray:
        """Subtract hip joint (index 0) at every frame."""
        return seq - seq[:, :1, :]  # broadcast over joints

    @staticmethod
    def _make_clips(seq: np.ndarray, clip_len: int) -> list[np.ndarray]:
        """Split a sequence into non-overlapping clips of length ``clip_len``."""
        T, J, C = seq.shape
        clips = []
        start = 0
        while start < T:
            end = start + clip_len
            chunk = seq[start:end]  # (≤clip_len, J, C)
            if chunk.shape[0] < clip_len:
                pad = np.zeros((clip_len - chunk.shape[0], J, C), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            clips.append(chunk.reshape(clip_len, J * C))
            start = end
        return clips

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.from_numpy(self.clips[idx])


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class PoseDataModule(L.LightningDataModule):
    """DataModule that loads a single dataset fold.

    Parameters
    ----------
    dataset:
        Dataset name, e.g. ``"BMCLab"``.
    fold:
        Fold number (1-based integer).
    num_folds:
        Total number of folds — used to look up the correct fold pickle.
    seq_len:
        Clip length in frames.
    batch_size:
        Batch size for both train and val loaders.
    num_workers:
        DataLoader worker count.
    """

    def __init__(
        self,
        dataset: str = "BMCLab",
        fold: int = 1,
        num_folds: int = 6,
        seq_len: int = 80,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.npz_path = PROJECT_ROOT / DATASET_NPZ[dataset]

        fold_key = (dataset, num_folds)
        if fold_key not in FOLD_PICKLE:
            raise ValueError(
                f"No fold pickle registered for dataset='{dataset}', "
                f"num_folds={num_folds}. Add it to FOLD_PICKLE."
            )
        fold_pkl_path = PROJECT_ROOT / FOLD_PICKLE[fold_key]
        fold_splits = pickle.load(open(fold_pkl_path, "rb"))
        if fold not in fold_splits:
            raise ValueError(
                f"Fold {fold} not found in {fold_pkl_path}. "
                f"Available folds: {sorted(fold_splits.keys())}"
            )
        self.train_pids: set[str] = set(fold_splits[fold]["train"])
        self.eval_pids:  set[str] = set(fold_splits[fold]["eval"])

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SequenceDataset(
            self.npz_path, self.train_pids, seq_len=self.hparams.seq_len
        )
        self.val_dataset = SequenceDataset(
            self.npz_path, self.eval_pids, seq_len=self.hparams.seq_len
        )
        print(
            f"[Data] train clips: {len(self.train_dataset)}, "
            f"val clips: {len(self.val_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=len(self.train_dataset) >= self.hparams.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class LstmVAELit(L.LightningModule):
    """Lightning wrapper around LstmVAE.

    Loss
    ----
    total = recon_loss + vel_w * velocity_loss + std_w * temporal_std_loss
            + beta * kl_loss  [+ gamma * kl_masked]

    where

    * ``recon_loss = L1(x_recon, x)``
    * ``velocity_loss = MSE(Δx_recon, Δx)``  (frame-to-frame deltas)
    * ``temporal_std_loss = MSE(σ_t(x_recon), σ_t(x))``  (per-feature temporal stds)
    * ``kl_loss = free-bits KL`` (per-dimension floor to prevent collapse)
    * ``kl_masked = MC-KL(q_φ ‖ r_ψ)``  (optional, when masked encoder is active)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        encoder_hidden_dim: int = 256,
        encoder_num_layers: int = 2,
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 4,
        latent_dim: int = 256,
        seq_len: int = 80,
        dropout: float = 0.1,
        beta: float = 0.5,
        kl_anneal_epochs: int = 10,
        free_nats: float = 0.1,
        vel_w: float = 2.0,
        std_w: float = 50.0,
        lr: float = 3e-4,
        # Masked encoder
        n_mix: int = 0,
        gamma: float = 1.0,
        mask_warmup_epochs: int = 20,
        mask_axis: str = "both",
        # Per-joint loss weighting (for addressing foot smearing, etc.)
        joint_weights: tuple[float, ...] | None = None,
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

        # Register (J, 3) per-joint weights as a buffer so they move with the
        # module and are saved inside the checkpoint. Defaults to all-ones
        # (i.e. behaviour identical to the unweighted losses).
        n_joints = input_dim // 3
        if joint_weights is None:
            w = torch.ones(n_joints)
        else:
            if len(joint_weights) != n_joints:
                raise ValueError(
                    f"joint_weights has {len(joint_weights)} entries but "
                    f"input_dim implies {n_joints} joints."
                )
            w = torch.tensor(joint_weights, dtype=torch.float32)
        # Normalise so that mean weight = 1; this keeps the recon_loss scale
        # comparable across runs even when we up-weight a few joints, so
        # {vel_w, std_w, beta} don't need re-tuning.
        w = w * (n_joints / w.sum())
        # ``persistent=False`` so the buffer is NOT written into the saved
        # state_dict. The vector is fully determined by the ``joint_weights``
        # hparam and regenerated in ``__init__`` every time — this lets us
        # resume old checkpoints that were saved before this field existed.
        self.register_buffer("joint_weight_vec", w, persistent=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_free_bits(mu: Tensor, log_var: Tensor, free_nats: float) -> Tensor:
        kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        return torch.clamp(kl_per_dim, min=free_nats).mean()

    def _current_beta(self) -> float:
        """Linear KL warm-up: beta ramps from 0 → hparams.beta over kl_anneal_epochs."""
        anneal = self.hparams.kl_anneal_epochs
        if anneal <= 0:
            return self.hparams.beta
        epoch = self.current_epoch
        return self.hparams.beta * min(1.0, epoch / anneal)

    @staticmethod
    def _mpjpe(recon: Tensor, target: Tensor) -> Tensor:
        B, T, D = target.shape
        J = D // 3
        pred = recon.view(B, T, J, 3)
        gt = target.view(B, T, J, 3)
        return torch.norm(pred - gt, dim=-1).mean()

    @staticmethod
    def _mpjpe_observed(recon: Tensor, target: Tensor, coalition_mask: Tensor) -> Tensor:
        """MPJPE only on joints/frames that are *observed* (coalition_mask=True).

        Handles both spatial masks ``(B, J)`` and temporal masks ``(B, T)``.
        """
        B, T, D = target.shape
        J = D // 3
        pred = recon.view(B, T, J, 3)
        gt = target.view(B, T, J, 3)
        per_joint_err = torch.norm(pred - gt, dim=-1)  # (B, T, J)

        if coalition_mask.shape[-1] == J:
            # Spatial mask (B, J) → broadcast over T
            obs = coalition_mask[:, None, :].expand_as(per_joint_err)
        else:
            # Temporal mask (B, T) → broadcast over J
            obs = coalition_mask[:, :, None].expand_as(per_joint_err)

        if obs.any():
            return per_joint_err[obs].mean()
        return per_joint_err.mean()

    def _weighted_reduce(self, per_elt: Tensor) -> Tensor:
        """Reduce a ``(B, T, J, 3)`` (or ``(B, J, 3)``) per-element loss map
        using the registered per-joint weight vector.

        We multiply by ``joint_weight_vec[J]`` and take the mean, which is
        equivalent to a weighted average because the vector is normalised to
        mean=1 in ``__init__``.
        """
        w = self.joint_weight_vec  # (J,)
        # Broadcast over leading and trailing dims.
        shape = [1] * per_elt.ndim
        shape[-2] = w.shape[0]
        return (per_elt * w.view(*shape)).mean()

    def _recon_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        B, T, D = target.shape
        J = D // 3
        r = recon.view(B, T, J, 3)
        g = target.view(B, T, J, 3)
        return self._weighted_reduce((r - g).abs())

    def _velocity_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        B, T, D = target.shape
        J = D // 3
        dr = (recon[:, 1:] - recon[:, :-1]).view(B, T - 1, J, 3)
        dt = (target[:, 1:] - target[:, :-1]).view(B, T - 1, J, 3)
        return self._weighted_reduce((dr - dt).pow(2))

    def _temporal_std_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        B, T, D = target.shape
        J = D // 3
        # std over time per feature → (B, J, 3)
        sr = recon.view(B, T, J, 3).std(dim=1)
        sg = target.view(B, T, J, 3).std(dim=1)
        return self._weighted_reduce((sr - sg).pow(2))

    def _sample_coalition_mask(self, x: Tensor) -> Tensor | None:
        """Sample a coalition mask for the masked encoder."""
        if self.model.masked_encoder is None:
            return None
        if self.current_epoch < self.hparams.mask_warmup_epochs:
            return None

        B, T, _D = x.shape
        axis = self.hparams.mask_axis
        if axis == "both":
            use_spatial = torch.rand(1).item() < 0.5
        else:
            use_spatial = axis == "spatial"

        if use_spatial:
            return sample_spatial_training_mask(B, x.device, n_joints=N_JOINTS)
        return sample_temporal_training_mask(B, T, x.device)

    def _step(self, batch: Tensor, stage: str) -> Tensor:
        x = batch  # (B, T, 51)
        recon, mu, log_var = self.model(x)

        recon_loss = self._recon_loss(recon, x)
        vel_loss   = self._velocity_loss(recon, x)
        std_loss   = self._temporal_std_loss(recon, x)
        kl_loss    = self._kl_free_bits(mu, log_var, self.hparams.free_nats)
        beta       = self._current_beta()

        loss = (recon_loss
                + self.hparams.vel_w * vel_loss
                + self.hparams.std_w * std_loss
                + beta * kl_loss)

        # Masked encoder regularisation
        kl_mask_val = torch.zeros(1, device=x.device)
        coalition_mask = self._sample_coalition_mask(x)
        mask_active = coalition_mask is not None
        if mask_active:
            log_pi, mu_mix, logvar_mix = self.model.forward_masked(x, coalition_mask)
            kl_mask_val = kl_full_vs_mixture(
                mu.detach(), log_var.detach(),
                log_pi, mu_mix, logvar_mix,
            )
            loss = loss + self.hparams.gamma * kl_mask_val

            z_masked = self.model.masked_encoder.sample(log_pi, mu_mix, logvar_mix)
            recon_masked = self.model.decode(z_masked, seq_len=x.shape[1])
            mpjpe_masked_obs = self._mpjpe_observed(recon_masked, x, coalition_mask)

        mpjpe = self._mpjpe(recon, x)

        # Per-joint MPJPE for the joints that are up-weighted, so we can see
        # whether the foot reconstruction actually improves over the run.
        B, T, D = x.shape
        J = D // 3
        per_joint_mpjpe = (
            recon.view(B, T, J, 3) - x.view(B, T, J, 3)
        ).norm(dim=-1).mean(dim=(0, 1))                # (J,)
        for j, w in enumerate(self.joint_weight_vec.tolist()):
            if abs(w - 1.0) > 1e-6:
                self.log(f"{stage}/mpjpe_j{j}", per_joint_mpjpe[j],
                         on_epoch=True, on_step=False)

        self.log(f"{stage}/beta",  beta,       on_epoch=True, on_step=False)
        self.log(f"{stage}/recon", recon_loss, prog_bar=(stage == "train"), on_epoch=True, on_step=False)
        self.log(f"{stage}/vel",   vel_loss,   prog_bar=False,              on_epoch=True, on_step=False)
        self.log(f"{stage}/tstd",  std_loss,   prog_bar=False,              on_epoch=True, on_step=False)
        self.log(f"{stage}/kl",    kl_loss,    prog_bar=False,              on_epoch=True, on_step=False)
        self.log(f"{stage}/loss",  loss,        prog_bar=True,              on_epoch=True, on_step=False)
        self.log(f"{stage}/mpjpe", mpjpe,       prog_bar=True,              on_epoch=True, on_step=False)
        if self.model.masked_encoder is not None:
            self.log(f"{stage}/kl_mask", kl_mask_val, prog_bar=False, on_epoch=True, on_step=False)
            # Log a large sentinel during warmup so the checkpoint callback
            # never picks a warmup epoch as "best".
            masked_obs_val = mpjpe_masked_obs if mask_active else torch.tensor(999.0, device=x.device)
            self.log(f"{stage}/mpjpe_masked_obs", masked_obs_val, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple LSTM VAE on CARE-PD pose sequences.")

    # Data
    p.add_argument("--dataset",    type=str, default="BMCLab",
                   choices=list(DATASET_NPZ.keys()),
                   help="Dataset name.")
    p.add_argument("--fold",       type=int, default=1,
                   help="Fold index (1-based).")
    p.add_argument("--num_folds",  type=int, default=6,
                   help="Total number of folds (used to select the fold pickle).")
    p.add_argument("--seq_len",    type=int, default=80,
                   help="Clip length in frames.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)

    # Architecture
    p.add_argument("--latent_dim",           type=int,   default=256)
    p.add_argument("--encoder_hidden_dim",   type=int,   default=256)
    p.add_argument("--encoder_num_layers",   type=int,   default=2)
    p.add_argument("--decoder_hidden_dim",   type=int,   default=256)
    p.add_argument("--decoder_num_layers",   type=int,   default=4)
    p.add_argument("--dropout",              type=float, default=0.1)

    # Logging
    p.add_argument("--wandb_project",  type=str, default="carepd_lstm",
                   help="Weights & Biases project name.")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="W&B run name. Defaults to '{dataset}_fold{fold}'.")

    # Training / loss weights
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--beta",             type=float, default=0.5,
                   help="KL weight target.")
    p.add_argument("--kl_anneal_epochs", type=int,   default=10,
                   help="Epochs over which beta linearly warms up from 0 to --beta. 0 disables annealing.")
    p.add_argument("--free_nats",        type=float, default=0.1,
                   help="Free-bits floor (nats per latent dim) to prevent posterior collapse.")
    p.add_argument("--vel_w",            type=float, default=2.0,
                   help="Weight on frame-to-frame velocity loss.")
    p.add_argument("--std_w",            type=float, default=50.0,
                   help="Weight on temporal standard-deviation matching loss.")
    # Masked encoder
    p.add_argument("--n_mix",    type=int,   default=0,
                   help="Number of GMM components for masked encoder. 0 disables.")
    p.add_argument("--gamma",    type=float, default=1.0,
                   help="Weight on KL(q_phi || r_psi) masked-encoder regularisation.")
    p.add_argument("--mask_warmup_epochs", type=int, default=20,
                   help="Train full encoder only for this many epochs before enabling masked encoder.")
    p.add_argument("--mask_axis", type=str, default="both",
                   choices=["spatial", "temporal", "both"],
                   help="Coalition mask axis (spatial=joints, temporal=frames, both=random).")

    # Per-joint loss weighting (helpful for addressing foot smearing).
    p.add_argument("--foot_weight", type=float, default=1.0,
                   help="Multiplier applied to the foot joints' reconstruction, "
                        "velocity, and temporal-std losses. 1.0 = no change.")
    p.add_argument("--foot_joints", type=str, default="3,6",
                   help="Comma-separated joint indices to treat as feet (default "
                        "3,6 = right foot + left foot for H36M 17-joint skeleton).")

    # Resume / fine-tune
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a .ckpt to resume from (Lightning ckpt_path). "
                        "Restores model weights + optimizer + epoch counter.")

    p.add_argument("--max_epochs", type=int,   default=150)
    p.add_argument("--devices", nargs="+", default=["4", "5", "6", "7"],
                   help="GPU ids or a single count, space- or comma-separated. "
                        "e.g. '--devices 4 5 6 7' or '--devices 4'.")
    p.add_argument("--accelerator", type=str, default="gpu")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Flatten comma-separated tokens users might pass (e.g. "4,5,6,7" as one token)
    raw = []
    for tok in args.devices:
        raw.extend(tok.split(","))
    ids = [int(x) for x in raw if x]

    if len(ids) == 1:
        # Single value → treat as a count
        args.devices = ids[0]
    else:
        # Multiple IDs → remap through CUDA_VISIBLE_DEVICES if set, so that
        # physical GPU 4 → logical index 0, etc.
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_vis:
            visible = [int(x) for x in cuda_vis.split(",") if x]
            args.devices = [visible.index(i) for i in ids if i in visible]
        else:
            args.devices = ids

    run_name = args.wandb_run_name or f"{args.dataset}_fold{args.fold}"
    out_dir  = PROJECT_ROOT / "experiment_outs" / "lstm_vae" / run_name

    # ---- Data ----
    dm = PoseDataModule(
        dataset=args.dataset,
        fold=args.fold,
        num_folds=args.num_folds,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- Per-joint weights ----
    n_joints = INPUT_DIM // 3
    joint_weights = [1.0] * n_joints
    if abs(args.foot_weight - 1.0) > 1e-6:
        foot_indices = [int(x) for x in args.foot_joints.split(",") if x.strip()]
        for j in foot_indices:
            if not 0 <= j < n_joints:
                raise ValueError(f"foot_joint index {j} out of range [0, {n_joints}).")
            joint_weights[j] = args.foot_weight
        print(f"[Loss] Per-joint weights: feet={foot_indices} at "
              f"×{args.foot_weight}, rest ×1.0 (normalised so mean=1).")

    # ---- Model ----
    lit = LstmVAELit(
        input_dim=INPUT_DIM,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        dropout=args.dropout,
        beta=args.beta,
        kl_anneal_epochs=args.kl_anneal_epochs,
        free_nats=args.free_nats,
        vel_w=args.vel_w,
        std_w=args.std_w,
        lr=args.lr,
        n_mix=args.n_mix,
        gamma=args.gamma,
        mask_warmup_epochs=args.mask_warmup_epochs,
        mask_axis=args.mask_axis,
        joint_weights=tuple(joint_weights),
    )

    # ---- Trainer ----
    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=str(out_dir),
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="epoch{epoch:03d}-mpjpe{val/mpjpe:.4f}",
        monitor="val/mpjpe",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_cb, LearningRateMonitor(logging_interval="epoch")]

    if args.n_mix > 0:
        masked_ckpt_cb = ModelCheckpoint(
            dirpath=str(out_dir / "checkpoints"),
            filename="epoch{epoch:03d}-masked_obs{val/mpjpe_masked_obs:.4f}",
            monitor="val/mpjpe_masked_obs",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        )
        callbacks.append(masked_ckpt_cb)

    strategy = "auto"
    if args.n_mix > 0 and isinstance(args.devices, list) and len(args.devices) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    if args.resume_from:
        print(f"[Resume] Loading state from {args.resume_from}")
        trainer.fit(lit, datamodule=dm, ckpt_path=args.resume_from)
    else:
        trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
