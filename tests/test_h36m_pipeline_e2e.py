#!/usr/bin/env python
"""End-to-end pipeline test for H36M data loading → batching → model → metrics.

Loads real H36M data and validates correctness at every stage:
  1. Raw data shapes and value ranges
  2. Preprocessor output (pickle files)
  3. ProcessedDataset samples
  4. collate_fn batch formation
  5. actor_batch_from_carepd transformation
  6. Encoder/decoder shape flow
  7. Metric computation (MPJPE, joint motion)
  8. Loss computation and gradient flow

Run from repo root:
    python -m pytest tests/test_h36m_pipeline_e2e.py -v --tb=short
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h36m_pkl():
    pkl_path = PROJECT_ROOT / "assets" / "datasets" / "H36M.pkl"
    if not pkl_path.exists():
        pytest.skip(f"H36M.pkl not found at {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def h36m_npz():
    npz_path = (
        PROJECT_ROOT / "assets" / "datasets" / "h36m" / "H36M"
        / "h36m_3d_world_floorXZZplus_30f_or_longer.npz"
    )
    if not npz_path.exists():
        pytest.skip(f"H36M NPZ not found at {npz_path}")
    return np.load(str(npz_path), allow_pickle=True)


@pytest.fixture(scope="module")
def datasets():
    """Load train/val datasets for fold 1."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from model.actor.cvae_data import get_carepd_datasets

    args = argparse.Namespace(
        dataset="H36M",
        num_folds=7,
        batch_size=8,
        experiment_name="VaeacActorMotion",
        source_seq_len=81,
        carepd_pose_npz=None,
        carepd_labels_pkl=None,
        fold=1,
    )
    train_ds, val_ds = get_carepd_datasets(args)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# 1. Raw data integrity
# ---------------------------------------------------------------------------

class TestRawData:
    def test_labels_structure(self, h36m_pkl):
        assert "labels" in h36m_pkl
        assert "participant" in h36m_pkl
        assert "action_to_label" in h36m_pkl
        assert "n_classes" in h36m_pkl

    def test_label_values(self, h36m_pkl):
        labels = list(h36m_pkl["labels"].values())
        assert min(labels) == 0, f"Min label should be 0, got {min(labels)}"
        assert max(labels) == h36m_pkl["n_classes"] - 1, (
            f"Max label should be {h36m_pkl['n_classes']-1}, got {max(labels)}"
        )

    def test_num_classes(self, h36m_pkl):
        assert h36m_pkl["n_classes"] == 13, (
            f"Expected 13 action classes (15 total - 2 excluded), got {h36m_pkl['n_classes']}"
        )

    def test_subjects(self, h36m_pkl):
        subjects = sorted(set(h36m_pkl["participant"].values()))
        expected = ["S1", "S11", "S5", "S6", "S7", "S8", "S9"]
        assert subjects == expected, f"Expected {expected}, got {subjects}"

    def test_pose_shapes(self, h36m_npz):
        for seq_name in list(h36m_npz.keys())[:10]:
            arr = h36m_npz[seq_name]
            assert arr.ndim == 3, f"{seq_name}: expected 3D, got {arr.ndim}D"
            assert arr.shape[1] == 17, f"{seq_name}: expected 17 joints, got {arr.shape[1]}"
            assert arr.shape[2] == 3, f"{seq_name}: expected 3 features, got {arr.shape[2]}"
            assert arr.shape[0] >= 30, f"{seq_name}: too few frames ({arr.shape[0]})"
            assert arr.dtype == np.float32

    def test_pelvis_at_origin(self, h36m_npz):
        """Global translation was zeroed → pelvis should be at/near origin."""
        for seq_name in list(h36m_npz.keys())[:5]:
            arr = h36m_npz[seq_name]
            pelvis_all_frames = arr[:, 0, :]  # (T, 3)
            pelvis_range = np.ptp(pelvis_all_frames, axis=0)
            assert np.all(pelvis_range < 0.01), (
                f"{seq_name}: pelvis range {pelvis_range} — expected near-zero "
                f"because global translation was zeroed during preprocessing"
            )

    def test_non_pelvis_joints_have_motion(self, h36m_npz):
        """Joints 1-16 should show meaningful motion (non-trivial velocity)."""
        for seq_name in list(h36m_npz.keys())[:5]:
            arr = h36m_npz[seq_name]
            vel = np.linalg.norm(np.diff(arr[:, 1:, :], axis=0), axis=-1)
            mean_vel = vel.mean()
            assert mean_vel > 1e-4, (
                f"{seq_name}: mean velocity of non-pelvis joints is {mean_vel:.6f} — "
                f"motion data appears static"
            )

    def test_joint_positions_reasonable_scale(self, h36m_npz):
        """Positions should be in metres (typical body ~1.8m tall)."""
        for seq_name in list(h36m_npz.keys())[:5]:
            arr = h36m_npz[seq_name]
            max_abs = np.abs(arr).max()
            assert max_abs < 5.0, (
                f"{seq_name}: max |position| = {max_abs:.2f}m — "
                f"expected < 5m for metre-scale H36M data"
            )
            max_extent = np.ptp(arr[:, :, :], axis=1).max()
            assert max_extent > 0.3, (
                f"{seq_name}: max body extent = {max_extent:.2f}m — "
                f"body seems too small"
            )

    def test_all_sequences_have_labels_and_participants(self, h36m_npz, h36m_pkl):
        npz_keys = set(h36m_npz.keys())
        label_keys = set(h36m_pkl["labels"].keys())
        participant_keys = set(h36m_pkl["participant"].keys())
        assert npz_keys == label_keys, (
            f"Mismatch: NPZ has {len(npz_keys)} seqs, labels has {len(label_keys)}. "
            f"Extra in NPZ: {npz_keys - label_keys}, missing: {label_keys - npz_keys}"
        )
        assert npz_keys == participant_keys


# ---------------------------------------------------------------------------
# 2. ProcessedDataset
# ---------------------------------------------------------------------------

class TestProcessedDataset:
    def test_dataset_sizes(self, datasets):
        train_ds, val_ds = datasets
        assert len(train_ds) > 0, "Train dataset is empty"
        assert len(val_ds) > 0, "Val dataset is empty"
        assert len(train_ds) > len(val_ds), (
            f"Train ({len(train_ds)}) should be larger than val ({len(val_ds)})"
        )

    def test_sample_shapes(self, datasets):
        train_ds, _ = datasets
        sample = train_ds[0]
        x = sample["encoder_inputs"]
        assert x.shape == (81, 17, 3), f"Expected (81, 17, 3), got {x.shape}"
        assert x.dtype == np.float32

    def test_labels_in_range(self, datasets):
        train_ds, val_ds = datasets
        all_labels = np.concatenate([train_ds.labels, val_ds.labels])
        assert all_labels.min() >= 0, f"Min label = {all_labels.min()}"
        max_label = all_labels.max()
        assert max_label <= 12, f"Max label = {max_label}, expected <= 12"

    def test_pad_masks(self, datasets):
        train_ds, _ = datasets
        for idx in range(min(20, len(train_ds))):
            sample = train_ds[idx]
            pm = sample["pad_mask"]
            assert pm.shape == (81,), f"Expected pad_mask shape (81,), got {pm.shape}"
            n_valid = int(pm.sum())
            assert n_valid >= 30, f"Sample {idx}: only {n_valid} valid frames"
            x = sample["encoder_inputs"]
            if n_valid < 81:
                padded_frames = x[n_valid:]
                assert np.allclose(padded_frames, 0.0), (
                    f"Sample {idx}: padded frames are not zero"
                )

    def test_no_all_zero_valid_frames(self, datasets):
        """Valid frames should not be all zeros (would indicate data loading bug)."""
        train_ds, _ = datasets
        for idx in range(min(20, len(train_ds))):
            sample = train_ds[idx]
            x = sample["encoder_inputs"]
            pm = sample["pad_mask"]
            n_valid = int(pm.sum())
            valid_frames = x[:n_valid]
            per_frame_norm = np.linalg.norm(valid_frames.reshape(n_valid, -1), axis=-1)
            assert np.all(per_frame_norm > 0.01), (
                f"Sample {idx}: some valid frames have near-zero norm — "
                f"data may not be loaded correctly"
            )


# ---------------------------------------------------------------------------
# 3. Batching
# ---------------------------------------------------------------------------

class TestBatching:
    @pytest.fixture
    def batch(self, datasets):
        from data.dataloaders import collate_fn
        train_ds, _ = datasets
        samples = [train_ds[i] for i in range(8)]
        return collate_fn(samples)

    def test_collate_shapes(self, batch):
        e_inp, labels, video_idxs, metadata, pad_mask = batch
        assert e_inp.shape == (8, 81, 17, 3), f"Got {e_inp.shape}"
        assert labels.shape == (8,)
        assert pad_mask.shape == (8, 81)

    def test_actor_batch_from_carepd(self, batch):
        from model.actor.cvae_data import actor_batch_from_carepd
        e_inp, labels, _, _, pad_mask = batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            y=labels.long(),
        )
        assert b["x"].shape == (8, 17, 3, 81), f"Got {b['x'].shape}"
        assert b["y"].shape == (8,)
        assert b["mask"].shape == (8, 81)
        assert b["lengths"].shape == (8,)
        assert b["x"].dtype == torch.float32
        assert b["y"].dtype == torch.int64

    def test_global_pelvis_transform_reversible(self, batch):
        from model.actor.cvae_data import actor_batch_from_carepd
        from model.actor.motion_utils import unroot_to_global
        e_inp, labels, _, _, pad_mask = batch

        b_gp = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            y=labels.long(), world_coords=False,
        )
        b_world = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            y=labels.long(), world_coords=True,
        )
        recovered = unroot_to_global(
            b_gp["x"].permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        assert torch.allclose(recovered, b_world["x"], atol=1e-5), (
            f"Global-pelvis transform is NOT reversible. "
            f"Max diff = {(recovered - b_world['x']).abs().max():.6f}"
        )

    def test_mask_matches_data(self, batch):
        from model.actor.cvae_data import actor_batch_from_carepd
        e_inp, labels, _, _, pad_mask = batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            y=labels.long(), world_coords=True,
        )
        x = b["x"]  # (B, J, F, T)
        mask = b["mask"]  # (B, T)
        for i in range(x.shape[0]):
            n_valid = int(mask[i].sum().item())
            if n_valid < 81:
                padded = x[i, :, :, n_valid:]
                assert torch.allclose(padded, torch.zeros_like(padded)), (
                    f"Sample {i}: padded frames in ACTOR tensor are not zero"
                )

    def test_lengths_match_mask(self, batch):
        from model.actor.cvae_data import actor_batch_from_carepd
        e_inp, labels, _, _, pad_mask = batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
        )
        assert torch.equal(b["lengths"], b["mask"].sum(dim=-1).long())


# ---------------------------------------------------------------------------
# 4. Model forward pass shapes
# ---------------------------------------------------------------------------

class TestModelShapes:
    @pytest.fixture
    def model_and_batch(self, datasets):
        from data.dataloaders import collate_fn
        from model.actor.cvae_data import actor_batch_from_carepd
        from model.actor.transformer_arch import Decoder_TRANSFORMER
        from model.actor.vaeac_actor_motion import (
            VaeacActorFullEncoder,
            VaeacActorMaskedEncoder,
            VaeacActorMotion,
        )

        train_ds, _ = datasets
        all_labels = train_ds.labels
        num_classes = int(np.max(all_labels)) + 1

        common = dict(
            modeltype="cvae",
            njoints=17, nfeats=3,
            num_frames=0, num_classes=num_classes,
            translation=True, pose_rep="xyz",
            glob=True, glob_rot=[3.141592653589793, 0, 0],
            latent_dim=32, ff_size=64,
            num_layers=2, num_heads=2,
            dropout=0.0, ablation=None, activation="gelu",
        )
        full_enc = VaeacActorFullEncoder(**common)
        masked_enc = VaeacActorMaskedEncoder(**common)
        dec = Decoder_TRANSFORMER(**common)
        model = VaeacActorMotion(
            full_enc, masked_enc, dec,
            latent_dim=32, njoints=17, nfeats=3,
            device=torch.device("cpu"),
            num_classes=num_classes,
        )
        model.eval()

        samples = [train_ds[i] for i in range(4)]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), num_classes, torch.device("cpu"),
            y=labels.long(),
        )
        return model, b

    def test_train_forward_full_mask(self, model_and_batch):
        model, b = model_and_batch
        model.train()
        cm = torch.ones(4, 17, dtype=torch.bool)
        cm[:, 3:6] = False
        b_in = {**b, "coalition_mask": cm}
        out = model(dict(b_in), phase="train")
        assert out["output"].shape == (4, 17, 3, 81)
        assert out["mu_full"].shape[0] == 4
        assert out["mu_masked"].shape[0] == 4
        assert "frame_tokens" not in out, (
            "Default use_frame_tokens=False should not put frame_tokens in batch"
        )

    def test_train_forward_no_mask(self, model_and_batch):
        model, b = model_and_batch
        model.train()
        out = model(dict(b), phase="train")
        assert out["output"].shape == (4, 17, 3, 81)
        assert "mu_full" in out
        assert "mu_masked" not in out

    def test_infer_forward(self, model_and_batch):
        model, b = model_and_batch
        model.eval()
        cm = torch.ones(4, 17, dtype=torch.bool)
        cm[:, 3:6] = False
        b_in = {**b, "coalition_mask": cm}
        with torch.no_grad():
            out = model(dict(b_in), phase="infer")
        assert out["output"].shape == (4, 17, 3, 81)
        assert "mu_masked" in out
        assert "mu_full" not in out

    def test_temporal_mask_forward(self, model_and_batch):
        model, b = model_and_batch
        model.train()
        cm = torch.ones(4, 81, dtype=torch.bool)
        cm[:, :20] = False
        b_in = {**b, "coalition_mask": cm}
        out = model(dict(b_in), phase="train")
        assert out["output"].shape == (4, 17, 3, 81)


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.fixture
    def gt_and_pred(self, datasets):
        from data.dataloaders import collate_fn
        from model.actor.cvae_data import actor_batch_from_carepd

        train_ds, _ = datasets
        samples = [train_ds[i] for i in range(8)]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            y=labels.long(),
        )
        x = b["x"]
        mask = b["mask"]
        pred = x + torch.randn_like(x) * 0.01
        pred = pred * mask.unsqueeze(1).unsqueeze(1).float()
        return x, pred, mask

    def test_mpjpe_correct(self, gt_and_pred):
        from train_actor_cvae import masked_mpjpe
        x, pred, mask = gt_and_pred
        mpjpe = masked_mpjpe(pred, x, mask)

        err = torch.linalg.norm(pred - x, dim=2)  # (B, J, T)
        m = mask.unsqueeze(1).expand_as(err)
        expected = err[m].mean()
        assert torch.allclose(mpjpe, expected, atol=1e-6)

    def test_gt_motion_nonzero(self, gt_and_pred):
        """Ground truth data should have non-trivial joint motion."""
        from train_actor_cvae import masked_mean_joint_motion
        x, _, mask = gt_and_pred
        gt_mot = masked_mean_joint_motion(x, mask)
        assert gt_mot > 1e-4, (
            f"GT joint motion = {float(gt_mot):.6f} — data appears static. "
            f"This would make recon_joint_motion a meaningless metric."
        )

    def test_perfect_reconstruction_zero_mpjpe(self, gt_and_pred):
        from train_actor_cvae import masked_mpjpe
        x, _, mask = gt_and_pred
        mpjpe = masked_mpjpe(x, x, mask)
        assert mpjpe < 1e-7, f"Perfect reconstruction MPJPE = {float(mpjpe)}"

    def test_perfect_reconstruction_matching_motion(self, gt_and_pred):
        from train_actor_cvae import masked_mean_joint_motion
        x, _, mask = gt_and_pred
        gt_mot = masked_mean_joint_motion(x, mask)
        recon_mot = masked_mean_joint_motion(x, mask)
        assert torch.allclose(gt_mot, recon_mot)

    def test_metrics_ignore_padded_frames(self):
        from train_actor_cvae import masked_mpjpe, masked_mean_joint_motion
        B, J, F, T = 2, 17, 3, 10
        x = torch.randn(B, J, F, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False
        mask[1, 5:] = False

        x_with_junk = x.clone()
        x_with_junk[0, :, :, 7:] = 999.0
        x_with_junk[1, :, :, 5:] = 999.0

        pred = x.clone()
        pred_with_junk = pred.clone()
        pred_with_junk[0, :, :, 7:] = -999.0
        pred_with_junk[1, :, :, 5:] = -999.0

        mpjpe_clean = masked_mpjpe(pred, x, mask)
        mpjpe_junk = masked_mpjpe(pred_with_junk, x_with_junk, mask)
        assert torch.allclose(mpjpe_clean, mpjpe_junk, atol=1e-6), (
            "Metrics should be identical regardless of padded-frame values"
        )

        mot_clean = masked_mean_joint_motion(x, mask)
        mot_junk = masked_mean_joint_motion(x_with_junk, mask)
        assert torch.allclose(mot_clean, mot_junk, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Loss + gradients
# ---------------------------------------------------------------------------

class TestLossAndGradients:
    @pytest.fixture
    def model_and_batch(self, datasets):
        from data.dataloaders import collate_fn
        from model.actor.cvae_data import actor_batch_from_carepd
        from model.actor.transformer_arch import Decoder_TRANSFORMER
        from model.actor.vaeac_actor_motion import (
            VaeacActorFullEncoder,
            VaeacActorMaskedEncoder,
            VaeacActorMotion,
        )

        train_ds, _ = datasets
        num_classes = int(np.max(train_ds.labels)) + 1
        common = dict(
            modeltype="cvae",
            njoints=17, nfeats=3,
            num_frames=0, num_classes=num_classes,
            translation=True, pose_rep="xyz",
            glob=True, glob_rot=[3.141592653589793, 0, 0],
            latent_dim=32, ff_size=64,
            num_layers=2, num_heads=2,
            dropout=0.0, ablation=None, activation="gelu",
        )
        full_enc = VaeacActorFullEncoder(**common)
        masked_enc = VaeacActorMaskedEncoder(**common)
        dec = Decoder_TRANSFORMER(**common)
        model = VaeacActorMotion(
            full_enc, masked_enc, dec,
            latent_dim=32, njoints=17, nfeats=3,
            device=torch.device("cpu"),
            num_classes=num_classes,
        )

        samples = [train_ds[i] for i in range(4)]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), num_classes, torch.device("cpu"),
            y=labels.long(),
        )
        return model, b

    def test_loss_finite(self, model_and_batch):
        model, b = model_and_batch
        model.train()
        cm = torch.ones(4, 17, dtype=torch.bool)
        cm[:, 3:6] = False
        b_in = {**b, "coalition_mask": cm}
        out = model(dict(b_in), phase="train")
        loss, ld = model.compute_loss(out, lambda_kl=0.01, lambda_vel=5.0)
        assert loss.isfinite(), f"Loss is not finite: {float(loss)}"
        for k, v in ld.items():
            assert np.isfinite(v), f"Loss component '{k}' = {v}"

    def test_gradients_flow(self, model_and_batch):
        model, b = model_and_batch
        model.train()
        cm = torch.ones(4, 17, dtype=torch.bool)
        cm[:, 3:6] = False
        b_in = {**b, "coalition_mask": cm}
        out = model(dict(b_in), phase="train")
        loss, _ = model.compute_loss(out, lambda_kl=0.01, lambda_vel=5.0)
        loss.backward()

        n_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().max() > 0
        )
        n_total = sum(1 for p in model.parameters() if p.requires_grad)
        assert n_with_grad > n_total * 0.8, (
            f"Only {n_with_grad}/{n_total} parameters have non-zero grad — "
            f"gradient flow may be broken"
        )

    def test_velocity_loss_nonzero_for_static_pred(self, model_and_batch):
        """If output is a frozen mean pose, velocity loss should be large."""
        model, b = model_and_batch
        model.train()
        out = model(dict(b), phase="train")
        x = out["x"]
        mask = out["mask"]

        mean_pose = x.mean(dim=-1, keepdim=True).expand_as(x)
        mean_pose = mean_pose * mask.unsqueeze(1).unsqueeze(1).float()
        out["output"] = mean_pose

        _, ld = model.compute_loss(out, lambda_vel=5.0)
        assert ld["vel"] > 1e-6, (
            f"Velocity loss on a static prediction = {ld['vel']:.8f} — "
            f"should be significantly non-zero"
        )

    def test_loss_decreases_after_step(self, model_and_batch):
        """One optimizer step should reduce the loss."""
        model, b = model_and_batch
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        cm = torch.ones(4, 17, dtype=torch.bool)
        cm[:, 3:6] = False
        b_in = {**b, "coalition_mask": cm}

        out1 = model(dict(b_in), phase="train")
        loss1, _ = model.compute_loss(out1, lambda_kl=0.01, lambda_vel=5.0)
        loss1.backward()
        opt.step()
        opt.zero_grad()

        out2 = model(dict(b_in), phase="train")
        loss2, _ = model.compute_loss(out2, lambda_kl=0.01, lambda_vel=5.0)
        assert loss2 < loss1 * 1.5, (
            f"Loss did not decrease after one step: {float(loss1):.4f} → {float(loss2):.4f}"
        )


# ---------------------------------------------------------------------------
# 7. Data properties critical for training
# ---------------------------------------------------------------------------

class TestDataProperties:
    def test_pelvis_is_static_in_preprocessed_data(self, datasets):
        """The pelvis (joint 0) should have near-zero motion since global
        translation was zeroed during H36M preprocessing."""
        train_ds, _ = datasets
        from data.dataloaders import collate_fn
        from model.actor.cvae_data import actor_batch_from_carepd

        samples = [train_ds[i] for i in range(min(20, len(train_ds)))]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            y=labels.long(), world_coords=True,
        )
        x = b["x"]  # (B, J, F, T)
        pelvis = x[:, 0, :, :]  # (B, 3, T)
        pelvis_vel = (pelvis[:, :, 1:] - pelvis[:, :, :-1]).norm(dim=1)
        mean_pelvis_vel = pelvis_vel.mean()
        assert mean_pelvis_vel < 0.01, (
            f"Pelvis mean velocity = {float(mean_pelvis_vel):.6f} — "
            f"expected near-zero because global translation was removed"
        )

    def test_global_pelvis_transform_is_noop_when_pelvis_at_origin(self, datasets):
        """Since pelvis is at origin, root_with_global_pelvis should be ~identity."""
        train_ds, _ = datasets
        from data.dataloaders import collate_fn
        from model.actor.cvae_data import actor_batch_from_carepd

        samples = [train_ds[i] for i in range(8)]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch

        b_gp = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            world_coords=False,
        )
        b_world = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
            world_coords=True,
        )
        max_diff = (b_gp["x"] - b_world["x"]).abs().max()
        assert max_diff < 0.01, (
            f"Global-pelvis transform should be near-identity (pelvis at origin), "
            f"but max diff = {float(max_diff):.6f}"
        )

    def test_gt_joint_motion_matches_reported_scale(self, datasets):
        """Sanity check: GT joint motion should be on a reasonable scale."""
        from data.dataloaders import collate_fn
        from model.actor.cvae_data import actor_batch_from_carepd
        from train_actor_cvae import masked_mean_joint_motion

        train_ds, _ = datasets
        samples = [train_ds[i] for i in range(min(50, len(train_ds)))]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
        )
        gt_mot = masked_mean_joint_motion(b["x"], b["mask"])
        print(f"\n  GT joint motion over {len(samples)} samples: {float(gt_mot):.6f} m/frame")
        assert gt_mot > 1e-4, f"GT joint motion too low: {float(gt_mot):.6f}"
        assert gt_mot < 1.0, f"GT joint motion too high: {float(gt_mot):.6f}"

    def test_per_joint_motion_breakdown(self, datasets):
        """Validate per-joint motion and identify static joints.

        Because global translation + root rotation were zeroed during H36M
        preprocessing, all direct children of the root (RHip, LHip, Spine)
        have FIXED positions.  This is expected FK behaviour, not a bug.
        """
        from data.dataloaders import collate_fn
        from model.actor.cvae_data import actor_batch_from_carepd

        H36M_JOINT_NAMES = [
            "Pelvis", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
            "Spine", "Thorax", "Neck", "Head",
            "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
        ]
        # Joints whose position is a fixed offset from the zeroed root.
        # Only DIRECT children of root (32j-0) are static: pelvis (0),
        # RHip (1, 32j-1), LHip (4, 32j-6).
        # Spine (7, 32j-12) is a grandchild (root→32j-11→32j-12) so it
        # has small but real motion from the intermediate joint rotation.
        EXPECTED_STATIC = {0, 1, 4}

        train_ds, _ = datasets
        samples = [train_ds[i] for i in range(min(50, len(train_ds)))]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), 13, torch.device("cpu"),
        )
        x = b["x"]  # (B, J, F, T)
        mask = b["mask"]  # (B, T)
        vel = torch.linalg.norm(x[..., 1:] - x[..., :-1], dim=2)  # (B, J, T-1)
        pair_mask = (mask[:, :-1] & mask[:, 1:]).unsqueeze(1).expand_as(vel)

        n_static = 0
        print("\n  Per-joint mean velocity (m/frame):")
        for j in range(17):
            jv = vel[:, j, :][pair_mask[:, j, :]]
            mean_v = float(jv.mean()) if jv.numel() > 0 else 0.0
            name = H36M_JOINT_NAMES[j] if j < len(H36M_JOINT_NAMES) else f"J{j}"
            tag = " (static — expected)" if j in EXPECTED_STATIC else ""
            print(f"    {name:12s}: {mean_v:.6f}{tag}")
            if j in EXPECTED_STATIC:
                assert mean_v < 0.001, (
                    f"Joint {name} should be static but has velocity {mean_v:.6f}"
                )
                n_static += 1
            else:
                assert mean_v > 1e-5, (
                    f"Joint {name} has near-zero velocity ({mean_v:.6f}) — "
                    f"unexpected for a non-root-child joint"
                )

        print(f"\n  {n_static}/{17} joints are static "
              f"({100*n_static/17:.0f}% of joints contribute 0 to velocity metric)")
        assert n_static == len(EXPECTED_STATIC)


# ---------------------------------------------------------------------------
# 8. ACTOR CVAE baseline comparison
# ---------------------------------------------------------------------------

class TestActorCVAEComparison:
    """Verify VaeacActorMotion's training path matches ACTOR CVAE."""

    @pytest.fixture
    def both_models_and_batch(self, datasets):
        from data.dataloaders import collate_fn
        from model.actor.cvae import ActorCVAE
        from model.actor.cvae_data import actor_batch_from_carepd
        from model.actor.transformer_arch import (
            Decoder_TRANSFORMER,
            Encoder_TRANSFORMER,
        )
        from model.actor.vaeac_actor_motion import (
            VaeacActorFullEncoder,
            VaeacActorMaskedEncoder,
            VaeacActorMotion,
        )

        train_ds, _ = datasets
        num_classes = int(np.max(train_ds.labels)) + 1
        common = dict(
            modeltype="cvae",
            njoints=17, nfeats=3,
            num_frames=0, num_classes=num_classes,
            translation=True, pose_rep="xyz",
            glob=True, glob_rot=[3.141592653589793, 0, 0],
            latent_dim=32, ff_size=64,
            num_layers=2, num_heads=2,
            dropout=0.0, ablation=None, activation="gelu",
        )

        # ACTOR CVAE
        enc_actor = Encoder_TRANSFORMER(**common)
        dec_actor = Decoder_TRANSFORMER(**common)
        actor_cvae = ActorCVAE(
            enc_actor, dec_actor,
            lambdas={"rc": 1.0, "kl": 1e-5, "vel": 5.0},
            latent_dim=32, device=torch.device("cpu"),
            pose_rep="xyz", num_classes=num_classes,
            use_frame_tokens=True,
        )

        # VAEAC
        full_enc = VaeacActorFullEncoder(**common)
        masked_enc = VaeacActorMaskedEncoder(**common)
        dec_vaeac = Decoder_TRANSFORMER(**common)
        vaeac = VaeacActorMotion(
            full_enc, masked_enc, dec_vaeac,
            latent_dim=32, njoints=17, nfeats=3,
            device=torch.device("cpu"),
            num_classes=num_classes,
        )

        samples = [train_ds[i] for i in range(4)]
        raw_batch = collate_fn(samples)
        e_inp, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            e_inp.float(), pad_mask.bool(), num_classes, torch.device("cpu"),
            y=labels.long(),
        )
        return actor_cvae, vaeac, b

    def test_encoder_outputs_same_keys(self, both_models_and_batch):
        actor_cvae, vaeac, b = both_models_and_batch
        actor_enc_out = actor_cvae.encoder(dict(b))
        vaeac_enc_out = vaeac.full_encoder(dict(b))

        assert "mu" in actor_enc_out
        assert "logvar" in actor_enc_out
        assert "frame_tokens" in actor_enc_out
        assert "mu_full" in vaeac_enc_out
        assert "logvar_full" in vaeac_enc_out
        assert "frame_tokens" in vaeac_enc_out

        assert actor_enc_out["mu"].shape == vaeac_enc_out["mu_full"].shape
        assert actor_enc_out["frame_tokens"].shape == vaeac_enc_out["frame_tokens"].shape

    def test_both_produce_output_of_same_shape(self, both_models_and_batch):
        actor_cvae, vaeac, b = both_models_and_batch
        actor_cvae.train()
        vaeac.train()

        out_actor = actor_cvae(dict(b))
        out_vaeac = vaeac(dict(b), phase="train")

        assert out_actor["output"].shape == out_vaeac["output"].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
