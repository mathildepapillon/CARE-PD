"""Adapter that exposes an LSTM VAE masked encoder as an ActorSHAP-compatible
``sample_completions`` interface, plus **batched** spatial/temporal SHAP
functions that process all coalitions through the LSTM VAE in GPU-friendly
chunks — typically 100–1000x faster than the sequential per-coalition loop in
``compute_spatial_shap``.

Data format bridge
------------------
SHAP pipeline (Actor format):  ``(B, J=17, F=3, T)`` in global-pelvis coords
LSTM VAE format:                ``(B, T, 51)``  root-centred XYZ (17 joints × 3)

The wrapper **automatically root-centres** before the LSTM VAE (zeroes joint 0)
and **restores the original pelvis trajectory** after decoding.  This means the
evaluation script should load data in the default global-pelvis format
(``root_centered=False``) so the classifier sees properly positioned sequences.

Coalition masks are the same in both systems:
- Spatial: ``(B, 17)``  bool, ``True`` = observed
- Temporal: ``(B, T)``  bool, ``True`` = observed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
from torch import Tensor

from model.lstm_vae.model import LstmVAE, N_JOINTS, FEAT_PER_JOINT

_INPUT_DIM = N_JOINTS * FEAT_PER_JOINT  # 51
_PELVIS_DIM = FEAT_PER_JOINT  # first 3 elements of the 51-d flat vector


# -----------------------------------------------------------------------
# Format conversion helpers
# -----------------------------------------------------------------------

def _actor_to_lstm(x: Tensor) -> Tensor:
    """``(B, J, F, T) -> (B, T, J*F)``."""
    return x.permute(0, 3, 1, 2).reshape(x.shape[0], x.shape[3], -1)


def _lstm_to_actor(x: Tensor) -> Tensor:
    """``(B, T, J*F) -> (B, J, F, T)``."""
    B, T, _ = x.shape
    return x.reshape(B, T, N_JOINTS, FEAT_PER_JOINT).permute(0, 2, 3, 1)


def _root_centre(x_lstm: Tensor) -> tuple[Tensor, Tensor]:
    """Zero out joint 0 (pelvis) for LSTM VAE input, return (centred, pelvis).

    In global-pelvis format joints 1-16 are already relative to the pelvis,
    so only the first 3 elements (joint 0) differ from root-centred.
    """
    pelvis = x_lstm[:, :, :_PELVIS_DIM].clone()  # (B, T, 3)
    x_rc = x_lstm.clone()
    x_rc[:, :, :_PELVIS_DIM] = 0.0
    return x_rc, pelvis


def _restore_pelvis(x_lstm: Tensor, pelvis: Tensor) -> Tensor:
    """Put the original global pelvis trajectory back into flat LSTM output."""
    x_lstm = x_lstm.clone()
    x_lstm[:, :, :_PELVIS_DIM] = pelvis
    return x_lstm


def _paste_observed_spatial(
    gt: Tensor,
    recon: Tensor,
    coalition_mask: Tensor,
) -> Tensor:
    """Paste observed joints from *gt* onto *recon*.

    ``gt``, ``recon``: ``(B, T, J*F)``
    ``coalition_mask``: ``(B, J)`` bool, True = observed.
    """
    B, T, D = gt.shape
    gt_4d = gt.reshape(B, T, N_JOINTS, FEAT_PER_JOINT)
    rc_4d = recon.reshape(B, T, N_JOINTS, FEAT_PER_JOINT)
    obs = coalition_mask[:, None, :, None].expand_as(gt_4d)
    return torch.where(obs, gt_4d, rc_4d).reshape(B, T, D)


def _paste_observed_temporal(
    gt: Tensor,
    recon: Tensor,
    coalition_mask: Tensor,
) -> Tensor:
    """Paste observed frames from *gt* onto *recon*.

    ``gt``, ``recon``: ``(B, T, J*F)``
    ``coalition_mask``: ``(B, T)`` bool, True = observed.
    """
    obs = coalition_mask[:, :, None].expand_as(gt)
    return torch.where(obs, gt, recon)


def _paste_observed(
    gt: Tensor,
    recon: Tensor,
    coalition_mask: Tensor,
) -> Tensor:
    """Auto-dispatch to spatial or temporal paste based on mask shape."""
    if coalition_mask.shape[-1] == N_JOINTS:
        return _paste_observed_spatial(gt, recon, coalition_mask)
    return _paste_observed_temporal(gt, recon, coalition_mask)


# -----------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------

class LstmVaeShapWrapper:
    """Wraps a trained :class:`LstmVAE` for use with the Actor-SHAP pipeline.

    Provides two interfaces:

    1.  ``sample_completions`` — drop-in replacement for
        ``ActorSHAP.sample_completions``.  All n_samples are decoded in one
        batched GPU call.

    2.  ``compute_spatial_shap_batched`` / ``compute_temporal_shap_batched`` —
        process *all* coalitions in chunked GPU passes, avoiding the sequential
        Python loop in ``compute_spatial_shap``.  **~100x faster** at default
        settings (3000 kernel samples, 20 completions).
    """

    def __init__(self, model: LstmVAE, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        if self.model.masked_encoder is None:
            raise RuntimeError(
                "LstmVAE checkpoint has no masked encoder (n_mix=0). "
                "Retrain with n_mix > 0."
            )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        device: torch.device,
    ) -> "LstmVaeShapWrapper":
        """Load an ``LstmVAELit`` Lightning checkpoint and extract the model."""
        from train_lstm_vae import LstmVAELit

        lit = LstmVAELit.load_from_checkpoint(
            str(ckpt_path), map_location=device,
        )
        return cls(lit.model, device)

    # -------------------------------------------------------------------
    # ActorSHAP-compatible interface (for faithfulness metrics, etc.)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def sample_completions(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        lengths: Tensor,
        coalition_mask: Tensor,
        n_samples: int = 20,
        paste_observed: bool = True,
    ) -> list[Tensor]:
        """Generate ``n_samples`` completions — batched decode, one encode.

        Input ``x`` is expected in **global-pelvis** format ``(1, J, F, T)``.
        The pelvis is internally stripped before encoding and restored after
        decoding so the LSTM VAE always sees root-centred data.

        Returns a list of ``n_samples`` tensors, each ``(1, J, F, T)`` in
        global-pelvis format.
        """
        x_lstm = _actor_to_lstm(x)  # (1, T, 51)
        T = x_lstm.size(1)
        cm = coalition_mask.to(self.device)

        x_rc, pelvis = _root_centre(x_lstm)

        log_pi, mu, logvar = self.model.forward_masked(x_rc, cm)

        log_pi_n = log_pi.expand(n_samples, -1)
        mu_n = mu.expand(n_samples, -1, -1)
        logvar_n = logvar.expand(n_samples, -1, -1)
        z_all = self.model.masked_encoder.sample(log_pi_n, mu_n, logvar_n)
        recon_all = self.model.decode(z_all, seq_len=T)  # (N, T, 51)

        pelvis_exp = pelvis.expand(n_samples, -1, -1)
        recon_all = _restore_pelvis(recon_all, pelvis_exp)

        if paste_observed:
            gt_exp = x_lstm.expand(n_samples, -1, -1)
            cm_exp = cm.expand(n_samples, -1) if cm.shape[0] == 1 else cm
            recon_all = _paste_observed(gt_exp, recon_all, cm_exp)

        actor_all = _lstm_to_actor(recon_all)  # (N, J, F, T)
        return [actor_all[i : i + 1] for i in range(n_samples)]

    # -------------------------------------------------------------------
    # Batched KernelSHAP — spatial
    # -------------------------------------------------------------------

    @torch.no_grad()
    def compute_spatial_shap_batched(
        self,
        classifier: Callable,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        lengths: Tensor,
        n_kernel_samples: int = 3000,
        n_completion_samples: int = 20,
        seed: int | None = None,
        coal_chunk: int = 64,
        cls_chunk: int = 256,
    ) -> dict[str, float]:
        """KernelSHAP over 17 joints — all coalitions evaluated in GPU chunks.

        Instead of 6002 sequential ``sample_completions`` calls, this
        processes ``coal_chunk`` coalitions at a time through the masked
        encoder, samples ``n_completion_samples`` z per coalition, decodes
        them all in one pass, pastes observed joints, classifies in
        ``cls_chunk``-sized GPU passes, and averages.

        Returns the same dict format as ``compute_spatial_shap``.
        """
        from model.actor.shap_compute import (
            _sample_kernel_coalitions,
            _solve_shapley_wls,
            _classify_chunked,
        )
        from model.actor.shap_masking import H36M_GROUPS, H36M_JOINT_NAMES

        M = 17
        N_s = n_completion_samples
        device = x.device
        rng = np.random.default_rng(seed)
        class_idx = int(y[0].item())

        coalitions, weights = _sample_kernel_coalitions(M, n_kernel_samples, rng)

        # (N_total, M) bool tensor of all coalition masks including boundaries.
        cms_np = np.vstack([
            np.zeros((1, M), dtype=bool),
            np.ones((1, M), dtype=bool),
            np.array(coalitions, dtype=bool),
        ])
        N_total = cms_np.shape[0]
        all_cms = torch.tensor(cms_np, dtype=torch.bool, device=device)

        x_lstm = _actor_to_lstm(x)  # (1, T, 51)
        T = x_lstm.size(1)

        x_rc, pelvis = _root_centre(x_lstm)

        all_values = np.empty(N_total, dtype=np.float64)

        for c0 in range(0, N_total, coal_chunk):
            c1 = min(c0 + coal_chunk, N_total)
            C = c1 - c0
            cm_batch = all_cms[c0:c1]  # (C, M) bool

            x_batch = x_rc.expand(C, -1, -1)

            log_pi, mu, logvar = self.model.forward_masked(x_batch, cm_batch)

            log_pi_r = log_pi.repeat_interleave(N_s, dim=0)
            mu_r = mu.repeat_interleave(N_s, dim=0)
            logvar_r = logvar.repeat_interleave(N_s, dim=0)

            z_all = self.model.masked_encoder.sample(log_pi_r, mu_r, logvar_r)
            recon = self.model.decode(z_all, seq_len=T)  # (C*N_s, T, 51)

            pelvis_exp = pelvis.expand(C * N_s, -1, -1)
            recon = _restore_pelvis(recon, pelvis_exp)

            gt_exp = x_lstm.expand(C * N_s, -1, -1)
            cm_paste = cm_batch.repeat_interleave(N_s, dim=0)
            pasted = _paste_observed_spatial(gt_exp, recon, cm_paste)

            actor_batch = _lstm_to_actor(pasted)
            probs = _classify_chunked(
                classifier, actor_batch, class_idx, chunk_size=cls_chunk,
            )

            per_coal = probs.reshape(C, N_s).mean(axis=1)
            all_values[c0:c1] = per_coal

        v_empty = float(all_values[0])
        v_full = float(all_values[1])
        values = all_values[2:]

        phi = _solve_shapley_wls(
            coalitions, values, weights, v_empty=v_empty, v_full=v_full,
        )
        result: dict[str, float] = {
            H36M_JOINT_NAMES[j]: float(phi[j]) for j in range(M)
        }
        for group_name, joint_indices in H36M_GROUPS.items():
            result[group_name] = float(sum(phi[j] for j in joint_indices))
        result["_v_empty"] = v_empty
        result["_v_full"] = v_full
        return result

    # -------------------------------------------------------------------
    # Batched exact SHAP — temporal (K=4 windows)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def compute_temporal_shap_batched(
        self,
        classifier: Callable,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        lengths: Tensor,
        window_assignments: list[list[int]] | None = None,
        n_completion_samples: int = 20,
        fps: int = 30,
        cls_chunk: int = 256,
    ) -> dict[str, float]:
        """Exact temporal SHAP (K=4 windows) — all 16 coalitions in one pass.

        Returns the same dict format as ``compute_temporal_shap``.
        """
        from model.actor.shap_compute import (
            _enumerate_all_coalitions,
            _solve_shapley_wls,
            _classify_chunked,
        )
        from model.actor.shap_masking import (
            build_temporal_shap_mask,
            build_temporal_windows,
            detect_stride_period,
        )

        K = 4
        T = x.shape[-1]
        device = x.device
        N_s = n_completion_samples
        PHASE_LABELS = [
            "IC_loading", "midstance_terminal",
            "preswing_initswing", "midswing_terminal",
        ]

        if window_assignments is None:
            x_np = x[0].permute(2, 0, 1).cpu().numpy()
            stride_period, fallback = detect_stride_period(x_np, fps=fps)
            window_assignments = build_temporal_windows(T, stride_period, K=K)
            window_names = (
                [f"window_{k}" for k in range(K)] if fallback else PHASE_LABELS
            )
        else:
            window_names = PHASE_LABELS

        coalitions, weights = _enumerate_all_coalitions(K)
        class_idx = int(y[0].item())

        # Build temporal masks for all 16 coalitions: (16, T) bool
        all_cms = torch.stack([
            build_temporal_shap_mask(
                [k for k in range(K) if z[k] == 1],
                window_assignments, T, device,
            )
            for z in coalitions
        ], dim=0)  # (16, T)

        x_lstm = _actor_to_lstm(x)  # (1, T, 51)
        C = all_cms.shape[0]  # 16

        x_rc, pelvis = _root_centre(x_lstm)

        x_batch = x_rc.expand(C, -1, -1)
        log_pi, mu, logvar = self.model.forward_masked(x_batch, all_cms)

        log_pi_r = log_pi.repeat_interleave(N_s, dim=0)
        mu_r = mu.repeat_interleave(N_s, dim=0)
        logvar_r = logvar.repeat_interleave(N_s, dim=0)

        z_all = self.model.masked_encoder.sample(log_pi_r, mu_r, logvar_r)
        recon = self.model.decode(z_all, seq_len=T)  # (C*N_s, T, 51)

        pelvis_exp = pelvis.expand(C * N_s, -1, -1)
        recon = _restore_pelvis(recon, pelvis_exp)

        gt_exp = x_lstm.expand(C * N_s, -1, -1)
        cm_paste = all_cms.repeat_interleave(N_s, dim=0)
        pasted = _paste_observed_temporal(gt_exp, recon, cm_paste)

        actor_batch = _lstm_to_actor(pasted)
        probs = _classify_chunked(
            classifier, actor_batch, class_idx, chunk_size=cls_chunk,
        )
        values = probs.reshape(C, N_s).mean(axis=1)

        v_empty = float(values[0])
        v_full = float(values[-1])

        phi = _solve_shapley_wls(
            coalitions, values.astype(np.float64), weights,
            v_empty=v_empty, v_full=v_full,
        )
        result = {window_names[k]: float(phi[k]) for k in range(K)}
        result["_v_empty"] = v_empty
        result["_v_full"] = v_full
        return result
