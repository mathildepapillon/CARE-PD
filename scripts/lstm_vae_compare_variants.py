"""Compare decoder noise floor across LSTM-VAE variants.

Measures quantities that are defined for *any* LstmVAE checkpoint, with or
without a masked encoder:

  full_recon_mpjpe:        full-encoder reconstruction MPJPE (floor of
                           everything downstream — determines SHAP SNR)
  recon_motion_ratio:      mean frame-to-frame joint speed(recon) / GT
  recon_tstd_ratio:        mean temporal std(recon) / GT
  prior_decode_mpjpe:      MPJPE of z ~ N(0,I) decodes vs val clips
  prior_decode_speed:      motion of unconditional prior samples
  per_joint_mpjpe_cm:      per-joint MPJPE (largest joints reveal where
                           the decoder blurs most)
  intra_sample_std:        std across 32 z~N(mu, σ²) samples for one fixed
                           input — this is the decoder's *stochastic noise
                           floor* at a single coalition (not just training
                           randomness).

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/lstm_vae_compare_variants.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from train_lstm_vae import LstmVAELit, PoseDataModule

CHECKPOINTS = {
    # The other two variants (fixed, std200) are from a pre-commit bidirectional
    # decoder architecture and cannot load against the current code. Their
    # logged val/mpjpe on the same val split is worse anyway:
    #   fixed  (50ep, beta=0.1, lat=128, dec=2L, no mask): val/mpjpe = 6.12 cm
    #   std200 (50ep, beta=0.1, lat=128, dec=2L, std_w=200, no mask): 6.37 cm
    #   fold1 (500ep, beta=0.5, lat=256, dec=4L, n_mix=10): 3.77 cm   ← best
    # So the useful comparison is "what does the fold1 decoder smear look
    # like, joint by joint, and is its intra-posterior noise floor really
    # the reason within-coalition diversity is tiny?"
    "fold1":  "experiment_outs/lstm_vae/BMCLab_fold1/checkpoints/epoch492-masked_obs0.0378.ckpt",
}


def _btj3(x_btd: torch.Tensor) -> torch.Tensor:
    B, T, D = x_btd.shape
    return x_btd.view(B, T, D // 3, 3)


@torch.no_grad()
def evaluate(ckpt_path: str, val_clips: torch.Tensor, device: torch.device) -> dict:
    lit = LstmVAELit.load_from_checkpoint(ckpt_path, map_location=device).eval().to(device)
    model = lit.model

    N_val, T, D = val_clips.shape
    J = D // 3

    # ---- Full-encoder reconstruction ---------------------------------------
    recon_batches = []
    mu_batches = []
    lv_batches = []
    for i in range(0, N_val, 64):
        x = val_clips[i:i + 64]
        recon, mu, lv = model(x)
        recon_batches.append(recon)
        mu_batches.append(mu)
        lv_batches.append(lv)
    recon = torch.cat(recon_batches, 0)
    mu = torch.cat(mu_batches, 0)
    lv = torch.cat(lv_batches, 0)

    full_err = torch.norm(_btj3(recon) - _btj3(val_clips), dim=-1)  # (N, T, J)
    full_mpjpe = full_err.mean().item()
    per_joint = full_err.mean(dim=(0, 1)).cpu().numpy()            # (J,)

    # Motion ratios
    gt_speed = (_btj3(val_clips)[:, 1:] - _btj3(val_clips)[:, :-1]).norm(dim=-1).mean().item()
    rc_speed = (_btj3(recon)[:, 1:] - _btj3(recon)[:, :-1]).norm(dim=-1).mean().item()
    gt_tstd  = val_clips.std(dim=1).mean().item()
    rc_tstd  = recon.std(dim=1).mean().item()

    # ---- Prior decode (unconditional generation) ---------------------------
    n_prior = 200
    z_prior = torch.randn(n_prior, model.latent_dim, device=device)
    x_prior = model.decode(z_prior, seq_len=T)
    prior_speed = (_btj3(x_prior)[:, 1:] - _btj3(x_prior)[:, :-1]).norm(dim=-1).mean().item()
    # min-MPJPE for each val clip vs prior pool (i.e. best nearest prior sample)
    diffs = []
    for vi in range(0, N_val, 16):
        vb = _btj3(val_clips[vi:vi + 16])
        # (n_prior, 1, T, J, 3) - (1, b, T, J, 3) → (n_prior, b)
        d = (_btj3(x_prior)[:, None] - vb[None]).norm(dim=-1).mean(dim=(2, 3))
        diffs.append(d.min(dim=0).values)
    min_prior = torch.cat(diffs).mean().item()

    # ---- Intra-sample std (decoder stochastic noise floor) -----------------
    # For a fixed input, sample K z's from q(z|x) and measure pairwise RMS.
    # This is the scale of "noise" that contaminates completion diversity.
    K = 32
    n_inputs = 8
    rng = np.random.default_rng(0)
    sel = rng.integers(0, N_val, size=n_inputs)
    intra_stds = []
    for idx in sel:
        x = val_clips[idx:idx + 1].expand(K, T, D).contiguous()
        mu_i, lv_i = model.encode(val_clips[idx:idx + 1])
        mu_i = mu_i.expand(K, -1)
        lv_i = lv_i.expand(K, -1)
        std = torch.exp(0.5 * lv_i)
        z = mu_i + std * torch.randn_like(std)
        recs = model.decode(z, seq_len=T)                            # (K, T, D)
        # RMS across K of per-joint position, then average
        intra = recs.view(K, T, J, 3).std(dim=0).mean().item()
        intra_stds.append(intra)
    intra_std = float(np.mean(intra_stds))

    return {
        "full_mpjpe_cm": full_mpjpe * 100,
        "motion_ratio_recon_over_gt": rc_speed / gt_speed,
        "tstd_ratio_recon_over_gt":   rc_tstd / gt_tstd,
        "prior_nearest_neighbor_mpjpe_cm": min_prior * 100,
        "prior_decode_speed_ratio_over_gt": prior_speed / gt_speed,
        "intra_posterior_std_cm": intra_std * 100,
        "latent_mean_std_cm": mu.std(dim=0).mean().item(),
        "latent_logvar_mean": lv.mean().item(),
        "per_joint_mpjpe_cm": [float(x * 100) for x in per_joint],
        "has_masked_encoder": model.masked_encoder is not None,
    }


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = PoseDataModule(dataset="BMCLab", fold=1, num_folds=23,
                        seq_len=80, batch_size=64, num_workers=0)
    dm.setup()
    val_clips = torch.stack(
        [dm.val_dataset[i] for i in range(len(dm.val_dataset))], dim=0
    ).to(device)
    N, T, D = val_clips.shape
    J = D // 3

    results = {}
    for name, path in CHECKPOINTS.items():
        full_path = str(_REPO / path)
        print(f"[*] Evaluating {name} ← {path}")
        results[name] = evaluate(full_path, val_clips, device)

    # Summary table
    print()
    print(json.dumps(results, indent=2))
    print()

    print(f"{'metric':35s} " + " ".join(f"{k:>10s}" for k in results))
    keys_num = [
        "full_mpjpe_cm",
        "motion_ratio_recon_over_gt",
        "tstd_ratio_recon_over_gt",
        "prior_nearest_neighbor_mpjpe_cm",
        "prior_decode_speed_ratio_over_gt",
        "intra_posterior_std_cm",
    ]
    for k in keys_num:
        vals = " ".join(f"{results[c][k]:>10.3f}" for c in results)
        print(f"{k:35s} {vals}")


if __name__ == "__main__":
    main()
