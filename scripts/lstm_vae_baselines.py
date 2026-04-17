"""LSTM-VAE SHAP-readiness: proper baselines + ablations.

Tests the user's concern: is the masked encoder doing real conditional
imputation, or just outputting "average gait"?

Baselines (all evaluated on BMCLab fold 1 val, 80-frame root-centred clips):

  1. random_pair:        MPJPE between two RANDOM val clips.
                         If imputation doesn't beat this, the model isn't
                         conditioning on x at all.
  2. mean_pose:          MPJPE of the training-set mean pose (broadcast over T)
                         vs each val clip.  Ceiling for "output constant pose".
  3. prior_sample:       Decode z ~ N(0, I).  Compare against each val clip.
                         What the masked encoder's completions should look like
                         if it had ZERO observation.
  4. held_out_imputation: Condition on n_obs joints, impute the rest.
                         What the model actually does.

Sensitivity / mode-activation checks:

  A. Does changing WHICH joint is observed change the output?
     sensitivity_to_conditioning = mean||completion(x, obs={j}) - completion(x, obs={j'})||
  B. Does changing the INPUT sequence (same coalition) change the output?
     sensitivity_to_input = mean||completion(x1, obs=S) - completion(x2, obs=S)||
  C. GMM mode activation: effective-number-of-components from mixture weights.

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/lstm_vae_baselines.py \
      --ckpt experiment_outs/lstm_vae/BMCLab_fold1/checkpoints/epoch492-masked_obs0.0378.ckpt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from train_lstm_vae import LstmVAELit, PoseDataModule, N_JOINTS


def _mpjpe(pred_btj3: torch.Tensor, gt_btj3: torch.Tensor) -> float:
    return torch.norm(pred_btj3 - gt_btj3, dim=-1).mean().item()


def _to_btj3(x_btd: torch.Tensor) -> torch.Tensor:
    B, T, D = x_btd.shape
    return x_btd.view(B, T, D // 3, 3)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="BMCLab")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--num_folds", type=int, default=23)
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_completions", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[*] Loading {args.ckpt}")
    lit = LstmVAELit.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    model = lit.model
    assert model.masked_encoder is not None

    dm = PoseDataModule(
        dataset=args.dataset, fold=args.fold, num_folds=args.num_folds,
        seq_len=args.seq_len, batch_size=args.batch_size, num_workers=0,
    )
    dm.setup()

    # ---- Collect all val clips into a single tensor (this dataset is small). ----
    val_clips = torch.stack(
        [dm.val_dataset[i] for i in range(len(dm.val_dataset))], dim=0
    ).to(device)                                                # (N_val, T, 51)
    N_val, T, D = val_clips.shape
    J = D // 3

    # Train pool for training-set mean.
    train_clips = torch.stack(
        [dm.train_dataset[i] for i in range(len(dm.train_dataset))], dim=0
    ).to(device)
    train_mean_pose = train_clips.view(-1, J, 3).mean(dim=0)     # (J, 3)

    # Also report mean frame-to-frame speed of GT.
    gt_speed = (val_clips.view(N_val, T, J, 3)[:, 1:] -
                val_clips.view(N_val, T, J, 3)[:, :-1]).norm(dim=-1).mean().item()
    gt_tstd  = val_clips.std(dim=1).mean().item()

    # =====================================================================
    # Baseline 1: MPJPE between two random val clips.
    # =====================================================================
    n_pairs = 500
    idx_a = rng.integers(0, N_val, size=n_pairs)
    idx_b = rng.integers(0, N_val, size=n_pairs)
    pair = []
    for a, b in zip(idx_a, idx_b):
        if a == b:
            continue
        pair.append(
            torch.norm(
                _to_btj3(val_clips[a:a + 1]) - _to_btj3(val_clips[b:b + 1]),
                dim=-1,
            ).mean().item()
        )
    baseline_random_pair = float(np.mean(pair))

    # =====================================================================
    # Baseline 2: training-mean pose (constant over T) vs val.
    # =====================================================================
    mean_seq = train_mean_pose[None, None].expand(1, T, J, 3)   # (1, T, J, 3)
    mean_mpjpe = torch.norm(
        _to_btj3(val_clips) - mean_seq.expand(N_val, T, J, 3),
        dim=-1,
    ).mean().item()

    # =====================================================================
    # Baseline 3: decode from prior z ~ N(0, I).  Unconditional generation.
    # =====================================================================
    n_prior = 200
    z_prior = torch.randn(n_prior, model.latent_dim, device=device)
    x_prior = model.decode(z_prior, seq_len=T)                   # (n_prior, T, 51)
    prior_vs_val = []
    for j in rng.integers(0, N_val, size=100):
        # for each val clip, compute min MPJPE over prior samples (best match)
        diffs = torch.norm(
            _to_btj3(x_prior) - _to_btj3(val_clips[j:j + 1]).expand_as(_to_btj3(x_prior)),
            dim=-1,
        ).mean(dim=(1, 2))                                       # (n_prior,)
        prior_vs_val.append(diffs.mean().item())
    prior_mean = float(np.mean(prior_vs_val))

    # Prior-sample motion ratio
    prior_speed = (
        _to_btj3(x_prior)[:, 1:] - _to_btj3(x_prior)[:, :-1]
    ).norm(dim=-1).mean().item()

    # =====================================================================
    # Sensitivity checks + imputation MPJPE at varying n_obs.
    # =====================================================================
    results_by_nobs: dict[int, dict] = {}
    for n_obs in [0, 1, 4, 8, 16]:
        held_out = []
        observed_pres = []
        div = []
        # Sensitivity-to-conditioning: same x, DIFFERENT joint observed.
        sens_cond = []
        # Sensitivity-to-input: same coalition, DIFFERENT x.
        sens_input = []
        # GMM effective-number-of-components
        eff_k = []
        # KL vs full encoder
        kls = []

        for trial in range(3):
            # Pick random indices of val clips to use this trial.
            idx = rng.integers(0, N_val, size=min(32, N_val))
            x = val_clips[idx]                                   # (B, T, 51)
            B = x.size(0)

            if n_obs == 0:
                # For n_obs=0 we don't call forward_masked — emulate it by
                # sampling from prior directly (no conditioning at all).
                z = torch.randn(
                    B * args.n_completions, model.latent_dim, device=device,
                )
                recs = model.decode(z, seq_len=T).view(
                    B, args.n_completions, T, J, 3,
                )
                # Compute MPJPE vs GT across samples (for context).
                gt = _to_btj3(x)[:, None].expand_as(recs)
                held_out.append(torch.norm(recs - gt, dim=-1).mean().item())
                observed_pres.append(float("nan"))
            else:
                cm = torch.zeros(B, J, dtype=torch.bool, device=device)
                for b in range(B):
                    which = rng.choice(J, size=n_obs, replace=False)
                    cm[b, which] = True

                log_pi, mu_mix, logvar_mix = model.forward_masked(x, cm)

                # GMM effective K
                pi = log_pi.exp()
                eff_k_val = torch.exp(-(pi * log_pi).sum(dim=-1)).mean().item()
                eff_k.append(eff_k_val)

                # KL vs full encoder
                _, mu_full, lv_full = model(x)
                from model.lstm_vae.model import kl_full_vs_mixture
                kls.append(
                    kl_full_vs_mixture(
                        mu_full, lv_full, log_pi, mu_mix, logvar_mix,
                        n_mc_samples=5,
                    ).item()
                )

                # Draw K completions
                pi_r = log_pi.repeat_interleave(args.n_completions, dim=0)
                mu_r = mu_mix.repeat_interleave(args.n_completions, dim=0)
                lv_r = logvar_mix.repeat_interleave(args.n_completions, dim=0)
                z = model.masked_encoder.sample(pi_r, mu_r, lv_r)
                recs = model.decode(z, seq_len=T).view(
                    B, args.n_completions, T, J, 3,
                )
                gt = _to_btj3(x)[:, None].expand_as(recs)
                err = torch.norm(recs - gt, dim=-1)                # (B, K, T, J)
                held = ~cm[:, None, None, :].expand_as(err)
                obs  =  cm[:, None, None, :].expand_as(err)
                if held.any():
                    held_out.append(err[held].mean().item())
                if obs.any():
                    observed_pres.append(err[obs].mean().item())

            # Completion diversity
            if args.n_completions > 1 and n_obs > 0:
                # mean pairwise MPJPE (joint-wise L2 between pairs of samples)
                pair_diffs = []
                for i in range(args.n_completions):
                    for k in range(i + 1, args.n_completions):
                        pair_diffs.append(
                            torch.norm(recs[:, i] - recs[:, k], dim=-1).mean().item()
                        )
                div.append(float(np.mean(pair_diffs)))

            # Sensitivity to conditioning: same x, different single joint
            if n_obs == 1:
                cm_a = torch.zeros(B, J, dtype=torch.bool, device=device)
                cm_a[:, 0] = True  # pelvis observed
                cm_b = torch.zeros(B, J, dtype=torch.bool, device=device)
                cm_b[:, 10] = True  # right ankle (index ~10 depends on skeleton)
                _, mu_a, lv_a = model.forward_masked(x, cm_a)
                _, mu_b, lv_b = model.forward_masked(x, cm_b)
                # Use mixture mean per-batch as z.
                # Sample K from each, compute cross-pair MPJPE.
                K = args.n_completions
                pi_a, _, _ = model.forward_masked(x, cm_a)
                z_a = model.masked_encoder.sample(
                    pi_a.repeat_interleave(K, dim=0),
                    mu_a.repeat_interleave(K, dim=0),
                    lv_a.repeat_interleave(K, dim=0),
                )
                rec_a = model.decode(z_a, seq_len=T).view(B, K, T, J, 3)
                pi_b, _, _ = model.forward_masked(x, cm_b)
                z_b = model.masked_encoder.sample(
                    pi_b.repeat_interleave(K, dim=0),
                    mu_b.repeat_interleave(K, dim=0),
                    lv_b.repeat_interleave(K, dim=0),
                )
                rec_b = model.decode(z_b, seq_len=T).view(B, K, T, J, 3)
                sens_cond.append(
                    torch.norm(rec_a - rec_b, dim=-1).mean().item()
                )

            # Sensitivity to input: SAME coalition S (random), DIFFERENT x
            # Shuffle x → x2, encode both with same S, compare first sample.
            if n_obs > 0:
                perm = torch.randperm(B, device=device)
                x2 = x[perm]
                pi1, mu1, lv1 = model.forward_masked(x, cm)
                pi2, mu2, lv2 = model.forward_masked(x2, cm)
                z1 = model.masked_encoder.sample(pi1, mu1, lv1)
                z2 = model.masked_encoder.sample(pi2, mu2, lv2)
                r1 = model.decode(z1, seq_len=T).view(B, T, J, 3)
                r2 = model.decode(z2, seq_len=T).view(B, T, J, 3)
                sens_input.append(torch.norm(r1 - r2, dim=-1).mean().item())

        results_by_nobs[n_obs] = {
            "held_out_mpjpe_m": float(np.mean(held_out)) if held_out else None,
            "observed_mpjpe_m": float(np.nanmean(observed_pres)) if observed_pres else None,
            "completion_diversity_m": float(np.mean(div)) if div else None,
            "sensitivity_to_conditioning_m": float(np.mean(sens_cond)) if sens_cond else None,
            "sensitivity_to_input_m": float(np.mean(sens_input)) if sens_input else None,
            "gmm_effective_components": float(np.mean(eff_k)) if eff_k else None,
            "kl_q_full_r_masked_nats": float(np.mean(kls)) if kls else None,
        }

    # =====================================================================
    # Report
    # =====================================================================
    import json
    report = {
        "data": {
            "n_val_clips": int(N_val),
            "seq_len_frames": int(T),
            "gt_mean_frame_speed_m": gt_speed,
            "gt_mean_temporal_std_m": gt_tstd,
        },
        "baselines": {
            "random_pair_mpjpe_m": baseline_random_pair,
            "mean_pose_mpjpe_m":   mean_mpjpe,
            "prior_decode_mpjpe_m": prior_mean,
            "prior_decode_speed_m_per_frame": prior_speed,
        },
        "imputation_vs_n_observed": results_by_nobs,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
