"""Quick LSTM-VAE SHAP-readiness diagnostic.

Evaluates a trained LstmVAELit checkpoint on the validation split and reports
the metrics that matter for manifold-constrained imputation:

  1. Full-encoder reconstruction fidelity (MPJPE).
  2. Temporal motion ratio:  recon temporal-std / gt temporal-std.
     1.0 => recon moves like GT.
     << 1 => mean-pose collapse.
     > 1 => recon jitters more than GT.
  3. Masked-encoder imputation MPJPE on HELD-OUT joints (the SHAP metric).
  4. Completion diversity across K GMM samples (pairwise RMS distance).
  5. KL(q_full || r_masked) at inference.

Spatial coalitions evaluated: `--mask_sizes` (default: mask 1, 4, 8, 12 joints,
which covers the KernelSHAP boundary and interior). For each size we sample
`--n_trials` random coalitions per batch.

Usage (from repo root, inside the manifoldshap conda env):

    CUDA_VISIBLE_DEVICES=6 python scripts/lstm_vae_shap_readiness.py \
        --ckpt experiment_outs/lstm_vae/BMCLab_fold1/checkpoints/epoch492-masked_obs0.0378.ckpt \
        --dataset BMCLab --fold 1 --num_folds 23
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


def _per_frame_speed(x_bt_jf: torch.Tensor) -> torch.Tensor:
    """Mean ||x[t]-x[t-1]||_2 over joints and frames. x: (B, T, J*3)."""
    B, T, D = x_bt_jf.shape
    J = D // 3
    x = x_bt_jf.view(B, T, J, 3)
    d = (x[:, 1:] - x[:, :-1]).norm(dim=-1)  # (B, T-1, J)
    return d.mean()


def _temporal_std(x_bt_jf: torch.Tensor) -> torch.Tensor:
    """Mean per-feature temporal std. x: (B, T, D)."""
    return x_bt_jf.std(dim=1).mean()


def _sample_spatial_coalitions(
    B: int, J: int, n_observed: int, device: torch.device, rng: np.random.Generator,
) -> torch.Tensor:
    """Return (B, J) bool coalition mask with exactly `n_observed` True entries / row."""
    cm = np.zeros((B, J), dtype=bool)
    for b in range(B):
        idx = rng.choice(J, size=n_observed, replace=False)
        cm[b, idx] = True
    return torch.from_numpy(cm).to(device)


@torch.no_grad()
def evaluate(
    lit: LstmVAELit,
    loader,
    device: torch.device,
    mask_sizes: list[int],
    n_completion_samples: int,
    n_coalition_trials: int,
    seed: int,
) -> dict:
    lit.eval().to(device)
    model = lit.model
    rng = np.random.default_rng(seed)

    full_mpjpe = []
    recon_speed = []
    gt_speed = []
    recon_tstd = []
    gt_tstd = []

    # Masked metrics indexed by n_observed
    held_out_mpjpe: dict[int, list[float]] = {n: [] for n in mask_sizes}
    observed_mpjpe: dict[int, list[float]] = {n: [] for n in mask_sizes}
    completion_div: dict[int, list[float]] = {n: [] for n in mask_sizes}
    kl_q_r: dict[int, list[float]] = {n: [] for n in mask_sizes}

    for x in loader:
        x = x.to(device)                       # (B, T, 51)
        B, T, D = x.shape
        J = D // 3

        recon, mu_full, logvar_full = model(x)

        full_mpjpe.append(
            torch.norm(recon.view(B, T, J, 3) - x.view(B, T, J, 3), dim=-1).mean().item()
        )
        gt_speed.append(_per_frame_speed(x).item())
        recon_speed.append(_per_frame_speed(recon).item())
        gt_tstd.append(_temporal_std(x).item())
        recon_tstd.append(_temporal_std(recon).item())

        if model.masked_encoder is None:
            continue

        for n_obs in mask_sizes:
            for _ in range(n_coalition_trials):
                cm = _sample_spatial_coalitions(B, J, n_obs, device, rng)  # (B, J)

                log_pi, mu_mix, logvar_mix = model.forward_masked(x, cm)

                # KL(q_full || r_masked) via MC, low n_samples since cheap.
                from model.lstm_vae.model import kl_full_vs_mixture
                kl = kl_full_vs_mixture(
                    mu_full, logvar_full, log_pi, mu_mix, logvar_mix,
                    n_mc_samples=5,
                ).item()
                kl_q_r[n_obs].append(kl)

                # n_completion_samples draws from the mixture
                log_pi_r = log_pi.repeat_interleave(n_completion_samples, dim=0)
                mu_r = mu_mix.repeat_interleave(n_completion_samples, dim=0)
                logvar_r = logvar_mix.repeat_interleave(n_completion_samples, dim=0)
                z_all = model.masked_encoder.sample(log_pi_r, mu_r, logvar_r)
                rec_all = model.decode(z_all, seq_len=T)      # (B*K, T, 51)

                rec_all = rec_all.view(B, n_completion_samples, T, J, 3)
                gt = x.view(B, 1, T, J, 3).expand_as(rec_all)
                per_joint = (rec_all - gt).norm(dim=-1)       # (B, K, T, J)

                obs_bjt = cm[:, None, None, :].expand(B, n_completion_samples, T, J)
                held = ~obs_bjt
                if held.any():
                    held_out_mpjpe[n_obs].append(per_joint[held].mean().item())
                if obs_bjt.any():
                    observed_mpjpe[n_obs].append(per_joint[obs_bjt].mean().item())

                # Completion diversity: mean pairwise RMS across K samples
                # in pose space (T, J, 3). O(K^2) but K small.
                if n_completion_samples > 1:
                    K = n_completion_samples
                    diffs = []
                    for i in range(K):
                        for j in range(i + 1, K):
                            d = (rec_all[:, i] - rec_all[:, j]).pow(2).mean().sqrt().item()
                            diffs.append(d)
                    completion_div[n_obs].append(float(np.mean(diffs)))

    res = {
        "full_mpjpe_m": float(np.mean(full_mpjpe)),
        "recon_speed_m_per_frame": float(np.mean(recon_speed)),
        "gt_speed_m_per_frame": float(np.mean(gt_speed)),
        "motion_ratio_recon_over_gt": float(np.mean(recon_speed) / max(np.mean(gt_speed), 1e-9)),
        "recon_temporal_std_m": float(np.mean(recon_tstd)),
        "gt_temporal_std_m": float(np.mean(gt_tstd)),
        "tstd_ratio_recon_over_gt": float(np.mean(recon_tstd) / max(np.mean(gt_tstd), 1e-9)),
        "n_val_batches": len(recon_speed),
    }
    for n_obs in mask_sizes:
        key = f"n_observed={n_obs}"
        res[key] = {
            "held_out_mpjpe_m": (
                float(np.mean(held_out_mpjpe[n_obs])) if held_out_mpjpe[n_obs] else None
            ),
            "observed_mpjpe_m": (
                float(np.mean(observed_mpjpe[n_obs])) if observed_mpjpe[n_obs] else None
            ),
            "completion_diversity_m": (
                float(np.mean(completion_div[n_obs])) if completion_div[n_obs] else None
            ),
            "kl_q_full_vs_r_masked_nats": (
                float(np.mean(kl_q_r[n_obs])) if kl_q_r[n_obs] else None
            ),
        }
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="BMCLab")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--num_folds", type=int, default=23)
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mask_sizes", nargs="+", type=int, default=[1, 4, 8, 12, 16])
    p.add_argument("--n_completion_samples", type=int, default=10)
    p.add_argument("--n_coalition_trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Loading checkpoint {args.ckpt}")
    lit = LstmVAELit.load_from_checkpoint(args.ckpt, map_location=device)

    dm = PoseDataModule(
        dataset=args.dataset, fold=args.fold, num_folds=args.num_folds,
        seq_len=args.seq_len, batch_size=args.batch_size, num_workers=0,
    )
    dm.setup()
    loader = dm.val_dataloader()

    res = evaluate(
        lit, loader, device,
        mask_sizes=args.mask_sizes,
        n_completion_samples=args.n_completion_samples,
        n_coalition_trials=args.n_coalition_trials,
        seed=args.seed,
    )

    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
