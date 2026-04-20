#!/usr/bin/env python3
"""evaluate_vaeac_conditional_vs_oracle.py — **direct** check that VAEAC’s
conditional completions approximate the **exact Gaussian conditional**
:p(x_hid | x_obs) from :class:`GaussianMotionBenchmark`.

This answers “are the conditionals correct?” more directly than EC1/EC2,
which mix in the classifier and KernelSHAP.

For random test sequences and random **non-degenerate** coalitions (both
observed and hidden players non-empty), we:

1. Draw ``n_oracle`` i.i.d. samples from ``benchmark.conditional_sample*``.
2. Draw ``n_vaeac`` completions from :class:`VAEACImputer.sample_completions`
   with the **same** coalition mask as the flow imputer uses.
3. On **hidden** (j, f, t) coordinates only, report:

   * **mse_conditional_mean** — :math:`\\frac{1}{|\\mathcal{H}|}\\sum_{h\\in\\mathcal{H}} (\\bar{x}^{\\mathrm{oracle}}_h - \\bar{x}^{\\mathrm{vaeac}}_h)^2`  between *sample means* (both are Monte-Carlo estimates of the true conditional mean).
   * **mean_rel_mse** — same MSE divided by the empirical variance of oracle samples on each hidden coordinate (dimensionless scale).
   * **mse_conditional_var** — mean squared difference between **per-coordinate sample variances** (oracle vs VAEAC), a coarse check of second-order match.

**Interpretation:** If ``mse_conditional_mean`` is small relative to the signal scale
and ``mean_rel_mse`` ≪ 1, the amortised model matches the **first moment** of the
true conditional. Variance gaps flag wrong uncertainty; large mean gaps flag
wrong imputations regardless of latent-space “interpretability”.

Usage::

    python scripts/evaluate_vaeac_conditional_vs_oracle.py \\
        --ckpt_dir experiment_outs/actor_shap_synthetic/synthetic_gaussian_k4 \\
        --vaeac_ckpt_dir experiment_outs/vaeac_synthetic/gaussian_k4 \\
        --vaeac_config configs/vaeac/synthetic_gaussian.json \\
        --n_sequences 50 --n_masks_per_seq 8 \\
        --n_oracle_samples 2000 --n_vaeac_samples 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic.gaussian_motion import GaussianMotionBenchmark  # noqa: E402


def _split_temporal(z: np.ndarray, K: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(k for k in range(K) if z[k] == 1),
        tuple(k for k in range(K) if z[k] == 0),
    )


def _split_spatial(z: np.ndarray, J: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(j for j in range(J) if z[j] == 1),
        tuple(j for j in range(J) if z[j] == 0),
    )


def _hidden_mask_jft(
    bench: GaussianMotionBenchmark,
    z_bool: np.ndarray,
) -> np.ndarray:
    """Boolean (J, F, T), True on coordinates VAEAC must fill in."""
    J, F, T = bench.J, bench.F, bench.T
    if bench.player_mode == "temporal":
        hid = np.zeros((J, F, T), dtype=bool)
        s_obs, s_hid = _split_temporal(z_bool, bench.K)
        for k in s_hid:
            hid[:, :, bench.window_assignments[k]] = True
        return hid
    hid = np.zeros((J, F, T), dtype=bool)
    _, j_hid = _split_spatial(z_bool, J)
    for j in j_hid:
        hid[j, :, :] = True
    return hid


def _coalition_to_flow_mask(
    z_bool: np.ndarray,
    bench: GaussianMotionBenchmark,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    if bench.player_mode == "temporal":
        t_mask = np.zeros(T, dtype=bool)
        for k in range(bench.K):
            if z_bool[k]:
                t_mask[bench.window_assignments[k]] = True
        return torch.from_numpy(t_mask[None]).to(device)
    return torch.from_numpy(z_bool[None, :, None]).to(device)


def _random_partial_coalition(rng: np.random.Generator, M: int) -> np.ndarray:
    """Bit vector with at least one 0 and one 1."""
    for _ in range(1000):
        z = rng.integers(0, 2, size=M, dtype=np.int64).astype(bool)
        if z.any() and (not z.all()):
            return z
    raise RuntimeError("failed to sample partial coalition")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="Dir with synthetic_benchmark.pkl + synthetic_test.pt")
    p.add_argument("--imputer", choices=["vaeac", "flow"], default="vaeac",
                   help="Which imputer to evaluate against the Gaussian oracle.")
    p.add_argument("--vaeac_ckpt_dir", default=None)
    p.add_argument("--vaeac_config", default=None)
    p.add_argument("--vaeac_temperature", type=float, default=1.0)
    p.add_argument("--flow_ckpt_dir", default=None,
                   help="Flow checkpoint dir (required when --imputer=flow).")
    p.add_argument("--flow_config", default=None,
                   help="Flow config path (required when --imputer=flow).")
    p.add_argument("--flow_num_steps", type=int, default=50)
    p.add_argument("--flow_solver", default="midpoint")
    p.add_argument("--n_sequences", type=int, default=20)
    p.add_argument("--n_masks_per_seq", type=int, default=5)
    p.add_argument("--n_oracle_samples", type=int, default=1000)
    p.add_argument("--n_vaeac_samples", type=int, default=1000,
                   help="Number of samples from the VAEAC / flow imputer.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_path", default=None,
                   help="Optional JSON output path (defaults to <ckpt_dir>/conditional_vs_oracle.json).")
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    with open(ckpt_dir / "synthetic_benchmark.pkl", "rb") as f:
        bench: GaussianMotionBenchmark = pickle.load(f)
    if not hasattr(bench, "player_mode"):
        bench.player_mode = "temporal"

    test_data = torch.load(ckpt_dir / "synthetic_test.pt", map_location="cpu")
    x_test = test_data["x"].permute(0, 2, 3, 1).numpy().astype(np.float32)  # (N, J, F, T)
    N = min(args.n_sequences, x_test.shape[0])

    _path = PROJECT_ROOT / "scripts" / "evaluate_shap_synthetic_gaussian.py"
    spec = importlib.util.spec_from_file_location("_eval_shap_syn", _path)
    _mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(_mod)

    if args.imputer == "vaeac":
        if not (args.vaeac_ckpt_dir and args.vaeac_config):
            raise ValueError("--imputer=vaeac requires --vaeac_ckpt_dir and --vaeac_config.")
        imputer = _mod.load_vaeac_imputer(
            args.vaeac_ckpt_dir, args.vaeac_config, device,
            temperature=args.vaeac_temperature, use_ema=False,
        )
        imputer_ckpt_dir = args.vaeac_ckpt_dir
        imputer_label    = "vaeac"
    else:
        if not (args.flow_ckpt_dir and args.flow_config):
            raise ValueError("--imputer=flow requires --flow_ckpt_dir and --flow_config.")
        imputer = _mod.load_flow_imputer(
            args.flow_ckpt_dir, args.flow_config, device,
            num_steps=args.flow_num_steps, solver=args.flow_solver,
        )
        imputer_ckpt_dir = args.flow_ckpt_dir
        imputer_label    = "flow"
    T = bench.T
    pad = torch.ones(1, T, dtype=torch.bool, device=device)

    M = bench.J if bench.player_mode == "spatial" else bench.K

    mse_means: list[float] = []
    rel_mses: list[float] = []
    mse_vars: list[float] = []

    for si in range(N):
        x_np = x_test[si]
        x_t = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1, J, F, T)

        for _ in range(args.n_masks_per_seq):
            z_bool = _random_partial_coalition(rng, M)
            hid_jft = _hidden_mask_jft(bench, z_bool)
            n_hid = int(hid_jft.sum())
            if n_hid == 0:
                continue

            if bench.player_mode == "temporal":
                s_obs, s_hid = _split_temporal(z_bool, bench.K)
                o = bench.conditional_sample(
                    x_np, s_obs, s_hid, args.n_oracle_samples, rng=rng,
                )
            else:
                j_obs, j_hid = _split_spatial(z_bool, bench.J)
                o = bench.conditional_sample_spatial(
                    x_np, j_obs, j_hid, args.n_oracle_samples, rng=rng,
                )

            flow_cm = _coalition_to_flow_mask(z_bool, bench, T, device)
            comps = imputer.sample_completions(
                x=x_t, y=None, mask=pad, lengths=None,
                coalition_mask=flow_cm, n_samples=args.n_vaeac_samples,
            )
            v = torch.cat(comps, dim=0).cpu().numpy().astype(np.float64)  # (n_v, J, F, T)

            o_h = o[:, hid_jft].reshape(o.shape[0], -1)
            v_h = v[:, hid_jft].reshape(v.shape[0], -1)

            mean_o = o_h.mean(axis=0)
            mean_v = v_h.mean(axis=0)
            m_mse = float(np.mean((mean_o - mean_v) ** 2))
            mse_means.append(m_mse)

            var_o = o_h.var(axis=0, ddof=1)
            var_v = v_h.var(axis=0, ddof=1)
            mse_vars.append(float(np.mean((var_o - var_v) ** 2)))

            # Scale-free: MSE of means / per-dim oracle variance (avoid div0)
            denom = np.maximum(var_o, 1e-12)
            rel_mses.append(float(np.mean(((mean_o - mean_v) ** 2) / denom)))

    print("\n" + "=" * 72)
    print(f"{imputer_label.upper()} vs exact Gaussian conditional (hidden coords only)")
    print("=" * 72)
    print(f"  ckpt_dir          {ckpt_dir}")
    print(f"  imputer           {imputer_label}  ({imputer_ckpt_dir})")
    print(f"  player_mode       {bench.player_mode}")
    print(f"  sequences × masks {N} × {args.n_masks_per_seq}")
    print(f"  n_oracle / n_vaeac {args.n_oracle_samples} / {args.n_vaeac_samples}")
    print(f"  trials            {len(mse_means)}")
    print()
    print(f"  mse_conditional_mean   mean={np.mean(mse_means):.6e}  std={np.std(mse_means):.6e}")
    print(f"  mean_rel_mse (÷var)    mean={np.mean(rel_mses):.6e}  std={np.std(rel_mses):.6e}")
    print(f"  mse_conditional_var    mean={np.mean(mse_vars):.6e}  std={np.std(mse_vars):.6e}")
    print()
    print("  Interpretation: mean_rel_mse ≪ 1 means sample means agree vs oracle noise scale.")
    print("=" * 72 + "\n")

    out_path = Path(args.out_path) if args.out_path else Path(imputer_ckpt_dir) / "conditional_vs_oracle.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "ckpt_dir": str(ckpt_dir),
                "imputer": imputer_label,
                "imputer_ckpt_dir": imputer_ckpt_dir,
                "vaeac_ckpt_dir": args.vaeac_ckpt_dir,
                "flow_ckpt_dir": args.flow_ckpt_dir,
                "n_sequences": N,
                "n_masks_per_seq": args.n_masks_per_seq,
                "n_oracle_samples": args.n_oracle_samples,
                "n_vaeac_samples": args.n_vaeac_samples,
                "player_mode": bench.player_mode,
                "mse_conditional_mean_mean": float(np.mean(mse_means)),
                "mse_conditional_mean_std": float(np.std(mse_means)),
                "mean_rel_mse_mean": float(np.mean(rel_mses)),
                "mean_rel_mse_std": float(np.std(rel_mses)),
                "mse_conditional_var_mean": float(np.mean(mse_vars)),
                "mse_conditional_var_std": float(np.std(mse_vars)),
            },
            f,
            indent=2,
        )
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
