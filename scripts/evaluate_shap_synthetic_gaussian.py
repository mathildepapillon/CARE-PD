"""evaluate_shap_synthetic_gaussian.py — EC1 / EC2 / EC3 benchmark for the
synthetic Gaussian motion task, comparing KernelSHAP under five imputers:

    zero               hidden features replaced by 0
    mean               hidden features replaced by per-joint training mean
    marginal           hidden features replaced from a random training clip
    gaussian_oracle    exact Gaussian conditional (temporal → AR(1) over t,
                       spatial → joint-conditional over j).  Upper bound on
                       achievable Shapley quality under correct on-manifold
                       imputation.
    flow_matching      conditional flow imputer via a trained VelocityNet.
    vaeac              conditional amortised-posterior imputer via a trained
                       :class:`model.vaeac.VAEAC` (JMLR 2022 baseline,
                       parameter-matched to the flow-matching model).

Both player modes from the generalised :mod:`synthetic.gaussian_motion`
benchmark are supported:

  * ``player_mode == "temporal"``: K temporal windows are the players.
    Ground-truth Shapley is exact by 2^K coalition enumeration.

  * ``player_mode == "spatial"``:  J joints are the players.  Ground-truth
    Shapley is estimated by KernelSHAP with the exact joint-conditional
    Gaussian as the imputer (2^J would be intractable).

Output:
  * ``ec_summary.json`` — per-method EC1/EC2/EC3 mean and std.
  * ``per_sequence.json`` — per-sequence φ, v for each method (diagnostic).
  * LaTeX snippet printed to stdout.

USAGE
-----
    python scripts/evaluate_shap_synthetic_gaussian.py \\
        --ckpt_dir   experiment_outs/actor_shap_synthetic/synthetic_gaussian \\
        --flow_ckpt_dir experiment_outs/flow_matching_synthetic/gaussian_gpu_xl \\
        --flow_config configs/flow_matching/synthetic_gaussian.json \\
        --output_dir experiment_outs/flow_matching_synthetic/ec_k4 \\
        --n_test_sequences 100 --K_mc_true 1000 --n_completion_samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic.gaussian_motion import (  # noqa: E402
    GaussianMotionBenchmark,
    SyntheticMLPClassifier,
    _enumerate_temporal_coalitions,
    _sample_kernelshap_coalitions,
    _solve_shapley_wls,
)


# ---------------------------------------------------------------------------
# Flow-matching imputer loader
# ---------------------------------------------------------------------------

def load_flow_imputer(
    flow_ckpt_dir: str,
    flow_cfg_path: str,
    device: torch.device,
    num_steps: int,
    solver: str,
):
    """Load VelocityNet + wrap in FlowImputer for synthetic-space sampling."""
    from model.flow_matching import VelocityNet
    from model.flow_shap.imputer import FlowImputer

    with open(flow_cfg_path) as f:
        flow_cfg = json.load(f)

    last = os.path.join(flow_ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        ckpt_path = last
    else:
        ckpts = [
            os.path.join(flow_ckpt_dir, f)
            for f in os.listdir(flow_ckpt_dir)
            if f.endswith(".ckpt")
        ]
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt in {flow_ckpt_dir}")
        ckpt_path = max(ckpts, key=os.path.getmtime)

    net = VelocityNet(
        n_joints=17, n_coords=3,
        d_model=int(flow_cfg["d_model"]),
        nhead=int(flow_cfg["nhead"]),
        num_layers=int(flow_cfg["num_layers"]),
        ff_dim=int(flow_cfg["ff_dim"]),
        dropout=float(flow_cfg.get("dropout", 0.0)),
        time_emb_dim=int(flow_cfg["time_emb_dim"]),
        max_len=max(int(flow_cfg["seq_len"]) + 16, 256),
        tokenization=str(flow_cfg.get("tokenization", "frame")),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    clean = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    if not clean:
        raise RuntimeError(f"No 'model.*' keys in {ckpt_path}")
    net.load_state_dict(clean, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"[flow] loaded VelocityNet from {ckpt_path} "
          f"(solver={solver}, num_steps={num_steps})", flush=True)
    # stats_mean/std = None → identity (data is already in flow space for the
    # synthetic benchmark).
    return FlowImputer(
        net, device, stats_mean=None, stats_std=None,
        num_steps=num_steps, solver=solver,
    )


# ---------------------------------------------------------------------------
# VAEAC imputer loader
# ---------------------------------------------------------------------------

def load_vaeac_imputer(
    vaeac_ckpt_dir: str,
    vaeac_cfg_path: str,
    device: torch.device,
    temperature: float,
    use_ema: bool,
):
    """Load :class:`VAEAC` + wrap in :class:`VAEACImputer`.

    Mirrors :func:`load_flow_imputer`: looks for ``last.ckpt`` in
    ``vaeac_ckpt_dir`` first, then the most-recently modified ``.ckpt``.
    If ``use_ema`` is True and the Lightning checkpoint's ``callback`` section
    exposes EMA state, the EMA weights are swapped in.  Otherwise the raw
    ``model.*`` weights are loaded.
    """
    from model.vaeac import VAEAC, VAEACImputer

    with open(vaeac_cfg_path) as f:
        vaeac_cfg = json.load(f)

    last = os.path.join(vaeac_ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        ckpt_path = last
    else:
        ckpts = [
            os.path.join(vaeac_ckpt_dir, f)
            for f in os.listdir(vaeac_ckpt_dir)
            if f.endswith(".ckpt")
        ]
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt in {vaeac_ckpt_dir}")
        ckpt_path = max(ckpts, key=os.path.getmtime)

    model = VAEAC(
        n_joints=17, n_coords=3,
        d_model=int(vaeac_cfg["d_model"]),
        nhead=int(vaeac_cfg["nhead"]),
        num_layers=int(vaeac_cfg["num_layers"]),
        ff_dim=int(vaeac_cfg["ff_dim"]),
        dropout=float(vaeac_cfg.get("dropout", 0.0)),
        d_latent=int(vaeac_cfg["d_latent"]),
        max_len=max(int(vaeac_cfg["seq_len"]) + 16, 256),
        decoder_head=str(vaeac_cfg.get("decoder_head", "gaussian_scalar")),
        ivanov_min_sigma=float(vaeac_cfg.get("ivanov_min_sigma", 1e-2)),
        use_prior_memory=bool(vaeac_cfg.get("use_prior_memory", False)),
        prior_reg_sigma_mu=float(vaeac_cfg.get("prior_reg_sigma_mu", 1e4)),
        prior_reg_sigma_sigma=float(vaeac_cfg.get("prior_reg_sigma_sigma", 1e-4)),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    clean = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    if not clean:
        raise RuntimeError(f"No 'model.*' keys in {ckpt_path}")
    # Optional EMA weight swap (if the checkpoint exposes it — our EMA is a
    # local python object, not serialised via Lightning hooks, so this is a
    # no-op by default; kept for forward-compat with future EMA callbacks).
    if use_ema and "ema" in ckpt:
        ema_sd = ckpt["ema"]
        clean_ema = {k[len("model."):]: v for k, v in ema_sd.items() if k.startswith("model.")}
        if clean_ema:
            clean = clean_ema
    model.load_state_dict(clean, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bd = model.count_parameters_breakdown()
    print(
        f"[vaeac] loaded from {ckpt_path} — total={bd['total']:,} params "
        f"(full_enc={bd['full_encoder']:,} dec={bd['decoder']:,}) "
        f"temperature={temperature}",
        flush=True,
    )
    return VAEACImputer(
        model, device, stats_mean=None, stats_std=None,
        temperature=float(temperature),
    )


# ---------------------------------------------------------------------------
# Temporal / spatial imputation helpers (zero / mean / marginal)
# ---------------------------------------------------------------------------

def _apply_temporal_mask(
    x_jct: torch.Tensor,                # (1, J, F, T) classifier layout
    fill: torch.Tensor,                 # (J, F, T) broadcastable
    cm: torch.Tensor,                   # (1, T) bool, True = observed
) -> torch.Tensor:
    hid_t = (~cm[0]).to(x_jct.device)   # (T,)
    out = x_jct.clone()
    out[0, :, :, hid_t] = fill[:, :, hid_t].to(x_jct.device, x_jct.dtype)
    return out


def _apply_spatial_mask(
    x_jct: torch.Tensor,
    fill: torch.Tensor,                 # (J, F, T)
    cm: torch.Tensor,                   # (1, J) bool
) -> torch.Tensor:
    hid_j = (~cm[0]).to(x_jct.device)   # (J,)
    out = x_jct.clone()
    out[0, hid_j, :, :] = fill[hid_j, :, :].to(x_jct.device, x_jct.dtype)
    return out


# ---------------------------------------------------------------------------
# Per-sequence SHAP evaluation for one method
# ---------------------------------------------------------------------------

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


def _v_to_phi(
    coalitions: np.ndarray,
    weights: np.ndarray,
    v: np.ndarray,
    v_empty: float,
    v_full: float,
) -> np.ndarray:
    return _solve_shapley_wls(coalitions, v, weights, v_empty=v_empty, v_full=v_full)


def _eval_probs(prob_fn: Callable, x: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    out = []
    for i in range(0, len(x), chunk):
        out.append(prob_fn(x[i : i + chunk]))
    return torch.cat(out, dim=0)


def evaluate_methods_one_sequence(
    *,
    bench: GaussianMotionBenchmark,
    prob_fn: Callable,
    x: torch.Tensor,                    # (1, J, F, T)
    train_pool: torch.Tensor,           # (N_tr, J, F, T) on CPU
    train_mean: torch.Tensor,           # (J, F, T)
    flow_imputer,                       # FlowImputer or None
    vaeac_imputer,                      # VAEACImputer or None
    coalitions: np.ndarray,             # (N, M)
    weights: np.ndarray,                # (N,)
    n_completion_samples: int,
    K_mc_true: int,
    rng: np.random.Generator,
    device: torch.device,
    zero_fill: torch.Tensor,            # (J, F, T), all-zero
) -> dict[str, dict[str, Any]]:
    """Run every imputer on one sequence and return {method → {"v", "phi"}}.

    The ``gaussian_oracle`` row doubles as the ground-truth Shapley when the
    number of coalitions is small enough to enumerate (temporal K<=12).
    For spatial mode the same oracle is used as "true" via KernelSHAP
    sampling at the caller's chosen ``n_kernel_samples`` (set outside).
    """
    player_mode = bench.player_mode
    J = bench.J
    T = bench.T
    M = J if player_mode == "spatial" else bench.K
    N = coalitions.shape[0]
    x_np = x[0].cpu().numpy().astype(np.float64)
    results: dict[str, dict[str, Any]] = {}

    # Build per-coalition masks in classifier layout ahead of time.
    cms_torch: list[torch.Tensor] = []
    for i in range(N):
        z_bool = coalitions[i].astype(bool)
        cm = torch.from_numpy(z_bool[None]).to(device)
        cms_torch.append(cm)

    # Also for v_empty / v_full.
    empty_z = np.zeros(M, dtype=bool)
    full_z  = np.ones(M,  dtype=bool)

    # Helper: convert a coalition bit-vector into the FlowImputer's expected
    # ``(1, T)`` or ``(1, J, 1)`` mask.
    def _coalition_to_flow_mask(z_bool: np.ndarray) -> torch.Tensor:
        if player_mode == "temporal":
            # Broadcast window-level mask to per-frame T-mask.
            t_mask = np.zeros(T, dtype=bool)
            for k in range(bench.K):
                if z_bool[k]:
                    t_mask[bench.window_assignments[k]] = True
            return torch.from_numpy(t_mask[None]).to(device)          # (1, T)
        return torch.from_numpy(z_bool[None, :, None]).to(device)     # (1, J, 1)

    # -- Zero -----------------------------------------------------------------
    t_start = time.time()
    v_zero = np.zeros(N)
    for i, cm in enumerate(cms_torch):
        z_bool = coalitions[i].astype(bool)
        if z_bool.all():
            with torch.no_grad():
                v_zero[i] = float(prob_fn(x.to(device)).item())
            continue
        if not z_bool.any():
            with torch.no_grad():
                x_imp = torch.zeros_like(x).to(device)
                v_zero[i] = float(prob_fn(x_imp).item())
            continue
        if player_mode == "temporal":
            t_frames_obs = np.concatenate([
                bench.window_assignments[k] for k in range(bench.K) if z_bool[k]
            ])
            obs_frame_mask = np.zeros(T, dtype=bool)
            obs_frame_mask[t_frames_obs] = True
            cm_t = torch.from_numpy(obs_frame_mask[None]).to(device)
            x_imp = _apply_temporal_mask(x, zero_fill, cm_t)
        else:
            x_imp = _apply_spatial_mask(x, zero_fill, cm)
        with torch.no_grad():
            v_zero[i] = float(prob_fn(x_imp).item())
    v_empty_zero, v_full_zero = float(v_zero[_empty_idx(coalitions)]), float(v_zero[_full_idx(coalitions)])
    results["zero"] = {
        "v":       v_zero,
        "phi":     _v_to_phi(coalitions, weights, v_zero, v_empty_zero, v_full_zero),
        "elapsed": time.time() - t_start,
    }

    # -- Mean -----------------------------------------------------------------
    t_start = time.time()
    v_mean = np.zeros(N)
    for i, cm in enumerate(cms_torch):
        z_bool = coalitions[i].astype(bool)
        if z_bool.all():
            with torch.no_grad():
                v_mean[i] = float(prob_fn(x.to(device)).item())
            continue
        if not z_bool.any():
            x_imp = train_mean.unsqueeze(0).to(device)                     # broadcast
            with torch.no_grad():
                v_mean[i] = float(prob_fn(x_imp).item())
            continue
        if player_mode == "temporal":
            t_frames_obs = np.concatenate([
                bench.window_assignments[k] for k in range(bench.K) if z_bool[k]
            ])
            obs_frame_mask = np.zeros(T, dtype=bool)
            obs_frame_mask[t_frames_obs] = True
            cm_t = torch.from_numpy(obs_frame_mask[None]).to(device)
            x_imp = _apply_temporal_mask(x, train_mean, cm_t)
        else:
            x_imp = _apply_spatial_mask(x, train_mean, cm)
        with torch.no_grad():
            v_mean[i] = float(prob_fn(x_imp).item())
    results["mean"] = {
        "v":       v_mean,
        "phi":     _v_to_phi(coalitions, weights, v_mean,
                             float(v_mean[_empty_idx(coalitions)]),
                             float(v_mean[_full_idx(coalitions)])),
        "elapsed": time.time() - t_start,
    }

    # -- Marginal -------------------------------------------------------------
    t_start = time.time()
    v_marg = np.zeros(N)
    for i in range(N):
        z_bool = coalitions[i].astype(bool)
        if z_bool.all():
            with torch.no_grad():
                v_marg[i] = float(prob_fn(x.to(device)).item())
            continue
        if not z_bool.any():
            idxs = rng.integers(0, len(train_pool), size=n_completion_samples)
            x_marg = train_pool[idxs].to(device, dtype=torch.float32)
            with torch.no_grad():
                v_marg[i] = float(prob_fn(x_marg).mean().item())
            continue
        donors_idx = rng.integers(0, len(train_pool), size=n_completion_samples)
        if player_mode == "temporal":
            t_frames_hid = np.concatenate([
                bench.window_assignments[k] for k in range(bench.K) if not z_bool[k]
            ])
            xs = x.expand(n_completion_samples, -1, -1, -1).clone().to(device)
            donors = train_pool[donors_idx].to(device)
            xs[:, :, :, t_frames_hid] = donors[:, :, :, t_frames_hid]
        else:
            hid_j = np.nonzero(~z_bool)[0]
            xs = x.expand(n_completion_samples, -1, -1, -1).clone().to(device)
            donors = train_pool[donors_idx].to(device)
            xs[:, hid_j, :, :] = donors[:, hid_j, :, :]
        with torch.no_grad():
            v_marg[i] = float(prob_fn(xs).mean().item())
    results["marginal"] = {
        "v":       v_marg,
        "phi":     _v_to_phi(coalitions, weights, v_marg,
                             float(v_marg[_empty_idx(coalitions)]),
                             float(v_marg[_full_idx(coalitions)])),
        "elapsed": time.time() - t_start,
    }

    # -- Gaussian oracle (exact conditional) ---------------------------------
    # For temporal: AR(1) conditional per (j, f).  For spatial: joint-
    # conditional per (f, t).  Both of these are the "ground truth" within the
    # family of imputation-based KernelSHAP estimators and serve as the v_true
    # reference for EC metrics.
    t_start = time.time()
    v_oracle = np.zeros(N)
    for i in range(N):
        z_bool = coalitions[i].astype(bool)
        if z_bool.all():
            with torch.no_grad():
                v_oracle[i] = float(prob_fn(x.to(device)).item())
            continue
        if not z_bool.any():
            x_marg = bench.sample(K_mc_true, seed=int(rng.integers(1 << 31)))
            with torch.no_grad():
                v_oracle[i] = float(
                    prob_fn(torch.tensor(x_marg, device=device, dtype=torch.float32)).mean().item()
                )
            continue
        if player_mode == "temporal":
            s_obs, s_hid = _split_temporal(coalitions[i], bench.K)
            samps = bench.conditional_sample(x_np, s_obs, s_hid, K_mc_true, rng=rng)
        else:
            j_obs, j_hid = _split_spatial(coalitions[i], J)
            samps = bench.conditional_sample_spatial(x_np, j_obs, j_hid, K_mc_true, rng=rng)
        samps_t = torch.tensor(samps, device=device, dtype=torch.float32)
        with torch.no_grad():
            v_oracle[i] = float(_eval_probs(prob_fn, samps_t).mean().item())
    v_empty_o = float(v_oracle[_empty_idx(coalitions)])
    v_full_o  = float(v_oracle[_full_idx(coalitions)])
    results["gaussian_oracle"] = {
        "v":       v_oracle,
        "phi":     _v_to_phi(coalitions, weights, v_oracle, v_empty_o, v_full_o),
        "elapsed": time.time() - t_start,
    }

    # -- Flow matching --------------------------------------------------------
    if flow_imputer is not None:
        t_start = time.time()
        v_flow = np.zeros(N)
        pad_mask = torch.ones(1, T, dtype=torch.bool, device=device)
        for i in range(N):
            z_bool = coalitions[i].astype(bool)
            if z_bool.all():
                with torch.no_grad():
                    v_flow[i] = float(prob_fn(x.to(device)).item())
                continue
            if not z_bool.any():
                idxs = rng.integers(0, len(train_pool), size=n_completion_samples)
                x_marg = train_pool[idxs].to(device, dtype=torch.float32)
                with torch.no_grad():
                    v_flow[i] = float(prob_fn(x_marg).mean().item())
                continue
            flow_cm = _coalition_to_flow_mask(z_bool)
            comps = flow_imputer.sample_completions(
                x=x.to(device), y=None, mask=pad_mask, lengths=None,
                coalition_mask=flow_cm, n_samples=n_completion_samples,
            )
            x_batch = torch.cat(comps, dim=0)
            with torch.no_grad():
                v_flow[i] = float(prob_fn(x_batch).mean().item())
        results["flow_matching"] = {
            "v":       v_flow,
            "phi":     _v_to_phi(coalitions, weights, v_flow,
                                 float(v_flow[_empty_idx(coalitions)]),
                                 float(v_flow[_full_idx(coalitions)])),
            "elapsed": time.time() - t_start,
        }

    # -- VAEAC ---------------------------------------------------------------
    # Uses the *same* coalition-mask shapes as the flow imputer, so we reuse
    # ``_coalition_to_flow_mask`` above.  Marginal fallback for the empty
    # coalition matches flow_matching's behaviour so v_empty is comparable.
    if vaeac_imputer is not None:
        t_start = time.time()
        v_vae = np.zeros(N)
        pad_mask = torch.ones(1, T, dtype=torch.bool, device=device)
        for i in range(N):
            z_bool = coalitions[i].astype(bool)
            if z_bool.all():
                with torch.no_grad():
                    v_vae[i] = float(prob_fn(x.to(device)).item())
                continue
            if not z_bool.any():
                idxs = rng.integers(0, len(train_pool), size=n_completion_samples)
                x_marg = train_pool[idxs].to(device, dtype=torch.float32)
                with torch.no_grad():
                    v_vae[i] = float(prob_fn(x_marg).mean().item())
                continue
            vae_cm = _coalition_to_flow_mask(z_bool)
            comps = vaeac_imputer.sample_completions(
                x=x.to(device), y=None, mask=pad_mask, lengths=None,
                coalition_mask=vae_cm, n_samples=n_completion_samples,
            )
            x_batch = torch.cat(comps, dim=0)
            with torch.no_grad():
                v_vae[i] = float(prob_fn(x_batch).mean().item())
        results["vaeac"] = {
            "v":       v_vae,
            "phi":     _v_to_phi(coalitions, weights, v_vae,
                                 float(v_vae[_empty_idx(coalitions)]),
                                 float(v_vae[_full_idx(coalitions)])),
            "elapsed": time.time() - t_start,
        }

    return results


def _empty_idx(coalitions: np.ndarray) -> int:
    row = (coalitions.sum(axis=1) == 0).nonzero()[0]
    if len(row) == 0:
        raise ValueError("coalitions list does not contain the empty coalition.")
    return int(row[0])


def _full_idx(coalitions: np.ndarray) -> int:
    row = (coalitions.sum(axis=1) == coalitions.shape[1]).nonzero()[0]
    if len(row) == 0:
        raise ValueError("coalitions list does not contain the full coalition.")
    return int(row[0])


# ---------------------------------------------------------------------------
# EC metrics
# ---------------------------------------------------------------------------

def compute_ec_table(
    per_seq: list[dict[str, dict[str, Any]]],
    truth_key: str,
    methods: list[str],
) -> dict[str, dict[str, float]]:
    """Compute EC1, EC2, EC3 (mean + std over sequences) for each method."""
    rows: dict[str, dict[str, list[float]]] = {
        m: {"ec1": [], "ec2": [], "ec3": []} for m in methods
    }
    for seq in per_seq:
        phi_true = seq[truth_key]["phi"]
        v_true   = seq[truth_key]["v"]
        # Non-trivial coalitions (skip the two boundary rows).
        nontriv = np.ones(len(v_true), dtype=bool)
        nontriv[_empty_idx(seq["_coalitions"])] = False
        nontriv[_full_idx(seq["_coalitions"])] = False
        f_x = float(v_true[_full_idx(seq["_coalitions"])])
        for m in methods:
            if m not in seq:
                continue
            phi_m = seq[m]["phi"]
            v_m   = seq[m]["v"]
            ec1 = float(np.abs(phi_m - phi_true).mean())
            ec2 = float(((v_m[nontriv] - v_true[nontriv]) ** 2).mean())
            ec3 = float(((f_x - v_m[nontriv]) ** 2).mean())
            rows[m]["ec1"].append(ec1)
            rows[m]["ec2"].append(ec2)
            rows[m]["ec3"].append(ec3)

    summary: dict[str, dict[str, float]] = {}
    for m in methods:
        if not rows[m]["ec1"]:
            continue
        summary[m] = {
            "EC1_mean": float(np.mean(rows[m]["ec1"])),
            "EC1_std":  float(np.std(rows[m]["ec1"])),
            "EC2_mean": float(np.mean(rows[m]["ec2"])),
            "EC2_std":  float(np.std(rows[m]["ec2"])),
            "EC3_mean": float(np.mean(rows[m]["ec3"])),
            "EC3_std":  float(np.std(rows[m]["ec3"])),
            "n":        int(len(rows[m]["ec1"])),
        }
    return summary


def print_ec_table(
    summary: dict[str, dict[str, float]],
    header: str,
) -> None:
    print(f"\n{'=' * 72}")
    print(header)
    print(f"{'=' * 72}")
    print(f"{'Method':<22s} {'EC1':>10s} {'EC2':>10s} {'EC3':>10s}  {'n':>4s}")
    print("-" * 62)
    # Sort methods by EC1 mean for readability.
    for m in sorted(summary.keys(), key=lambda k: summary[k]["EC1_mean"]):
        s = summary[m]
        print(f"{m:<22s} {s['EC1_mean']:>10.4f} {s['EC2_mean']:>10.4f} "
              f"{s['EC3_mean']:>10.4f}  {s['n']:>4d}")
    print("\n% LaTeX snippet:")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print("Method & EC1 & EC2 & EC3 \\\\")
    print("\\midrule")
    for m in sorted(summary.keys(), key=lambda k: summary[k]["EC1_mean"]):
        s = summary[m]
        print(
            f"{m} & {s['EC1_mean']:.4f} $\\pm$ {s['EC1_std']:.4f}"
            f" & {s['EC2_mean']:.4f} $\\pm$ {s['EC2_std']:.4f}"
            f" & {s['EC3_mean']:.4f} $\\pm$ {s['EC3_std']:.4f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory produced by scripts/build_synthetic_gaussian_data.py "
                        "(contains synthetic_benchmark.pkl + synthetic_clf.pt).")
    p.add_argument("--flow_ckpt_dir", default=None,
                   help="Flow-matching checkpoint directory; if omitted, the "
                        "flow_matching row is skipped.")
    p.add_argument("--flow_config", default=None,
                   help="Flow-matching JSON config (required with --flow_ckpt_dir).")
    p.add_argument("--flow_num_steps", type=int, default=100)
    p.add_argument("--flow_solver", choices=("euler", "midpoint"), default="midpoint")
    p.add_argument("--vaeac_ckpt_dir", default=None,
                   help="VAEAC checkpoint directory; if omitted, the vaeac "
                        "row is skipped.")
    p.add_argument("--vaeac_config", default=None,
                   help="VAEAC JSON config (required with --vaeac_ckpt_dir).")
    p.add_argument("--vaeac_temperature", type=float, default=1.0,
                   help="Prior-latent sampling temperature (1.0 = full variance, "
                        "0.0 = mean-field).")
    p.add_argument("--vaeac_use_ema", action="store_true",
                   help="Prefer EMA weights if present in the checkpoint.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write ec_summary.json and per_sequence.json.")
    p.add_argument("--n_test_sequences", type=int, default=100,
                   help="Number of test sequences to evaluate.")
    p.add_argument("--K_mc_true", type=int, default=1000,
                   help="MC samples per coalition for oracle v(S).")
    p.add_argument("--n_completion_samples", type=int, default=20,
                   help="Completions per coalition for stochastic imputers "
                        "(marginal, flow_matching).")
    p.add_argument("--n_kernel_samples", type=int, default=250,
                   help="Only for spatial mode: paired coalition samples for "
                        "KernelSHAP (total non-boundary coalitions = 2*this).")
    p.add_argument("--class_idx", type=int, default=None,
                   help="SHAP target class; defaults to the test-set majority class.")
    p.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    print(f"[eval] ckpt_dir={ckpt_dir}", flush=True)

    with open(ckpt_dir / "synthetic_benchmark.pkl", "rb") as f:
        bench: GaussianMotionBenchmark = pickle.load(f)
    # Back-compat: older pickles have no player_mode attribute.
    if not hasattr(bench, "player_mode"):
        bench.player_mode = "temporal"
        bench.signal_joints = None
    print(f"[eval] benchmark: J={bench.J}, F={bench.F}, T={bench.T}, K={bench.K}, "
          f"rho={bench.rho}, alpha={bench.alpha}, player_mode={bench.player_mode}",
          flush=True)

    with open(ckpt_dir / "synthetic_clf_meta.json") as f:
        clf_meta = json.load(f)
    clf_pm = clf_meta.get("player_mode", "temporal")
    if clf_pm != bench.player_mode:
        raise RuntimeError(
            f"classifier player_mode={clf_pm!r} but benchmark is {bench.player_mode!r}"
        )
    clf = SyntheticMLPClassifier(
        J=clf_meta["J"], F=clf_meta["F"], T=clf_meta["T"],
        K=clf_meta.get("K", 4), num_classes=clf_meta["num_classes"],
        player_mode=clf_pm,
    )
    clf.load_state_dict(torch.load(ckpt_dir / "synthetic_clf.pt", map_location="cpu"))
    clf.to(device).eval()

    test_data = torch.load(ckpt_dir / "synthetic_test.pt", map_location="cpu")
    x_test_tjf = test_data["x"]                                             # (N, T, J, F)
    y_test     = test_data["y"]
    x_test_jft = x_test_tjf.permute(0, 2, 3, 1).contiguous()                # (N, J, F, T)
    N_test = min(args.n_test_sequences, len(x_test_jft))
    x_test_jft = x_test_jft[:N_test]
    y_test     = y_test[:N_test]
    print(f"[eval] evaluating {N_test} test sequences", flush=True)

    # Classifier test accuracy for context.
    with torch.no_grad():
        preds = clf(x_test_jft.to(device)).argmax(dim=-1).cpu()
    acc = (preds == y_test).float().mean().item()
    print(f"[eval] classifier test accuracy: {acc:.1%}", flush=True)

    x_train_jft = np.load(ckpt_dir / "x_train_jft.npy")
    train_pool = torch.from_numpy(x_train_jft).float()
    train_mean = torch.from_numpy(x_train_jft.mean(axis=0)).float()
    zero_fill  = torch.zeros_like(train_mean)

    if args.class_idx is None:
        class_idx = int(np.bincount(y_test.numpy()).argmax())
    else:
        class_idx = int(args.class_idx)
    print(f"[eval] SHAP target class: {class_idx}", flush=True)
    prob_fn = clf.class_prob_fn(class_idx)

    # ---- Coalitions ---------------------------------------------------------
    if bench.player_mode == "temporal":
        coalitions, weights = _enumerate_temporal_coalitions(bench.K)
        player_count = bench.K
        truth_key = "gaussian_oracle"
        kshap_label = f"enumerate 2^{bench.K}={2**bench.K}"
    else:
        coalitions, weights = _sample_kernelshap_coalitions(
            bench.J, args.n_kernel_samples, np.random.default_rng(args.seed + 1),
        )
        empty_row = np.zeros((1, bench.J), dtype=int)
        full_row  = np.ones((1, bench.J),  dtype=int)
        coalitions = np.vstack([empty_row, coalitions, full_row])
        boundary_w = 1e6
        weights = np.concatenate([[boundary_w], weights, [boundary_w]])
        player_count = bench.J
        truth_key = "gaussian_oracle"
        kshap_label = f"KernelSHAP paired sampling (N={len(coalitions)})"
    print(f"[eval] coalitions: {kshap_label}, M={player_count} players", flush=True)

    # ---- Flow imputer (optional) -------------------------------------------
    flow_imputer = None
    if args.flow_ckpt_dir is not None:
        if args.flow_config is None:
            raise ValueError("--flow_config is required with --flow_ckpt_dir")
        flow_imputer = load_flow_imputer(
            args.flow_ckpt_dir, args.flow_config, device,
            num_steps=args.flow_num_steps, solver=args.flow_solver,
        )

    # ---- VAEAC imputer (optional) ------------------------------------------
    vaeac_imputer = None
    if args.vaeac_ckpt_dir is not None:
        if args.vaeac_config is None:
            raise ValueError("--vaeac_config is required with --vaeac_ckpt_dir")
        vaeac_imputer = load_vaeac_imputer(
            args.vaeac_ckpt_dir, args.vaeac_config, device,
            temperature=args.vaeac_temperature, use_ema=args.vaeac_use_ema,
        )

    # ---- Per-sequence loop --------------------------------------------------
    per_seq: list[dict] = []
    t_global = time.time()
    for i in range(N_test):
        xi = x_test_jft[i : i + 1]                                          # (1, J, F, T)
        res = evaluate_methods_one_sequence(
            bench=bench, prob_fn=prob_fn, x=xi,
            train_pool=train_pool, train_mean=train_mean,
            flow_imputer=flow_imputer,
            vaeac_imputer=vaeac_imputer,
            coalitions=coalitions, weights=weights,
            n_completion_samples=args.n_completion_samples,
            K_mc_true=args.K_mc_true,
            rng=rng, device=device, zero_fill=zero_fill,
        )
        res["_coalitions"] = coalitions
        per_seq.append(res)

        step = max(1, N_test // 20)
        if (i + 1) % step == 0:
            elapsed = time.time() - t_global
            avg = elapsed / (i + 1)
            eta = avg * (N_test - i - 1)
            print(f"[eval] {i+1}/{N_test}  t={elapsed:6.1f}s  "
                  f"avg={avg:5.2f}s/seq  eta={eta/60:5.1f}m", flush=True)

    # ---- Aggregate ----------------------------------------------------------
    methods = ["zero", "mean", "marginal", "gaussian_oracle"]
    if flow_imputer is not None:
        methods.append("flow_matching")
    if vaeac_imputer is not None:
        methods.append("vaeac")
    summary = compute_ec_table(per_seq, truth_key=truth_key, methods=methods)

    # Persist summary + per-sequence (light-weight: only v + phi per method).
    summary_path = out_dir / "ec_summary.json"
    summary_full = {
        "_config": {
            "ckpt_dir":            str(ckpt_dir),
            "flow_ckpt_dir":       args.flow_ckpt_dir,
            "flow_config":         args.flow_config,
            "flow_num_steps":      args.flow_num_steps,
            "flow_solver":         args.flow_solver,
            "vaeac_ckpt_dir":      args.vaeac_ckpt_dir,
            "vaeac_config":        args.vaeac_config,
            "vaeac_temperature":   float(args.vaeac_temperature),
            "vaeac_use_ema":       bool(args.vaeac_use_ema),
            "n_test_sequences":    int(N_test),
            "K_mc_true":           int(args.K_mc_true),
            "n_completion_samples": int(args.n_completion_samples),
            "n_kernel_samples":    int(args.n_kernel_samples),
            "class_idx":           int(class_idx),
            "player_mode":         bench.player_mode,
            "K":                   int(bench.K),
            "J":                   int(bench.J),
            "truth_key":           truth_key,
        },
        "methods": summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_full, f, indent=2)
    print(f"[eval] wrote {summary_path}", flush=True)

    per_seq_light = []
    for seq in per_seq:
        row = {}
        for m in methods + [truth_key]:
            if m in seq:
                row[m] = {
                    "v":   np.asarray(seq[m]["v"]).tolist(),
                    "phi": np.asarray(seq[m]["phi"]).tolist(),
                }
        per_seq_light.append(row)
    with open(out_dir / "per_sequence.json", "w") as f:
        json.dump(per_seq_light, f)
    print(f"[eval] wrote {out_dir / 'per_sequence.json'}", flush=True)

    header = (
        f"Gaussian benchmark  rho={bench.rho}  alpha={bench.alpha}  "
        f"J={bench.J}  T={bench.T}  K={bench.K}  player_mode={bench.player_mode}  "
        f"class={class_idx}  n_seqs={N_test}"
    )
    print_ec_table(summary, header=header)


if __name__ == "__main__":
    main()
