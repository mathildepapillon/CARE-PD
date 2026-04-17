"""
diagnose_flow_matching.py — SHAP-readiness diagnostics for the flow-matching
velocity predictor trained with ``train_flow_matching.py``.

Runs two independent tests on the held-out validation set:

1. **t-bucketed val loss.**
   Masked MSE between ``v_theta(x_t, t)`` and the CondOT target ``u_t`` is
   evaluated separately for ``t`` drawn from 6 buckets:
   ``[0.0, 0.1) [0.1, 0.3) [0.3, 0.5) [0.5, 0.7) [0.7, 0.9) [0.9, 1.0]``.
   A model that is accurate on average but biased near ``t=0`` or ``t=1``
   will bias any path-integrated SHAP attribution.

2. **Noise-denoise reconstruction MPJPE.**
   For each real val clip ``x_1`` and sampled ``x_0 ~ N(0,I)``, we form
   ``x_s = (1-s) x_0 + s x_1`` at several ``s`` values and integrate
   ``v_theta`` forward from ``s`` to ``1`` with a midpoint ODE solver, then
   measure the mm-level MPJPE between the integrated clip and the original
   ``x_1`` (after denormalisation via the cached z-score stats). A round-trip
   test (``x_1 -> x_0_hat`` via reverse integration, then forward) is also
   reported.

Usage::

    python scripts/diagnose_flow_matching.py \
        --config configs/flow_matching/bmclab_h36m3d_fold1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

from model.flow_matching import VelocityNet
from train_flow_matching import FlowClipDataset, VelocityModelWrapper, masked_mse


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _strip_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    """Lightning wraps the ``VelocityNet`` under ``self.model``; strip it."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def load_velocity_net(cfg: dict, ckpt_path: str, device: torch.device) -> VelocityNet:
    net = VelocityNet(
        n_joints=17,
        n_coords=3,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
        time_emb_dim=int(cfg["time_emb_dim"]),
        max_len=max(int(cfg["seq_len"]) + 16, 256),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    clean_state = _strip_prefix(raw_state, "model.")
    if not clean_state:
        raise RuntimeError(
            f"No 'model.*' keys found in {ckpt_path}; keys look like "
            f"{list(raw_state.keys())[:3]}..."
        )
    missing, unexpected = net.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (first few: {missing[:3]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (first few: {unexpected[:3]})")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# Metric 1: t-bucketed val loss
# ---------------------------------------------------------------------------

def bucketed_val_loss(
    net: VelocityNet,
    val_loader: DataLoader,
    buckets: list[tuple[float, float]],
    k_per_bucket: int,
    device: torch.device,
) -> np.ndarray:
    """Masked MSE for each ``t`` bucket, weighted by real-frame count."""
    path = AffineProbPath(scheduler=CondOTScheduler())
    n_b = len(buckets)
    loss_sum = np.zeros(n_b, dtype=np.float64)
    count_sum = np.zeros(n_b, dtype=np.float64)

    with torch.no_grad():
        for batch in val_loader:
            x1 = batch["x1"].to(device)
            mask = batch["mask"].to(device)
            J, C = x1.shape[-2], x1.shape[-1]
            real_scalars = float(mask.sum().item() * J * C)

            for bi, (lo, hi) in enumerate(buckets):
                for _ in range(k_per_bucket):
                    t = torch.rand(x1.shape[0], device=device) * (hi - lo) + lo
                    x0 = torch.randn_like(x1)
                    sample = path.sample(t=t, x_0=x0, x_1=x1)
                    v = net(sample.x_t, sample.t, mask=mask)
                    loss = masked_mse(v, sample.dx_t, mask).item()
                    loss_sum[bi] += loss * real_scalars
                    count_sum[bi] += real_scalars

    return loss_sum / np.maximum(count_sum, 1.0)


# ---------------------------------------------------------------------------
# Metric 2: reconstruction MPJPE
# ---------------------------------------------------------------------------

def masked_mpjpe_mm(
    x_hat_z: torch.Tensor,
    x1_z: torch.Tensor,
    mask: torch.Tensor,
    stats_mean: torch.Tensor,
    stats_std: torch.Tensor,
) -> torch.Tensor:
    """MPJPE in mm between z-score clips after denormalisation.

    Shapes: ``x_hat_z``, ``x1_z``: ``(B, T, 17, 3)``; ``mask``: ``(B, T)``;
    ``stats_mean``, ``stats_std``: ``(17, 3)``. Returns a ``(B,)`` tensor.
    """
    x_hat_m = x_hat_z * stats_std + stats_mean
    x_m = x1_z * stats_std + stats_mean
    per_joint_err = (x_hat_m - x_m).pow(2).sum(dim=-1).sqrt()  # (B, T, 17) metres
    per_frame = per_joint_err.mean(dim=-1)                      # (B, T)
    m = mask.to(per_frame.dtype)
    per_clip = (per_frame * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return per_clip * 1000.0  # mm


def _uniform_grid(start: float, end: float, step: float, device: torch.device) -> torch.Tensor:
    """Linspace between ``start`` and ``end`` with spacing ~= ``step``."""
    span = abs(end - start)
    n_pts = max(int(round(span / step)) + 1, 2)
    return torch.linspace(float(start), float(end), n_pts, device=device)


def partial_reconstruction(
    net: VelocityNet,
    val_loader: DataLoader,
    s_list: list[float],
    ode_steps: int,
    stats_mean: torch.Tensor,
    stats_std: torch.Tensor,
    device: torch.device,
) -> dict[float, np.ndarray]:
    path = AffineProbPath(scheduler=CondOTScheduler())
    wrapper = VelocityModelWrapper(net)
    solver = ODESolver(velocity_model=wrapper)

    step_size = 1.0 / max(ode_steps - 1, 1)
    per_s: dict[float, list[torch.Tensor]] = {s: [] for s in s_list}

    with torch.no_grad():
        for batch in val_loader:
            x1 = batch["x1"].to(device)
            mask = batch["mask"].to(device)
            x0 = torch.randn_like(x1)

            for s in s_list:
                t_s = torch.full((x1.shape[0],), float(s), device=device,
                                 dtype=x1.dtype)
                sample = path.sample(t=t_s, x_0=x0, x_1=x1)
                time_grid = _uniform_grid(float(s), 1.0, step_size, device)
                x_end = solver.sample(
                    x_init=sample.x_t,
                    step_size=step_size,
                    method="midpoint",
                    time_grid=time_grid,
                )
                per_s[s].append(
                    masked_mpjpe_mm(x_end, x1, mask, stats_mean, stats_std).cpu()
                )

    return {s: torch.cat(vs).numpy() for s, vs in per_s.items()}


def round_trip_reconstruction(
    net: VelocityNet,
    val_loader: DataLoader,
    ode_steps: int,
    stats_mean: torch.Tensor,
    stats_std: torch.Tensor,
    device: torch.device,
    limit_batches: int | None,
) -> np.ndarray:
    wrapper = VelocityModelWrapper(net)
    solver = ODESolver(velocity_model=wrapper)
    step_size = 1.0 / max(ode_steps - 1, 1)
    grid_rev = torch.linspace(1.0, 0.0, ode_steps, device=device)
    grid_fwd = torch.linspace(0.0, 1.0, ode_steps, device=device)

    errs: list[torch.Tensor] = []
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if limit_batches is not None and bi >= limit_batches:
                break
            x1 = batch["x1"].to(device)
            mask = batch["mask"].to(device)

            x0_hat = solver.sample(
                x_init=x1,
                step_size=step_size,
                method="midpoint",
                time_grid=grid_rev,
            )
            x1_hat = solver.sample(
                x_init=x0_hat,
                step_size=step_size,
                method="midpoint",
                time_grid=grid_fwd,
            )
            errs.append(
                masked_mpjpe_mm(x1_hat, x1, mask, stats_mean, stats_std).cpu()
            )
    return torch.cat(errs).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_ckpt(cfg: dict) -> Path:
    d = PROJECT_ROOT / cfg["checkpoint_dir"]
    return d / f"flow_matching_{cfg['dataset']}_fold{cfg['fold']}_best.ckpt"


def _pct_of_baseline(x: float, baseline: float = 2.0) -> float:
    return 100.0 * x / baseline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Path to a Lightning .ckpt. Defaults to the best ckpt for this config.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override cfg batch_size for the val loader.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ode_steps", type=int, default=100,
                   help="Number of midpoint-ODE substeps over the full [0, 1] interval.")
    p.add_argument("--k_per_bucket", type=int, default=8,
                   help="Number of (t, x_0) draws per bucket per batch.")
    p.add_argument("--round_trip_batches", type=int, default=5,
                   help="Cap on val batches used for the round-trip test (expensive).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = json.load(f)

    ckpt_path = args.checkpoint or str(_default_ckpt(cfg))
    print(f"[cfg]    config     = {args.config}")
    print(f"[cfg]    checkpoint = {ckpt_path}")
    print(f"[cfg]    device     = {args.device}")

    device = torch.device(args.device)

    cache_path = PROJECT_ROOT / cfg["cache_dir"] / "cache.npz"
    cache = np.load(cache_path, allow_pickle=True)
    x1_val = cache["x1_val"]
    mask_val = cache["mask_val"]
    stats_mean = torch.from_numpy(cache["stats_mean"]).to(device)
    stats_std = torch.from_numpy(cache["stats_std"]).to(device)
    print(f"[data]   val clips  = {x1_val.shape[0]}  shape={tuple(x1_val.shape)}")
    print(f"[data]   stats_mean = {tuple(stats_mean.shape)}  stats_std = {tuple(stats_std.shape)}")

    batch_size = int(args.batch_size or cfg.get("batch_size", 128))
    val_ds = FlowClipDataset(x1_val, mask_val, mirror_augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    net = load_velocity_net(cfg, ckpt_path, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model]  VelocityNet params = {n_params:,}")

    # ------------------------------------------------------------------
    # 1. Bucketed val loss
    # ------------------------------------------------------------------
    buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print("\n=== 1) t-bucketed val loss (masked MSE, z-score units) ===")
    print(f"    # of (t, x_0) samples per bucket per clip: {args.k_per_bucket}")
    losses = bucketed_val_loss(net, val_loader, buckets, args.k_per_bucket, device)
    hdr = f"{'bucket':>14}  {'loss':>8}  {'% baseline (2.0)':>18}"
    print("    " + hdr)
    for (lo, hi), L in zip(buckets, losses):
        print(f"    [{lo:.1f}, {hi:.1f}){'':>6}  {L:>8.4f}  {_pct_of_baseline(float(L)):>17.1f}%")
    overall = float(losses.mean())
    print(f"    {'mean':>14}  {overall:>8.4f}  {_pct_of_baseline(overall):>17.1f}%")

    # ------------------------------------------------------------------
    # 2. Partial noise-denoise reconstruction MPJPE
    # ------------------------------------------------------------------
    s_list = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    print("\n=== 2) Partial noise-denoise reconstruction MPJPE (mm) ===")
    print("    x_s = (1-s) x_0 + s x_1;  then integrate from s to 1.")
    print(f"    ode_steps over [0,1] = {args.ode_steps}  step_size = {1.0/max(args.ode_steps-1,1):.5f}")
    recon = partial_reconstruction(
        net, val_loader, s_list,
        ode_steps=args.ode_steps,
        stats_mean=stats_mean, stats_std=stats_std,
        device=device,
    )
    hdr = f"{'start s':>8}  {'mean':>8}  {'median':>8}  {'p90':>8}  {'p95':>8}  {'max':>8}"
    print("    " + hdr)
    for s in s_list:
        mm = recon[s]
        print(
            f"    {s:>8.2f}  {mm.mean():>8.1f}  {np.median(mm):>8.1f}  "
            f"{np.percentile(mm, 90):>8.1f}  {np.percentile(mm, 95):>8.1f}  {mm.max():>8.1f}"
        )

    # ------------------------------------------------------------------
    # 3. Round-trip reconstruction MPJPE
    # ------------------------------------------------------------------
    print("\n=== 3) Round-trip reconstruction MPJPE (mm) ===")
    print("    x_1 --ODE(1 -> 0)-->  x_0_hat  --ODE(0 -> 1)-->  x_1_hat;  MPJPE(x_1_hat, x_1).")
    rt = round_trip_reconstruction(
        net, val_loader,
        ode_steps=args.ode_steps,
        stats_mean=stats_mean, stats_std=stats_std,
        device=device,
        limit_batches=args.round_trip_batches,
    )
    print(f"    clips evaluated = {len(rt)}  (capped to {args.round_trip_batches} batches)")
    print(f"    {'mean':>8}  {'median':>8}  {'p90':>8}  {'p95':>8}  {'max':>8}")
    print(
        f"    {rt.mean():>8.1f}  {np.median(rt):>8.1f}  "
        f"{np.percentile(rt, 90):>8.1f}  {np.percentile(rt, 95):>8.1f}  {rt.max():>8.1f}"
    )

    print("\n[done]")


if __name__ == "__main__":
    main()
