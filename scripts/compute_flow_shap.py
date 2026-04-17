"""
compute_flow_shap.py — OTFlow-SHAP attribution driver for CARE-PD classifiers.

Given (a) a trained flow-matching velocity field and (b) a pretrained CARE-PD
classifier (POTR, PoseFormerV2, MotionBERT, ...), this script:

1. Loads the flow cache (pelvis-centered z-scored clips + z-score stats).
2. Re-builds the world pelvis trajectory per val clip (so the classifier
   sees an in-distribution walking path).
3. Instantiates a differentiable flow-space → classifier-logit adapter via
   :func:`model.flow_shap.build_classifier_fn`.
4. Runs :func:`model.flow_shap.compute_flow_shap` on every val clip in
   batches, writing per-clip attributions and diagnostics to disk.

Output:

- ``psi.npz``: ``{psi, x_star, x0_hat, pelvis_world, mask, f_xstar, f_x0,
  completeness_rel, pelvis_leak_abs, pelvis_leak_frac, fce_per_sample,
  clip_id, participant_id, label, class_idx}``.
- ``summary.json``: dataset-wide quality stats (medians / fractions above
  thresholds / etc.).

Usage::

    python scripts/compute_flow_shap.py \\
        --config configs/flow_shap/bmclab_potr_fold1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.flow_matching import VelocityNet
from model.flow_shap import (
    build_classifier_fn,
    completeness_residual,
    compute_flow_shap,
    flow_consistency_error,
    pelvis_leak,
)
from model.flow_shap.data_loading import (
    iter_flow_shap_batches,
    load_flow_cache,
    load_pelvis_world_for_val,
)


# ---------------------------------------------------------------------------
# VelocityNet loading (mirrors scripts/diagnose_flow_matching.py)
# ---------------------------------------------------------------------------

def _strip_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _load_velocity_net(flow_cfg: dict, ckpt_path: str, device: torch.device) -> VelocityNet:
    net = VelocityNet(
        n_joints=17, n_coords=3,
        d_model=int(flow_cfg["d_model"]),
        nhead=int(flow_cfg["nhead"]),
        num_layers=int(flow_cfg["num_layers"]),
        ff_dim=int(flow_cfg["ff_dim"]),
        dropout=float(flow_cfg.get("dropout", 0.0)),
        time_emb_dim=int(flow_cfg["time_emb_dim"]),
        max_len=max(int(flow_cfg["seq_len"]) + 16, 256),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    clean_state = _strip_prefix(raw_state, "model.")
    if not clean_state:
        raise RuntimeError(
            f"No 'model.*' keys found in {ckpt_path}; "
            f"available: {list(raw_state.keys())[:5]}..."
        )
    missing, unexpected = net.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[load_velocity_net] missing keys: {len(missing)} "
              f"(first few: {missing[:3]})", flush=True)
    if unexpected:
        print(f"[load_velocity_net] unexpected keys: {len(unexpected)} "
              f"(first few: {unexpected[:3]})", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# Class policy resolution
# ---------------------------------------------------------------------------

def _resolve_class_idx(
    policy: Any,
    full_logits_fn,
    x_flow: torch.Tensor,
    ctx: Dict[str, Any],
    num_classes: int,
    labels: np.ndarray,
    batch_ids: list,
) -> torch.Tensor:
    """Return a ``(B,)`` long tensor of class indices for this batch.

    Policies:
        - int in ``[0, num_classes)``: that fixed class everywhere.
        - ``"predicted"`` / ``"argmax"``: argmax of classifier on ``x_star``.
        - ``"label"``: use the ground-truth UPDRS label from cache metadata.
    """
    B = x_flow.shape[0]
    device = x_flow.device
    if isinstance(policy, int):
        if not (0 <= policy < num_classes):
            raise ValueError(f"class_idx {policy} out of [0, {num_classes})")
        return torch.full((B,), policy, dtype=torch.long, device=device)
    if isinstance(policy, str):
        pol = policy.lower()
        if pol in ("predicted", "argmax"):
            with torch.no_grad():
                logits = full_logits_fn(x_flow, ctx)
            return logits.argmax(dim=-1).long()
        if pol == "label":
            lbls = torch.as_tensor(labels[batch_ids], dtype=torch.long, device=device)
            if ((lbls < 0) | (lbls >= num_classes)).any():
                raise ValueError(
                    f"Some labels out of [0, {num_classes}): {lbls.tolist()}"
                )
            return lbls
    raise ValueError(
        f"Unknown class_policy {policy!r}. Use int, 'predicted', 'argmax', or 'label'."
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _summarise(arr: np.ndarray, name: str) -> Dict[str, float]:
    if arr.size == 0:
        return {f"{name}_mean": float("nan"), f"{name}_median": float("nan"),
                f"{name}_p95": float("nan"), f"{name}_max": float("nan")}
    return {
        f"{name}_mean":   float(arr.mean()),
        f"{name}_median": float(np.median(arr)),
        f"{name}_p95":    _percentile(arr, 95.0),
        f"{name}_max":    float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to flow-SHAP JSON config.")
    p.add_argument("--max_clips", type=int, default=None,
                   help="Override config data.max_clips (cap clip count for debug).")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--fce_batches", type=int, default=2,
                   help="Number of batches to run trajectory-returning compute_flow_shap "
                        "on for FCE stats. Trajectory tensors are expensive in memory.")
    p.add_argument("--num_ode_steps", type=int, default=None,
                   help="Override config num_ode_steps (handy for K-robustness sweeps).")
    p.add_argument("--output_dir", default=None,
                   help="Override config output_dir (handy for K-robustness sweeps).")
    p.add_argument("--flow_checkpoint", default=None,
                   help="Override config flow_checkpoint (handy for multi-seed sweeps).")
    p.add_argument("--flow_config", default=None,
                   help="Override config flow_config (handy for multi-seed sweeps).")
    p.add_argument("--cache_dir", default=None,
                   help="Override flow cache_dir (e.g. to run flow-SHAP on the "
                        "classifier's held-out eval subjects instead of the "
                        "flow's own val split).")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path) as f:
        cfg = json.load(f)

    flow_cfg_rel = args.flow_config if args.flow_config is not None else cfg["flow_config"]
    flow_cfg_path = Path(flow_cfg_rel)
    if not flow_cfg_path.is_absolute():
        flow_cfg_path = PROJECT_ROOT / flow_cfg_path
    with open(flow_cfg_path) as f:
        flow_cfg = json.load(f)

    device = torch.device(args.device or cfg.get("device", "cuda:0"))
    batch_size = int(args.batch_size or cfg.get("batch_size", 16))
    max_clips = args.max_clips if args.max_clips is not None else cfg.get("data", {}).get("max_clips", None)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- flow cache + pelvis world ---
    cache_dir_raw = args.cache_dir if args.cache_dir is not None else flow_cfg["cache_dir"]
    cache_dir_path = Path(cache_dir_raw)
    if not cache_dir_path.is_absolute():
        cache_dir_path = PROJECT_ROOT / cache_dir_path
    cache_path = cache_dir_path / "cache.npz"
    print(f"[flow_shap] loading flow cache: {cache_path}", flush=True)
    cache = load_flow_cache(cache_path, split=cfg.get("data", {}).get("split", "val"))

    x1_val        = cache["x1"]                                # (N, T, 17, 3)
    mask_val      = cache["mask"]                              # (N, T) bool
    seq_len       = int(cache["seq_len"])
    stats_mean_np = cache["stats_mean"]                         # (17, 3)
    stats_std_np  = cache["stats_std"]                          # (17, 3)
    dataset_name  = str(cache["dataset"])
    clip_stride_val = int(cache["clip_stride_val"])

    meta_seq_key   = cache["meta_seq_key"]                      # (N,) object
    meta_clip_idx  = cache["meta_clip_idx"]                     # (N,) int32
    meta_pid       = cache["meta_pid"]                          # (N,) object
    meta_walk_id   = cache["meta_walk_id"]                      # (N,) object
    meta_updrs     = cache["meta_updrs_gait"]                   # (N,) int32

    N = int(x1_val.shape[0])
    print(f"[flow_shap] split contains {N} clips from dataset={dataset_name!r}", flush=True)

    print("[flow_shap] recovering world pelvis trajectories from raw NPZ...", flush=True)
    pelvis_world = load_pelvis_world_for_val(
        dataset=dataset_name,
        meta_seq_key_val=meta_seq_key,
        meta_clip_idx_val=meta_clip_idx,
        seq_len=seq_len,
        clip_stride_val=clip_stride_val,
    )  # (N, T, 3)
    assert pelvis_world.shape == (N, seq_len, 3)

    # --- velocity net ---
    flow_ckpt_rel = args.flow_checkpoint if args.flow_checkpoint is not None else cfg["flow_checkpoint"]
    flow_ckpt = Path(flow_ckpt_rel)
    if not flow_ckpt.is_absolute():
        flow_ckpt = PROJECT_ROOT / flow_ckpt
    print(f"[flow_shap] loading velocity net: {flow_ckpt}", flush=True)
    velocity_net = _load_velocity_net(flow_cfg, str(flow_ckpt), device)

    stats_mean = torch.from_numpy(stats_mean_np).to(device)
    stats_std  = torch.from_numpy(stats_std_np).to(device)

    # --- classifier adapter ---
    print(f"[flow_shap] loading classifier: backbone={cfg['backbone']} "
          f"ckpt={cfg['classifier_checkpoint']}", flush=True)
    classifier_fn, full_logits_fn, motion_encoder, backbone_params = build_classifier_fn(
        backbone_name=cfg["backbone"],
        classifier_ckpt=str(PROJECT_ROOT / cfg["classifier_checkpoint"]),
        flow_stats_mean=stats_mean,
        flow_stats_std=stats_std,
        class_idx=0,     # overridden per-sample via ctx["class_idx"]
        device=device,
        config_file=cfg.get("classifier_config_file", "BMCLab.json"),
        num_folds=int(cfg.get("classifier_num_folds", 23)),
        fold=int(cfg["data"].get("fold", flow_cfg.get("fold", 1))),
    )
    num_classes = int(backbone_params["num_classes"])

    # --- iterate ---
    indices = list(range(N))
    if max_clips is not None and max_clips > 0:
        indices = indices[:int(max_clips)]
        print(f"[flow_shap] limiting to first {len(indices)} clips (max_clips)", flush=True)

    out_dir_raw = args.output_dir if args.output_dir is not None else cfg["output_dir"]
    out_dir = Path(out_dir_raw)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    psi_path = out_dir / "psi.npz"
    summary_path = out_dir / "summary.json"

    psi_all: list = []
    x0_hat_all: list = []
    f_xstar_all: list = []
    f_x0_all: list = []
    comp_rel_all: list = []
    pelvis_leak_abs_all: list = []
    pelvis_leak_frac_all: list = []
    fce_per_sample_all: list = []
    fce_indices: list = []
    class_idx_all: list = []
    used_indices: list = []

    num_ode_steps = int(args.num_ode_steps if args.num_ode_steps is not None
                        else cfg.get("num_ode_steps", 100))
    solver_method = str(cfg.get("solver", "rk4"))
    class_policy_raw = cfg.get("class_policy", "predicted")
    class_policy: Any = class_policy_raw
    if isinstance(class_policy_raw, str) and class_policy_raw.isdigit():
        class_policy = int(class_policy_raw)

    t0 = time.time()
    batch_count = 0
    for batch_ids, batch in iter_flow_shap_batches(
        x1=x1_val, mask=mask_val, pelvis_world=pelvis_world,
        batch_size=batch_size, indices=indices,
    ):
        x_star_flow = torch.from_numpy(batch["x_star_flow"]).to(device=device, dtype=torch.float32)
        mask_b = torch.from_numpy(batch["mask"]).to(device=device, dtype=torch.bool)
        pelvis_b = torch.from_numpy(batch["pelvis_world"]).to(device=device, dtype=torch.float32)

        ctx: Dict[str, Any] = {
            "pelvis_world": pelvis_b,
            "mask":         mask_b,
        }
        # Resolve per-sample class idx and bake into ctx so ``classifier_fn``
        # gathers the right logit for every sample in the batch.
        cls_idx = _resolve_class_idx(
            class_policy, full_logits_fn, x_star_flow, ctx, num_classes,
            labels=meta_updrs, batch_ids=batch_ids,
        )
        ctx["class_idx"] = cls_idx

        keep_traj = batch_count < args.fce_batches
        result = compute_flow_shap(
            velocity_net=velocity_net,
            classifier_fn=classifier_fn,
            x_star_flow=x_star_flow,
            ctx=ctx,
            num_steps=num_ode_steps,
            solver_method=solver_method,
            zero_pelvis=False,   # keep leak for bookkeeping; we zero manually below
            return_trajectory=keep_traj,
        )

        psi = result["psi"]                      # (B, T, J, C)  not yet pelvis-zeroed
        pleak = pelvis_leak(psi, mask=mask_b)
        # zero the pelvis channel AFTER measuring leak
        psi_clean = psi.clone()
        psi_clean[..., 0, :] = 0.0

        comp = completeness_residual(
            psi=psi_clean,                        # report completeness of what we save
            f_xstar=result["f_xstar"], f_x0=result["f_x0"], mask=mask_b,
        )

        if keep_traj:
            fce = flow_consistency_error(
                velocity_net=velocity_net,
                trajectory=result["trajectory"],
                t_grid=result["t_grid"],
                mask=mask_b,
            )
            fce_per_sample_all.append(fce["per_sample"].numpy())
            fce_indices.extend(batch_ids)

        psi_all.append(psi_clean.detach().cpu().numpy())
        x0_hat_all.append(result["x0_hat"].detach().cpu().numpy())
        f_xstar_all.append(result["f_xstar"].detach().cpu().numpy())
        f_x0_all.append(result["f_x0"].detach().cpu().numpy())
        comp_rel_all.append(comp["rel"].numpy())
        pelvis_leak_abs_all.append(pleak["pelvis_abs"].numpy())
        pelvis_leak_frac_all.append(pleak["fraction"].numpy())
        class_idx_all.append(cls_idx.detach().cpu().numpy())
        used_indices.extend(batch_ids)

        batch_count += 1
        if batch_count % max(1, len(indices) // batch_size // 10 or 1) == 0:
            elapsed = time.time() - t0
            done = len(used_indices)
            print(f"[flow_shap] {done}/{len(indices)} clips  "
                  f"t={elapsed:6.1f}s  "
                  f"median comp_rel={float(np.median(np.concatenate(comp_rel_all))):.3f}  "
                  f"median |pelvis leak frac|={float(np.median(np.concatenate(pelvis_leak_frac_all))):.4f}",
                  flush=True)

    # ----- stack + save -----
    psi_arr           = np.concatenate(psi_all, axis=0)
    x0_hat_arr        = np.concatenate(x0_hat_all, axis=0)
    f_xstar_arr       = np.concatenate(f_xstar_all, axis=0)
    f_x0_arr          = np.concatenate(f_x0_all, axis=0)
    comp_rel_arr      = np.concatenate(comp_rel_all, axis=0)
    pelvis_leak_abs_arr  = np.concatenate(pelvis_leak_abs_all, axis=0)
    pelvis_leak_frac_arr = np.concatenate(pelvis_leak_frac_all, axis=0)
    class_idx_arr     = np.concatenate(class_idx_all, axis=0)
    x_star_arr        = x1_val[used_indices]
    mask_arr          = mask_val[used_indices]
    pelvis_world_arr  = pelvis_world[used_indices]

    fce_arr = np.concatenate(fce_per_sample_all) if fce_per_sample_all else np.array([], dtype=np.float32)

    payload = {
        "psi":               psi_arr.astype(np.float32),
        "x_star":            x_star_arr.astype(np.float32),
        "x0_hat":            x0_hat_arr.astype(np.float32),
        "mask":              mask_arr.astype(bool),
        "pelvis_world":      pelvis_world_arr.astype(np.float32),
        "f_xstar":           f_xstar_arr.astype(np.float32),
        "f_x0":              f_x0_arr.astype(np.float32),
        "completeness_rel":  comp_rel_arr.astype(np.float32),
        "pelvis_leak_abs":   pelvis_leak_abs_arr.astype(np.float32),
        "pelvis_leak_frac":  pelvis_leak_frac_arr.astype(np.float32),
        "class_idx":         class_idx_arr.astype(np.int32),
        "fce_per_sample":    fce_arr.astype(np.float32),
        "fce_indices":       np.array(fce_indices, dtype=np.int32),
        "clip_idx":          np.asarray(used_indices, dtype=np.int32),
        "meta_pid":          np.asarray([meta_pid[i] for i in used_indices], dtype=object),
        "meta_walk_id":      np.asarray([meta_walk_id[i] for i in used_indices], dtype=object),
        "meta_seq_key":      np.asarray([meta_seq_key[i] for i in used_indices], dtype=object),
        "meta_clip_idx":     np.asarray([meta_clip_idx[i] for i in used_indices], dtype=np.int32),
        "meta_updrs_gait":   np.asarray([meta_updrs[i]     for i in used_indices], dtype=np.int32),
        "stats_mean":        stats_mean_np.astype(np.float32),
        "stats_std":         stats_std_np.astype(np.float32),
    }
    np.savez_compressed(psi_path, **payload)
    print(f"[flow_shap] wrote {psi_path}", flush=True)

    # ----- summary -----
    summary = {
        "config_path":        str(cfg_path.relative_to(PROJECT_ROOT)) if cfg_path.is_relative_to(PROJECT_ROOT) else str(cfg_path),
        "flow_config":        cfg["flow_config"],
        "flow_checkpoint":    cfg["flow_checkpoint"],
        "backbone":           cfg["backbone"],
        "classifier_checkpoint": cfg["classifier_checkpoint"],
        "dataset":            dataset_name,
        "split":              cfg.get("data", {}).get("split", "val"),
        "num_folds":          int(cache.get("num_folds", flow_cfg.get("num_folds", -1))) if "num_folds" in cache else flow_cfg.get("num_folds", -1),
        "fold":               int(cache.get("fold", flow_cfg.get("fold", -1))) if "fold" in cache else flow_cfg.get("fold", -1),
        "num_clips_processed": int(psi_arr.shape[0]),
        "num_classes":        num_classes,
        "num_ode_steps":      num_ode_steps,
        "solver":             solver_method,
        "class_policy":       cfg.get("class_policy", "predicted"),
        **_summarise(comp_rel_arr, "completeness_rel"),
        "frac_completeness_above_10pct": float((comp_rel_arr > 0.10).mean()) if comp_rel_arr.size else float("nan"),
        "frac_completeness_above_20pct": float((comp_rel_arr > 0.20).mean()) if comp_rel_arr.size else float("nan"),
        **_summarise(pelvis_leak_frac_arr, "pelvis_leak_frac"),
        **_summarise(pelvis_leak_abs_arr,  "pelvis_leak_abs"),
        **_summarise(fce_arr, "fce"),
        **_summarise(np.abs(f_xstar_arr - f_x0_arr), "delta_f_abs"),
        "class_idx_histogram": {
            str(c): int((class_idx_arr == c).sum()) for c in range(num_classes)
        },
        "elapsed_seconds":    float(time.time() - t0),
        "output_psi_path":    str(psi_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[flow_shap] wrote {summary_path}", flush=True)
    print(json.dumps(
        {k: summary[k] for k in [
            "num_clips_processed", "completeness_rel_median", "completeness_rel_p95",
            "frac_completeness_above_10pct", "pelvis_leak_frac_median",
            "pelvis_leak_frac_p95", "fce_median", "delta_f_abs_median",
            "class_idx_histogram",
        ]},
        indent=2,
    ))


if __name__ == "__main__":
    main()
