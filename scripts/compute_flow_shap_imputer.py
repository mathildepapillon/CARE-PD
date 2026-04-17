"""
compute_flow_shap_imputer.py — Shapley-via-imputer attribution using a trained
flow-matching velocity field as a RePaint-style conditional imputer.

This is the *imputer* counterpart to :mod:`scripts.compute_flow_shap`
(which implements Integrated-Gradients-along-flow a.k.a. IG-flow /
Aumann-Shapley). Where IG-flow answers "what would a per-coordinate
Aumann-Shapley value of :math:`f` be along the learned flow path", this
driver answers "what is the KernelSHAP value of joint :math:`j` under the
conditional distribution :math:`p_{\\theta}(x_{\\bar S} \\mid x_S)` induced by
the flow field?".

Both methods emit a ``psi.npz`` with the exact same schema
(``psi``, ``x_star``, ``x0_hat``, ``pelvis_world``, ``mask``, ``f_xstar``,
``f_x0``, ``class_idx``, and metadata) so downstream faithfulness /
comparison scripts (``compute_flow_shap_faithfulness.py``,
``compare_flow_shap_faithfulness.py``) consume them interchangeably.

Algorithm per clip:
  1. Sample ``N`` KernelSHAP coalitions over the 16 non-pelvis joints plus
     their complements (Lundberg & Lee 2017 variance-reduction trick).
  2. For each coalition, use :class:`model.flow_shap.imputer.FlowImputer`
     to draw ``n_completion_samples`` RePaint-style completions where
     observed joints follow the CondOT linear path and hidden joints follow
     the learned ODE.
  3. Compute the classifier's target-class *probability* on every
     completion, average across samples to get :math:`\\hat v(S)`.
  4. Run weighted-least-squares Shapley (``_solve_shapley_wls``) with the
     empty/full coalitions anchored as soft constraints.
  5. Broadcast the resulting ``phi_j`` across ``(T, C)`` and zero the
     pelvis channel so downstream rankers agree with the IG-flow schema.

Usage::

    python scripts/compute_flow_shap_imputer.py \\
        --config configs/flow_shap/bmclab_potr_fold1.json \\
        --flow_checkpoint experiment_outs/flow_matching/bmclab_h36m3d_fold1_seed123_xl/last.ckpt \\
        --n_kernel_samples 250 --n_completion_samples 20 \\
        --num_ode_steps 100 --solver midpoint \\
        --output_dir experiment_outs/flow_shap/bmclab_potr_fold1/seed123_xl_imputer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.actor.shap_compute import (
    _sample_kernel_coalitions,
    _solve_shapley_wls,
)
from model.flow_shap import FlowImputer, build_classifier_fn
from model.flow_shap.data_loading import (
    load_flow_cache,
    load_pelvis_world_for_val,
)
from scripts.compute_flow_shap import (
    _load_velocity_net,
    _resolve_class_idx,
    _summarise,
)


PELVIS_IDX = 0
J_FULL = 17
M_PLAYERS = J_FULL - 1                    # 16 non-pelvis joints are the players


# ---------------------------------------------------------------------------
# Per-clip Shapley driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def _classifier_prob(
    full_logits_fn,
    x_flow: torch.Tensor,           # (B, T, J, 3) flow space
    ctx_template: Dict[str, torch.Tensor],
    class_idx: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Softmax probability of ``class_idx`` for each flow-space clip.

    ``ctx_template`` is built per-clip (a single ``(T, 3)`` pelvis trajectory
    and ``(T,)`` mask); we expand it to the completion batch size here.
    """
    N = x_flow.shape[0]
    outs: List[torch.Tensor] = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B = end - start
        ctx = {
            "pelvis_world": ctx_template["pelvis_world"].unsqueeze(0).expand(B, -1, -1),
            "mask":         ctx_template["mask"].unsqueeze(0).expand(B, -1),
            "class_idx":    torch.full(
                (B,), int(class_idx),
                device=x_flow.device, dtype=torch.long,
            ),
        }
        logits = full_logits_fn(x_flow[start:end], ctx)                  # (B, C)
        probs = torch.softmax(logits, dim=-1)
        outs.append(probs[:, int(class_idx)].detach())
    return torch.cat(outs, dim=0)


def _attribute_one_clip(
    *,
    imputer: FlowImputer,
    full_logits_fn,
    classifier_fn,
    x_flow: torch.Tensor,              # (T, J, 3) flow space
    pelvis_world: torch.Tensor,        # (T, 3)
    mask_t: torch.Tensor,              # (T,) bool
    class_idx: int,
    n_kernel_samples: int,
    n_completion_samples: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Run KernelSHAP-over-joints on a single clip.

    Returns a dict containing ``phi`` (M,), ``v_empty``, ``v_full``, plus
    the classifier readouts at :math:`x_\\star` and :math:`x_0=\\mathbf 0`.
    """
    device = x_flow.device
    T, J, C = x_flow.shape
    assert J == J_FULL and C == 3, f"expected (T, 17, 3); got {(T, J, C)}"

    # ``x_jct`` in classifier / ActorSHAP layout (1, J, C, T) — FlowImputer's
    # expected input. Because we leave stats_mean / stats_std as ``None`` the
    # imputer treats this input as already being in flow space and returns
    # flow-space completions in the same layout.
    x_jct = x_flow.permute(1, 2, 0).contiguous().unsqueeze(0)            # (1, J, C, T)
    pad_mask = mask_t.unsqueeze(0).contiguous()                          # (1, T)

    # --- KernelSHAP coalitions over the 16 non-pelvis joints. ---------
    coalitions, weights = _sample_kernel_coalitions(
        M_PLAYERS, n_kernel_samples, rng,
    )                                                                    # (2N, 16), (2N,)
    N_coal = coalitions.shape[0]

    # Boundary coalitions (all-zeros, all-ones) for v_empty / v_full; we
    # feed these as soft constraints into _solve_shapley_wls.
    empty_c = np.zeros((1, M_PLAYERS), dtype=int)
    full_c  = np.ones((1, M_PLAYERS),  dtype=int)
    all_coalitions = np.concatenate([coalitions, empty_c, full_c], axis=0)  # (2N+2, 16)

    # --- Value function per coalition via FlowImputer + classifier. ---
    v_vals = np.zeros(N_coal + 2, dtype=np.float64)

    ctx_template = {"pelvis_world": pelvis_world, "mask": mask_t}

    for i in range(all_coalitions.shape[0]):
        # Observed = always pelvis (joint 0) + joints where coalition bit = 1.
        obs_j = np.zeros(J_FULL, dtype=bool)
        obs_j[PELVIS_IDX] = True
        obs_j[1:] = all_coalitions[i].astype(bool)
        obs_mask = torch.from_numpy(obs_j).to(device).view(1, J_FULL, 1).bool()

        # Draw n_completion_samples conditional completions (flow-space).
        comps = imputer.sample_completions(
            x_jct, None, pad_mask, None,
            coalition_mask=obs_mask,
            n_samples=n_completion_samples,
        )                                                                # list of (1, J, C, T)
        x_compl = torch.cat(comps, dim=0)                                # (N, J, C, T)
        x_compl_flow = x_compl.permute(0, 3, 1, 2).contiguous()          # (N, T, J, C)

        probs = _classifier_prob(
            full_logits_fn, x_compl_flow, ctx_template,
            class_idx=class_idx, chunk_size=chunk_size,
        )
        v_vals[i] = float(probs.mean().item())

    v_empty = v_vals[-2]
    v_full  = v_vals[-1]
    interior_coals   = all_coalitions[:N_coal]
    interior_values  = v_vals[:N_coal]
    interior_weights = weights

    phi = _solve_shapley_wls(
        interior_coals, interior_values, interior_weights,
        v_empty=v_empty, v_full=v_full,
    )                                                                    # (16,)

    # --- Classifier readouts at x_star and x0=zeros for bookkeeping. ---
    with torch.no_grad():
        # f_xstar as *logit* (matches compute_flow_shap schema).
        x1 = x_flow.unsqueeze(0)                                         # (1, T, J, 3)
        f_xstar = classifier_fn(
            x1, {
                "pelvis_world": pelvis_world.unsqueeze(0),
                "mask":         mask_t.unsqueeze(0),
                "class_idx":    torch.tensor([int(class_idx)], device=device, dtype=torch.long),
            },
        ).detach().cpu().item()
        x0 = torch.zeros_like(x1)
        f_x0 = classifier_fn(
            x0, {
                "pelvis_world": pelvis_world.unsqueeze(0),
                "mask":         mask_t.unsqueeze(0),
                "class_idx":    torch.tensor([int(class_idx)], device=device, dtype=torch.long),
            },
        ).detach().cpu().item()

    return {
        "phi":         phi.astype(np.float64),        # (16,)
        "v_empty":     float(v_empty),
        "v_full":      float(v_full),
        "f_xstar":     float(f_xstar),
        "f_x0":        float(f_x0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Flow-SHAP JSON config.")
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--n_kernel_samples", type=int, default=250,
                   help="KernelSHAP sample pairs (total coalitions = 2x this).")
    p.add_argument("--n_completion_samples", type=int, default=20,
                   help="Conditional ODE completions per coalition.")
    p.add_argument("--num_ode_steps", type=int, default=100)
    p.add_argument("--solver", default="midpoint", choices=["euler", "midpoint"],
                   help="Imputer ODE solver. Midpoint (synthetic-validated) "
                        "suffices because we only need sample quality, not "
                        "gradient fidelity.")
    p.add_argument("--output_dir", default=None,
                   help="Override config output_dir.")
    p.add_argument("--flow_checkpoint", default=None,
                   help="Override config flow_checkpoint.")
    p.add_argument("--flow_config", default=None,
                   help="Override config flow_config.")
    p.add_argument("--cache_dir", default=None,
                   help="Override flow cache_dir.")
    p.add_argument("--device", default=None)
    p.add_argument("--chunk_size", type=int, default=64,
                   help="Classifier mini-batch size for per-coalition completions.")
    p.add_argument("--seed", type=int, default=None)
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
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- flow cache + pelvis world ---
    cache_dir_raw = args.cache_dir if args.cache_dir is not None else flow_cfg["cache_dir"]
    cache_dir_path = Path(cache_dir_raw)
    if not cache_dir_path.is_absolute():
        cache_dir_path = PROJECT_ROOT / cache_dir_path
    cache_path = cache_dir_path / "cache.npz"
    print(f"[imputer-shap] loading flow cache: {cache_path}", flush=True)
    cache = load_flow_cache(cache_path, split=cfg.get("data", {}).get("split", "val"))

    x1_val        = cache["x1"]
    mask_val      = cache["mask"]
    seq_len       = int(cache["seq_len"])
    stats_mean_np = cache["stats_mean"]
    stats_std_np  = cache["stats_std"]
    dataset_name  = str(cache["dataset"])
    clip_stride_val = int(cache["clip_stride_val"])

    meta_seq_key   = cache["meta_seq_key"]
    meta_clip_idx  = cache["meta_clip_idx"]
    meta_pid       = cache["meta_pid"]
    meta_walk_id   = cache["meta_walk_id"]
    meta_updrs     = cache["meta_updrs_gait"]

    N = int(x1_val.shape[0])
    print(f"[imputer-shap] split contains {N} clips from dataset={dataset_name!r}", flush=True)

    print("[imputer-shap] recovering world pelvis trajectories...", flush=True)
    pelvis_world = load_pelvis_world_for_val(
        dataset=dataset_name,
        meta_seq_key_val=meta_seq_key,
        meta_clip_idx_val=meta_clip_idx,
        seq_len=seq_len,
        clip_stride_val=clip_stride_val,
    )                                                                    # (N, T, 3)

    # --- velocity net + imputer ---
    flow_ckpt_rel = args.flow_checkpoint if args.flow_checkpoint is not None else cfg["flow_checkpoint"]
    flow_ckpt = Path(flow_ckpt_rel)
    if not flow_ckpt.is_absolute():
        flow_ckpt = PROJECT_ROOT / flow_ckpt
    print(f"[imputer-shap] loading velocity net: {flow_ckpt}", flush=True)
    velocity_net = _load_velocity_net(flow_cfg, str(flow_ckpt), device)

    imputer = FlowImputer(
        velocity_net, device,
        stats_mean=None,   # we pass flow-space x directly, no normalisation needed
        stats_std=None,
        num_steps=int(args.num_ode_steps),
        solver=str(args.solver),
    )
    print(f"[imputer-shap] FlowImputer(solver={args.solver}, K={args.num_ode_steps})", flush=True)

    # --- classifier adapter ---
    stats_mean = torch.from_numpy(stats_mean_np).to(device)
    stats_std  = torch.from_numpy(stats_std_np).to(device)
    print(f"[imputer-shap] loading classifier: backbone={cfg['backbone']} "
          f"ckpt={cfg['classifier_checkpoint']}", flush=True)
    classifier_fn, full_logits_fn, motion_encoder, backbone_params = build_classifier_fn(
        backbone_name=cfg["backbone"],
        classifier_ckpt=str(PROJECT_ROOT / cfg["classifier_checkpoint"]),
        flow_stats_mean=stats_mean,
        flow_stats_std=stats_std,
        class_idx=0,
        device=device,
        config_file=cfg.get("classifier_config_file", "BMCLab.json"),
        num_folds=int(cfg.get("classifier_num_folds", 23)),
        fold=int(cfg["data"].get("fold", flow_cfg.get("fold", 1))),
    )
    num_classes = int(backbone_params["num_classes"])

    # --- indices / output dir ---
    indices = list(range(N))
    max_clips = (args.max_clips if args.max_clips is not None
                 else cfg.get("data", {}).get("max_clips", None))
    if max_clips is not None and max_clips > 0:
        indices = indices[:int(max_clips)]
        print(f"[imputer-shap] limiting to {len(indices)} clips (max_clips)", flush=True)

    out_dir_raw = args.output_dir if args.output_dir is not None else cfg["output_dir"]
    out_dir = Path(out_dir_raw)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    psi_path = out_dir / "psi.npz"
    summary_path = out_dir / "summary.json"

    # --- class policy resolution (batched on x_star for speed) ---
    class_policy_raw = cfg.get("class_policy", "predicted")
    class_policy: Any = class_policy_raw
    if isinstance(class_policy_raw, str) and class_policy_raw.isdigit():
        class_policy = int(class_policy_raw)

    print(f"[imputer-shap] resolving class_policy={class_policy!r} on all x_star...",
          flush=True)
    x_star_device = torch.from_numpy(x1_val[indices]).to(device=device, dtype=torch.float32)
    mask_device   = torch.from_numpy(mask_val[indices]).to(device=device, dtype=torch.bool)
    pelvis_device = torch.from_numpy(pelvis_world[indices]).to(device=device, dtype=torch.float32)
    cls_ctx = {"pelvis_world": pelvis_device, "mask": mask_device}
    cls_idx_all = _resolve_class_idx(
        class_policy, full_logits_fn, x_star_device, cls_ctx, num_classes,
        labels=meta_updrs, batch_ids=indices,
    ).cpu().numpy().astype(np.int32)
    del x_star_device, mask_device, pelvis_device, cls_ctx
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- main loop ---
    phi_all    = np.zeros((len(indices), M_PLAYERS), dtype=np.float64)
    v_empty_all = np.zeros(len(indices), dtype=np.float64)
    v_full_all  = np.zeros(len(indices), dtype=np.float64)
    f_xstar_all = np.zeros(len(indices), dtype=np.float64)
    f_x0_all    = np.zeros(len(indices), dtype=np.float64)
    used_indices: List[int] = []

    t0 = time.time()
    for k, idx in enumerate(indices):
        x_flow = torch.from_numpy(x1_val[idx]).to(device=device, dtype=torch.float32)
        pelvis_t = torch.from_numpy(pelvis_world[idx]).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(mask_val[idx]).to(device=device, dtype=torch.bool)

        result = _attribute_one_clip(
            imputer=imputer,
            full_logits_fn=full_logits_fn,
            classifier_fn=classifier_fn,
            x_flow=x_flow,
            pelvis_world=pelvis_t,
            mask_t=mask_t,
            class_idx=int(cls_idx_all[k]),
            n_kernel_samples=int(args.n_kernel_samples),
            n_completion_samples=int(args.n_completion_samples),
            chunk_size=int(args.chunk_size),
            rng=rng,
        )
        phi_all[k]    = result["phi"]
        v_empty_all[k] = result["v_empty"]
        v_full_all[k]  = result["v_full"]
        f_xstar_all[k] = result["f_xstar"]
        f_x0_all[k]    = result["f_x0"]
        used_indices.append(int(idx))

        if (k + 1) % max(1, len(indices) // 20) == 0:
            elapsed = time.time() - t0
            avg = elapsed / (k + 1)
            eta = avg * (len(indices) - k - 1)
            print(f"[imputer-shap] {k+1}/{len(indices)}  "
                  f"t={elapsed:6.1f}s  avg={avg:5.1f}s/clip  eta={eta/60:6.1f}m  "
                  f"median |v_full-v_empty|={np.median(np.abs(v_full_all[:k+1]-v_empty_all[:k+1])):.4f}",
                  flush=True)

    # --- broadcast per-joint Shapley to (N, T, J, C) for faithfulness pipeline. ---
    N_used = len(used_indices)
    T = int(x1_val.shape[1])
    psi_arr = np.zeros((N_used, T, J_FULL, 3), dtype=np.float32)
    # phi_all[k] holds 16 values for joints 1..16; joint 0 (pelvis) stays at 0.
    psi_joint_level = np.zeros((N_used, J_FULL), dtype=np.float32)
    psi_joint_level[:, 1:] = phi_all.astype(np.float32)
    # Broadcast to (N, T, J, C) by dividing across T*C so that summing
    # |psi|.sum(axis=(T, C)) in _rank_joints_by_psi recovers |phi_j| * T * C,
    # which preserves the ranking. We don't need the magnitudes themselves.
    # Faithfulness scripts only use the ranking + magnitudes for lift stats.
    psi_arr[:] = psi_joint_level[:, None, :, None] / float(T * 3)        # (N, T, J, C)

    x_star_arr       = x1_val[used_indices].astype(np.float32)
    mask_arr         = mask_val[used_indices].astype(bool)
    pelvis_world_arr = pelvis_world[used_indices].astype(np.float32)
    x0_hat_arr       = np.zeros_like(x_star_arr)                         # flow-space zero
    comp_rel = np.abs(phi_all.sum(axis=1) - (v_full_all - v_empty_all)) / \
        np.maximum(np.abs(v_full_all - v_empty_all), 1e-8)

    payload = {
        "psi":               psi_arr.astype(np.float32),
        "x_star":            x_star_arr,
        "x0_hat":            x0_hat_arr,
        "mask":              mask_arr,
        "pelvis_world":      pelvis_world_arr,
        "f_xstar":           f_xstar_all.astype(np.float32),
        "f_x0":              f_x0_all.astype(np.float32),
        "completeness_rel":  comp_rel.astype(np.float32),
        "pelvis_leak_abs":   np.zeros(N_used, dtype=np.float32),
        "pelvis_leak_frac":  np.zeros(N_used, dtype=np.float32),
        "class_idx":         np.asarray(cls_idx_all[:N_used], dtype=np.int32),
        "fce_per_sample":    np.array([], dtype=np.float32),
        "fce_indices":       np.array([], dtype=np.int32),
        "clip_idx":          np.asarray(used_indices, dtype=np.int32),
        "meta_pid":          np.asarray([meta_pid[i]     for i in used_indices], dtype=object),
        "meta_walk_id":      np.asarray([meta_walk_id[i] for i in used_indices], dtype=object),
        "meta_seq_key":      np.asarray([meta_seq_key[i] for i in used_indices], dtype=object),
        "meta_clip_idx":     np.asarray([meta_clip_idx[i] for i in used_indices], dtype=np.int32),
        "meta_updrs_gait":   np.asarray([meta_updrs[i]     for i in used_indices], dtype=np.int32),
        "stats_mean":        stats_mean_np.astype(np.float32),
        "stats_std":         stats_std_np.astype(np.float32),
        # Extra diagnostic fields unique to the imputer pipeline.
        "phi_joint":         psi_joint_level[:N_used],                   # (N, 17)
        "v_empty":           v_empty_all.astype(np.float32),
        "v_full":            v_full_all.astype(np.float32),
    }
    np.savez_compressed(psi_path, **payload)
    print(f"[imputer-shap] wrote {psi_path}", flush=True)

    summary = {
        "config_path":        str(cfg_path.relative_to(PROJECT_ROOT)) if cfg_path.is_relative_to(PROJECT_ROOT) else str(cfg_path),
        "method":             "shapley_via_imputer",
        "flow_config":        cfg["flow_config"],
        "flow_checkpoint":    cfg["flow_checkpoint"],
        "backbone":           cfg["backbone"],
        "classifier_checkpoint": cfg["classifier_checkpoint"],
        "dataset":            dataset_name,
        "split":              cfg.get("data", {}).get("split", "val"),
        "fold":               int(cache.get("fold", flow_cfg.get("fold", -1))) if "fold" in cache else flow_cfg.get("fold", -1),
        "num_clips_processed": int(N_used),
        "num_classes":        num_classes,
        "n_kernel_samples":   int(args.n_kernel_samples),
        "n_completion_samples": int(args.n_completion_samples),
        "num_ode_steps":      int(args.num_ode_steps),
        "solver":             str(args.solver),
        "class_policy":       cfg.get("class_policy", "predicted"),
        **_summarise(comp_rel.astype(np.float32), "completeness_rel"),
        **_summarise(np.abs(f_xstar_all - f_x0_all).astype(np.float32), "delta_f_abs"),
        **_summarise(np.abs(v_full_all - v_empty_all).astype(np.float32), "delta_v_abs"),
        "class_idx_histogram": {
            str(c): int((cls_idx_all[:N_used] == c).sum()) for c in range(num_classes)
        },
        "elapsed_seconds":    float(time.time() - t0),
        "output_psi_path":    str(psi_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[imputer-shap] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
