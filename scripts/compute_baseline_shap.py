"""
compute_baseline_shap.py — KernelSHAP-via-imputer driver for SIMPLE
baseline imputers on real-world CARE-PD data, emitting a ``psi.npz``
in the exact same schema as :mod:`scripts.compute_flow_shap` and
:mod:`scripts.compute_flow_shap_imputer`.

Why this exists
---------------
``evaluate_shap_baselines.py`` already computes PGI/PGU/AUC for
``zero / mean / marginal`` baselines *directly* from its own SHAP
implementation; those headline numbers can be reshaped into the
common schema via :mod:`scripts.jsonl_to_faithfulness_table`.
However, the comparison-plot pipeline
(``compute_flow_shap_faithfulness.py`` → ``compare_flow_shap_faithfulness.py``)
is built around ``psi.npz`` inputs.  To enable fully *uniform*
end-to-end plotting (including averaged deletion / insertion curves
with per-step probabilities), this script emits a ``psi.npz`` for each
baseline method using exactly the same KernelSHAP machinery as
``compute_flow_shap_imputer.py``.

Supported methods
-----------------
* ``zero``      — hidden joints set to 0 (flow space).
* ``mean``      — hidden joints set to flow-space mean
                  (which is 0 by construction after z-scoring,
                  but we provide the hook for datasets where the
                  cache's ``stats_mean`` is not pre-subtracted).
* ``marginal``  — hidden joints replaced by the same joint's value
                  in a random TRAIN clip (classical BarShap-style
                  marginal baseline).

Not implemented here
--------------------
* ``gaussian_full`` — tractable only if we fit per-coalition
  conditional parameters once across a fixed coalition set; on
  real-world BMCLab (D≈4131) the per-coalition 3.6k×3.6k Cholesky
  is too slow for the 500-coalition budget.  Use the synthetic
  driver (``evaluate_shap_synthetic.py``) for this baseline.
* ``actor`` — plug-point exists (see ``_build_baseline_imputer``)
  but requires routing through classifier-space normalization;
  left as future work.
* ``lstm_vae`` — ditto.

Usage::

    python scripts/compute_baseline_shap.py \\
        --config configs/flow_shap/bmclab_potr_fold1.json \\
        --method zero \\
        --n_kernel_samples 250 --n_completion_samples 20 \\
        --output_dir experiment_outs/flow_shap/bmclab_potr_fold1/baseline_zero
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.actor.shap_compute import (
    _sample_kernel_coalitions,
    _solve_shapley_wls,
)
from model.flow_shap import build_classifier_fn
from model.flow_shap.data_loading import (
    load_flow_cache,
    load_pelvis_world_for_val,
)
from scripts.compute_flow_shap import (
    _resolve_class_idx,
    _summarise,
)
from scripts.compute_flow_shap_imputer import (
    PELVIS_IDX,
    J_FULL,
    M_PLAYERS,
    _classifier_prob,
)


# ---------------------------------------------------------------------------
# Baseline imputers — (x_jct, obs_mask, n_samples) → list[(1, J, C, T)]
# ---------------------------------------------------------------------------

class ZeroBaselineImputer:
    """Fill hidden joints with zero (flow space)."""

    name = "zero"

    def sample_completions(
        self,
        x_jct: Tensor,
        coalition_mask: Tensor,       # (1, J, 1) bool
        n_samples: int,
        _rng: np.random.Generator,
    ) -> List[Tensor]:
        assert x_jct.ndim == 4, f"expected (1,J,C,T); got {x_jct.shape}"
        assert coalition_mask.shape[-1] == 1
        obs = coalition_mask.to(x_jct.device).bool()            # (1, J, 1)
        hid = (~obs).to(dtype=x_jct.dtype)                      # (1, J, 1) of 1.0 on hidden
        x_fill = x_jct * obs.to(dtype=x_jct.dtype)              # observed kept; hidden → 0
        _ = hid                                                 # zero is the fill value
        return [x_fill for _ in range(n_samples)]


class MeanBaselineImputer:
    """Fill hidden joints with flow-space *per-joint-per-coord-per-time* mean.

    In the CARE-PD flow cache ``stats_mean`` has already been subtracted
    during preprocessing, so the flow-space mean is (approximately) zero.
    Therefore ``mean`` is numerically similar to ``zero`` here; we still
    expose it for schema parity with ``evaluate_shap_baselines.py``.
    """

    name = "mean"

    def __init__(self, mean_jct: Tensor) -> None:
        # mean_jct: (J, C, T) in flow space.
        self.mean_jct = mean_jct

    def sample_completions(
        self,
        x_jct: Tensor,
        coalition_mask: Tensor,
        n_samples: int,
        _rng: np.random.Generator,
    ) -> List[Tensor]:
        obs = coalition_mask.to(x_jct.device).bool()
        obs_f = obs.to(dtype=x_jct.dtype)
        hid_f = (~obs).to(dtype=x_jct.dtype)
        mu = self.mean_jct.to(x_jct.device).unsqueeze(0)        # (1, J, C, T)
        x_fill = x_jct * obs_f + mu * hid_f
        return [x_fill for _ in range(n_samples)]


class MarginalBaselineImputer:
    """Fill hidden joints with donor joints from a random training clip.

    Draws ``n_samples`` donor clips IID per coalition so the expectation
    over the marginal distribution is Monte-Carlo estimated (same
    Lundberg-Lee ``marginal`` convention used in ``evaluate_shap_baselines.py``).
    """

    name = "marginal"

    def __init__(self, train_pool_jct: Tensor) -> None:
        # train_pool_jct: (N_train, J, C, T) flow-space tensor on CPU (float32).
        self.train_pool = train_pool_jct
        self.N_train = int(self.train_pool.shape[0])

    def sample_completions(
        self,
        x_jct: Tensor,
        coalition_mask: Tensor,
        n_samples: int,
        rng: np.random.Generator,
    ) -> List[Tensor]:
        obs = coalition_mask.to(x_jct.device).bool()
        obs_f = obs.to(dtype=x_jct.dtype)
        hid_f = (~obs).to(dtype=x_jct.dtype)

        donor_idx = rng.integers(0, self.N_train, size=n_samples).tolist()
        outs: List[Tensor] = []
        for di in donor_idx:
            donor = self.train_pool[int(di)].to(x_jct.device, dtype=x_jct.dtype)
            donor = donor.unsqueeze(0)                          # (1, J, C, T)
            x_fill = x_jct * obs_f + donor * hid_f
            outs.append(x_fill)
        return outs


BaselineImputer = Any  # protocol above


def _build_baseline_imputer(
    method: str,
    *,
    device: torch.device,
    x1_val: np.ndarray,
    cache_dir: Path,
    dataset_name: str,
) -> BaselineImputer:
    if method == "zero":
        return ZeroBaselineImputer()
    if method == "mean":
        # Estimate mean over the VAL split — in practice ~0 after z-score,
        # but this makes the baseline explicit and dataset-agnostic.
        mean_tjc = x1_val.mean(axis=0)                          # (T, J, C)
        mean_jct = torch.from_numpy(
            np.transpose(mean_tjc, (1, 2, 0))                   # (J, C, T)
        ).to(device, dtype=torch.float32)
        return MeanBaselineImputer(mean_jct)
    if method == "marginal":
        # Need the TRAIN split x1 → load the same cache with split="train".
        cache_path = cache_dir / "cache.npz"
        train_cache = load_flow_cache(cache_path, split="train")
        x1_train = train_cache["x1"]                            # (N, T, J, C)
        x1_train_jct = np.transpose(x1_train, (0, 2, 3, 1))     # (N, J, C, T)
        return MarginalBaselineImputer(torch.from_numpy(x1_train_jct).float())
    if method in ("actor", "lstm_vae", "gaussian_full", "gaussian_temporal"):
        raise NotImplementedError(
            f"method={method!r} not yet supported in compute_baseline_shap.py; "
            "use evaluate_shap_synthetic.py (synthetic) or the flow/actor "
            "trained checkpoints via their dedicated drivers."
        )
    raise ValueError(f"unknown --method {method!r}")


# ---------------------------------------------------------------------------
# Per-clip Shapley driver (mirrors compute_flow_shap_imputer._attribute_one_clip)
# ---------------------------------------------------------------------------

def _attribute_one_clip(
    *,
    imputer: BaselineImputer,
    full_logits_fn: Callable,
    classifier_fn: Callable,
    x_flow: Tensor,
    pelvis_world: Tensor,
    mask_t: Tensor,
    class_idx: int,
    n_kernel_samples: int,
    n_completion_samples: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    device = x_flow.device
    T, J, C = x_flow.shape
    assert J == J_FULL and C == 3, f"expected (T, 17, 3); got {(T, J, C)}"

    x_jct = x_flow.permute(1, 2, 0).contiguous().unsqueeze(0)   # (1, J, C, T)

    coalitions, weights = _sample_kernel_coalitions(
        M_PLAYERS, n_kernel_samples, rng,
    )
    N_coal = coalitions.shape[0]

    empty_c = np.zeros((1, M_PLAYERS), dtype=int)
    full_c  = np.ones((1, M_PLAYERS),  dtype=int)
    all_coalitions = np.concatenate([coalitions, empty_c, full_c], axis=0)

    v_vals = np.zeros(N_coal + 2, dtype=np.float64)
    ctx_template = {"pelvis_world": pelvis_world, "mask": mask_t}

    for i in range(all_coalitions.shape[0]):
        obs_j = np.zeros(J_FULL, dtype=bool)
        obs_j[PELVIS_IDX] = True
        obs_j[1:] = all_coalitions[i].astype(bool)
        obs_mask = torch.from_numpy(obs_j).to(device).view(1, J_FULL, 1).bool()

        comps = imputer.sample_completions(
            x_jct, obs_mask, n_completion_samples, rng,
        )                                                        # list of (1, J, C, T)
        x_compl = torch.cat(comps, dim=0)
        x_compl_flow = x_compl.permute(0, 3, 1, 2).contiguous()  # (N, T, J, C)

        probs = _classifier_prob(
            full_logits_fn, x_compl_flow, ctx_template,
            class_idx=class_idx, chunk_size=chunk_size,
        )
        v_vals[i] = float(probs.mean().item())

    v_empty = v_vals[-2]
    v_full  = v_vals[-1]
    phi = _solve_shapley_wls(
        all_coalitions[:N_coal], v_vals[:N_coal], weights,
        v_empty=v_empty, v_full=v_full,
    )

    with torch.no_grad():
        x1 = x_flow.unsqueeze(0)
        ctx_single = {
            "pelvis_world": pelvis_world.unsqueeze(0),
            "mask":         mask_t.unsqueeze(0),
            "class_idx":    torch.tensor([int(class_idx)], device=device, dtype=torch.long),
        }
        f_xstar = classifier_fn(x1, ctx_single).detach().cpu().item()
        x0 = torch.zeros_like(x1)
        f_x0 = classifier_fn(x0, ctx_single).detach().cpu().item()

    return {
        "phi":     phi.astype(np.float64),
        "v_empty": float(v_empty),
        "v_full":  float(v_full),
        "f_xstar": float(f_xstar),
        "f_x0":    float(f_x0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Flow-SHAP JSON config (reused: gives classifier ckpt, "
                        "fold, backbone).")
    p.add_argument("--method", required=True,
                   choices=["zero", "mean", "marginal"],
                   help="Baseline imputation method.")
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--n_kernel_samples", type=int, default=250,
                   help="KernelSHAP sample pairs (total coalitions = 2x this).")
    p.add_argument("--n_completion_samples", type=int, default=20,
                   help="Donor / fill samples per coalition. For zero/mean "
                        "the extra samples are redundant (deterministic fill) "
                        "so values>1 only increase wall time.")
    p.add_argument("--output_dir", default=None,
                   help="Override config output_dir.")
    p.add_argument("--flow_config", default=None,
                   help="Override config flow_config.")
    p.add_argument("--cache_dir", default=None,
                   help="Override flow cache_dir (needed because we re-load "
                        "the cache for the TRAIN split when --method=marginal).")
    p.add_argument("--device", default=None)
    p.add_argument("--chunk_size", type=int, default=64)
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
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
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
    print(f"[baseline-shap:{args.method}] loading flow cache: {cache_path}", flush=True)
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
    print(f"[baseline-shap:{args.method}] split contains {N} clips from "
          f"dataset={dataset_name!r}", flush=True)

    print(f"[baseline-shap:{args.method}] recovering world pelvis trajectories...",
          flush=True)
    pelvis_world = load_pelvis_world_for_val(
        dataset=dataset_name,
        meta_seq_key_val=meta_seq_key,
        meta_clip_idx_val=meta_clip_idx,
        seq_len=seq_len,
        clip_stride_val=clip_stride_val,
    )

    # --- baseline imputer ---
    imputer = _build_baseline_imputer(
        args.method, device=device,
        x1_val=x1_val, cache_dir=cache_dir_path, dataset_name=dataset_name,
    )
    print(f"[baseline-shap:{args.method}] imputer={imputer.__class__.__name__}", flush=True)

    # --- classifier adapter ---
    stats_mean = torch.from_numpy(stats_mean_np).to(device)
    stats_std  = torch.from_numpy(stats_std_np).to(device)
    print(f"[baseline-shap:{args.method}] loading classifier: backbone={cfg['backbone']} "
          f"ckpt={cfg['classifier_checkpoint']}", flush=True)
    classifier_fn, full_logits_fn, _motion_encoder, backbone_params = build_classifier_fn(
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
        print(f"[baseline-shap:{args.method}] limiting to {len(indices)} clips",
              flush=True)

    out_dir_raw = args.output_dir if args.output_dir is not None else cfg["output_dir"]
    out_dir = Path(out_dir_raw)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    psi_path = out_dir / "psi.npz"
    summary_path = out_dir / "summary.json"

    # --- class policy resolution ---
    class_policy_raw = cfg.get("class_policy", "predicted")
    class_policy: Any = class_policy_raw
    if isinstance(class_policy_raw, str) and class_policy_raw.isdigit():
        class_policy = int(class_policy_raw)

    print(f"[baseline-shap:{args.method}] resolving class_policy={class_policy!r}...",
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

    # --- For deterministic baselines (zero/mean) we can safely set
    # n_completion_samples=1 to skip redundant work; we warn if the user
    # asked for more.
    n_comp_samples = int(args.n_completion_samples)
    if args.method in ("zero", "mean") and n_comp_samples > 1:
        print(f"[baseline-shap:{args.method}] method is deterministic; "
              f"reducing n_completion_samples {n_comp_samples} → 1",
              flush=True)
        n_comp_samples = 1

    # --- main loop ---
    phi_all     = np.zeros((len(indices), M_PLAYERS), dtype=np.float64)
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
            n_completion_samples=n_comp_samples,
            chunk_size=int(args.chunk_size),
            rng=rng,
        )
        phi_all[k]     = result["phi"]
        v_empty_all[k] = result["v_empty"]
        v_full_all[k]  = result["v_full"]
        f_xstar_all[k] = result["f_xstar"]
        f_x0_all[k]    = result["f_x0"]
        used_indices.append(int(idx))

        step = max(1, len(indices) // 20)
        if (k + 1) % step == 0:
            elapsed = time.time() - t0
            avg = elapsed / (k + 1)
            eta = avg * (len(indices) - k - 1)
            print(f"[baseline-shap:{args.method}] {k+1}/{len(indices)}  "
                  f"t={elapsed:6.1f}s  avg={avg:5.2f}s/clip  eta={eta/60:5.1f}m",
                  flush=True)

    # --- broadcast per-joint phi to (N, T, J, C) ---
    N_used = len(used_indices)
    T = int(x1_val.shape[1])
    psi_arr = np.zeros((N_used, T, J_FULL, 3), dtype=np.float32)
    psi_joint_level = np.zeros((N_used, J_FULL), dtype=np.float32)
    psi_joint_level[:, 1:] = phi_all.astype(np.float32)
    psi_arr[:] = psi_joint_level[:, None, :, None] / float(T * 3)

    x_star_arr       = x1_val[used_indices].astype(np.float32)
    mask_arr         = mask_val[used_indices].astype(bool)
    pelvis_world_arr = pelvis_world[used_indices].astype(np.float32)
    x0_hat_arr       = np.zeros_like(x_star_arr)
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
        "phi_joint":         psi_joint_level[:N_used],
        "v_empty":           v_empty_all.astype(np.float32),
        "v_full":            v_full_all.astype(np.float32),
    }
    np.savez_compressed(psi_path, **payload)
    print(f"[baseline-shap:{args.method}] wrote {psi_path}", flush=True)

    summary = {
        "config_path":        str(cfg_path.relative_to(PROJECT_ROOT))
                              if cfg_path.is_relative_to(PROJECT_ROOT) else str(cfg_path),
        "method":             f"baseline_{args.method}",
        "flow_config":        cfg["flow_config"],
        "backbone":           cfg["backbone"],
        "classifier_checkpoint": cfg["classifier_checkpoint"],
        "dataset":            dataset_name,
        "split":              cfg.get("data", {}).get("split", "val"),
        "fold":               int(flow_cfg.get("fold", -1)),
        "num_clips_processed": int(N_used),
        "num_classes":        num_classes,
        "n_kernel_samples":   int(args.n_kernel_samples),
        "n_completion_samples": n_comp_samples,
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
    print(f"[baseline-shap:{args.method}] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
