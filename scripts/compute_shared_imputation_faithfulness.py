"""
compute_shared_imputation_faithfulness.py — Evaluate *all* SHAP rankings
(zero / mean / marginal KernelSHAP baselines + OTFlow-SHAP across multiple
flow seeds) on the same 240-sequence SUB01 test fold, using a **single,
shared imputation method** chosen by ``--imputation``.

Motivation
==========
The self-referential evaluation in ``evaluate_shap_baselines.py`` lets each
KernelSHAP baseline evaluate its own ranking using its own imputation
function (zero uses zero imputation, marginal uses marginal donors, etc.).
That protocol entangles two distinct questions:

1. Does this method's *ranking* identify the joints the classifier relies on?
2. Does this method's *imputation* produce realistic/informative counterfactuals?

Because zero-KernelSHAP's ranking is literally a Shapley decomposition of
``f(x) − f(0)``, running it with zero imputation at eval time gives it a
built-in alignment advantage that the other methods do not share. OTFlow-
SHAP's native reference is the flow back-solve ``x0_hat`` (on-manifold),
not zero, so zero imputation is *not* its home turf.

This script fixes the imputation to a single shared function for a given
run and evaluates every ranking source under it. Any remaining difference
then isolates ranking quality on that particular evaluation counterfactual.

Recommended runs:

* ``--imputation marginal`` — each joint is replaced by real donor samples
  from the training pool. On-manifold, but not any method's native
  reference ⇒ equally "foreign" to zero-/mean-/flow-SHAP.
* ``--imputation flow_imputer`` — joints are filled with RePaint-style
  conditional samples from the trained flow-matching velocity net
  (:class:`model.flow_shap.imputer.FlowImputer`). Also on-manifold and is
  OTFlow-SHAP's semantic home turf.
* ``--imputation zero`` — zero imputation, useful as a reference row.

Outputs ``per_sequence.jsonl`` + ``aggregate.json`` with a ``faithfulness``
dict keyed by ranking source: ``zero``, ``mean``, ``marginal``,
``flow_seed{42,123,456}``, etc.

Usage::

    python scripts/compute_shared_imputation_faithfulness.py \\
        --imputation marginal \\
        --baseline_jsonl results/shap_baselines_potr_bmclab_fold1_current/per_sequence.jsonl \\
        --flow_psi_paths \\
            experiment_outs/flow_shap/bmclab_potr_fold1_classifier_eval/seed42/psi.npz \\
            experiment_outs/flow_shap/bmclab_potr_fold1_classifier_eval/seed123/psi.npz \\
            experiment_outs/flow_shap/bmclab_potr_fold1_classifier_eval/seed456/psi.npz \\
        --backbone potr --config BMCLab.json --num_folds 23 --fold 1 \\
        --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \\
        --output_dir results/shap_shared_imputation_marginal_fold1 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: eagerly import the PyPI ``flow_matching`` package before anything in
# ``model.potr.*`` is imported. Several POTR modules call
# ``sys.path.insert(0, thispath + "/../")``, which exposes our local
# ``model/flow_matching/`` directory at the top-level import root and
# shadows the installed Meta ``flow_matching`` package (which has the
# ``solver`` submodule we need). Caching the real package in
# ``sys.modules`` here prevents that collision when ``FlowImputer`` is
# lazily imported later.
import flow_matching  # noqa: E402,F401
import flow_matching.solver  # noqa: E402,F401

from data.dataloaders import collate_fn  # noqa: E402
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets  # noqa: E402
from model.actor.shap_eval_shared import (  # noqa: E402
    _load_backbone_params,
    _raw_data_args,
    build_classifier_fn,
    build_train_pool,
    build_zscore_stats_for_potr,
    check_p_full_warning,
    load_motion_encoder,
)
from model.actor.shap_masking import H36M_JOINT_NAMES  # noqa: E402
from model.actor.shap_metrics import (  # noqa: E402
    _class_prob,
    compute_spatial_faithfulness_batched,
)


_VIEW_SUFFIX = re.compile(r"_view\d+$")


def _strip_view(video_name: str) -> str:
    return _VIEW_SUFFIX.sub("", video_name)


# ---------------------------------------------------------------------------
# Ranking loaders
# ---------------------------------------------------------------------------

def aggregate_psi_signed(psi_path: Path) -> dict[str, np.ndarray]:
    """Axiomatic Shapley per-joint attributions from ``psi.npz``.

    phi_j = Σ_clips Σ_{t,c} psi[clip, t, j, c]   (signed).

    Returns a dict mapping stripped ``seq_key`` → ``(17,)`` array.
    """
    d = np.load(str(psi_path), allow_pickle=True)
    psi = d["psi"]                               # (N_clips, T, J, C)
    seq_keys = d["meta_seq_key"]                 # (N_clips,)

    per_clip = psi.sum(axis=(1, 3))              # (N_clips, J), signed
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, sk in enumerate(seq_keys.tolist()):
        acc[str(sk)].append(per_clip[i])
    return {k: np.sum(np.stack(vs, axis=0), axis=0) for k, vs in acc.items()}


def load_baseline_rankings(jsonl_path: Path) -> dict[int, dict[str, dict[str, float]]]:
    """Load per-sequence KernelSHAP scalars for zero/mean/marginal.

    Returns ``{seq_idx: {method: {joint_name: phi}}}``.
    """
    out: dict[int, dict[str, dict[str, float]]] = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx = int(r["seq_idx"])
            sv = r["shap_values"]
            out[idx] = {}
            for m in ("zero", "mean", "marginal"):
                if m not in sv:
                    continue
                out[idx][m] = {
                    j: float(sv[m][j]) for j in H36M_JOINT_NAMES if j in sv[m]
                }
    return out


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def _build_shap_dict(phi_vec: np.ndarray) -> dict[str, float]:
    return {H36M_JOINT_NAMES[j]: float(phi_vec[j]) for j in range(17)}


def _compute_p_ref(
    classifier_fn,
    x: torch.Tensor,
    class_idx: int,
    imputation: str,
    *,
    joint_means: torch.Tensor | None,
    train_pool: torch.Tensor | None,
    flow_imputer,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    n_samples: int,
    seq_idx: int,
) -> float:
    """Empty-coalition value under the shared imputation.

    p_ref = E_{x'∼imputation of all joints}[classifier(x')[class]].

    Used to compute Shapley completeness error; it is NOT used for ranking.
    For rankings whose native reference != this p_ref, the completeness
    field under the shared protocol is expected to be non-zero — that is
    precisely the mismatch we are highlighting.
    """
    device = x.device

    if imputation == "zero":
        return float(_class_prob(classifier_fn, torch.zeros_like(x), class_idx))

    if imputation == "mean":
        assert joint_means is not None
        # joint_means: (J, F, T_ref) broadcast to x shape.
        x_mean = joint_means[None].expand_as(x).contiguous()
        return float(_class_prob(classifier_fn, x_mean, class_idx))

    if imputation == "marginal":
        assert train_pool is not None
        rng = np.random.default_rng(seq_idx + (2 << 30))
        idx = rng.integers(0, train_pool.shape[0], size=n_samples)
        donors = train_pool[idx].to(device)     # (n_samples, J, F, T)
        probs = []
        for d in donors:
            probs.append(_class_prob(classifier_fn, d[None], class_idx))
        return float(np.mean(probs))

    if imputation == "flow_imputer":
        assert flow_imputer is not None
        J = x.shape[1]
        empty = torch.zeros(1, J, dtype=torch.bool, device=device)
        y_tensor = torch.tensor([class_idx], device=device, dtype=torch.long)
        comps = flow_imputer.sample_completions(
            x, y_tensor, mask, lengths, empty, n_samples=n_samples,
        )
        probs = []
        for c in comps:
            c_in = c if c.ndim == 4 else c[None]
            probs.append(_class_prob(classifier_fn, c_in, class_idx))
        return float(np.mean(probs))

    raise ValueError(f"unknown imputation {imputation!r}")


def _maybe_load_flow_imputer(args, device):
    """Instantiate a :class:`FlowImputer` when ``--imputation flow_imputer``.

    Reads z-score stats from the training flow cache so inputs in
    classifier-native space are normalised into flow space before the ODE
    and de-normalised on the way out, matching the pipeline used by
    :mod:`scripts.compute_flow_shap_imputer`.
    """
    if args.imputation != "flow_imputer":
        return None
    if not args.flow_config:
        raise ValueError(
            "--flow_config is required when --imputation flow_imputer."
        )
    if not args.flow_checkpoint:
        raise ValueError(
            "--flow_checkpoint is required when --imputation flow_imputer."
        )

    sys.path.insert(0, str(PROJECT_ROOT))
    from model.flow_shap import FlowImputer  # noqa: E402
    from scripts.compute_flow_shap import _load_velocity_net  # noqa: E402

    flow_cfg_path = Path(args.flow_config)
    if not flow_cfg_path.is_absolute():
        flow_cfg_path = PROJECT_ROOT / flow_cfg_path
    with open(flow_cfg_path) as fh:
        flow_cfg = json.load(fh)

    flow_ckpt = Path(args.flow_checkpoint)
    if not flow_ckpt.is_absolute():
        flow_ckpt = PROJECT_ROOT / flow_ckpt
    print(f"[flow] velocity net ← {flow_ckpt}")
    velocity_net = _load_velocity_net(flow_cfg, str(flow_ckpt), device)

    # Pull z-score stats from the flow cache so the imputer can round-trip
    # between classifier-native root-centered coordinates and flow space.
    cache_dir = flow_cfg.get("cache_dir")
    stats_mean = stats_std = None
    if cache_dir:
        cache_path = Path(cache_dir) / "cache.npz"
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        if cache_path.exists():
            d = np.load(str(cache_path))
            if "stats_mean" in d.files and "stats_std" in d.files:
                stats_mean = torch.from_numpy(d["stats_mean"]).to(device).float()
                stats_std = torch.from_numpy(d["stats_std"]).to(device).float()
                print(f"[flow] loaded z-score stats from {cache_path}")

    imputer = FlowImputer(
        velocity_net, device,
        stats_mean=stats_mean, stats_std=stats_std,
        num_steps=int(args.flow_num_steps),
        solver=str(args.flow_solver),
    )
    print(f"[flow] FlowImputer(solver={args.flow_solver}, "
          f"K={args.flow_num_steps})")
    return imputer


def evaluate_sequence(
    *,
    seq_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    motion_encoder: Any,
    backbone_name: str,
    zscore_mean: torch.Tensor | None,
    zscore_std: torch.Tensor | None,
    rankings: dict[str, dict[str, float]],
    imputation: str,
    joint_means: torch.Tensor | None,
    train_pool: torch.Tensor | None,
    flow_imputer: Any | None,
    n_samples: int,
    k_list: tuple[int, ...],
) -> dict:
    """Compute faithfulness for *every* ranking in ``rankings`` under the
    shared ``imputation`` method on this sequence.
    """
    classifier_fn = build_classifier_fn(
        motion_encoder, mask, backbone_name,
        zscore_mean=zscore_mean, zscore_std=zscore_std, x_orig=x,
    )
    class_idx = int(y[0].item())
    p_full = _class_prob(classifier_fn, x, class_idx)

    # One p_ref per (sequence, imputation). All rankings share it — this
    # matches the baseline protocol and makes completeness error
    # directly comparable between ranking sources under the same imputation.
    p_ref = _compute_p_ref(
        classifier_fn, x, class_idx, imputation,
        joint_means=joint_means, train_pool=train_pool,
        flow_imputer=flow_imputer, mask=mask, lengths=lengths,
        n_samples=n_samples, seq_idx=seq_idx,
    )

    faithfulness: dict[str, dict] = {}
    for ranking_name, shap_dict in rankings.items():
        metrics = compute_spatial_faithfulness_batched(
            classifier_fn, x, y, mask, lengths, shap_dict, imputation,
            joint_means=joint_means,
            train_pool=train_pool,
            flow_imputer=flow_imputer,
            n_samples=n_samples,
            k_list=k_list, p_full=p_full, p_ref=p_ref, seq_idx=seq_idx,
        )
        faithfulness[ranking_name] = metrics

    return {
        "seq_idx":   seq_idx,
        "true_class": class_idx,
        "p_full":    p_full,
        "shap_values": rankings,
        "faithfulness": faithfulness,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(per_seq: list[dict]) -> dict:
    agg: dict[str, list[float]] = defaultdict(list)
    for r in per_seq:
        agg["p_full"].append(r["p_full"])
        for ranking_name, f in r["faithfulness"].items():
            prefix = f"spatial/{ranking_name}"
            agg[f"{prefix}/deletion_auc"].append(f["deletion_auc"])
            agg[f"{prefix}/insertion_auc"].append(f["insertion_auc"])
            agg[f"{prefix}/random_deletion_auc"].append(f["random_deletion_auc"])
            agg[f"{prefix}/random_insertion_auc"].append(f["random_insertion_auc"])
            agg[f"{prefix}/completeness_error"].append(f["completeness_error"])
            for k, vals in f["pgi_pgu"].items():
                agg[f"{prefix}/pgi@{k}"].append(vals["pgi"])
                agg[f"{prefix}/pgu@{k}"].append(vals["pgu"])
            for k, vals in f["pgi_pgu_rand"].items():
                agg[f"{prefix}/pgi_rand@{k}"].append(vals["pgi"])
                agg[f"{prefix}/pgu_rand@{k}"].append(vals["pgu"])
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Ranking sources
    p.add_argument("--baseline_jsonl", type=str, default=None,
                   help="per_sequence.jsonl produced by evaluate_shap_baselines.py "
                        "from which to pull zero / mean / marginal rankings.")
    p.add_argument("--flow_psi_paths", nargs="*", default=[],
                   help="Any number of psi.npz files (one per flow seed).")
    p.add_argument("--flow_seed_labels", nargs="*", default=[],
                   help="Optional labels (same order as --flow_psi_paths). "
                        "Default: inferred from the seed<N> parent dir.")

    # Imputation
    p.add_argument("--imputation", required=True,
                   choices=["zero", "mean", "marginal", "flow_imputer"],
                   help="Shared imputation protocol applied to every ranking.")
    p.add_argument("--n_samples", type=int, default=20,
                   help="Donor draws (marginal) or completions (flow_imputer) "
                        "per coalition.")

    # Backbone / classifier
    p.add_argument("--backbone", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--num_folds", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--classifier_ckpt", required=True)

    # Flow imputer (optional; required when --imputation flow_imputer)
    p.add_argument("--flow_config", type=str, default=None,
                   help="Flow-matching training config (JSON) used to train "
                        "--flow_checkpoint. Required for --imputation flow_imputer.")
    p.add_argument("--flow_checkpoint", type=str, default=None,
                   help="Trained VelocityNet checkpoint (*.ckpt). Required for "
                        "--imputation flow_imputer.")
    p.add_argument("--flow_num_steps", type=int, default=100,
                   help="ODE integration steps for FlowImputer.")
    p.add_argument("--flow_solver", type=str, default="midpoint",
                   choices=["euler", "midpoint"],
                   help="ODE solver for FlowImputer.")

    # Bookkeeping
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--max_test_sequences", type=int, default=None)
    p.add_argument("--start_seq", type=int, default=0)
    p.add_argument("--root_centered", action="store_true", default=False)
    p.add_argument("--max_train_pool", type=int, default=2000)
    p.add_argument("--train_pool_batch_size", type=int, default=64)
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- load flow rankings ----------
    flow_rankings_per_seq: dict[str, dict[str, np.ndarray]] = {}
    for i, path in enumerate(args.flow_psi_paths):
        if args.flow_seed_labels and i < len(args.flow_seed_labels):
            label = args.flow_seed_labels[i]
        else:
            parent = Path(path).parent.name   # e.g. "seed42"
            label = f"flow_{parent}" if parent.startswith("seed") else f"flow_{i}"
        print(f"[ranks] {label} ← {path}")
        flow_rankings_per_seq[label] = aggregate_psi_signed(Path(path))

    # ---------- load baseline rankings ----------
    baseline_rankings: dict[int, dict[str, dict[str, float]]] = {}
    if args.baseline_jsonl:
        print(f"[ranks] baselines ← {args.baseline_jsonl}")
        baseline_rankings = load_baseline_rankings(Path(args.baseline_jsonl))
        print(f"        {len(baseline_rankings)} baseline seq_idx entries "
              f"with methods={sorted(next(iter(baseline_rankings.values())).keys())}")

    # ---------- load classifier, train pool, flow imputer ----------
    backbone_params = _load_backbone_params(args.backbone, args.config, args.num_folds)
    backbone_name = backbone_params["backbone"]
    motion_encoder = load_motion_encoder(args.classifier_ckpt, backbone_params, device)

    need_train_pool = args.imputation in ("mean", "marginal")
    need_zscore = backbone_name == "potr"
    train_pool = None
    joint_means = None
    zscore_mean = zscore_std = None

    if need_train_pool:
        print("[data] building raw training pool …")
        train_pool, joint_means = build_train_pool(
            backbone_params, args.fold, device,
            max_sequences=args.max_train_pool,
            batch_size=args.train_pool_batch_size,
            root_centered=args.root_centered,
        )
        print(f"       train_pool: {tuple(train_pool.shape)}")

    if need_zscore:
        print("[data] computing z-score stats (POTR) …")
        zscore_mean, zscore_std = build_zscore_stats_for_potr(
            backbone_params, args.fold, device,
            batch_size=args.train_pool_batch_size,
            root_centered=args.root_centered,
        )

    flow_imputer = _maybe_load_flow_imputer(args, device)

    # ---------- iterate test set ----------
    data_args = _raw_data_args(backbone_params, args.fold, batch_size=1)
    _, test_ds = get_carepd_datasets(data_args)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)
    print(f"[data] test_ds: {len(test_ds)} sequences")

    # Sanity: every test sequence must have flow coverage (if any flow seed is used)
    if flow_rankings_per_seq:
        stripped = [_strip_view(v) for v in test_ds.video_names]
        for label, seqs in flow_rankings_per_seq.items():
            missing = [v for v in set(stripped) if v not in seqs]
            if missing:
                raise RuntimeError(
                    f"{label}: {len(missing)} test sequences missing from psi.npz. "
                    f"First: {missing[:3]}"
                )

    jsonl_path = os.path.join(args.output_dir, "per_sequence.jsonl")
    open(jsonl_path, "w").close()

    per_seq_results: list[dict] = []
    k_list = tuple(args.k_list)

    for seq_idx, raw_batch in enumerate(test_loader):
        if seq_idx < args.start_seq:
            continue
        if args.max_test_sequences is not None and seq_idx >= args.start_seq + args.max_test_sequences:
            break

        x_raw, labels, _, _, pad_mask = raw_batch
        batch = actor_batch_from_carepd(
            x_raw.float(), pad_mask, backbone_params["num_classes"], device,
            y=labels, root_centered=args.root_centered,
        )
        x, y, mask, lengths = batch["x"], batch["y"], batch["mask"], batch["lengths"]

        # Assemble the ranking dict for this sequence.
        rankings: dict[str, dict[str, float]] = {}
        if seq_idx in baseline_rankings:
            rankings.update(baseline_rankings[seq_idx])

        if flow_rankings_per_seq:
            seq_key = _strip_view(test_ds.video_names[seq_idx])
            for label, seq_map in flow_rankings_per_seq.items():
                phi = seq_map[seq_key]
                rankings[label] = _build_shap_dict(phi)

        if not rankings:
            print(f"  seq {seq_idx}: no rankings available, skipping")
            continue

        result = evaluate_sequence(
            seq_idx=seq_idx, x=x, y=y, mask=mask, lengths=lengths,
            motion_encoder=motion_encoder, backbone_name=backbone_name,
            zscore_mean=zscore_mean, zscore_std=zscore_std,
            rankings=rankings, imputation=args.imputation,
            joint_means=joint_means, train_pool=train_pool,
            flow_imputer=flow_imputer, n_samples=args.n_samples,
            k_list=k_list,
        )
        result["seq_key"] = _strip_view(test_ds.video_names[seq_idx])

        per_seq_results.append(result)
        with open(jsonl_path, "a") as fh:
            fh.write(json.dumps(result) + "\n")

        # Progress line showing the most-interesting ranking's deletion/insertion.
        primary = next(iter(rankings))
        fm = result["faithfulness"][primary]
        print(
            f"  seq {seq_idx} ({result['seq_key']}): class={result['true_class']} "
            f"p_full={result['p_full']:.3f}  "
            f"[{primary}] del={fm['deletion_auc']:.3f} ins={fm['insertion_auc']:.3f}  "
            f"pgi@1={fm['pgi_pgu']['1']['pgi']:.3f}",
            flush=True,
        )

    # ---------- save ----------
    p_full_warning = check_p_full_warning(
        per_seq_results, backbone_params["num_classes"], backbone_name,
    )
    agg = aggregate(per_seq_results)
    agg["_meta"] = {
        "backbone":          args.backbone,
        "dataset":           backbone_params["dataset"],
        "config":            args.config,
        "num_folds":         args.num_folds,
        "fold":              args.fold,
        "n_test_sequences":  len(per_seq_results),
        "k_list":            list(k_list),
        "classifier_ckpt":   args.classifier_ckpt,
        "imputation":        args.imputation,
        "n_samples":         args.n_samples,
        "baseline_jsonl":    args.baseline_jsonl,
        "flow_psi_paths":    list(args.flow_psi_paths),
        "flow_seed_labels":  list(flow_rankings_per_seq.keys()),
        "flow_config":       args.flow_config,
        "flow_checkpoint":   args.flow_checkpoint,
        "flow_num_steps":    args.flow_num_steps,
        "flow_solver":       args.flow_solver,
        "psi_aggregation":   "signed_sum",
        "p_full_warning":    p_full_warning,
        "notes": (
            "Shared-imputation cross-evaluation. Every ranking source is "
            "evaluated with the same imputation function (imputation=..) "
            "via compute_spatial_faithfulness_batched on the same 240 "
            "SUB01 test sequences. Any difference between rankings under "
            "this protocol isolates ranking quality."
        ),
    }
    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as fh:
        json.dump(agg, fh, indent=2)

    print(f"\nDone. Wrote {jsonl_path} ({len(per_seq_results)} sequences)")
    print(f"Wrote {os.path.join(args.output_dir, 'aggregate.json')}")


if __name__ == "__main__":
    main()
