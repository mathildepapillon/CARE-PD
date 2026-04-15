#!/usr/bin/env python3
"""Merge baseline SHAP JSONL with physics-cache JSON files and emit aggregate stats + a markdown table.

Baseline rows must come from the same fold / classifier as the physics precompute
(e.g. ``results/shap_baselines_potr_bmclab_fold1_full/per_sequence.jsonl`` paired with
``results/physics_cache_fold1/``).  Only sequence indices present in both inputs are used.

Example:

    python scripts/make_shap_physics_comparison_table.py \\
        --baseline_jsonl results/shap_baselines_potr_bmclab_fold1_full/per_sequence.jsonl \\
        --physics_dir results/physics_cache_fold1 \\
        --out_aggregate results/shap_physics_fold1_aggregate.json \\
        --out_md results/shap_physics_fold1_table.md
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def aggregate_results(per_seq: list[dict]) -> dict[str, dict[str, float]]:
    """Same structure as ``evaluate_shap_baselines.aggregate_results``."""
    agg: dict[str, list[float]] = defaultdict(list)
    spatial_methods = list(per_seq[0]["faithfulness"].keys()) if per_seq else []
    temporal_methods = ("zero", "mean", "marginal", "physics")

    for r in per_seq:
        agg["p_full"].append(r["p_full"])
        agg["stride_detected"].append(float(r.get("stride_detected", True)))

        for method in spatial_methods:
            f = r["faithfulness"].get(method)
            if f is None:
                continue
            prefix = f"spatial/{method}"
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

    # Temporal: if physics is present, pool only those sequences (same n for all four columns).
    temporal_only = [r for r in per_seq if "physics" in r.get("temporal_faithfulness", {})]
    temporal_pool = temporal_only if temporal_only else per_seq
    methods_temporal = temporal_methods if temporal_only else ("zero", "mean", "marginal")
    for r in temporal_pool:
        for method in methods_temporal:
            tf = r.get("temporal_faithfulness", {}).get(method, {})
            prefix = f"temporal/{method}"
            for key in ("deletion_auc", "insertion_auc", "completeness_error"):
                if key in tf:
                    agg[f"{prefix}/{key}"].append(tf[key])

    return {
        key: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for key, vals in agg.items()
    }


def _load_baseline_by_seq(jsonl_path: Path) -> dict[int, dict]:
    by_seq: dict[int, dict] = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_seq[int(r["seq_idx"])] = r
    return by_seq


def _load_physics_by_seq(physics_dir: Path) -> dict[int, dict]:
    by_seq: dict[int, dict] = {}
    for p in sorted(physics_dir.glob("seq_*.json")):
        idx = int(p.stem.split("_")[1])
        with open(p, encoding="utf-8") as fh:
            by_seq[idx] = json.load(fh)
    return by_seq


def _load_temporal_physics_jsonl(path: Path) -> dict[int, dict]:
    by_seq: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_seq[int(r["seq_idx"])] = r
    return by_seq


def _merge_records(
    baseline_by_seq: dict[int, dict],
    physics_by_seq: dict[int, dict],
) -> tuple[list[dict], list[int]]:
    common = sorted(set(baseline_by_seq.keys()) & set(physics_by_seq.keys()))
    merged: list[dict] = []
    for i in common:
        b = baseline_by_seq[i]
        ph = physics_by_seq[i]
        faith = dict(b["faithfulness"])
        faith["physics"] = ph["faithfulness"]
        merged.append({
            "seq_idx": b["seq_idx"],
            "p_full": b["p_full"],
            "stride_detected": b["stride_detected"],
            "faithfulness": faith,
            "temporal_faithfulness": b["temporal_faithfulness"],
        })
    return merged, common


def _attach_temporal_physics(
    merged: list[dict],
    temporal_by_seq: dict[int, dict],
) -> None:
    """In-place: add ``temporal_faithfulness['physics']`` from JSONL rows."""
    for r in merged:
        t = temporal_by_seq.get(int(r["seq_idx"]))
        if t is None:
            continue
        tf = dict(r.get("temporal_faithfulness") or {})
        tf["physics"] = t["temporal_faithfulness"]["physics"]
        r["temporal_faithfulness"] = tf


def _fmt(agg: dict, key: str) -> str:
    e = agg.get(key, {})
    m, s = e.get("mean", float("nan")), e.get("std", float("nan"))
    return f"{m:.3f} ± {s:.3f}"


def _table_row_bold_best(
    label: str,
    agg: dict,
    key_tmpl: str,
    method_keys: list[str],
) -> str:
    """Bold cell(s) whose mean is best for this row (↑ higher better, ↓ lower better)."""
    means = [agg[key_tmpl.format(m)]["mean"] for m in method_keys]
    lower_is_better = "↓" in label
    if lower_is_better:
        best = min(means)
    else:
        best = max(means)
    cells: list[str] = []
    for m, mu in zip(method_keys, means):
        txt = _fmt(agg, key_tmpl.format(m))
        if np.isclose(mu, best, rtol=0.0, atol=1e-12):
            cells.append(f"**{txt}**")
        else:
            cells.append(txt)
    return f"| {label} | " + " | ".join(cells) + " |"


def _build_markdown(
    agg: dict,
    n: int,
    seq_range: tuple[int, int],
    n_temporal: int,
    temporal_range: tuple[int, int],
    temporal_has_physics: bool,
) -> str:
    zm = ["zero", "mean", "marginal", "physics"]
    zm3 = ["zero", "mean", "marginal"]
    zm_temporal = ["zero", "mean", "marginal", "physics"] if temporal_has_physics else zm3
    lines = [
        "### SHAP baselines vs physics imputation (spatial KernelSHAP)",
        "",
        f"Pooled over **n = {n}** test sequences (indices **{seq_range[0]}–{seq_range[1]}**) "
        "present in both the baseline JSONL and ``physics_cache_fold1``.  "
        "Baselines: zero / mean / marginal.  Fourth column: **physics** from the offline cache.",
        "",
        "| Metric | Zero | Mean | Marginal | Physics |",
        "|--------|------|------|----------|---------|",
    ]
    # Omitted: spatial completeness (diagnostic; dominated by |v_full−p_full|).
    spatial_rows = [
        ("Deletion AUC ↓", "spatial/{}/deletion_auc"),
        ("Insertion AUC ↑", "spatial/{}/insertion_auc"),
        ("PGI@1 ↑", "spatial/{}/pgi@1"),
        ("PGU@1 ↓", "spatial/{}/pgu@1"),
        ("PGI−Rand@1 ↑", "spatial/{}/pgi_rand@1"),
        ("PGU−Rand@1 ↓", "spatial/{}/pgu_rand@1"),
        ("PGI@3 ↑", "spatial/{}/pgi@3"),
        ("PGU@3 ↓", "spatial/{}/pgu@3"),
        ("PGI−Rand@3 ↑", "spatial/{}/pgi_rand@3"),
        ("PGU−Rand@3 ↓", "spatial/{}/pgu_rand@3"),
        ("PGI@5 ↑", "spatial/{}/pgi@5"),
        ("PGU@5 ↓", "spatial/{}/pgu@5"),
        ("PGI−Rand@5 ↑", "spatial/{}/pgi_rand@5"),
        ("PGU−Rand@5 ↓", "spatial/{}/pgu_rand@5"),
    ]
    for label, tmpl in spatial_rows:
        lines.append(_table_row_bold_best(label, agg, tmpl, zm))

    lines.extend([
        "",
        f"**Classifier probability at full input:** mean ± std over the same **n = {n}** sequences: "
        f"{_fmt(agg, 'p_full')}.",
        "",
        "### Temporal SHAP (exact stride windows"
        + ("; physics imputation)" if temporal_has_physics else "; baselines only)"),
        "",
    ])
    if temporal_has_physics:
        lines.extend([
            f"Pooled over **n = {n_temporal}** test sequences (indices **{temporal_range[0]}–{temporal_range[1]}**) "
            "with temporal **physics** faithfulness from ``temporal_physics`` JSONL; baselines use the same sequences.",
            "",
            "| Metric | Zero | Mean | Marginal | Physics |",
            "|--------|------|------|----------|---------|",
        ])
    else:
        lines.extend([
            "Physics cache is spatial-only; temporal rows are baselines on all merged sequences (no physics column).",
            "",
            "| Metric | Zero | Mean | Marginal |",
            "|--------|------|------|----------|",
        ])
    temporal_rows = [
        ("Shapley completeness err ↓", "temporal/{}/completeness_error"),
        ("Deletion AUC ↓", "temporal/{}/deletion_auc"),
        ("Insertion AUC ↑", "temporal/{}/insertion_auc"),
    ]
    for label, tmpl in temporal_rows:
        lines.append(_table_row_bold_best(label, agg, tmpl, zm_temporal))

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--baseline_jsonl",
        type=Path,
        default=Path("results/shap_baselines_potr_bmclab_fold1_full/per_sequence.jsonl"),
        help="Per-sequence baseline results (zero / mean / marginal + temporal).",
    )
    p.add_argument(
        "--physics_dir",
        type=Path,
        default=Path("results/physics_cache_fold1"),
        help="Directory of seq_*.json files from precompute_physics_shap.py.",
    )
    p.add_argument("--out_aggregate", type=Path, default=Path("results/shap_physics_fold1_aggregate.json"))
    p.add_argument("--out_md", type=Path, default=Path("results/shap_physics_fold1_table.md"))
    p.add_argument(
        "--temporal_physics_jsonl",
        type=Path,
        default=Path("results/temporal_physics_fold1.jsonl"),
        help="Per-sequence temporal physics faithfulness; merged into temporal_faithfulness['physics'].",
    )
    args = p.parse_args()

    baseline_by_seq = _load_baseline_by_seq(args.baseline_jsonl)
    physics_by_seq = _load_physics_by_seq(args.physics_dir)
    merged, common = _merge_records(baseline_by_seq, physics_by_seq)
    if not merged:
        raise SystemExit("No overlapping seq_idx between baseline JSONL and physics cache.")

    temporal_by_seq: dict[int, dict] = {}
    if args.temporal_physics_jsonl.is_file():
        temporal_by_seq = _load_temporal_physics_jsonl(args.temporal_physics_jsonl)
        _attach_temporal_physics(merged, temporal_by_seq)

    agg = aggregate_results(merged)
    temporal_seqs = sorted(
        set(int(r["seq_idx"]) for r in merged if "physics" in r.get("temporal_faithfulness", {}))
    )
    n_temporal = len(temporal_seqs)
    temporal_range = (min(temporal_seqs), max(temporal_seqs)) if temporal_seqs else (0, -1)
    temporal_has_physics = n_temporal > 0
    if not temporal_has_physics:
        # Baselines-only temporal: use full merged set (no physics column in table).
        temporal_seqs = sorted(int(r["seq_idx"]) for r in merged)
        n_temporal = len(merged)
        temporal_range = (min(temporal_seqs), max(temporal_seqs)) if temporal_seqs else (0, -1)

    meta = {
        "n_sequences": len(common),
        "seq_indices": common,
        "n_temporal_sequences": n_temporal,
        "temporal_seq_indices": temporal_seqs,
        "baseline_jsonl": str(args.baseline_jsonl.resolve()),
        "physics_dir": str(args.physics_dir.resolve()),
        "temporal_physics_jsonl": str(args.temporal_physics_jsonl.resolve())
        if args.temporal_physics_jsonl.is_file()
        else None,
    }
    out_obj = {"_meta": meta, **agg}

    args.out_aggregate.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_aggregate, "w", encoding="utf-8") as fh:
        json.dump(out_obj, fh, indent=2)

    md = _build_markdown(
        agg,
        len(common),
        (min(common), max(common)),
        n_temporal,
        temporal_range,
        temporal_has_physics,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"Wrote {args.out_aggregate} and {args.out_md}  (n={len(common)})")


if __name__ == "__main__":
    main()
