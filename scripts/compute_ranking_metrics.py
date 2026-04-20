#!/usr/bin/env python3
"""compute_ranking_metrics.py

Computes *ranking-based* attribution metrics from a ``per_sequence*.json`` file
produced by the synthetic Gaussian EC pipeline. These are the standard
saliency/attribution metrics used e.g. by Zhang et al. (2026) "OTFlow-SHAP"
and by the broader Integrated-Gradients literature.

Unlike EC1/EC2/EC3 — which penalise a method for not reproducing the exact
Shapley numbers — these metrics only ask "does the method *rank* features in
the right order?" This is the fair comparison axis for Aumann–Shapley methods
(OTFlow-SHAP, Integrated Gradients, GradientSHAP) that do not target classical
Shapley values on non-additive classifiers.

Metrics (per sequence, then averaged):

  • Spearman rho (on |phi|): rank correlation between method attributions
    (in magnitude) and oracle Shapley attributions (in magnitude). 1 = perfect
    ranking, 0 = random, -1 = inverted.

  • Top-1 agreement: does the method's most-important feature match the
    oracle's most-important feature? Averaged as accuracy in [0, 1].

  • Top-K/2 overlap: fraction of the oracle's top-K/2 players that appear in
    the method's top-K/2 set. For K=4 this is top-2 overlap, for K=8 top-4.

  • Insertion AUC (↑): normalised area under the v-curve as we add features
    from most- to least-important (by |phi_method|). Curve is sign-corrected
    per sequence so 1.0 = the full-coalition value is reached immediately and
    0.0 = no progress until all features are added.

  • Deletion AUC (↑): normalised area under the v-curve as we remove features
    from most- to least-important (by |phi_method|). Curve is sign-corrected
    so 1.0 = removing the top-1 feature immediately collapses confidence and
    0.0 = confidence is retained until all features are removed.

Insertion/Deletion AUC rely on ``v_true(S)`` being present for every
cumulative coalition. For enumerated tasks (k4: 2^4=16 coalitions, k8:
2^8=256) this is always the case. For the spatial task (J=17 players,
typically 250 sampled coalitions) the cumulative lookup usually misses, and
those metrics gracefully fall back to ``n/a``.

Usage:
    python scripts/compute_ranking_metrics.py \
        --ec_dir experiment_outs/flow_matching_synthetic/ec_olsen_vs_flow_k4

The script auto-prefers ``per_sequence_with_otflow.json`` over
``per_sequence.json`` so it works out-of-the-box once the OTFlow-SHAP
post-processing step has been run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Coalition lookup helpers
# ---------------------------------------------------------------------------


def _coalition_key(row) -> tuple:
    return tuple(int(x) for x in row)


def build_coalition_lookup(coalitions: np.ndarray) -> Dict[tuple, int]:
    """Map coalition pattern -> row index in the per-sequence ``v`` arrays."""
    return {_coalition_key(row): i for i, row in enumerate(coalitions.tolist())}


def cumulative_coalition_mask(player_order: np.ndarray, M: int, k: int) -> np.ndarray:
    """Binary mask of the coalition containing the top-``k`` players."""
    mask = np.zeros(M, dtype=int)
    if k > 0:
        mask[player_order[:k]] = 1
    return mask


# ---------------------------------------------------------------------------
# Insertion / Deletion curves
# ---------------------------------------------------------------------------


def insertion_curve(
    phi: np.ndarray,
    lookup: Dict[tuple, int],
    v_true: np.ndarray,
) -> Optional[np.ndarray]:
    """Return v_true evaluated at the cumulative top-k coalitions.

    If any cumulative coalition is absent from the stored coalition set (can
    happen for sampled regimes like spatial), returns ``None``.
    """
    M = len(phi)
    order = np.argsort(-np.abs(phi), kind="stable")
    curve = np.empty(M + 1)
    for k in range(M + 1):
        key = _coalition_key(cumulative_coalition_mask(order, M, k))
        idx = lookup.get(key)
        if idx is None:
            return None
        curve[k] = v_true[idx]
    return curve


def deletion_curve(
    phi: np.ndarray,
    lookup: Dict[tuple, int],
    v_true: np.ndarray,
) -> Optional[np.ndarray]:
    """Return v_true starting from the full coalition and removing the top-k."""
    M = len(phi)
    order = np.argsort(-np.abs(phi), kind="stable")
    curve = np.empty(M + 1)
    for k in range(M + 1):
        mask = np.ones(M, dtype=int)
        if k > 0:
            mask[order[:k]] = 0
        idx = lookup.get(_coalition_key(mask))
        if idx is None:
            return None
        curve[k] = v_true[idx]
    return curve


def normalised_insertion_auc(curve: np.ndarray) -> float:
    """Sign-corrected, [0, 1]-normalised insertion AUC.

    1.0 = curve jumps to v(full) immediately at k=1.
    0.0 = curve stays at v(empty) until k=M.
    """
    v0, vF = float(curve[0]), float(curve[-1])
    target = vF - v0
    if abs(target) < 1e-8:
        return 0.0
    sign = 1.0 if target > 0 else -1.0
    shifted = (curve - v0) * sign  # monotone upward, reaches |target|
    span = abs(target)
    xs = np.linspace(0.0, 1.0, len(curve))
    return float(np.trapz(shifted, xs) / span)


def normalised_deletion_auc(curve: np.ndarray) -> float:
    """Sign-corrected, [0, 1]-normalised deletion AUC.

    1.0 = curve drops from v(full) to v(empty) after removing the top-1.
    0.0 = curve stays at v(full) until all features are removed.
    """
    vF, v0 = float(curve[0]), float(curve[-1])
    target = v0 - vF
    if abs(target) < 1e-8:
        return 0.0
    sign = 1.0 if target > 0 else -1.0
    shifted = (curve - vF) * sign  # monotone upward away from v(full)
    span = abs(target)
    xs = np.linspace(0.0, 1.0, len(curve))
    return float(np.trapz(shifted, xs) / span)


# ---------------------------------------------------------------------------
# Per-sequence metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    per_seq: List[dict],
    truth_key: str = "gaussian_oracle",
) -> Tuple[Dict[str, Dict[str, List[float]]], int]:
    method_names = set()
    for s in per_seq:
        method_names.update(k for k in s if not k.startswith("_"))
    method_names.discard(truth_key)
    methods = sorted(method_names)

    results: Dict[str, Dict[str, List[float]]] = {
        m: {
            "spearman": [],
            "top1": [],
            "topkhalf": [],
            "insertion_auc": [],
            "deletion_auc": [],
        }
        for m in methods
    }

    M_seen = 0
    for seq in per_seq:
        if truth_key not in seq or "_coalitions" not in seq:
            continue
        coalitions = np.asarray(seq["_coalitions"], dtype=int)
        lookup = build_coalition_lookup(coalitions)
        phi_true = np.asarray(seq[truth_key]["phi"], dtype=float)
        v_true = np.asarray(seq[truth_key]["v"], dtype=float)
        M = len(phi_true)
        M_seen = M
        true_order = np.argsort(-np.abs(phi_true), kind="stable")

        for m in methods:
            if m not in seq or "phi" not in seq[m]:
                continue
            phi_m = np.asarray(seq[m]["phi"], dtype=float)
            if len(phi_m) != M:
                continue

            # Spearman on |phi|
            if np.ptp(np.abs(phi_m)) < 1e-12 or np.ptp(np.abs(phi_true)) < 1e-12:
                rho = 0.0  # degenerate (flat attribution)
            else:
                rho_raw, _ = spearmanr(np.abs(phi_m), np.abs(phi_true))
                rho = 0.0 if np.isnan(rho_raw) else float(rho_raw)
            results[m]["spearman"].append(rho)

            # Top-1 and top-K/2 ranking agreement
            m_order = np.argsort(-np.abs(phi_m), kind="stable")
            results[m]["top1"].append(float(m_order[0] == true_order[0]))
            k_half = max(1, M // 2)
            overlap = (
                len(set(m_order[:k_half].tolist()) & set(true_order[:k_half].tolist()))
                / k_half
            )
            results[m]["topkhalf"].append(float(overlap))

            # Insertion / Deletion AUC (requires all cumulative coalitions)
            ins = insertion_curve(phi_m, lookup, v_true)
            if ins is not None:
                results[m]["insertion_auc"].append(normalised_insertion_auc(ins))
            dele = deletion_curve(phi_m, lookup, v_true)
            if dele is not None:
                results[m]["deletion_auc"].append(normalised_deletion_auc(dele))

    return results, M_seen


def aggregate(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for m, metrics in results.items():
        row: Dict[str, float] = {}
        for key, vals in metrics.items():
            if vals:
                arr = np.asarray(vals, dtype=float)
                row[f"{key}_mean"] = float(arr.mean())
                row[f"{key}_std"] = float(arr.std())
                row[f"{key}_sem"] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                row[f"{key}_n"] = int(len(arr))
            else:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_std"] = float("nan")
                row[f"{key}_sem"] = float("nan")
                row[f"{key}_n"] = 0
        out[m] = row
    return out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


_METHOD_DISPLAY = {
    "gaussian_oracle": "gaussian_oracle (ref)",
    "mean": "mean",
    "zero": "zero",
    "marginal": "marginal",
    "flow_matching": "flow_matching (imp.)",
    "vaeac": "vaeac (imp.)",
    "otflow_shap": "otflow_shap (IG)",
}

_METHOD_ORDER = [
    "mean",
    "zero",
    "marginal",
    "vaeac",
    "flow_matching",
    "otflow_shap",
]


def _cell(value: float, sem: float, best: float, best_sem: float, direction: str) -> str:
    if np.isnan(value):
        return "n/a"
    if value == best:
        return f"**{value:.4f}**"
    pooled = float(np.sqrt(max(sem, 0.0) ** 2 + max(best_sem, 0.0) ** 2))
    if direction == "max":
        within = best - value <= pooled
    else:
        within = value - best <= pooled
    if within:
        return f"__{value:.4f}__"
    return f"{value:.4f}"


def render_table(
    summary: Dict[str, Dict[str, float]],
    header: str,
    metrics_spec: List[Tuple[str, str, str]],
) -> str:
    """Render a markdown-style table. ``metrics_spec``: list of (key, direction, display)."""
    ordered = [m for m in _METHOD_ORDER if m in summary]
    # Append any unexpected methods at the end.
    for m in summary:
        if m not in ordered:
            ordered.append(m)

    # Find best per column
    best_val: Dict[str, float] = {}
    best_sem: Dict[str, float] = {}
    for key, direction, _ in metrics_spec:
        cands = [
            (m, summary[m][f"{key}_mean"], summary[m][f"{key}_sem"])
            for m in ordered
            if summary[m][f"{key}_n"] > 0 and not np.isnan(summary[m][f"{key}_mean"])
        ]
        if not cands:
            continue
        if direction == "max":
            winner = max(cands, key=lambda t: t[1])
        else:
            winner = min(cands, key=lambda t: t[1])
        best_val[key] = winner[1]
        best_sem[key] = winner[2]

    col_names = ["Method"] + [d[2] for d in metrics_spec]
    widths = [max(24, max(len(_METHOD_DISPLAY.get(m, m)) for m in ordered) + 2)]
    widths += [max(len(name), 14) for name in col_names[1:]]

    lines: List[str] = []
    lines.append("")
    lines.append(header)
    total_w = sum(widths) + 3 * (len(widths) - 1)
    lines.append("=" * total_w)
    lines.append(" | ".join(f"{n:<{w}}" for n, w in zip(col_names, widths)))
    lines.append("-" * total_w)

    for m in ordered:
        display = _METHOD_DISPLAY.get(m, m)
        row = [f"{display:<{widths[0]}}"]
        for i, (key, direction, _) in enumerate(metrics_spec, start=1):
            val = summary[m][f"{key}_mean"]
            sem = summary[m][f"{key}_sem"]
            n = summary[m][f"{key}_n"]
            if n == 0:
                cell = "n/a"
            elif key in best_val:
                cell = _cell(val, sem, best_val[key], best_sem[key], direction)
            else:
                cell = f"{val:.4f}" if not np.isnan(val) else "n/a"
            row.append(f"{cell:<{widths[i]}}")
        lines.append(" | ".join(row))

    lines.append("")
    lines.append("Legend: **bold** = best in column, __underlined__ = within pooled SE of best.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_per_seq(ec_dir: Path) -> Tuple[List[dict], dict, Path]:
    for name in ("per_sequence_with_otflow.json", "per_sequence.json"):
        p = ec_dir / name
        if p.exists():
            with open(p) as f:
                per_seq = json.load(f)
            summary_full: dict = {}
            for sname in ("ec_summary_with_otflow.json", "ec_summary.json"):
                sp = ec_dir / sname
                if sp.exists():
                    with open(sp) as f:
                        summary_full = json.load(f)
                    break
            return per_seq, summary_full, p
    raise FileNotFoundError(f"No per_sequence*.json in {ec_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ec_dir", required=True, help="EC output dir containing per_sequence*.json")
    ap.add_argument("--truth_key", default="gaussian_oracle")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    ec_dir = Path(args.ec_dir)
    per_seq, summary_full, src_path = _load_per_seq(ec_dir)
    cfg = summary_full.get("_config", {}) if isinstance(summary_full, dict) else {}

    results, M = compute_metrics(per_seq, truth_key=args.truth_key)
    agg = aggregate(results)

    khalf = max(1, M // 2)
    metrics_spec = [
        ("spearman", "max", "Spearman rho ↑"),
        ("top1", "max", "Top-1 acc ↑"),
        ("topkhalf", "max", f"Top-{khalf} overlap ↑"),
        ("insertion_auc", "max", "Insertion AUC ↑"),
        ("deletion_auc", "max", "Deletion AUC ↑"),
    ]

    header = (
        f"Ranking metrics  src={src_path.name}  "
        f"player_mode={cfg.get('player_mode', '?')}  "
        f"K/J={cfg.get('K', '?')}/{cfg.get('J', '?')}  "
        f"class={cfg.get('class_idx', '?')}  "
        f"n_seqs={cfg.get('n_test_sequences', len(per_seq))}  M={M}"
    )
    table = render_table(agg, header, metrics_spec)
    print(table)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "_config": cfg,
                    "_source": str(src_path),
                    "_M": M,
                    "metrics": agg,
                },
                f,
                indent=2,
            )
        print(f"\n[ranking] wrote {out_path}")


if __name__ == "__main__":
    main()
