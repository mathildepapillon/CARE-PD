"""make_tables.py — Generate publishable evaluation tables for ActorSHAP.

Produces four tables following the metric hierarchy from the evaluation plan:

  Table 1 — Primary (non-confounded): rank stability + Shapley completeness error
  Table 2 — Temporal faithfulness (all folds)
  Table 3 — Self-evaluated spatial PGI / PGU with PGI−Rand / PGU−Rand
             (filtered to p_full ≥ min_p_full; self-eval caveat noted in caption)
  Table 4 — Cross-evaluated PGI under Zero protocol (diagnostic)

Usage
=====
    # Fold 1 only (while folds 2/8 are still running):
    python make_tables.py --folds 1

    # All three folds once complete:
    python make_tables.py --folds 1 2 8

    # With confidence filter:
    python make_tables.py --folds 1 2 8 --min_p_full 0.35

    # Print markdown (default) or LaTeX:
    python make_tables.py --folds 1 --format latex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")

_ACTOR_DIRS = {
    1: "shap_actor_potr_bmclab_fold1_full",
    2: "shap_actor_potr_bmclab_fold2",
    8: "shap_actor_potr_bmclab_fold8",
}
_BASELINE_DIRS = {
    1: "shap_baselines_potr_bmclab_fold1_full",
    2: "shap_baselines_potr_bmclab_fold2",
    8: "shap_baselines_potr_bmclab_fold8",
}
_CROSSEVAL_DIRS = {
    1: "shap_crosseval_potr_bmclab_fold1",
    2: "shap_crosseval_potr_bmclab_fold2",
    8: "shap_crosseval_potr_bmclab_fold8",
}

METHODS = ["actor", "zero", "mean", "marginal"]
METHOD_LABELS = {
    "actor":    "ActorSHAP (ours)",
    "zero":     "Zero",
    "mean":     "Mean",
    "marginal": "Marginal",
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_actor_seqs(fold: int) -> list[dict]:
    d = os.path.join(RESULTS_ROOT, _ACTOR_DIRS[fold])
    seqs: dict[int, dict] = {}
    for f in sorted(glob.glob(os.path.join(d, "shards", "per_sequence_shard*.jsonl"))):
        for r in _load_jsonl(f):
            if r["seq_idx"] not in seqs:
                seqs[r["seq_idx"]] = r
    if not seqs:
        p = os.path.join(d, "per_sequence.jsonl")
        if os.path.exists(p):
            for r in _load_jsonl(p):
                if r["seq_idx"] not in seqs:
                    seqs[r["seq_idx"]] = r
    return list(seqs.values())


def _load_baseline_seqs(fold: int) -> list[dict]:
    p = os.path.join(RESULTS_ROOT, _BASELINE_DIRS[fold], "per_sequence.jsonl")
    seqs: dict[int, dict] = {}
    for r in _load_jsonl(p):
        if r["seq_idx"] not in seqs:
            seqs[r["seq_idx"]] = r
    return list(seqs.values())


def _load_crosseval_seqs(fold: int, min_p_full: Optional[float] = None) -> list[dict]:
    p = os.path.join(RESULTS_ROOT, _CROSSEVAL_DIRS[fold], "per_sequence.jsonl")
    if not os.path.exists(p):
        return []
    seqs = _load_jsonl(p)
    if min_p_full is not None:
        seqs = [s for s in seqs if s.get("p_full", 0) >= min_p_full]
    return seqs


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _arr(seqs: list[dict], *path) -> np.ndarray:
    vals = []
    for s in seqs:
        node = s
        for p in path:
            node = node.get(p, {}) if isinstance(node, dict) else None
            if node is None:
                break
        if node is not None and not isinstance(node, dict):
            v = float(node)
            if v == v:  # not NaN
                vals.append(v)
    return np.array(vals)


def _ms(v: np.ndarray) -> tuple[float, float]:
    return (float(v.mean()), float(v.std())) if len(v) > 0 else (float("nan"), float("nan"))


def _fmt(mn: float, sd: float, signed: bool = False) -> str:
    if mn != mn:
        return "—"
    fmt = f"{mn:+.3f}" if signed else f"{mn:.3f}"
    return f"{fmt} ± {sd:.3f}"


def _bold(s: str, val: float, best: float) -> str:
    return f"**{s}**" if abs(val - best) < 1e-9 and val == val else s


def _best(vals: list[float], higher: bool) -> float:
    clean = [v for v in vals if v == v]
    return (max(clean) if higher else min(clean)) if clean else float("nan")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _md_table(title: str, header: list[str], rows: list[tuple]) -> str:
    col_w = 26
    lines = [f"\n### {title}\n"]
    lines.append("| " + " | ".join(f"{h:{col_w}}" for h in header) + " |")
    lines.append("|" + "|".join("-" * (col_w + 2) for _ in header) + "|")
    for row in rows:
        cells = []
        for c in row:
            cells.append(f"{str(c):{col_w}}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _latex_table(title: str, header: list[str], rows: list[tuple]) -> str:
    n = len(header)
    spec = "l" + "r" * (n - 1)
    lines = [
        f"% {title}",
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c).replace("**", "").replace("±", r"$\pm$")
                                .replace("−", "--") for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 1: Rank stability + completeness error (non-confounded)
# ---------------------------------------------------------------------------

def make_table1(folds: list[int], fmt: str) -> str:
    """Primary metrics — no imputation confound."""
    rows: list[tuple] = []
    header = ["Fold / n", "Rank Stability ↑", "Completeness Err ↓"]
    for m in ["actor", "zero", "mean", "marginal"]:
        header.append(f"Compl. Err ({METHOD_LABELS[m]})")

    # Collect per-fold
    fold_data = []
    for fold in folds:
        try:
            bl_seqs_all = _load_baseline_seqs(fold)
        except (FileNotFoundError, OSError):
            continue

        bl_by_idx   = {s["seq_idx"]: s for s in bl_seqs_all}
        actor_seqs  = []
        try:
            actor_seqs = _load_actor_seqs(fold)
        except (FileNotFoundError, OSError):
            pass

        has_actor = len(actor_seqs) > 0
        common    = [s for s in actor_seqs if s["seq_idx"] in bl_by_idx] if has_actor else []
        n         = len(common)

        rs_vals      = _arr(common, "rank_stability", "mean_rank_corr") if has_actor else np.array([])
        rs_mn, rs_sd = _ms(rs_vals)

        comp_by_method: dict[str, tuple[float, float]] = {}
        for m in METHODS:
            if m == "actor":
                if not has_actor:
                    comp_by_method[m] = (float("nan"), float("nan"))
                    continue
                src  = common
                path = ("faithfulness", "actor", "completeness_error")
            else:
                src  = [bl_by_idx[s["seq_idx"]] for s in common] if has_actor else list(bl_by_idx.values())
                path = ("faithfulness", m, "completeness_error")
            v = _arr(src, *path)
            comp_by_method[m] = _ms(v)

        n_label = n if has_actor else len(bl_by_idx)
        fold_data.append((fold, n_label, has_actor, rs_mn, rs_sd, comp_by_method))

    # Build rows
    for fold, n, has_actor, rs_mn, rs_sd, comp in fold_data:
        actor_note = "" if has_actor else " (baselines only)"
        row: list = [f"Fold {fold} (n={n}{actor_note})"]
        row.append(_fmt(rs_mn, rs_sd))
        # best completeness
        comps = [comp[m][0] for m in METHODS]
        best_c = _best(comps, higher=False)
        for m in METHODS:
            mn, sd = comp[m]
            s = _fmt(mn, sd)
            row.append(_bold(s, mn, best_c))
        rows.append(tuple(row))

    title = "Table 1 — Primary (non-confounded): Rank Stability and Shapley Completeness Error"
    h = ["Fold / n", "Rank Stability ↑"] + [f"Compl Err — {METHOD_LABELS[m]} ↓" for m in METHODS]
    if fmt == "latex":
        return _latex_table(title, h, rows)
    return _md_table(title, h, rows)


# ---------------------------------------------------------------------------
# Table 2: Temporal faithfulness
# ---------------------------------------------------------------------------

def make_table2(folds: list[int], fmt: str) -> str:
    """Temporal deletion / insertion AUC and completeness error."""
    metrics = [
        ("Deletion AUC ↓", False, "temporal_faithfulness", "{m}", "deletion_auc"),
        ("Insertion AUC ↑", True,  "temporal_faithfulness", "{m}", "insertion_auc"),
        ("Completeness Err ↓", False, "temporal_faithfulness", "{m}", "completeness_error"),
    ]
    h = ["Metric"] + [METHOD_LABELS[m] for m in METHODS]
    rows: list[tuple] = []

    for fold in folds:
        try:
            bl_seqs = _load_baseline_seqs(fold)
        except (FileNotFoundError, OSError):
            continue
        bl_by_idx  = {s["seq_idx"]: s for s in bl_seqs}
        actor_seqs = []
        try:
            actor_seqs = _load_actor_seqs(fold)
        except (FileNotFoundError, OSError):
            pass
        has_actor  = len(actor_seqs) > 0
        common     = [s for s in actor_seqs if s["seq_idx"] in bl_by_idx] if has_actor else []
        n          = len(common) if has_actor else len(bl_by_idx)
        actor_note = "" if has_actor else " (baselines only)"
        rows.append((f"** Fold {fold} (n={n}{actor_note}) **", "", "", "", ""))
        for label, higher, *path_tmpl in metrics:
            vals_per_method = {}
            for m in METHODS:
                path = [p.replace("{m}", m) for p in path_tmpl]
                if m == "actor":
                    src = common if has_actor else []
                else:
                    src = [bl_by_idx[s["seq_idx"]] for s in common] if has_actor else list(bl_by_idx.values())
                vals_per_method[m] = _arr(src, *path)

            means = [v.mean() if len(v) > 0 else float("nan")
                     for v in vals_per_method.values()]
            best  = _best(means, higher)
            row   = [label]
            for m, v in zip(METHODS, vals_per_method.values()):
                mn, sd = _ms(v)
                s = _fmt(mn, sd)
                row.append(_bold(s, mn, best))
            rows.append(tuple(row))

    title = "Table 2 — Temporal Faithfulness (all folds)"
    if fmt == "latex":
        return _latex_table(title, h, rows)
    return _md_table(title, h, rows)


# ---------------------------------------------------------------------------
# Table 3: Self-evaluated spatial PGI / PGU
# ---------------------------------------------------------------------------

def make_table3(folds: list[int], min_p_full: float, fmt: str) -> str:
    """Self-evaluated spatial PGI / PGU with PGI−Rand and PGU−Rand."""
    h = ["Metric"] + [METHOD_LABELS[m] for m in METHODS]
    rows: list[tuple] = []

    for fold in folds:
        try:
            bl_seqs = _load_baseline_seqs(fold)
        except (FileNotFoundError, OSError):
            continue

        bl_by_idx = {s["seq_idx"]: s for s in bl_seqs}
        actor_seqs = []
        try:
            actor_seqs = _load_actor_seqs(fold)
        except (FileNotFoundError, OSError):
            pass

        actor_by_idx = {s["seq_idx"]: s for s in actor_seqs}
        has_actor    = len(actor_seqs) > 0

        # Source sequences: use common set if actor available, else all baselines.
        if has_actor:
            common = [s for s in actor_seqs if s["seq_idx"] in bl_by_idx]
        else:
            # No actor data — show baselines on the full baseline set.
            common = bl_seqs

        if min_p_full:
            common = [s for s in common if s.get("p_full", 0) >= min_p_full]
        n = len(common)
        actor_note = "" if has_actor else ", baselines only"
        rows.append((f"** Fold {fold} (n={n}, p_full≥{min_p_full}{actor_note}) **", "", "", "", ""))

        for k in [1, 3, 5]:
            # PGI row
            pgi_vals = {}
            rand_vals = {}
            pgu_vals = {}
            for m in METHODS:
                if m == "actor":
                    if not has_actor:
                        pgi_vals[m]  = np.array([])
                        pgu_vals[m]  = np.array([])
                        rand_vals[m] = np.array([])
                        continue
                    src = [actor_by_idx[s["seq_idx"]] for s in common
                           if s["seq_idx"] in actor_by_idx]
                else:
                    src = [bl_by_idx[s["seq_idx"]] for s in common
                           if s["seq_idx"] in bl_by_idx]
                pgi_vals[m]  = _arr(src, "faithfulness", m, "pgi_pgu",      str(k), "pgi")
                pgu_vals[m]  = _arr(src, "faithfulness", m, "pgi_pgu",      str(k), "pgu")
                rand_vals[m] = _arr(src, "faithfulness", m, "pgi_pgu_rand", str(k), "pgi")

            # PGI row
            means = [v.mean() if len(v) else float("nan") for v in pgi_vals.values()]
            best  = _best(means, higher=True)
            row   = [f"PGI@{k} ↑"]
            for m in METHODS:
                mn, sd = _ms(pgi_vals[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            # PGU row
            means = [v.mean() if len(v) else float("nan") for v in pgu_vals.values()]
            best  = _best(means, higher=False)
            row   = [f"PGU@{k} ↓"]
            for m in METHODS:
                mn, sd = _ms(pgu_vals[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            # Rand row (shared reference)
            means = [v.mean() if len(v) else float("nan") for v in rand_vals.values()]
            best  = _best(means, higher=False)
            row   = [f"Rand@{k} ↓"]
            for m in METHODS:
                mn, sd = _ms(rand_vals[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            # PGI − Rand row
            diff_pgi = {m: pgi_vals[m] - rand_vals[m] for m in METHODS}
            means    = [v.mean() if len(v) else float("nan") for v in diff_pgi.values()]
            best     = _best(means, higher=True)
            row      = [f"PGI−Rand@{k} ↑"]
            for m in METHODS:
                mn, sd = _ms(diff_pgi[m])
                row.append(_bold(_fmt(mn, sd, signed=True), mn, best))
            rows.append(tuple(row))

            # PGU − Rand row: bold based on lowest absolute PGU, not most negative
            # difference. A very negative PGU−Rand can simply reflect a high Rand
            # baseline (e.g. zero-imputation always causes large shocks), not a
            # genuinely low PGU. The winner is whoever has the smallest absolute PGU.
            diff_pgu     = {m: pgu_vals[m] - rand_vals[m] for m in METHODS}
            pgu_means    = [pgu_vals[m].mean() if len(pgu_vals[m]) else float("nan")
                            for m in METHODS]
            best_pgu     = _best(pgu_means, higher=False)
            pgu_mn_by_m  = {m: (pgu_vals[m].mean() if len(pgu_vals[m]) else float("nan"))
                            for m in METHODS}
            row = [f"PGU−Rand@{k} ↓"]
            for m in METHODS:
                mn, sd = _ms(diff_pgu[m])
                row.append(_bold(_fmt(mn, sd, signed=True), pgu_mn_by_m[m], best_pgu))
            rows.append(tuple(row))

            rows.append(("", "", "", "", ""))  # spacer

    title = (
        f"Table 3 — Self-evaluated Spatial Faithfulness "
        f"(p_full ≥ {min_p_full}; each method evaluated with its own imputation)"
    )
    if fmt == "latex":
        return _latex_table(title, h, rows)
    return _md_table(title, h, rows)


# ---------------------------------------------------------------------------
# Table 4: Cross-evaluated PGI under Zero protocol (diagnostic)
# ---------------------------------------------------------------------------

def make_table4(folds: list[int], min_p_full: Optional[float], fmt: str) -> str:
    """Cross-evaluated PGI: all methods' rankings evaluated under Zero imputation."""
    h = ["Metric"] + [METHOD_LABELS[m] for m in METHODS]
    rows: list[tuple] = []

    for fold in folds:
        seqs = _load_crosseval_seqs(fold, min_p_full)
        if not seqs:
            continue
        n = len(seqs)
        label = f"p_full≥{min_p_full}" if min_p_full else "all"
        rows.append((f"** Fold {fold} (n={n}, {label}) **", "", "", "", ""))

        for k in [1, 3, 5]:
            pgi_vals  = {}
            rand_vals = {}
            pgu_vals  = {}
            for m in METHODS:
                if m not in [r.get("crosseval", {}).keys() for r in seqs[:1]][0]:
                    pgi_vals[m]  = np.array([])
                    rand_vals[m] = np.array([])
                    pgu_vals[m]  = np.array([])
                    continue
                pgi_vals[m]  = _arr(seqs, "crosseval", m, "pgi_pgu",      str(k), "pgi")
                rand_vals[m] = _arr(seqs, "crosseval", m, "pgi_pgu_rand", str(k), "pgi")
                pgu_vals[m]  = _arr(seqs, "crosseval", m, "pgi_pgu",      str(k), "pgu")

            # PGI row
            means = [v.mean() if len(v) else float("nan") for v in pgi_vals.values()]
            best  = _best(means, higher=True)
            row   = [f"PGI@{k} ↑ (eval: Zero)"]
            for m in METHODS:
                mn, sd = _ms(pgi_vals[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            # Rand row
            means = [v.mean() if len(v) else float("nan") for v in rand_vals.values()]
            best  = _best(means, higher=False)
            row   = [f"Rand@{k} ↓"]
            for m in METHODS:
                mn, sd = _ms(rand_vals[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            # PGI_norm row (normalized by p_full)
            pgi_norm_vals = {}
            for m in METHODS:
                pgi_norm_vals[m] = _arr(seqs, "crosseval", m, "pgi_norm", str(k))
            means = [v.mean() if len(v) else float("nan") for v in pgi_norm_vals.values()]
            best  = _best(means, higher=True)
            row   = [f"PGI_norm@{k} ↑"]
            for m in METHODS:
                mn, sd = _ms(pgi_norm_vals[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            # PGI − Rand row
            diff_pgi = {m: (pgi_vals[m] - rand_vals[m]
                            if len(pgi_vals[m]) and len(rand_vals[m])
                            else np.array([]))
                        for m in METHODS}
            means = [v.mean() if len(v) else float("nan") for v in diff_pgi.values()]
            best  = _best(means, higher=True)
            row   = [f"PGI−Rand@{k} ↑"]
            for m in METHODS:
                mn, sd = _ms(diff_pgi[m])
                row.append(_bold(_fmt(mn, sd, signed=True), mn, best))
            rows.append(tuple(row))

            rows.append(("", "", "", "", ""))  # spacer

    title = (
        "Table 4 — Cross-evaluated Spatial PGI under Zero Protocol (diagnostic)\n"
        "All methods' SHAP rankings evaluated with the same (Zero) imputation. "
        "Positive PGI−Rand indicates rankings correctly identify joints that matter "
        "under zero-masking; near-zero for ActorSHAP shows it computes a distinct "
        "on-manifold attribution game."
    )
    if fmt == "latex":
        return _latex_table(title, h, rows)
    return _md_table(title, h, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_table_pooled(folds: list[int], min_p_full: float, fmt: str) -> str:
    """Pooled multi-fold summary: mean ± std across all sequences from all folds.

    Uses a single combined collection of sequences (as if all folds were one set)
    for rank stability, completeness error, and temporal metrics.
    Only includes sequences for which both actor and baseline data are available.
    """
    # --- pool actor + baseline seqs across folds ---
    all_actor:   list[dict] = []
    all_bl:      dict[str, list[dict]] = defaultdict(list)  # method -> seqs
    all_crosseval: list[dict] = []

    for fold in folds:
        try:
            bl_seqs = _load_baseline_seqs(fold)
        except (FileNotFoundError, OSError):
            continue
        bl_by_idx = {s["seq_idx"]: s for s in bl_seqs}

        actor_seqs = []
        try:
            actor_seqs = _load_actor_seqs(fold)
        except (FileNotFoundError, OSError):
            pass

        if actor_seqs:
            common = [s for s in actor_seqs if s["seq_idx"] in bl_by_idx]
            all_actor.extend(common)
            for m in ("zero", "mean", "marginal"):
                all_bl[m].extend([bl_by_idx[s["seq_idx"]] for s in common])

        ce_seqs = _load_crosseval_seqs(fold, min_p_full)
        all_crosseval.extend(ce_seqs)

    h = ["Metric"] + [METHOD_LABELS[m] for m in METHODS]
    rows: list[tuple] = []

    n_actor = len(all_actor)
    n_ce    = len(all_crosseval)

    # Rank Stability + Completeness Error (Table 1 content)
    rows.append((f"** Primary Metrics (pooled, n={n_actor}) **", "", "", "", ""))
    rs_mn, rs_sd = _ms(_arr(all_actor, "rank_stability", "mean_rank_corr"))
    row = ["Rank Stability ↑", _fmt(rs_mn, rs_sd), "—", "—", "—"]
    rows.append(tuple(row))

    comp_per_method = {}
    for m in METHODS:
        if m == "actor":
            v = _arr(all_actor, "faithfulness", "actor", "completeness_error")
        else:
            v = _arr(all_bl[m], "faithfulness", m, "completeness_error")
        comp_per_method[m] = _ms(v)
    comps = [comp_per_method[m][0] for m in METHODS]
    best_c = _best(comps, higher=False)
    row = ["Spatial Compl Err ↓"]
    for m in METHODS:
        mn, sd = comp_per_method[m]
        row.append(_bold(_fmt(mn, sd), mn, best_c))
    rows.append(tuple(row))

    temp_comp = {}
    for m in METHODS:
        if m == "actor":
            v = _arr(all_actor, "temporal_faithfulness", "actor", "completeness_error")
        else:
            v = _arr(all_bl[m], "temporal_faithfulness", m, "completeness_error")
        temp_comp[m] = _ms(v)
    comps = [temp_comp[m][0] for m in METHODS]
    best_c = _best(comps, higher=False)
    row = ["Temporal Compl Err ↓"]
    for m in METHODS:
        mn, sd = temp_comp[m]
        row.append(_bold(_fmt(mn, sd), mn, best_c))
    rows.append(tuple(row))
    rows.append(("", "", "", "", ""))

    # Temporal Ins/Del AUC
    rows.append((f"** Temporal Faithfulness (pooled, n={n_actor}) **", "", "", "", ""))
    for label, higher, key in [("Temporal Del AUC ↓", False, "deletion_auc"),
                                ("Temporal Ins AUC ↑", True,  "insertion_auc")]:
        vals = {}
        for m in METHODS:
            if m == "actor":
                v = _arr(all_actor, "temporal_faithfulness", "actor", key)
            else:
                v = _arr(all_bl[m], "temporal_faithfulness", m, key)
            vals[m] = _ms(v)
        means = [vals[m][0] for m in METHODS]
        best  = _best(means, higher)
        row   = [label]
        for m in METHODS:
            mn, sd = vals[m]
            row.append(_bold(_fmt(mn, sd), mn, best))
        rows.append(tuple(row))
    rows.append(("", "", "", "", ""))

    # Self-eval PGI/PGU (pooled, filtered)
    actor_filt = [s for s in all_actor if s.get("p_full", 0) >= min_p_full]
    n_filt     = len(actor_filt)
    rows.append((f"** Self-eval Spatial (pooled, n={n_filt}, p_full≥{min_p_full}) **", "", "", "", ""))
    for k in [1, 3, 5]:
        pgi_v, pgu_v, rand_v = {}, {}, {}
        for m in METHODS:
            if m == "actor":
                src = actor_filt
            else:
                src = [bl_by_idx[s["seq_idx"]] for bl_by_idx in
                       [{s2["seq_idx"]: s2 for s2 in fold_bl}
                        for fold_bl in [_load_baseline_seqs(fold)
                                        for fold in folds
                                        if os.path.exists(os.path.join(RESULTS_ROOT, _BASELINE_DIRS.get(fold, ""), "per_sequence.jsonl"))]]
                       for s in actor_filt
                       if s["seq_idx"] in bl_by_idx]
                # simpler: use all_bl[m] filtered to common seqs
                actor_idxs = {s["seq_idx"] for s in actor_filt}
                src = [s for s in all_bl[m] if s["seq_idx"] in actor_idxs]
            pgi_v[m]  = _arr(src, "faithfulness", m, "pgi_pgu",      str(k), "pgi")
            pgu_v[m]  = _arr(src, "faithfulness", m, "pgi_pgu",      str(k), "pgu")
            rand_v[m] = _arr(src, "faithfulness", m, "pgi_pgu_rand", str(k), "pgi")

        diff_pgi = {m: pgi_v[m] - rand_v[m] for m in METHODS}
        for label, higher, vals_d, signed in [
            (f"PGI@{k} ↑",      True,  pgi_v,    False),
            (f"PGU@{k} ↓",      False, pgu_v,    False),
            (f"PGI−Rand@{k} ↑", True,  diff_pgi, True),
        ]:
            means = [vals_d[m].mean() if len(vals_d[m]) else float("nan") for m in METHODS]
            best  = _best(means, higher)
            row   = [label]
            for m in METHODS:
                mn, sd = _ms(vals_d[m])
                row.append(_bold(_fmt(mn, sd, signed=signed), mn, best))
            rows.append(tuple(row))
        rows.append(("", "", "", "", ""))

    # Cross-eval PGI (pooled, filtered)
    rows.append((f"** Cross-eval PGI (Zero protocol, pooled, n={n_ce}) **", "", "", "", ""))
    for k in [1, 3]:
        pgi_n = {m: _arr(all_crosseval, "crosseval", m, "pgi_norm", str(k)) for m in METHODS}
        means = [pgi_n[m].mean() if len(pgi_n[m]) else float("nan") for m in METHODS]
        best  = _best(means, higher=True)
        row   = [f"PGI_norm@{k} ↑"]
        for m in METHODS:
            mn, sd = _ms(pgi_n[m])
            row.append(_bold(_fmt(mn, sd), mn, best))
        rows.append(tuple(row))

        pgi_r = {m: _arr(all_crosseval, "crosseval", m, "pgi_minus_rand", str(k)) for m in METHODS}
        means = [pgi_r[m].mean() if len(pgi_r[m]) else float("nan") for m in METHODS]
        best  = _best(means, higher=True)
        row   = [f"PGI−Rand@{k} ↑"]
        for m in METHODS:
            mn, sd = _ms(pgi_r[m])
            row.append(_bold(_fmt(mn, sd, signed=True), mn, best))
        rows.append(tuple(row))

    title = f"Table 0 — Pooled Multi-Fold Summary (folds {folds})"
    if fmt == "latex":
        return _latex_table(title, h, rows)
    return _md_table(title, h, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publishable SHAP evaluation tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folds",      nargs="+", type=int, default=[1])
    parser.add_argument("--min_p_full", type=float, default=0.35,
                        help="Confidence filter for Table 3 and Table 4.")
    parser.add_argument("--format",     choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--tables",     nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which tables to print (0=pooled, 1-4=per-fold).")
    parser.add_argument("--output",     default=None,
                        help="Write output to file instead of stdout.")
    args = parser.parse_args()

    fmt  = "latex" if args.format == "latex" else "markdown"
    out  = []

    if 0 in args.tables:
        out.append(make_table_pooled(args.folds, args.min_p_full, fmt))
    if 1 in args.tables:
        out.append(make_table1(args.folds, fmt))
    if 2 in args.tables:
        out.append(make_table2(args.folds, fmt))
    if 3 in args.tables:
        out.append(make_table3(args.folds, args.min_p_full, fmt))
    if 4 in args.tables:
        out.append(make_table4(args.folds, args.min_p_full, fmt))

    result = "\n\n".join(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
        print(f"Tables written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
