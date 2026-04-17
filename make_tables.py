"""make_tables.py — publishable spatial-faithfulness tables for the flow-
matching SHAP branch.

Three tables on the classifier-held-out SUB01 fold-1 test set (n=240
sequences, J=17 H36M joints, softmax-prob readout):

  Table 1 — **Shared zero imputation**. Zero-KernelSHAP's native
            perturbation, so Zero gets a home-turf alignment advantage
            while OTFlow-SHAP evaluates off-home-turf.

  Table 2 — **Shared marginal imputation** (donor joints drawn uniformly
            from the POTR training pool). On-manifold but not any method's
            native reference ⇒ every ranking is measured off-home-turf.

  Table 3 — **Shared flow_imputer imputation** (RePaint-style conditional
            completions from the trained flow-matching velocity net).
            On-manifold and the semantic home turf for OTFlow-SHAP.

Across all three tables:
  * the **rankings** are unchanged (zero/mean/marginal KernelSHAP and
    OTFlow-SHAP phi_j = Σ_{t,c} psi[t,j,c], summed over clips per
    sequence and pooled across the 3 flow seeds);
  * the **imputation** used at faithfulness time is what differs;
  * any gap that survives all three is attributable to ranking quality.

Usage
=====
    python make_tables.py                        # all three tables, markdown
    python make_tables.py --tables 1             # Table 1 only
    python make_tables.py --format latex         # LaTeX booktabs
    python make_tables.py --output tables.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")

# Shared-imputation cross-evaluation results (one per_sequence.jsonl per fold
# per imputation). Produced by
# :mod:`scripts.compute_shared_imputation_faithfulness`.
_SHARED_IMPUTATION_DIRS = {
    "zero": {
        1: "shap_shared_imputation_zero_fold1",
    },
    "marginal": {
        1: "shap_shared_imputation_marginal_fold1",
    },
    "flow_imputer": {
        1: "shap_shared_imputation_flow_imputer_fold1",
    },
}

# Column order and labels. Tables 1/2/3 all share the same 4 ranking sources.
_RANKING_ORDER  = ["zero", "mean", "marginal", "flow_shap"]
_RANKING_LABELS = {
    "zero":      "Zero",
    "mean":      "Mean",
    "marginal":  "Marginal",
    "flow_shap": "OTFlow-SHAP (ours)",
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
            if v == v:
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
        cells = [f"{str(c):{col_w}}" for c in row]
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
# Shared-imputation cross-evaluation (Tables 1 / 2 / 3)
# ---------------------------------------------------------------------------

def _flatten_shared_rankings(per_seq: list[dict]) -> dict[str, list[dict]]:
    """Turn a shared-imputation per_sequence.jsonl into per-ranking lists.

    Each input record stores
    ``faithfulness = {zero: {...}, mean: {...}, marginal: {...},
                      flow_seed42: {...}, flow_seed123: {...}, ...}``.

    We emit one list per ranking column in ``_RANKING_ORDER``. The
    flow-seed columns are flattened into a single ``flow_shap`` list so
    pooled mean/std reflect both sequence and seed variability.
    """
    out: dict[str, list[dict]] = {k: [] for k in _RANKING_ORDER}
    for r in per_seq:
        fmap = r.get("faithfulness", {})
        for m in ("zero", "mean", "marginal"):
            if m in fmap:
                out[m].append({
                    "seq_idx": r["seq_idx"],
                    "p_full":  r["p_full"],
                    "faithfulness": {m: fmap[m]},
                })
        for key, val in fmap.items():
            if key.startswith("flow_") or key.startswith("otflow"):
                out["flow_shap"].append({
                    "seq_idx": r["seq_idx"],
                    "p_full":  r["p_full"],
                    "seed":    key,
                    "faithfulness": {"flow_shap": val},
                })
    return out


def make_table_shared_imputation(
    folds: list[int], fmt: str, imputation: str, table_num: int,
) -> str:
    """Shared-imputation cross-evaluation of SHAP rankings.

    Same 240 SUB01 sequences, same classifier_fn, same J=17 universe, same
    k-grid; only the imputation used at faithfulness time differs across
    the three tables. Under an imputation that is not any method's native
    reference, ranking quality is decoupled from home-turf alignment bias.
    """
    rel_map = _SHARED_IMPUTATION_DIRS[imputation]
    h = ["Metric"] + [_RANKING_LABELS[m] for m in _RANKING_ORDER]
    rows: list[tuple] = []

    for fold in folds:
        rel = rel_map.get(fold)
        if rel is None:
            continue
        p = os.path.join(RESULTS_ROOT, rel, "per_sequence.jsonl")
        if not os.path.exists(p):
            rows.append((f"** Fold {fold}: {p} not found **",
                         *["" for _ in _RANKING_ORDER]))
            continue

        per_seq = list(_load_jsonl(p))
        per_ranking = _flatten_shared_rankings(per_seq)

        n_seqs = len(per_seq)
        flow_seed_set   = {r.get("seed") for r in per_ranking["flow_shap"]}
        n_flow_rankings = len(per_ranking["flow_shap"])
        n_flow_seeds    = len([s for s in flow_seed_set if s])
        header_note = (
            f"baselines: n={len(per_ranking['zero'])} sequences; "
            f"flow_shap: {n_flow_seeds} seed(s) × {n_seqs} sequences "
            f"({n_flow_rankings} pooled records). "
            f"Imputation = {imputation}."
        )
        rows.append((f"** Fold {fold} — {header_note} **",
                     *["" for _ in _RANKING_ORDER]))

        def _collect(method: str, path: tuple) -> np.ndarray:
            key = "flow_shap" if method == "flow_shap" else method
            return _arr(per_ranking[method], "faithfulness", key, *path)

        for k in [1, 3, 5]:
            pgi = {m: _collect(m, ("pgi_pgu", str(k), "pgi")) for m in _RANKING_ORDER}
            pgu = {m: _collect(m, ("pgi_pgu", str(k), "pgu")) for m in _RANKING_ORDER}
            rnd = {m: _collect(m, ("pgi_pgu_rand", str(k), "pgi")) for m in _RANKING_ORDER}
            diff = {m: (pgi[m] - rnd[m]) if (len(pgi[m]) and len(pgi[m]) == len(rnd[m]))
                    else np.array([]) for m in _RANKING_ORDER}

            means = [v.mean() if len(v) else float("nan") for v in pgi.values()]
            best  = _best(means, higher=True)
            row   = [f"PGI@{k} ↑"]
            for m in _RANKING_ORDER:
                mn, sd = _ms(pgi[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            means = [v.mean() if len(v) else float("nan") for v in pgu.values()]
            best  = _best(means, higher=False)
            row   = [f"PGU@{k} ↓"]
            for m in _RANKING_ORDER:
                mn, sd = _ms(pgu[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

            row = [f"Rand@{k}"]
            for m in _RANKING_ORDER:
                mn, sd = _ms(rnd[m])
                row.append(_fmt(mn, sd))
            rows.append(tuple(row))

            means = [v.mean() if len(v) else float("nan") for v in diff.values()]
            best  = _best(means, higher=True)
            row   = [f"PGI−Rand@{k} ↑"]
            for m in _RANKING_ORDER:
                mn, sd = _ms(diff[m])
                row.append(_bold(_fmt(mn, sd, signed=True), mn, best))
            rows.append(tuple(row))

            rows.append(tuple(["" for _ in range(len(_RANKING_ORDER) + 1)]))

        for label, path, higher in [
            ("Deletion AUC ↓",        ("deletion_auc",),         False),
            ("Random Deletion AUC",   ("random_deletion_auc",),  False),
            ("Insertion AUC ↑",       ("insertion_auc",),        True),
            ("Random Insertion AUC",  ("random_insertion_auc",), True),
        ]:
            vals = {m: _collect(m, path) for m in _RANKING_ORDER}
            means = [v.mean() if len(v) else float("nan") for v in vals.values()]
            best  = _best(means, higher)
            row   = [label]
            for m in _RANKING_ORDER:
                mn, sd = _ms(vals[m])
                row.append(_bold(_fmt(mn, sd), mn, best))
            rows.append(tuple(row))

        pf = _arr(per_seq, "p_full")
        if len(pf):
            rows.append(tuple(["p_full (sanity)",
                               _fmt(float(pf.mean()), float(pf.std())),
                               *["—" for _ in _RANKING_ORDER[1:]]]))
        rows.append(tuple(["" for _ in range(len(_RANKING_ORDER) + 1)]))

    imp_notes = {
        "zero": (
            "hidden joints set to 0 in the raw pose space (pelvis restored). "
            "This is Zero-KernelSHAP's native perturbation, which gives Zero "
            "a home-turf alignment advantage; OTFlow-SHAP evaluates off-home-"
            "turf here since its native reference is the flow back-solve "
            "x0_hat."
        ),
        "marginal": (
            "donor joints drawn uniformly from the POTR training pool. "
            "On-manifold, but not any method's native reference, so "
            "zero-/mean-/flow-SHAP rankings are all evaluated off-home-turf."
        ),
        "flow_imputer": (
            "joints filled with RePaint-style conditional completions from "
            "the trained flow-matching velocity net (FlowImputer). "
            "On-manifold and the semantic home turf for OTFlow-SHAP."
        ),
    }[imputation]
    title = (
        f"Table {table_num} — Cross-evaluation of SHAP rankings under a "
        f"shared **{imputation} imputation** (SUB01 fold-1 test set, "
        "n=240 sequences, softmax prob readout, J=17, k∈{1,2,3,5}, "
        "n_random_repeats=10). Rankings unchanged from Table 1; only the "
        f"perturbation differs — {imp_notes} OTFlow-SHAP's per-joint "
        "attributions use the axiomatic signed-sum phi_j = Σ_{t,c} "
        "psi[t,j,c] (summed over clips per sequence). Flow-SHAP seeds are "
        "pooled, so its ± reflects both sequence and seed variability."
    )
    if fmt == "latex":
        return _latex_table(title, h, rows)
    return _md_table(title, h, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publishable spatial-faithfulness tables for the "
                    "flow-matching SHAP branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folds",  nargs="+", type=int, default=[1])
    parser.add_argument("--format", choices=["markdown", "latex"],
                        default="markdown")
    parser.add_argument("--tables", nargs="+", type=int, default=[1, 2, 3],
                        help="Which tables to print "
                             "(1=zero-imputation, 2=marginal-imputation, "
                             "3=flow_imputer-imputation).")
    parser.add_argument("--output", default=None,
                        help="Write output to file instead of stdout.")
    args = parser.parse_args()

    fmt = "latex" if args.format == "latex" else "markdown"
    out: list[str] = []

    if 1 in args.tables:
        out.append(make_table_shared_imputation(
            args.folds, fmt, "zero", table_num=1))
    if 2 in args.tables:
        out.append(make_table_shared_imputation(
            args.folds, fmt, "marginal", table_num=2))
    if 3 in args.tables:
        out.append(make_table_shared_imputation(
            args.folds, fmt, "flow_imputer", table_num=3))

    result = "\n\n".join(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
        print(f"Tables written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
