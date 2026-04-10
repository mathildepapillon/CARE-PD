"""evaluate_shap_consistency.py — Cross-method SHAP faithfulness comparison.

Implements two model-free faithfulness checks across UPDRS severity groups,
comparing ActorSHAP to baseline methods (zero, marginal) with no checkpoint
loading required at evaluation time.

Option B — Within-group rank consistency
    For each method and UPDRS class, compute all pairwise Spearman rank
    correlations of the |SHAP| joint vectors. Higher within-group consistency
    means the method identifies a stable class-specific signal. The key
    comparison is: does ActorSHAP's within-group consistency beat baselines?
    Additionally computes between-group distance: are the average SHAP profiles
    for different UPDRS classes more distinct for ActorSHAP?

Option C1 — SHAP partial-sum sufficiency  (no raw data needed)
    Within each UPDRS class, compute R²(p_baseline + Σ_{j∈top-k} φ_j, p_full).
    If SHAP is faithful, the top-k joints always dominate the prediction change,
    regardless of severity group.

Option C2 — ROAR-lite kinematic probe  (requires the preprocessed PKL)
    Per-joint kinematic statistics (std, velocity) → Ridge LOO-CV → predict
    p_full. Top-k SHAP joints should outperform random-k and bottom-k.

Multi-fold usage (class 0 = fold 2, class 1 = fold 8, class 2 = fold 1)::

    python evaluate_shap_consistency.py \\
        --actor_shap_dirs  results/shap_actor_potr_bmclab_fold1_full \\
        --baseline_dirs    results/shap_baselines_potr_bmclab_fold1_full \\
                           results/shap_baselines_potr_bmclab_fold2 \\
                           results/shap_baselines_potr_bmclab_fold8 \\
        --eval_pkls        assets/.../BMCLab_eval_1.pkl \\
                           assets/.../BMCLab_eval_2.pkl \\
                           assets/.../BMCLab_eval_8.pkl \\
        --output_dir       results/shap_consistency_multigroup

Outputs::

    results/shap_consistency_multigroup/
        rank_consistency.json   – Option B per-method per-group
        shap_sufficiency.json   – Option C1 per-method per-group
        joint_probe.json        – Option C2 per-method per-group
        report.txt              – human-readable summary tables
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
H36M_JOINT_NAMES = [
    "Pelvis", "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck", "Head",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
]
J = len(H36M_JOINT_NAMES)   # 17
FEAT_PER_JOINT = 4           # std_x, std_y, std_z, mean_vel


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_actor_rows(shap_dir: str) -> list[dict]:
    """Deduplicated actor shard rows (last write wins)."""
    rows_by_seq: dict[int, dict] = {}
    for path in sorted(glob.glob(
            os.path.join(shap_dir, "shards", "per_sequence_shard*_of_*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows_by_seq[r["seq_idx"]] = r
    rows = sorted(rows_by_seq.values(), key=lambda r: r["seq_idx"])
    print(f"  ActorSHAP {shap_dir}: {len(rows)} unique seqs", flush=True)
    return rows


def _load_baseline_rows(baseline_dir: str) -> list[dict]:
    path = os.path.join(baseline_dir, "per_sequence.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    rows.sort(key=lambda r: r["seq_idx"])
    print(f"  Baselines  {baseline_dir}: {len(rows)} seqs", flush=True)
    return rows


def _load_pkl(pkl_path: str) -> list[np.ndarray]:
    """(T_real, J, 3) float32 pose arrays, one per sequence."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    out = []
    for p, m in zip(data["pose"], data["pad_mask"]):
        out.append(p[m.astype(bool)].astype(np.float32))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _joint_features(pose: np.ndarray) -> np.ndarray:
    """(J * 4,) kinematic vector: std_xyz + mean_velocity per joint."""
    std_pos  = pose.std(axis=0)                                # (J, 3)
    vel      = np.diff(pose, axis=0)                           # (T-1, J, 3)
    mean_vel = np.linalg.norm(vel, axis=-1).mean(axis=0)      # (J,)
    feat     = np.concatenate([std_pos, mean_vel[:, None]], axis=-1)  # (J, 4)
    return feat.reshape(-1).astype(np.float32)


def _ridge_loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Analytical Ridge LOO-CV R² via the hat-matrix identity (O(n p²))."""
    if y.std() < 1e-8 or len(y) < 4:
        return float("nan")
    sc   = StandardScaler()
    X_c  = sc.fit_transform(X)
    y_c  = y - y.mean()
    gcv  = RidgeCV(alphas=np.logspace(-3, 4, 30), fit_intercept=False)
    gcv.fit(X_c, y_c)
    alpha = float(gcv.alpha_)
    n, p  = X_c.shape
    inv   = np.linalg.inv(X_c.T @ X_c + alpha * np.eye(p))
    H_d   = (X_c @ inv * X_c).sum(axis=1)
    y_hat = X_c @ (inv @ X_c.T @ y_c)
    loo_r = (y_c - y_hat) / (1.0 - H_d)
    return float(1.0 - (loo_r ** 2).sum() / (y_c ** 2).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Core metrics (operate on a list of rows with the same UPDRS class)
# ──────────────────────────────────────────────────────────────────────────────

def _rank_consistency_for_rows(rows: list[dict], method: str) -> dict:
    """Pairwise Spearman |SHAP| rank correlation (vectorised)."""
    vecs = []
    for r in rows:
        sv = r["shap_values"].get(method, {})
        vecs.append(np.array([abs(sv.get(jn, 0.0)) for jn in H36M_JOINT_NAMES],
                              dtype=np.float64))
    N = len(vecs)
    if N < 2:
        return {"mean": float("nan"), "std": float("nan"), "n_pairs": 0, "n": N}
    ranks = np.array([rankdata(v) for v in vecs], dtype=np.float64)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms  = np.sqrt((ranks ** 2).sum(axis=1, keepdims=True)) + 1e-12
    ranks /= norms
    C    = ranks @ ranks.T
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    corrs = C[mask]
    return {
        "mean": float(corrs.mean()),
        "std":  float(corrs.std()),
        "median": float(np.median(corrs)),
        "n_pairs": int(mask.sum()),
        "n": N,
    }


def _mean_shap_profile(rows: list[dict], method: str) -> np.ndarray:
    """Mean signed SHAP vector (J,) across rows."""
    mats = []
    for r in rows:
        sv = r["shap_values"].get(method, {})
        mats.append([sv.get(jn, 0.0) for jn in H36M_JOINT_NAMES])
    return np.array(mats).mean(axis=0)


def _shap_sufficiency_for_rows(
    rows: list[dict],
    method: str,
    k_list: tuple[int, ...],
    n_random: int,
    rng: np.random.Generator,
    global_rank: list[int],
) -> dict:
    """Partial-sum R² for top-k / random-k / bottom-k subsets."""
    p_fulls  = np.array([r["p_full"] for r in rows], dtype=np.float64)
    shap_mat = np.array(
        [[r["shap_values"].get(method, {}).get(jn, 0.0) for jn in H36M_JOINT_NAMES]
         for r in rows], dtype=np.float64
    )
    shap_sum_all = shap_mat.sum(axis=1)
    p_baseline   = float(np.mean(p_fulls - shap_sum_all))

    def _r2(joints):
        pred = shap_mat[:, joints].sum(axis=1) + p_baseline
        ss_r = ((p_fulls - pred) ** 2).sum()
        ss_t = ((p_fulls - p_fulls.mean()) ** 2).sum()
        return float(1.0 - ss_r / ss_t) if ss_t > 1e-10 else float("nan")

    result = {}
    for k in k_list:
        k = min(k, J)
        top_j = global_rank[:k]
        bot_j = global_rank[-k:]
        r2_rands = [_r2(rng.choice(J, size=k, replace=False).tolist())
                    for _ in range(n_random)]
        result[k] = {
            "top_k":    _r2(top_j),
            "bottom_k": _r2(bot_j),
            "random_k": float(np.mean(r2_rands)),
        }
    return result


def _probe_for_rows(
    rows: list[dict],
    all_poses: list[np.ndarray],
    method: str,
    k_list: tuple[int, ...],
    n_random: int,
    rng: np.random.Generator,
    global_rank_inf: list[int],
    informative_joints: list[int],
) -> dict:
    """Kinematic LOO-CV R² for top-k / random-k / bottom-k."""
    feats = np.stack([_joint_features(all_poses[r["seq_idx"]]) for r in rows])
    p     = np.array([r["p_full"] for r in rows], dtype=np.float32)

    def _sel(joints):
        cols = np.concatenate([
            np.arange(j * FEAT_PER_JOINT, (j + 1) * FEAT_PER_JOINT) for j in joints
        ])
        return feats[:, cols]

    result = {}
    for k in k_list:
        k = min(k, len(informative_joints))
        top_j = global_rank_inf[:k]
        bot_j = global_rank_inf[-k:]
        r2_rands = []
        for _ in range(n_random):
            rj = rng.choice(informative_joints, size=k, replace=False).tolist()
            v  = _ridge_loo_r2(_sel(rj), p)
            if not np.isnan(v):
                r2_rands.append(v)
        result[k] = {
            "top_k":    _ridge_loo_r2(_sel(top_j), p),
            "bottom_k": _ridge_loo_r2(_sel(bot_j), p),
            "random_k": float(np.mean(r2_rands)) if r2_rands else float("nan"),
        }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

_UPDRS_LABEL = {0: "mild (0)", 1: "moderate (1)", 2: "severe (2)"}


def run_analysis(
    actor_rows_by_fold:    list[list[dict]],   # one list per fold (or empty)
    baseline_rows_by_fold: list[list[dict]],   # one list per fold
    all_poses_by_fold:     list[list[np.ndarray]],
    k_list:     tuple[int, ...] = (1, 2, 3, 5),
    n_random:   int = 20,
    methods:    list[str] | None = None,
) -> dict[str, Any]:
    if methods is None:
        methods = ["actor", "zero", "marginal"]

    rng = np.random.default_rng(42)

    # ── Flatten: tag each row with its fold and true_class ───────────────────
    all_rows_by_method: dict[str, list[dict]] = {m: [] for m in methods}

    for fold_idx, (actor_fold, base_fold) in enumerate(
            zip(actor_rows_by_fold, baseline_rows_by_fold)):
        for row in actor_fold:
            row["_fold"] = fold_idx
            all_rows_by_method["actor"].append(row)
        for row in base_fold:
            row["_fold"] = fold_idx
            for m in ("zero", "mean", "marginal"):
                if m in methods:
                    all_rows_by_method[m].append(row)

    # Group rows by true_class
    def _by_class(rows):
        groups: dict[int, list[dict]] = {}
        for r in rows:
            groups.setdefault(r["true_class"], []).append(r)
        return groups

    # ── Compute global SHAP importance rank per method (all sequences) ───────
    def _global_rank(rows, method):
        mats = []
        for r in rows:
            sv = r["shap_values"].get(method, {})
            mats.append([abs(sv.get(jn, 0.0)) for jn in H36M_JOINT_NAMES])
        mag = np.array(mats).mean(axis=0)
        return np.argsort(mag)[::-1].tolist()

    # ── Identify informative joints (for C2) ─────────────────────────────────
    # Use all poses across folds
    all_poses_flat = []
    all_rows_flat  = []
    for fold_idx, (actor_fold, poses_fold) in enumerate(
            zip(actor_rows_by_fold, all_poses_by_fold)):
        for r in actor_fold:
            all_poses_flat.append(poses_fold[r["seq_idx"]])
            all_rows_flat.append(r)

    if all_rows_flat:
        X_all = np.stack([_joint_features(p) for p in all_poses_flat])
        feat_std = X_all.std(axis=0)
        informative_joints = [
            j for j in range(J)
            if feat_std[j * FEAT_PER_JOINT:(j + 1) * FEAT_PER_JOINT].max() > 1e-5
        ]
        excluded = [H36M_JOINT_NAMES[j] for j in range(J)
                    if j not in informative_joints]
        print(f"  Zero-variance joints excluded from C2: {excluded}", flush=True)
    else:
        informative_joints = list(range(J))
        excluded = []

    # ── Per-method, per-class statistics ─────────────────────────────────────
    results: dict[str, Any] = {
        "k_list": list(k_list),
        "methods": methods,
        "excluded_joints": excluded,
        "by_method": {},
    }

    for method in methods:
        rows_all = all_rows_by_method[method]
        if not rows_all:
            continue

        global_rank   = _global_rank(rows_all, method)
        global_rank_inf = [j for j in global_rank if j in informative_joints]

        by_class_rows = _by_class(rows_all)
        classes_found = sorted(by_class_rows.keys())

        method_result: dict[str, Any] = {
            "global_joint_ranking": [H36M_JOINT_NAMES[j] for j in global_rank],
            "n_total": len(rows_all),
            "classes": {},
        }

        for cls in classes_found:
            cls_rows  = by_class_rows[cls]
            cls_label = _UPDRS_LABEL.get(cls, str(cls))

            # B: within-class rank consistency
            rc = _rank_consistency_for_rows(cls_rows, method)

            # mean SHAP profile
            profile = _mean_shap_profile(cls_rows, method)  # (J,)

            # C1: sufficiency (actor method uses actor SHAP; baselines use their own)
            suff = _shap_sufficiency_for_rows(
                cls_rows, method, k_list, n_random, rng, global_rank
            )

            # C2: kinematic probe (only for actor, only if poses available)
            probe: dict | None = None
            if method == "actor" and all_poses_by_fold:
                # find poses for this fold's actor rows
                probe_poses: list[np.ndarray] = []
                probe_rows: list[dict] = []
                for r in cls_rows:
                    fi = r["_fold"]
                    if fi < len(all_poses_by_fold) and r["seq_idx"] < len(all_poses_by_fold[fi]):
                        probe_poses.append(all_poses_by_fold[fi][r["seq_idx"]])
                        probe_rows.append(r)
                if len(probe_rows) >= 4:
                    feats = np.stack([_joint_features(p) for p in probe_poses])
                    p_arr = np.array([r["p_full"] for r in probe_rows], dtype=np.float32)
                    probe = {}
                    def _sel(joints):
                        cols = np.concatenate([
                            np.arange(j * FEAT_PER_JOINT, (j + 1) * FEAT_PER_JOINT)
                            for j in joints
                        ])
                        return feats[:, cols]
                    for k in k_list:
                        k = min(k, len(informative_joints))
                        top_j = global_rank_inf[:k]
                        bot_j = global_rank_inf[-k:]
                        r2_rands = []
                        for _ in range(n_random):
                            rj = rng.choice(informative_joints, size=k, replace=False).tolist()
                            v  = _ridge_loo_r2(_sel(rj), p_arr)
                            if not np.isnan(v):
                                r2_rands.append(v)
                        probe[k] = {
                            "top_k":    _ridge_loo_r2(_sel(top_j), p_arr),
                            "bottom_k": _ridge_loo_r2(_sel(bot_j), p_arr),
                            "random_k": float(np.mean(r2_rands)) if r2_rands else float("nan"),
                        }

            method_result["classes"][cls] = {
                "label":             cls_label,
                "n":                 len(cls_rows),
                "rank_consistency":  rc,
                "mean_shap_profile": {H36M_JOINT_NAMES[j]: float(profile[j])
                                       for j in range(J)},
                "shap_sufficiency":  suff,
                "kinematic_probe":   probe,
            }

            # Print summary
            suf_k1 = suff.get(min(k_list), {})
            probe_k1 = (probe or {}).get(min(k_list), {})
            print(
                f"  [{method:>8}] class {cls} (n={len(cls_rows):>3}): "
                f"rank_ρ={rc['mean']:.3f}  "
                f"suf_top1={suf_k1.get('top_k', float('nan')):+.3f}/"
                f"rand={suf_k1.get('random_k', float('nan')):+.3f}"
                + (f"  probe_top1={probe_k1.get('top_k', float('nan')):.3f}/"
                   f"rand={probe_k1.get('random_k', float('nan')):.3f}"
                   if probe else ""),
                flush=True,
            )

        # Between-class profile distance (L1 of mean SHAP profiles)
        if len(classes_found) >= 2:
            profiles = {cls: _mean_shap_profile(by_class_rows[cls], method)
                        for cls in classes_found}
            dists = {}
            for i, ci in enumerate(classes_found):
                for cj in classes_found[i + 1:]:
                    d = float(np.abs(profiles[ci] - profiles[cj]).mean())
                    dists[f"{ci}_vs_{cj}"] = d
            method_result["between_class_profile_distance"] = dists
            print(
                f"  [{method:>8}] between-class profile distance: "
                + "  ".join(f"{k}: {v:.4f}" for k, v in dists.items()),
                flush=True,
            )

        results["by_method"][method] = method_result

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _format_report(results: dict) -> str:
    methods = results["methods"]
    k_list  = results["k_list"]
    k1      = k_list[0] if k_list else 1

    lines = [
        "=" * 78,
        "ACTORSHAP vs BASELINE — UPDRS GROUP CONSISTENCY EVALUATION",
        "=" * 78,
        "",
    ]

    # ── B: Within-group rank consistency ─────────────────────────────────────
    lines += [
        "OPTION B — Within-group rank consistency (pairwise Spearman ρ)",
        "-" * 78,
        f"{'Method':<12}  " +
        "  ".join(f"UPDRS {cls:>1} (mean ρ)" for cls in [0, 1, 2]),
        "-" * 78,
    ]
    for m in methods:
        mr = results["by_method"].get(m, {})
        cells = []
        for cls in [0, 1, 2]:
            cr = mr.get("classes", {}).get(cls, {}).get("rank_consistency", {})
            v  = cr.get("mean", float("nan"))
            n  = cr.get("n", 0)
            cells.append(f"{v:+.3f} (n={n:>3})" if not np.isnan(v) else "     —      ")
        lines.append(f"{m:<12}  " + "  ".join(cells))

    lines += [
        "",
        "Interpretation: high within-group ρ = explanations agree across",
        "  sequences of the same UPDRS class (stable, class-specific signal).",
        "",
    ]

    # Between-class profile distance
    lines += [
        "Between-class profile distance (mean |Δ SHAP|): higher = more discriminative",
        "-" * 78,
        f"{'Method':<12}  {'0 vs 1':>10}  {'0 vs 2':>10}  {'1 vs 2':>10}",
        "-" * 78,
    ]
    for m in methods:
        mr    = results["by_method"].get(m, {})
        dists = mr.get("between_class_profile_distance", {})
        d01   = dists.get("0_vs_1", float("nan"))
        d02   = dists.get("0_vs_2", float("nan"))
        d12   = dists.get("1_vs_2", float("nan"))
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "  —   "
        lines.append(f"{m:<12}  {_fmt(d01):>10}  {_fmt(d02):>10}  {_fmt(d12):>10}")

    lines += [
        "",
        "Interpretation: higher between-class distance = explanations are more",
        "  discriminative across UPDRS severity levels.",
        "",
    ]

    # ── C1: Sufficiency ───────────────────────────────────────────────────────
    lines += [
        f"OPTION C1 — SHAP partial-sum sufficiency at k={k1} (top-k R² vs random R²)",
        "-" * 78,
        f"{'Method':<12}  " +
        "  ".join(f"UPDRS {cls} (top/rand)" for cls in [0, 1, 2]),
        "-" * 78,
    ]
    for m in methods:
        mr = results["by_method"].get(m, {})
        cells = []
        for cls in [0, 1, 2]:
            suf = mr.get("classes", {}).get(cls, {}).get("shap_sufficiency", {}).get(k1, {})
            t   = suf.get("top_k",    float("nan"))
            r   = suf.get("random_k", float("nan"))
            if np.isnan(t):
                cells.append("         —        ")
            else:
                flag = "✓" if t > r else "✗"
                cells.append(f"{t:+.3f}/{r:+.3f} {flag}")
        lines.append(f"{m:<12}  " + "  ".join(cells))

    lines += [
        "",
        "Interpretation: top-k R² > random R² → attribution is concentrated.",
        "",
    ]

    # ── C2: Kinematic probe (actor only) ──────────────────────────────────────
    if "actor" in methods:
        lines += [
            f"OPTION C2 — Kinematic probe at k={k1} (ActorSHAP only, excluded: "
            f"{results.get('excluded_joints', [])})",
            "-" * 78,
            f"{'UPDRS class':>12}  {'Top-k R²':>10}  {'Random R²':>10}  {'Bot-k R²':>10}  {'Top>Rand':>9}",
            "-" * 78,
        ]
        mr = results["by_method"].get("actor", {})
        for cls in [0, 1, 2]:
            pr = mr.get("classes", {}).get(cls, {}).get("kinematic_probe")
            if pr is None:
                lines.append(f"{_UPDRS_LABEL.get(cls, str(cls)):>12}  (no ActorSHAP data)")
                continue
            bk = pr.get(k1, {})
            r2t = bk.get("top_k",    float("nan"))
            r2r = bk.get("random_k", float("nan"))
            r2b = bk.get("bottom_k", float("nan"))
            lines.append(
                f"{_UPDRS_LABEL.get(cls, str(cls)):>12}  "
                f"{r2t:>10.3f}  {r2r:>10.3f}  {r2b:>10.3f}  "
                f"{'✓' if r2t > r2r else '✗':>9}"
            )

    lines += ["", "=" * 78]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-method SHAP consistency comparison by UPDRS group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--actor_shap_dirs", nargs="*", default=[],
        help="One directory per fold containing shards/ from evaluate_shap.py.",
    )
    parser.add_argument(
        "--baseline_dirs", nargs="+", required=True,
        help="One directory per fold containing per_sequence.jsonl from "
             "evaluate_shap_baselines.py (must parallel --eval_pkls order).",
    )
    parser.add_argument(
        "--eval_pkls", nargs="+", required=True,
        help="Preprocessed fold PKL (pose/label/pad_mask) — one per fold, "
             "in the same order as --baseline_dirs.",
    )
    parser.add_argument("--output_dir", default="results/shap_consistency")
    parser.add_argument("--methods", nargs="+", default=["actor", "zero", "marginal"])
    parser.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--n_random_seeds", type=int, default=20)
    args = parser.parse_args()

    if len(args.baseline_dirs) != len(args.eval_pkls):
        raise SystemExit("--baseline_dirs and --eval_pkls must have the same length.")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/3] Loading SHAP results …", flush=True)
    # Pair actor dirs with baseline dirs by fold order
    # If fewer actor dirs supplied, pad with empty lists
    n_folds = len(args.baseline_dirs)
    actor_rows_by_fold = []
    for i in range(n_folds):
        if i < len(args.actor_shap_dirs):
            actor_rows_by_fold.append(_load_actor_rows(args.actor_shap_dirs[i]))
        else:
            actor_rows_by_fold.append([])

    baseline_rows_by_fold = [_load_baseline_rows(d) for d in args.baseline_dirs]

    print("\n[2/3] Loading pose sequences …", flush=True)
    all_poses_by_fold = [_load_pkl(p) for p in args.eval_pkls]
    for i, (poses, pk) in enumerate(zip(all_poses_by_fold, args.eval_pkls)):
        print(f"  fold {i}: {len(poses)} seqs from {os.path.basename(pk)}", flush=True)

    print("\n[3/3] Running analysis …", flush=True)
    # Filter methods to what's actually available
    available_methods = list(args.methods)
    if "actor" in available_methods and not any(actor_rows_by_fold):
        available_methods.remove("actor")
        print("  (actor method skipped: no ActorSHAP data supplied)", flush=True)

    results = run_analysis(
        actor_rows_by_fold   = actor_rows_by_fold,
        baseline_rows_by_fold = baseline_rows_by_fold,
        all_poses_by_fold    = all_poses_by_fold,
        k_list   = tuple(args.k_list),
        n_random = args.n_random_seeds,
        methods  = available_methods,
    )

    out_json   = os.path.join(args.output_dir, "consistency_results.json")
    out_report = os.path.join(args.output_dir, "report.txt")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    report = _format_report(results)
    print("\n" + report, flush=True)
    with open(out_report, "w") as f:
        f.write(report + "\n")

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_report}")


if __name__ == "__main__":
    main()
