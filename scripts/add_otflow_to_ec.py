"""add_otflow_to_ec.py — add an ``otflow_shap`` row to an existing EC1/EC2/EC3
benchmark produced by :mod:`scripts.evaluate_shap_synthetic_gaussian`.

OTFlow-SHAP attributions (``psi`` of shape ``(N, T, J, F)``) are aggregated to
per-player ``phi_IG`` and compared to the Gaussian-oracle ``phi_true``.  Because
IG does not define a coalition value ``v(S)``, we use the additive
reconstruction ``v_IG(S) = f(x0_hat) + sum_{i in S} phi_IG_i``.  By the OTFlow
completeness identity this satisfies ``v_IG(full) = f(x*)`` up to solver error,
and ``v_IG(empty) = f(x0_hat)``.  Note that ``v_IG(empty)`` differs from the
oracle's empty-reference (the marginal expected prediction), so EC2/EC3 carry
an inherent floor for IG — we report all three metrics and flag this.

Usage
-----

    python scripts/add_otflow_to_ec.py \\
        --ec_dir   experiment_outs/flow_matching_synthetic/ec_ec_synthetic_gaussian_k4 \\
        --psi_path experiment_outs/flow_matching_synthetic/gaussian_gpu_xl/ig_ec_synthetic_gaussian_k4_c1/psi.npz \\
        --bench    experiment_outs/actor_shap_synthetic/ec_synthetic_gaussian_k4/synthetic_benchmark.pkl

Inputs
------
* ``ec_summary.json``  and ``per_sequence.json`` produced by the EC evaluator.
* ``psi.npz`` produced by :mod:`scripts.compute_flow_shap_synthetic`.
* ``synthetic_benchmark.pkl`` (for ``player_mode``, ``window_assignments``,
  ``J``, ``K``).

Outputs
-------
* Writes ``ec_summary_with_otflow.json`` and ``per_sequence_with_otflow.json``
  alongside the originals.
* Prints the updated EC table (markdown + LaTeX).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_psi_temporal(psi: np.ndarray, window_assignments: list[list[int]]) -> np.ndarray:
    """psi: (N, T, J, F) -> phi: (N, K) by summing per-window frames, all (J, F)."""
    N = psi.shape[0]
    K = len(window_assignments)
    phi = np.zeros((N, K), dtype=np.float64)
    for k, frames in enumerate(window_assignments):
        phi[:, k] = psi[:, frames, :, :].sum(axis=(1, 2, 3))
    return phi


def aggregate_psi_spatial(psi: np.ndarray) -> np.ndarray:
    """psi: (N, T, J, F) -> phi: (N, J) by summing over T and F."""
    return psi.sum(axis=(1, 3)).astype(np.float64)  # (N, J)


# ---------------------------------------------------------------------------
# v_IG(S) by additive reconstruction
# ---------------------------------------------------------------------------

def v_ig_from_phi(coalitions: np.ndarray, phi: np.ndarray, f_x0: float) -> np.ndarray:
    """coalitions: (M_coal, M), phi: (M,), f_x0 scalar  ->  v_IG: (M_coal,).

    v_IG(S) = f_x0 + sum_{i in S} phi_i.
    """
    return f_x0 + coalitions.astype(np.float64) @ phi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _empty_idx(coalitions: np.ndarray) -> int:
    return int((coalitions.sum(axis=1) == 0).nonzero()[0][0])


def _full_idx(coalitions: np.ndarray) -> int:
    return int((coalitions.sum(axis=1) == coalitions.shape[1]).nonzero()[0][0])


def compute_ec_row(
    per_seq: list[dict[str, Any]],
    truth_key: str = "gaussian_oracle",
) -> dict[str, dict[str, Any]]:
    """Recompute EC1/EC2/EC3 (mean + std) for every method from per_seq."""
    methods_in_data: set[str] = set()
    for seq in per_seq:
        methods_in_data.update(k for k in seq if not k.startswith("_"))
    methods = sorted(m for m in methods_in_data if m != truth_key)
    rows: dict[str, dict[str, list[float]]] = {
        m: {"ec1": [], "ec2": [], "ec3": []} for m in list(methods) + [truth_key]
    }

    for seq in per_seq:
        if truth_key not in seq:
            continue
        coalitions = np.asarray(seq["_coalitions"], dtype=np.int64)
        phi_true = np.asarray(seq[truth_key]["phi"], dtype=np.float64)
        v_true   = np.asarray(seq[truth_key]["v"], dtype=np.float64)
        nontriv = np.ones(len(v_true), dtype=bool)
        nontriv[_empty_idx(coalitions)] = False
        nontriv[_full_idx(coalitions)] = False
        f_x = float(v_true[_full_idx(coalitions)])
        for m in rows:
            if m not in seq:
                continue
            phi_m = np.asarray(seq[m]["phi"], dtype=np.float64)
            v_m   = np.asarray(seq[m]["v"],   dtype=np.float64)
            rows[m]["ec1"].append(float(np.abs(phi_m - phi_true).mean()))
            rows[m]["ec2"].append(float(((v_m[nontriv] - v_true[nontriv]) ** 2).mean()))
            rows[m]["ec3"].append(float(((f_x - v_m[nontriv]) ** 2).mean()))

    summary: dict[str, dict[str, Any]] = {}
    for m, d in rows.items():
        if not d["ec1"]:
            continue
        summary[m] = {
            "EC1_mean": float(np.mean(d["ec1"])),
            "EC1_std":  float(np.std(d["ec1"])),
            "EC2_mean": float(np.mean(d["ec2"])),
            "EC2_std":  float(np.std(d["ec2"])),
            "EC3_mean": float(np.mean(d["ec3"])),
            "EC3_std":  float(np.std(d["ec3"])),
            "n":        int(len(d["ec1"])),
        }
    return summary


def print_table(summary: dict[str, dict[str, Any]], header: str) -> None:
    print(f"\n{'=' * 72}")
    print(header)
    print(f"{'=' * 72}")
    print(f"{'Method':<22s} {'EC1':>10s} {'EC2':>10s} {'EC3':>10s}  {'n':>4s}")
    print("-" * 62)
    ordered = sorted(summary.keys(), key=lambda k: summary[k]["EC1_mean"])
    for m in ordered:
        s = summary[m]
        print(f"{m:<22s} {s['EC1_mean']:>10.4f} {s['EC2_mean']:>10.4f} "
              f"{s['EC3_mean']:>10.4f}  {s['n']:>4d}")
    print("\n% LaTeX snippet:")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print("Method & EC1 & EC2 & EC3 \\\\")
    print("\\midrule")
    for m in ordered:
        s = summary[m]
        print(
            f"{m} & {s['EC1_mean']:.4f} $\\pm$ {s['EC1_std']:.4f}"
            f" & {s['EC2_mean']:.4f} $\\pm$ {s['EC2_std']:.4f}"
            f" & {s['EC3_mean']:.4f} $\\pm$ {s['EC3_std']:.4f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ec_dir", required=True,
                   help="Directory holding ec_summary.json + per_sequence.json.")
    p.add_argument("--psi_path", required=True,
                   help="Path to psi.npz from scripts/compute_flow_shap_synthetic.py.")
    p.add_argument("--bench", required=True,
                   help="synthetic_benchmark.pkl for window_assignments / player_mode.")
    p.add_argument("--method_name", default="otflow_shap",
                   help="Row label to insert.")
    args = p.parse_args()

    ec_dir = Path(args.ec_dir)
    summary_path = ec_dir / "ec_summary.json"
    per_seq_path = ec_dir / "per_sequence.json"
    with open(summary_path) as f:
        summary_full = json.load(f)
    with open(per_seq_path) as f:
        per_seq = json.load(f)

    cfg = summary_full.get("_config", {})
    print(f"[add_otflow] ec_dir={ec_dir}")
    print(f"[add_otflow] existing methods: {sorted(summary_full['methods'].keys())}")
    print(f"[add_otflow] EC class_idx={cfg.get('class_idx')}  "
          f"player_mode={cfg.get('player_mode')}  K={cfg.get('K')}  "
          f"n_sequences={cfg.get('n_test_sequences')}")

    # Reconstruct coalitions from config (per_sequence.json doesn't store them
    # since they are identical across sequences).
    from synthetic.gaussian_motion import (
        _enumerate_temporal_coalitions,
        _sample_kernelshap_coalitions,
    )
    pm_cfg = cfg.get("player_mode", "temporal")
    if pm_cfg == "temporal":
        coalitions, _ = _enumerate_temporal_coalitions(int(cfg["K"]))
    else:
        rng_c = np.random.default_rng(int(cfg.get("seed", 0)) + 1)
        J_cfg = int(cfg["J"])
        coalitions_mid, _ = _sample_kernelshap_coalitions(
            J_cfg, int(cfg.get("n_kernel_samples", 250)), rng_c,
        )
        empty_row = np.zeros((1, J_cfg), dtype=int)
        full_row  = np.ones((1, J_cfg),  dtype=int)
        coalitions = np.vstack([empty_row, coalitions_mid, full_row])
    # Sanity: must match any existing v in per_sequence.json.
    ref_v = per_seq[0]["gaussian_oracle"]["v"]
    if len(ref_v) != len(coalitions):
        raise RuntimeError(
            f"reconstructed coalitions (N={len(coalitions)}) do not match "
            f"len(v)={len(ref_v)} in per_sequence.json"
        )
    # Attach to each row for compute_ec_row.
    for seq in per_seq:
        seq["_coalitions"] = coalitions.tolist()

    # --- load OTFlow psi ------------------------------------------------------
    d = np.load(args.psi_path)
    psi = d["psi"]                              # (N, T, J, F)
    f_xstar = d["f_xstar"].astype(np.float64)   # (N,)
    f_x0    = d["f_x0"].astype(np.float64)      # (N,)
    psi_class = int(d["class_idx"][0])
    print(f"[add_otflow] psi: shape={psi.shape}  class_idx={psi_class}  "
          f"completeness_rel_mean={float(d['completeness_rel'].mean()):.4f}")
    if cfg.get("class_idx") is not None and psi_class != cfg["class_idx"]:
        raise RuntimeError(
            f"class_idx mismatch: EC used {cfg['class_idx']} but psi used {psi_class}"
        )

    # --- load benchmark ------------------------------------------------------
    with open(args.bench, "rb") as fb:
        bench = pickle.load(fb)
    pm = getattr(bench, "player_mode", "temporal")
    if pm == "temporal":
        window_assignments = [list(w) for w in bench.window_assignments]
        phi_ig = aggregate_psi_temporal(psi, window_assignments)   # (N, K)
    elif pm == "spatial":
        phi_ig = aggregate_psi_spatial(psi)                         # (N, J)
    else:
        raise RuntimeError(f"Unknown player_mode {pm!r}")
    N_psi, M = phi_ig.shape
    N_seq = len(per_seq)
    if N_psi < N_seq:
        raise RuntimeError(f"psi has only {N_psi} sequences but per_sequence.json has {N_seq}")
    phi_ig = phi_ig[:N_seq]
    f_x0   = f_x0[:N_seq]
    f_xstar = f_xstar[:N_seq]

    # --- inject otflow_shap row into per_seq ---------------------------------
    for i, seq in enumerate(per_seq):
        coalitions = np.asarray(seq["_coalitions"], dtype=np.int64)
        if coalitions.shape[1] != M:
            raise RuntimeError(
                f"coalition width {coalitions.shape[1]} != phi_ig width {M} at seq {i}"
            )
        v_ig = v_ig_from_phi(coalitions, phi_ig[i], f_x0[i])
        seq[args.method_name] = {
            "phi": phi_ig[i].tolist(),
            "v":   v_ig.tolist(),
        }

    # --- recompute EC metrics ------------------------------------------------
    new_summary = compute_ec_row(per_seq, truth_key="gaussian_oracle")

    # --- persist --------------------------------------------------------------
    summary_full["methods"] = new_summary
    summary_full["_config"]["otflow_psi_path"] = str(args.psi_path)
    out_summary = ec_dir / "ec_summary_with_otflow.json"
    out_per_seq = ec_dir / "per_sequence_with_otflow.json"
    with open(out_summary, "w") as f:
        json.dump(summary_full, f, indent=2)
    with open(out_per_seq, "w") as f:
        json.dump(per_seq, f)
    print(f"[add_otflow] wrote {out_summary}")
    print(f"[add_otflow] wrote {out_per_seq}")

    header = (
        f"EC benchmark (with OTFlow-SHAP)  player_mode={pm}  "
        f"K={cfg.get('K')}  J={getattr(bench, 'J', '?')}  "
        f"class={cfg.get('class_idx')}  n_seqs={len(per_seq)}"
    )
    print_table(new_summary, header=header)


if __name__ == "__main__":
    main()
