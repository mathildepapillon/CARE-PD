"""build_synthetic_burr_data.py — Burr tabular benchmark data generator.

Produces the files consumed downstream by

* ``scripts/generate_velocity_synthetic.py`` — builds the flow-matching cache
  from ``x_train_jft.npy`` and ``synthetic_test.pt``.
* ``scripts/compute_flow_shap_synthetic.py`` — loads ``synthetic_clf.pkl`` and
  ``synthetic_test.pt`` to explain the Burr RF regressor with OTFlow-SHAP.
* ``scripts/evaluate_shap_synthetic_gaussian.py`` — loads benchmark + classifier
  to compute true Shapley values and evaluation metrics.

The benchmark is defined in :mod:`synthetic.burr_tabular`: M scalar Burr
features embedded into (J=17, F=3, T=M) tensors. The black-box model is a
RandomForestRegressor with 500 trees.  This exactly mirrors Olsen et al.
(JMLR 2022) Section 4.2 Simulation Study: Continuous Data.

Outputs
-------

* ``<out_dir>/x_train_jft.npy``           (N_train, J, F, T=M)
* ``<out_dir>/synthetic_test.pt``          dict(x, y, pad_mask)
* ``<out_dir>/synthetic_clf.pkl``          BurrRFWrapper (pickle)
* ``<out_dir>/synthetic_clf_meta.json``    {type, task, J, F, M, ...}
* ``<out_dir>/synthetic_benchmark.pkl``    BurrTabularBenchmark (for true Shapley)
* ``<out_dir>/config.json``                {data_mode, J, F, M, kappa, seed, ...}

Usage
-----

    python scripts/build_synthetic_burr_data.py \\
        --out_dir experiment_outs/burr_synthetic/burr_m10 \\
        --M 10 --kappa 2.0 \\
        --n_train 1000 --n_val 200 --n_test 100 \\
        --n_estimators 500 --seed 0

    # Low-dim (paper: N_train ∈ {100, 1000, 5000}, M ∈ {5, 10})
    python scripts/build_synthetic_burr_data.py \\
        --out_dir experiment_outs/burr_synthetic/burr_m5_n100 \\
        --M 5 --n_train 100 --n_test 100 --seed 0

    # High-dim (paper: M ∈ {25, 50, 100, 250})
    python scripts/build_synthetic_burr_data.py \\
        --out_dir experiment_outs/burr_synthetic/burr_m25 \\
        --M 25 --n_train 5000 --n_test 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic.burr_tabular import build_burr_benchmark_and_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True,
                   help="Directory to write data, classifier and config to.")
    p.add_argument("--M", type=int, default=10,
                   help="Number of Burr features = SHAP players. Must be divisible by 5.")
    p.add_argument("--kappa", type=float, default=2.0,
                   help="Burr shape parameter κ controlling inter-feature dependence.")
    p.add_argument("--J", type=int, default=17,
                   help="Number of joints in the embedded tensor (default: 17).")
    p.add_argument("--F", type=int, default=3,
                   help="Number of features per joint (default: 3).")
    p.add_argument("--n_train", type=int, default=1000,
                   help="Training samples. Paper experiments: 100, 1000, or 5000.")
    p.add_argument("--n_val",   type=int, default=200)
    p.add_argument("--n_test",  type=int, default=100,
                   help="Test samples. Paper default: 100.")
    p.add_argument("--n_estimators", type=int, default=500,
                   help="Number of trees in RandomForestRegressor (paper: 500).")
    p.add_argument("--noise_std", type=float, default=1.0,
                   help="Scale for heteroscedastic label noise (0 = no noise).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    print(f"[build_synthetic_burr] M={args.M}  κ={args.kappa}  "
          f"n_train={args.n_train}  n_test={args.n_test}  seed={args.seed}")
    print("  Sampling data and training RandomForestRegressor …")

    bench, clf, train_ds, val_ds, test_ds = build_burr_benchmark_and_classifier(
        M=args.M, kappa=args.kappa,
        J=args.J, F=args.F,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n_estimators=args.n_estimators,
        seed=args.seed,
        noise_std=args.noise_std,
    )

    # ---- Evaluate RF on held-out test set -----------------------------------
    x_te_tjf, y_te_t, pm_te = test_ds.tensors
    x_te_jft = x_te_tjf.numpy().transpose(0, 2, 3, 1)   # (N_test, J, F, M)
    y_te_np   = y_te_t.numpy()
    preds_te  = clf.predict_np(x_te_jft)
    from sklearn.metrics import r2_score, mean_squared_error
    r2  = r2_score(y_te_np, preds_te)
    mse = mean_squared_error(y_te_np, preds_te)
    print(f"  RF test R²={r2:.4f}  RMSE={mse**0.5:.4f}")

    # ---- Save benchmark + classifier ----------------------------------------
    bench.save(str(out_dir / "synthetic_benchmark.pkl"))
    clf.save(str(out_dir / "synthetic_clf.pkl"))

    clf_meta = {
        "type":         "BurrRFWrapper",
        "task":         "regression",
        "J":            args.J,
        "F":            args.F,
        "T":            args.M,     # T = M (one time step per feature)
        "M":            args.M,
        "kappa":        args.kappa,
        "n_estimators": args.n_estimators,
        "num_classes":  1,
        "player_mode":  "temporal",
        "rf_r2_test":   float(r2),
    }
    with open(out_dir / "synthetic_clf_meta.json", "w") as f:
        json.dump(clf_meta, f, indent=2)

    # ---- Extract raw training sequences in (N, J, F, T) layout -------------
    x_tr_tjf = train_ds.tensors[0].numpy()                        # (N, T, J, F)
    x_tr_jft = x_tr_tjf.transpose(0, 2, 3, 1).astype(np.float32) # (N, J, F, T)
    np.save(str(out_dir / "x_train_jft.npy"), x_tr_jft)

    # ---- Save test tensors (for flow cache and flow-SHAP) -------------------
    torch.save(
        {"x": x_te_tjf, "y": y_te_t, "pad_mask": pm_te},
        str(out_dir / "synthetic_test.pt"),
    )

    # ---- Save config --------------------------------------------------------
    cfg = {
        "data_mode":    "synthetic_burr",
        "J":            args.J,
        "F":            args.F,
        "T":            args.M,     # seq_len = M for flow-matching model
        "seq_len":      args.M,
        "M":            args.M,
        "kappa":        args.kappa,
        "n_train":      args.n_train,
        "n_val":        args.n_val,
        "n_test":       args.n_test,
        "n_estimators": args.n_estimators,
        "noise_std":    args.noise_std,
        "seed":         args.seed,
        "player_mode":  "temporal",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[build_synthetic_burr] wrote to {out_dir}")
    print(f"  x_train_jft.npy           shape={x_tr_jft.shape}")
    print(f"  synthetic_test.pt         N_test={len(x_te_tjf)}")
    print(f"  synthetic_clf.pkl         RF n_estimators={args.n_estimators}")
    print(f"  synthetic_benchmark.pkl   M={args.M}  κ={args.kappa}")


if __name__ == "__main__":
    main()
