"""build_synthetic_gaussian_data.py — self-contained synthetic Gaussian benchmark generator.

Produces the files consumed downstream by

* ``scripts/generate_velocity_synthetic.py`` — builds the flow-matching cache
  from ``x_train_jft.npy`` and ``synthetic_test.pt``.
* ``scripts/compute_flow_shap_synthetic.py`` — loads ``synthetic_clf.pt`` and
  ``synthetic_test.pt`` to explain the Gaussian classifier with OTFlow-SHAP.

The benchmark is defined in :mod:`synthetic.gaussian_motion`: equicorrelated
multivariate Gaussian joints with AR(1) temporal structure and K non-overlapping
temporal windows that the classifier partitions into three UPDRS-like classes.

Unlike ``train_actor_shap_synthetic.py`` (removed in the flow-matching cleanup),
this script does **not** train any SHAP model — it only materialises the data
and the classifier so the flow-matching + flow-SHAP pipelines can run without
ActorSHAP code on the import path.

Outputs
-------

* ``<out_dir>/x_train_jft.npy``           (N_train, J, F, T)
* ``<out_dir>/synthetic_test.pt``          dict(x, y, pad_mask)
* ``<out_dir>/synthetic_clf.pt``           classifier state_dict
* ``<out_dir>/synthetic_clf_meta.json``    {J, F, T, K, num_classes}
* ``<out_dir>/synthetic_benchmark.pkl``    GaussianMotionBenchmark (for true Shapley)
* ``<out_dir>/config.json``                {data_mode, J, F, T, K, rho, alpha, seed, ...}

Usage
-----

    python scripts/build_synthetic_gaussian_data.py \\
        --out_dir experiment_outs/actor_shap_synthetic/synthetic_gaussian \\
        --rho 0.5 --alpha 0.8 \\
        --n_train 2000 --n_val 500 --n_test 100 \\
        --clf_epochs 80 --seed 0
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

from synthetic.gaussian_motion import build_gaussian_benchmark_and_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True,
                   help="Directory to write data, classifier and config to.")
    p.add_argument("--rho", type=float, default=0.5,
                   help="Inter-joint correlation (equicorrelation Sigma).")
    p.add_argument("--alpha", type=float, default=0.8,
                   help="AR(1) temporal coefficient.")
    p.add_argument("--J", type=int, default=17)
    p.add_argument("--F", type=int, default=3)
    p.add_argument("--T", type=int, default=81)
    p.add_argument("--K", type=int, default=4,
                   help="Number of temporal windows (must be a multiple of 4; "
                        "coalitions = 2^K).")
    p.add_argument("--players", choices=("temporal", "spatial"), default="temporal",
                   help="SHAP player mode: 'temporal' = K windows as players, "
                        "'spatial' = J joints as players.")
    p.add_argument("--signal_joints", type=int, nargs=4, default=None,
                   metavar=("J0", "J1", "J2", "J3"),
                   help="Only used when --players=spatial: indices of the 4 "
                        "joints that drive the label (default: 0 1 2 3).")
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val",   type=int, default=500)
    p.add_argument("--n_test",  type=int, default=100)
    p.add_argument("--clf_epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bench, clf, train_ds, val_ds, test_ds = build_gaussian_benchmark_and_classifier(
        rho=args.rho, alpha=args.alpha,
        J=args.J, F=args.F, T=args.T, K=args.K,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        clf_epochs=args.clf_epochs, seed=args.seed, device=device,
        player_mode=args.players,
        signal_joints=tuple(args.signal_joints) if args.signal_joints is not None else None,
    )

    # ---- Save benchmark + classifier ---------------------------------------
    bench.save(str(out_dir / "synthetic_benchmark.pkl"))
    torch.save(clf.state_dict(), str(out_dir / "synthetic_clf.pt"))
    clf_meta = {
        "type":        "SyntheticMLPClassifier",
        "J":           args.J,
        "F":           args.F,
        "T":           args.T,
        "K":           args.K,
        "num_classes": 3,
        "player_mode": args.players,
        "signal_joints": (
            list(bench.signal_joints) if bench.signal_joints is not None else None
        ),
    }
    with open(out_dir / "synthetic_clf_meta.json", "w") as f:
        json.dump(clf_meta, f, indent=2)

    # ---- Extract raw training sequences in (N, J, F, T) layout -------------
    x_tr_tjf = train_ds.tensors[0].numpy()                    # (N, T, J, F)
    x_tr_jft = x_tr_tjf.transpose(0, 2, 3, 1).astype(np.float32)  # (N, J, F, T)
    np.save(str(out_dir / "x_train_jft.npy"), x_tr_jft)

    # ---- Save test tensors (for both flow cache and flow-SHAP) --------------
    x_te_tjf, y_te, pm_te = test_ds.tensors
    torch.save(
        {"x": x_te_tjf, "y": y_te, "pad_mask": pm_te},
        str(out_dir / "synthetic_test.pt"),
    )

    # ---- Save config --------------------------------------------------------
    cfg = {
        "data_mode":     "synthetic_gaussian",
        "J":             args.J,
        "F":             args.F,
        "T":             args.T,
        "K":             args.K,
        "rho":           args.rho,
        "alpha":         args.alpha,
        "n_train":       args.n_train,
        "n_val":         args.n_val,
        "n_test":        args.n_test,
        "clf_epochs":    args.clf_epochs,
        "seed":          args.seed,
        "player_mode":   args.players,
        "signal_joints": clf_meta["signal_joints"],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[build_synthetic] wrote to {out_dir}")
    print(f"  x_train_jft.npy           shape={x_tr_jft.shape}")
    print(f"  synthetic_test.pt         N_test={len(x_te_tjf)}")
    print(f"  synthetic_clf.pt          num_classes=3")
    print(f"  synthetic_benchmark.pkl   J={args.J}, K={args.K}, T={args.T}")


if __name__ == "__main__":
    main()
