# Synthetic SHAP Benchmarks

Controlled-ground-truth benchmarks used to evaluate SHAP-imputation and
Aumann–Shapley methods against **analytically-computable Shapley values**.
If you want to add a new benchmark, read
[`docs/CONTRIBUTING-DATASETS.md`](../docs/CONTRIBUTING-DATASETS.md); it walks
through the interface end-to-end.

---

## What lives here

```
synthetic/
├── __init__.py
├── gaussian_motion.py   ← the benchmark currently wired into the pipeline
└── README.md            ← you are here
```

### `gaussian_motion.py`

Temporal motion drawn from

$$ x \sim \mathcal{N}(0,\; \Sigma_{\text{joints}} \otimes I_F \otimes \Sigma_{\text{time}}) $$

with `Σ_joints[j,j'] = ρ` (equicorrelation) and `Σ_time[t,t'] = αⁱᵗ⁻ᵗ'ⁱ` (AR(1)).
Because the full distribution is Gaussian with Kronecker structure, the exact
conditional `p(x_hidden | x_observed)` is Gaussian with closed-form mean and
covariance — which is what `GaussianMotionBenchmark.conditional_sample(...)`
and `conditional_sample_spatial(...)` return. That in turn makes the oracle
Shapley values computable:

- **`player_mode="temporal"`**: K temporal windows are the players
  (`K ∈ {4, 8, 12}`, must be divisible by 4 so the Olsen-style label tiles
  cleanly). Oracle Shapley is **exact by 2^K coalition enumeration**.
- **`player_mode="spatial"`**: J = 17 joints are the players. Oracle Shapley
  is estimated by **KernelSHAP using the exact joint-conditional Gaussian**
  as the imputer.

The black-box classifier is `SyntheticMLPClassifier`: a small MLP on top of
per-window (or per-joint) grand means, with BatchNorm + ReLU non-linearities
so that classical Shapley and Aumann–Shapley are guaranteed to **diverge**
(the label interactions cannot be reproduced by an additive decomposition).

Label function — adapted from Olsen et al. (JMLR 2022) Eq. (12):

$$ s(u) = c_1 \sin(\pi u_0 u_1) + c_2 u_2 \exp(c_3 u_2 u_3) $$

tiled across groups of 4 windows for `K > 4`.

---

## Pipeline (current)

The synthetic benchmark is consumed by five scripts that together reproduce
the EC1 / EC2 / EC3 tables:

| Stage                       | Script                                                        | Output                                                             |
|-----------------------------|---------------------------------------------------------------|--------------------------------------------------------------------|
| 1. Build data + classifier  | `scripts/build_synthetic_gaussian_data.py`                    | `synthetic_benchmark.pkl`, `synthetic_clf.pt`, `synthetic_test.pt` |
| 2. Flow-training cache      | `scripts/generate_velocity_synthetic.py`                      | `cache.npz`, `sanity.npz`                                          |
| 3a. Train flow model        | `train_flow_matching.py --config configs/flow_matching/...`   | `last.ckpt` (VelocityNet)                                          |
| 3b. Train VAEAC baseline    | `train_vaeac.py --config configs/vaeac/synthetic_gaussian.json` | `last.ckpt` (Ivanov/Olsen-faithful VAEAC)                        |
| 4. OTFlow-SHAP attributions | `scripts/compute_flow_shap_synthetic.py`                      | `psi.npz` (per-element IG)                                         |
| 5. EC1 / EC2 / EC3 sweep    | `scripts/evaluate_shap_synthetic_gaussian.py`                 | `ec_summary.json`, `per_sequence.json`                             |

**One-shot orchestration** (all five stages):

```bash
GPU=0 TAG=synthetic_gaussian_k4 K=4 PLAYERS=temporal \
    bash scripts/run_flow_matching_synthetic.sh
```

Key environment-variable knobs (see the script header for the full list):

| Variable            | Default                                                                   | Meaning                                          |
|---------------------|---------------------------------------------------------------------------|--------------------------------------------------|
| `K`                 | `4`                                                                       | Number of temporal windows (multiple of 4)       |
| `PLAYERS`           | `temporal`                                                                | `temporal` (K windows) or `spatial` (J joints)   |
| `SIGNAL_JOINTS`     | `"0 1 2 3"`                                                               | Signal-carrying joints (spatial mode only)       |
| `TRAIN_VAEAC`       | `0`                                                                       | Set to `1` to also train the VAEAC baseline      |
| `FLOW_CKPT_DIR`     | `experiment_outs/flow_matching_synthetic/synthetic_gaussian`              | Reuse an existing flow checkpoint                |
| `VAEAC_CKPT_DIR`    | `experiment_outs/vaeac_synthetic/synthetic_gaussian`                      | Reuse an existing VAEAC checkpoint               |
| `N_TEST`            | `100`                                                                     | Number of test sequences for the EC sweep        |
| `K_MC_TRUE`         | `1000`                                                                    | Monte-Carlo samples for oracle `v(S)`            |

---

## What gets compared

The EC sweep runs **all of the following SHAP methods against the same oracle**:

| Method            | Imputer                                            | Notes                          |
|-------------------|----------------------------------------------------|--------------------------------|
| `zero`            | Replace hidden entries with 0                      | Off-manifold baseline          |
| `mean`            | Replace with per-joint training mean               | Off-manifold baseline          |
| `marginal`        | Replace with random training clip                  | On-marginal baseline           |
| `gaussian_oracle` | Exact Gaussian conditional (analytic)              | Upper bound                    |
| `flow_matching`   | Trained flow model, RePaint-style sampling         | Our main method                |
| `vaeac`           | Trained VAEAC, amortised posterior                 | JMLR 2022 baseline (Olsen-faithful) |

Ranking-based metrics (Spearman ρ, Top-1, Insertion AUC, Deletion AUC) for
the same methods plus OTFlow-SHAP are produced by
`scripts/compute_ranking_metrics.py`, which consumes the
`per_sequence.json` emitted by stage 5.

---

## Metrics

| Metric  | Formula                          | Direction |
|---------|----------------------------------|-----------|
| **EC1** | `mean_i \|φ_true − φ_method\|`   | lower     |
| **EC2** | `mean_S (v_true(S) − v̂(S))²`    | lower     |
| **EC3** | `mean_S (f(x*) − v̂(S))²`        | lower     |

For the spatial mode the coalition space is sampled (paired KernelSHAP),
so EC2/EC3 are estimated over those sampled coalitions.

---

## Adding a new benchmark

See [`docs/CONTRIBUTING-DATASETS.md`](../docs/CONTRIBUTING-DATASETS.md).
