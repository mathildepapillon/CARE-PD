# Synthetic SHAP Benchmarks

Two synthetic benchmarks for validating that ActorSHAP (and LSTM-VAE) produce
Shapley values closer to the analytically correct ones than simpler baselines.

---

## Benchmarks

### 1. Gaussian Motion (`synthetic_gaussian`)

Temporal motion drawn from a known multivariate Gaussian distribution
(equicorrelation across joints × AR(1) across time). Because the true
conditional distribution is analytically Gaussian, the exact Shapley values can
be computed via Monte Carlo from the known conditionals.

- **K = 4 temporal windows** are the SHAP players.
- **Label function**: nonlinear with interactions across windows, adapted from
Olsen et al. (JMLR 2022) Eq. (12) — `c₁·sin(π·u₀·u₁) + c₂·u₂·exp(c₃·u₂·u₃)`.
- **Black-box**: a small MLP trained to predict the quantile-binned label from
per-window grand means.
- **Metrics**: EC1 (MAE of Shapley values), EC2 (MSE of contribution functions),
EC3 (Expected Prediction Error).

### 2. Diagnostic Gait (`synthetic_diagnostic`)

Fourier-series synthetic gait with a known subset of diagnostic joints (right
leg, spine) that carry label-relevant amplitude signal.

- **Players**: J = 17 joints (spatial SHAP via KernelSHAP).
- **Black-box**: linear classifier — true Shapley values are analytically exact.
- **Metrics**: EC1, Top-k joint recovery, Spearman rank correlation.

---

## Directory Structure

After training, both models save all artifacts to the **same directory**:

```
experiment_outs/actor_shap_synthetic/actor_shap_synthetic_synthetic_gaussian/
├── actor_shap_synthetic_last.ckpt  ← ActorSHAP model weights
├── lstm_vae_model.pt               ← LstmVAE model weights (after Step 2)
├── lstm_vae_config.json            ← LstmVAE architecture config
├── synthetic_benchmark.pkl         ← GaussianMotionBenchmark instance
├── synthetic_clf.pt                ← MLP classifier weights
├── synthetic_clf_meta.json         ← Classifier metadata (J, F, T, K)
├── synthetic_test.pt               ← Test tensors (x, y, pad_mask)
├── x_train_jft.npy                 ← (N_train, J, F, T) training sequences
└── config.json                     ← CLI arguments used for ActorSHAP training
```

---

## Quick Start: Gaussian Benchmark

### Step 1 — Train ActorSHAP

```bash
python train_actor_shap_synthetic.py \
    --data_mode synthetic_gaussian \
    --rho 0.7 --alpha 0.9 \
    --n_train 10000 --n_val 2000 --n_test 500 \
    --latent_dim 128 --num_layers 6 --num_heads 4 \
    --epochs 300 --phase0_epochs 0 \
    --lambda_kl 0.01 --lambda_rc_psi 1.0 \
    --mask_axis temporal \
    --checkpoint_dir experiment_outs/actor_shap_synthetic \
    2>&1 | tee /tmp/actor_train.log
```

Progress is printed per epoch:

```
epoch    1/300  train/loss=2.3141  val/loss=2.1892  val/mixed=2.1892
epoch    2/300  ...
```

Output directory: `experiment_outs/actor_shap_synthetic/actor_shap_synthetic_synthetic_gaussian/`

### Step 2 — Train LstmVAE on the Same Benchmark

Passing `--actor_data_dir` reuses the exact same distribution, benchmark, and
test split — ensuring a fair comparison.

```bash
ACTOR_DIR=experiment_outs/actor_shap_synthetic/actor_shap_synthetic_synthetic_gaussian

python train_lstm_vae_synthetic.py \
    --actor_data_dir "$ACTOR_DIR" \
    --epochs 300 \
    --n_mix 5 \
    --mask_warmup_epochs 20 \
    --mask_axis temporal \
    2>&1 | tee /tmp/lstm_train.log
```

Saves `lstm_vae_model.pt` and `lstm_vae_config.json` into `$ACTOR_DIR`.

### Step 3 — Evaluate Both Models

```bash
ACTOR_DIR=experiment_outs/actor_shap_synthetic/actor_shap_synthetic_synthetic_gaussian

python evaluate_shap_synthetic.py gaussian \
    --ckpt_dir "$ACTOR_DIR" \
    --lstm_vae_ckpt_dir "$ACTOR_DIR" \
    --device cuda:0 \
    --K_mc_true 2000 \
    --n_completion_samples 50 \
    --n_test_sequences 500 \
    2>&1 | tee /tmp/eval_gaussian.log
```

Prints classifier test accuracy, then an EC1/EC2/EC3 table and a LaTeX snippet.

---

## Quick Start: Diagnostic Benchmark

### Step 1 — Train ActorSHAP

```bash
python train_actor_shap_synthetic.py \
    --data_mode synthetic_diagnostic \
    --n_train 2000 --n_val 500 --n_test 200 \
    --mask_axis spatial \
    --epochs 300 --phase0_epochs 0 \
    --checkpoint_dir experiment_outs/actor_shap_diagnostic \
    2>&1 | tee /tmp/actor_diag_train.log
```

### Step 2 — Train LstmVAE

```bash
ACTOR_DIR=experiment_outs/actor_shap_diagnostic/actor_shap_synthetic_synthetic_diagnostic

python train_lstm_vae_synthetic.py \
    --actor_data_dir "$ACTOR_DIR" \
    --epochs 300 \
    --mask_axis spatial \
    2>&1 | tee /tmp/lstm_diag_train.log
```

### Step 3 — Evaluate

```bash
ACTOR_DIR=experiment_outs/actor_shap_diagnostic/actor_shap_synthetic_synthetic_diagnostic

python evaluate_shap_synthetic.py diagnostic \
    --ckpt_dir "$ACTOR_DIR" \
    --lstm_vae_ckpt_dir "$ACTOR_DIR" \
    --device cuda:0 \
    --n_completion_samples 50 \
    --n_test_sequences 200 \
    2>&1 | tee /tmp/eval_diag.log
```

---

## Key Arguments

### `train_actor_shap_synthetic.py`


| Argument           | Default                                | Description                                                                                   |
| ------------------ | -------------------------------------- | --------------------------------------------------------------------------------------------- |
| `--data_mode`      | `synthetic_gaussian`                   | `synthetic_gaussian` or `synthetic_diagnostic`                                                |
| `--rho`            | `0.5`                                  | Joint equicorrelation strength (Gaussian only)                                                |
| `--alpha`          | `0.8`                                  | AR(1) temporal correlation (Gaussian only) (making this higher should break marginal further) |
| `--n_train`        | `2000`                                 | Training sequences (use ≥10000 for paper results)                                             |
| `--n_val`          | `500`                                  | Validation sequences                                                                          |
| `--n_test`         | `100`                                  | Test sequences (use ≥500 for stable metrics)                                                  |
| `--epochs`         | `200`                                  | Training epochs                                                                               |
| `--phase0_epochs`  | `0`                                    | Warm-up epochs for masked encoder only                                                        |
| `--mask_axis`      | `temporal`                             | `temporal` (Gaussian) or `spatial` (diagnostic)                                               |
| `--lambda_kl`      | `0.01`                                 | KL divergence weight                                                                          |
| `--lambda_rc_psi`  | `1.0`                                  | Masked encoder reconstruction weight                                                          |
| `--checkpoint_dir` | `experiment_outs/actor_shap_synthetic` | Output root                                                                                   |
| `--devices`        | *(all GPUs)*                           | Override GPUs, e.g. `0,1`                                                                     |
| `--seed`           | `0`                                    | Random seed                                                                                   |


### `train_lstm_vae_synthetic.py`


| Argument               | Default    | Description                                      |
| ---------------------- | ---------- | ------------------------------------------------ |
| `--actor_data_dir`     | `None`     | **Recommended**: reuse ActorSHAP's benchmark dir |
| `--epochs`             | `200`      | Training epochs                                  |
| `--n_mix`              | `5`        | Mixture components in masked encoder             |
| `--mask_warmup_epochs` | `20`       | Epochs before masked encoder activates           |
| `--mask_axis`          | `temporal` | Must match ActorSHAP's `--mask_axis`             |
| `--latent_dim`         | `128`      | Latent space dimension                           |


### `evaluate_shap_synthetic.py gaussian`


| Argument                 | Default      | Description                                                         |
| ------------------------ | ------------ | ------------------------------------------------------------------- |
| `--ckpt_dir`             | *(required)* | ActorSHAP output directory                                          |
| `--lstm_vae_ckpt_dir`    | `None`       | LstmVAE output directory (usually same as `--ckpt_dir`)             |
| `--K_mc_true`            | `1000`       | MC samples for ground-truth Shapley values (higher = more accurate) |
| `--n_completion_samples` | `50`         | Stochastic completions per coalition                                |
| `--n_test_sequences`     | `100`        | Number of test sequences to evaluate                                |
| `--device`               | `cuda:0`     | Evaluation device                                                   |
| `--use_gaussian_full`    | `False`      | Also run expensive LedoitWolf full-covariance baseline              |


### `evaluate_shap_synthetic.py diagnostic`


| Argument                 | Default      | Description                             |
| ------------------------ | ------------ | --------------------------------------- |
| `--ckpt_dir`             | *(required)* | ActorSHAP output directory              |
| `--lstm_vae_ckpt_dir`    | `None`       | LstmVAE output directory                |
| `--n_kernel_samples`     | `200`        | KernelSHAP coalition pairs per sequence |
| `--n_completion_samples` | `50`         | Stochastic completions per coalition    |


---

## Metrics

### Gaussian benchmark


| Metric  | Formula                     | Interpretation                                      |
| ------- | --------------------------- | --------------------------------------------------- |
| **EC1** | `mean                       | φ_true − φ_method                                   |
| **EC2** | `mean (v_true(S) − v̂(S))²` | MSE of contribution functions — **lower is better** |
| **EC3** | `mean (f(x*) − v̂(S))²`     | Expected Prediction Error — **lower is better**     |


### Diagnostic benchmark


| Metric             | Interpretation                                                          |
| ------------------ | ----------------------------------------------------------------------- |
| **EC1**            | MAE of per-joint Shapley values — lower is better                       |
| **Top-k recovery** | Fraction of true top-k joints in estimated top-k — **higher is better** |
| **Spearman ρ**     | Rank correlation of Shapley values — **higher is better**               |


---

## Notes

- All three steps use the **same random seed and data split** when
`--actor_data_dir` is passed to `train_lstm_vae_synthetic.py`. This ensures
both models are trained and evaluated on identical data.
- Re-running a training script **overwrites** the previous run in the same
output directory. Rename or copy the directory first if you want to keep
previous results.
- EC1/EC2/EC3 compare all methods against the **same black-box MLP**. The MLP's
accuracy vs the true label function does not affect the relative ranking of
methods — only imputation quality matters.

