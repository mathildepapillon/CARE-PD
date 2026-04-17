# Flow matching for axiomatic SHAP on motion classifiers

This branch is a streamlined implementation of two flow-matching-based SHAP
methods for motion classifiers, evaluated side-by-side with classical
KernelSHAP baselines on a single BMCLab fold and on a synthetic Gaussian
benchmark with ground-truth Shapley structure.

Everything here is built on a single core idea: once you have a trained
flow-matching velocity field `v_θ(x, t)` that transports `N(0, I)` onto the
data manifold, you get **two** ways to do SHAP for a downstream classifier
`f(x)`, both axiomatically grounded:

1. **Integrated gradients along the flow geodesic** (OTFlow-SHAP,
   Aumann-Shapley). Attribute `f(x) - f(x₀_hat)` where `x₀_hat` is the
   backward-integrated noise endpoint of the flow.
2. **KernelSHAP with RePaint-style conditional sampling from the flow**
   (Flow-imputer KernelSHAP). Use the flow as a *conditional sampler*
   `p_θ(x_{\bar S} | x_S)` and plug it into standard KernelSHAP.

Method (1) is the gradient-only path; method (2) is the sampling path.
Both consume the same trained velocity net, and both are compared against
`zero` / `mean` / `marginal` KernelSHAP baselines under a shared
evaluation protocol.

---

## Contents

```
README_FLOW_MATCHING_SHAP.md   ← you are here
train_flow_matching.py         ← Lightning trainer for the velocity net
evaluate_shap_baselines.py     ← zero/mean/marginal KernelSHAP rankings

configs/
  flow_matching/
    bmclab_h36m3d_fold1.json   ← BMCLab flow model (XL: d_model=256, L=6)
    synthetic_gaussian.json    ← synthetic benchmark flow model
  flow_shap/
    bmclab_potr_fold1.json     ← OTFlow-SHAP on BMCLab / POTR
    bmclab_pfv2_fold1.json     ← OTFlow-SHAP on BMCLab / PoseFormerV2

model/
  flow_matching/velocity_net.py   ← VelocityNet (transformer + time emb)
  flow_shap/
    attribution.py                ← OTFlow-SHAP IG-along-flow computation
    imputer.py                    ← RePaint-style FlowImputer
    classifier_adapter.py         ← flow-space → classifier-logit adapter
    data_loading.py               ← flow cache I/O
    diagnostics.py                ← completeness / leak diagnostics

scripts/
  # Data / caches
  build_synthetic_gaussian_data.py   ← synthetic data + MLP classifier
  generate_velocity.py               ← BMCLab flow training cache
  generate_velocity_synthetic.py     ← synthetic flow training cache
  build_flow_shap_eval_cache.py      ← eval cache on classifier's held-out fold

  # Attribution (real-world BMCLab, both SHAP methods)
  compute_flow_shap.py               ← OTFlow-SHAP (IG-along-flow)
  compute_flow_shap_imputer.py       ← Flow-imputer KernelSHAP
  compute_baseline_shap.py           ← zero/mean/marginal KernelSHAP psi.npz

  # Attribution (synthetic Gaussian, OTFlow-SHAP only)
  compute_flow_shap_synthetic.py

  # Evaluation
  compute_shared_imputation_faithfulness.py  ← unified faithfulness
  visualize_flow_shap.py                     ← per-joint / per-time plots

  run_flow_matching_synthetic.sh     ← end-to-end synthetic pipeline

tests/
  test_flow_matching_velocity_net.py
  test_flow_shap.py
  test_attribution_completeness.py

make_tables.py                 ← publishable Tables 1/2/3

```

---

## 1. Flow-matching velocity net

The backbone is a conditional-OT flow-matching model implementing
`∂_t x_t = v_θ(x_t, t)` with linear interpolation paths
`x_t = (1-t) x_0 + t x_1`, `u_t = x_1 - x_0`. Training minimizes masked
MSE between `v_θ` and `u_t` with `x_1 ~ data`, `x_0 ~ N(0, I)`,
`t ~ U(0, 1)`.

`VelocityNet` (`model/flow_matching/velocity_net.py`) is a transformer
with per-frame positional embeddings (temporal context within a clip) and
a sinusoidal flow-time embedding. Outputs have the same shape as inputs
(`(B, T, J, C)`).

Train:

```bash
python train_flow_matching.py --config configs/flow_matching/bmclab_h36m3d_fold1.json
python train_flow_matching.py --config configs/flow_matching/synthetic_gaussian.json
```

The BMCLab config trains on a 6-fold train split (fold 1) at `seq_len=80`,
pelvis-centered and z-scored with per-joint statistics cached alongside
`cache.npz`. The synthetic config trains on i.i.d. Gaussian-structure
"clips" with known Shapley ground truth.

---

## 2. Two SHAP methods

Both methods share the same classifier adapter
(`model/flow_shap/classifier_adapter.py`): a flow-space tensor is
de-normalized, pelvis-restored, and fed through the classifier exactly as
during training, so attributions live in classifier-native coordinates
(`(B, J, F, T)`).

### 2a. OTFlow-SHAP (integrated gradients along the flow geodesic)

```
φ_j,t,c = ∫₀¹ (∂ f(x_s) / ∂ x_{j,t,c}) · v_{θ,j,t,c}(x_s, s) ds
```

where `x_s = ODE(x_0_hat, s)` with `x_0_hat = ODE⁻¹(x*, 0)` the backward
flow solve of the real clip. This is the Aumann-Shapley continuous
decomposition of `f(x*) - f(x_0_hat)` along the learned geodesic.

Run on BMCLab / POTR:

```bash
# 1. Build an eval cache on the classifier's held-out LOSO fold (SUB01)
python scripts/build_flow_shap_eval_cache.py \
    --flow_config   configs/flow_matching/bmclab_h36m3d_fold1.json \
    --classifier_num_folds 23 --classifier_fold 1

# 2. Compute attributions (per-clip psi.npz + summary.json)
python scripts/compute_flow_shap.py \
    --config configs/flow_shap/bmclab_potr_fold1.json
```

Output: `psi.npz` with `(psi, x_star, x0_hat, mask, f_xstar, f_x0,
completeness_rel, fce_per_sample, clip_id, participant_id, class_idx)`.

### 2b. Flow-imputer KernelSHAP (RePaint-style)

Same classifier, same test clips, but use the flow as a conditional
sampler. For each KernelSHAP coalition `S ⊆ {1, …, J}` the observed
joints `x_S` follow the CondOT linear path
`x_S,t = (1-t) x_0,S + t x_*,S` while the hidden joints `x_{\bar S}`
follow the learned ODE starting from noise, harmonized each step so the
observed subspace is never violated (RePaint-style, per-step re-conditioning).

```bash
python scripts/compute_flow_shap_imputer.py \
    --config configs/flow_shap/bmclab_potr_fold1.json \
    --n_kernel_samples 200 --n_completion_samples 20
```

The output `psi.npz` has the same schema as OTFlow-SHAP so downstream
faithfulness and comparison tooling consumes the two interchangeably.

`FlowImputer` (`model/flow_shap/imputer.py`) is also used by
`scripts/compute_shared_imputation_faithfulness.py` as an on-manifold
imputation protocol at evaluation time.

---

## 3. Evaluation

Faithfulness is computed by
`scripts/compute_shared_imputation_faithfulness.py`, which takes a
directory of SHAP rankings (baselines + flow-SHAP psi) and evaluates
every ranking under the same imputation at faithfulness time. This is
the central "apples-to-apples" script: same 240 SUB01 sequences, same
`build_classifier_fn`, same J=17 universe, same k-grid, same random-
control RNG; only the ranking source differs.

```bash
# 0. Generate the SHAP rankings themselves (run once per method)
python evaluate_shap_baselines.py \
    --backbone potr --config BMCLab.json --num_folds 23 --fold 1 \
    --output_dir results/shap_baselines_potr_bmclab_fold1_current
python scripts/compute_flow_shap.py --config configs/flow_shap/bmclab_potr_fold1.json \
    --output_dir experiment_outs/flow_shap/bmclab_potr_fold1_seeds/seed42

# 1. Evaluate every ranking under shared zero imputation
python scripts/compute_shared_imputation_faithfulness.py \
    --backbone potr --config BMCLab.json --num_folds 23 --fold 1 \
    --classifier_ckpt experiment_outs/Hypertune/POTR_BMCLab/0/models/train_BMCLab_23fold/fold1/latest_epoch.pth.tr \
    --baseline_jsonl results/shap_baselines_potr_bmclab_fold1_current/per_sequence.jsonl \
    --flow_psi_paths experiment_outs/flow_shap/bmclab_potr_fold1_seeds/seed42/psi.npz \
    --imputation zero \
    --output_dir results/shap_shared_imputation_zero_fold1

# 2. Same, under shared marginal imputation
python scripts/compute_shared_imputation_faithfulness.py \
    ... --imputation marginal \
    --output_dir results/shap_shared_imputation_marginal_fold1

# 3. Same, under shared flow-imputer imputation (on-manifold, ours)
python scripts/compute_shared_imputation_faithfulness.py \
    ... --imputation flow_imputer \
    --flow_config configs/flow_matching/bmclab_h36m3d_fold1.json \
    --flow_checkpoint experiment_outs/flow_matching/bmclab_h36m3d_fold1/flow_matching_BMCLab_fold1_best.ckpt \
    --output_dir results/shap_shared_imputation_flow_imputer_fold1
```

Each run writes `per_sequence.jsonl` + `aggregate.json` with PGI/PGU,
PGI-Rand/PGU-Rand, deletion/insertion AUC, and Shapley completeness
error for every ranking at every `k ∈ {1, 2, 3, 5}`.

### Tables

```bash
python make_tables.py                         # Tables 1/2/3 on fold 1
python make_tables.py --format latex          # booktabs LaTeX
python make_tables.py --output tables.md      # write to file
```

- **Table 1**: shared zero imputation (Zero-KernelSHAP's home turf).
- **Table 2**: shared marginal imputation (nobody's home turf).
- **Table 3**: shared flow-imputer imputation (OTFlow-SHAP's home turf).

Because the imputation is held fixed inside each table, any gap between
rankings isolates **ranking quality**. Differences between Tables 1/2/3
for a given ranking isolate **imputation severity**.

---

## 4. Synthetic Gaussian benchmark

The synthetic pipeline exists to check OTFlow-SHAP against **ground-truth
Shapley values** on a problem where they can be computed in closed form.
A tiny Gaussian-structure dataset and an MLP classifier are built by
`scripts/build_synthetic_gaussian_data.py`; a dedicated flow matching
model is trained on that dataset, and `scripts/compute_flow_shap_synthetic.py`
evaluates OTFlow-SHAP against the analytical Shapley ground truth.

End-to-end on a single GPU:

```bash
GPU=0 TAG=synthetic_gaussian bash scripts/run_flow_matching_synthetic.sh
```

This runs four phases sequentially:
1. Build synthetic data + classifier.
2. Generate the flow-training cache.
3. Train the velocity net (`train_flow_matching.py` with
   `configs/flow_matching/synthetic_gaussian.json`).
4. Compute OTFlow-SHAP attributions and compare to ground-truth Shapley.

---

## Tests

```bash
pytest tests/test_flow_matching_velocity_net.py \
       tests/test_flow_shap.py \
       tests/test_attribution_completeness.py
```

- `test_flow_matching_velocity_net.py`: shape / time-embedding sanity
  for `VelocityNet`.
- `test_flow_shap.py`: smoke test for the OTFlow-SHAP IG computation.
- `test_attribution_completeness.py`: asserts
  `Σ_{j,t,c} psi[j,t,c] ≈ f(x*) - f(x_0_hat)` within tolerance.
