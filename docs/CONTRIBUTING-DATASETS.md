# Contributing a new synthetic SHAP benchmark

This document is a step-by-step guide for adding a new **synthetic SHAP
benchmark** — a controlled-ground-truth dataset used to evaluate imputation-
based Shapley methods (flow-matching, VAEAC, zero / mean / marginal) and
Aumann–Shapley methods (OTFlow-SHAP) against analytically-computable true
Shapley values.

The existing reference implementation is `synthetic/gaussian_motion.py`
(Gaussian-with-Kronecker-structure motion). A collaborator should be able to
add a new benchmark — e.g. a heavier-tailed distribution, a different
correlation structure, or a non-Gaussian generative process — without
touching the flow-matching / VAEAC / EC-evaluator code paths.

Not in scope: adding a real-world motion dataset (e.g. another cohort under
the CARE-PD umbrella). That has a separate surface (`data/*_datareader.py`);
see the top-level `README.md`.

---

## What "adding a benchmark" actually means

The pipeline consumes a benchmark through **five well-defined contact points**:

1. A Python class in `synthetic/<your_benchmark>.py` that produces data and
  oracle Shapley values.
2. A build-data script `scripts/build_<your_benchmark>_data.py` that
  materialises that class to disk in the canonical layout below.
3. A flow-training cache generator. For anything that emits `(N, J, F, T)`
  tensors with `J=17, F=3` you can reuse `scripts/generate_velocity_synthetic.py`
   unchanged.
4. A flow-matching config `configs/flow_matching/<your_benchmark>.json`
  and optionally a VAEAC config `configs/vaeac/<your_benchmark>.json`.
5. A `data_mode` branch in `scripts/compute_flow_shap_synthetic.py`
  (classifier adapter) and `scripts/evaluate_shap_synthetic_gaussian.py`
   (benchmark loader). See §4 for the exact hook points.

Everything else — flow training, VAEAC training, OTFlow-SHAP, the EC sweep,
the ranking-metrics script — is **benchmark-agnostic** and does not need to
be modified.

---

## 1. Hard constraints (things you cannot change without also patching models)

Two constants are baked into `VelocityNet` and the classifier adapter today:


| Constant | Value | Where                                                              |
| -------- | ----- | ------------------------------------------------------------------ |
| `J`      | 17    | `VelocityNet(n_joints=17)` in `train_flow_matching.py` and loaders |
| `F`      | 3     | `VelocityNet(n_coords=3)` in `train_flow_matching.py` and loaders  |


If your benchmark has different joint or feature counts you have two
options: (a) pad to `(J=17, F=3)` in the build-data script and carry a mask;
(b) patch every `VelocityNet(...)` call-site and the classifier-adapter
permutation in `scripts/compute_flow_shap_synthetic.py`. Option (a) is much
cheaper; option (b) is a first-class refactor (please flag it as a separate
PR).

`T` (sequence length) is free per-config (via `seq_len` in the flow config),
so temporal window count `K` and per-window frame counts are up to you as
long as `T % K == 0` is enforced by your benchmark class.

---

## 2. The benchmark class interface

Your class must expose the following attributes and methods. `gaussian_motion.GaussianMotionBenchmark`
is the reference implementation; copy its shape and replace the data-generation
internals.

### Required attributes

```python
class YourBenchmark:
    J: int                    # number of joints (17 today)
    F: int                    # number of features per joint (3 today)
    T: int                    # frames per sequence
    K: int                    # number of temporal windows (multiple of 4 if
                              # you want the Olsen-style label to tile cleanly)
    player_mode: str          # "temporal" or "spatial"
    window_assignments: list[list[int]]
                              # K lists of frame indices defining each window
    signal_joints: tuple[int, ...] | None
                              # 4 joint indices for spatial mode; None otherwise
    label_config: dict | None # populated by setup_label_fn
```

### Required methods

```python
def sample(N: int, seed: int | None = None) -> np.ndarray:
    """Sample N unconditioned sequences of shape (N, J, F, T) float32."""

def conditional_sample(
    x: np.ndarray,                              # (J, F, T) the conditioning sequence
    s_obs: tuple[int, ...],                     # observed window indices
    s_hid: tuple[int, ...],                     # hidden window indices
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw from p(x_hidden | x_observed) where hidden/observed are
    specified at *window* granularity. Must preserve observed frames
    exactly. Shape: (n_samples, J, F, T) float32."""

def conditional_sample_spatial(
    x: np.ndarray,                              # (J, F, T)
    j_obs: tuple[int, ...],                     # observed joint indices
    j_hid: tuple[int, ...],                     # hidden joint indices
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Same as conditional_sample but at *joint* granularity. Shape:
    (n_samples, J, F, T) float32. Only needed if you support
    player_mode='spatial'."""

def setup_label_fn(n_calib: int = 5000, seed: int = 999, noise_std: float = 0.0) -> None:
    """Calibrate and store the canonical label function in self.label_config.
    Called once by the build script; cached in the pickle afterwards."""

def canonical_label_fn(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Map (N, J, F, T) sequences to (N,) int64 class labels.
    Typically quantile-bin a scalar score into {0, 1, 2}."""

def compute_v_true_all_coalitions(
    x: torch.Tensor,                            # (1, J, F, T)
    classifier_fn: Callable,                    # (B, J, F, T) -> (B,)
    K_mc: int = 1000,
    device: torch.device | None = None,
    seed: int | None = None,
) -> dict[tuple, float]:
    """Temporal mode only. Return {coalition_tuple: v_true(S)} for all 2^K
    coalitions."""

def compute_true_shapley(v_all: dict[tuple, float]) -> np.ndarray:
    """Temporal mode only. Convert the 2^K v_true dict to (K,) Shapley values
    via the constrained Lundberg–Lee WLS. Typically reuses
    gaussian_motion._solve_shapley_wls."""

# Spatial-mode additions (if supported):

def sample_spatial_coalitions(n_pairs: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Paired KernelSHAP sampler. Returns (2*n_pairs, J) binary coalitions
    and their kernel weights."""

def compute_v_spatial(x, classifier_fn, coalitions, K_mc, device, seed,
                     v_empty_samples=None) -> np.ndarray:
    """Evaluate oracle v(S) for a list of spatial coalitions."""

def compute_true_shapley_spatial(x, classifier_fn, coalitions, weights,
                                 K_mc, device, seed) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (phi, v_on_coalitions, v_empty, v_full)."""

def save(path: str) -> None: ...
@staticmethod
def load(path: str) -> "YourBenchmark": ...
```

Plus a classifier. You have two reasonable options:

- **Reuse `SyntheticMLPClassifier`** from `gaussian_motion.py`. It already
accepts `player_mode ∈ {"temporal", "spatial"}` and defines the black-box
on per-window-or-per-joint grand means. This is the fastest path.
- **Write your own** `nn.Module` subclass. It must expose `.J`, `.F`, `.T`,
`.K`, `.player_mode`, a `.fit(x_train, y_train, epochs, ...)` method, and
a `.class_prob_fn(class_idx)` factory that returns a callable
`(B, J, F, T) -> (B,)`. Keep the body non-linear (ReLU / BatchNorm) so
classical Shapley and Aumann–Shapley diverge; otherwise your benchmark
will silently stop distinguishing methods.

A convenience factory that stitches benchmark + sampler + classifier together
should live alongside your class, mirroring
`gaussian_motion.build_gaussian_benchmark_and_classifier(...)`.

---

## 3. Canonical on-disk layout

The build-data script for your benchmark must produce **exactly** these files
in its output directory. Anything else the pipeline does not know how to
read.

```
<your_out_dir>/
├── synthetic_benchmark.pkl       # pickle of YourBenchmark instance
├── synthetic_clf.pt              # classifier state_dict
├── synthetic_clf_meta.json       # {"type": "...", "J": 17, "F": 3, "T": ..., "K": ...,
│                                  #  "num_classes": 3, "player_mode": "...",
│                                  #  "signal_joints": [...] or null}
├── synthetic_test.pt             # torch.save({"x": (N, T, J, F), "y": (N,), "pad_mask": (N, T)})
├── x_train_jft.npy               # (N_train, J, F, T) float32
└── config.json                   # {"data_mode": "<your_data_mode>", "J": ..., "F": ..., "T": ...,
                                  #  "K": ..., "n_train": ..., "n_val": ..., "n_test": ...,
                                  #  "seed": ..., "player_mode": "...", <plus your hyperparams>}
```

Key points:

- `x_train_jft.npy` is in `(N, J, F, T)` layout. The flow-cache generator
transposes to `(N, T, J, F)` internally, so don't pre-transpose.
- `synthetic_test.pt` is in `(N, T, J, F)` layout (the ACTOR convention
propagated through the rest of the pipeline).
- `config.json["data_mode"]` is the string key the pipeline uses to dispatch
your benchmark (see §4). Pick something unambiguous like
`"synthetic_<yourname>"`.

Use `scripts/build_synthetic_gaussian_data.py` as a template — the Gaussian
build script is ~160 lines and the structure is 1:1 with what you need.

---

## 4. Where to plug in — exact hook points

Only two scripts hard-code the `data_mode` string. Patch both.

### 4a. Classifier adapter in `scripts/compute_flow_shap_synthetic.py`

Around line 141, `_make_classifier_fn(clf, data_mode, num_classes)` currently
dispatches on `data_mode == "synthetic_gaussian"`. Add an `elif` branch for
your benchmark that returns a callable `(x_flow, ctx) -> (B,)` taking
flow-space `(B, T, 17, 3)` tensors, permuting to classifier layout, and
returning the target class probability.

If your classifier has the same `(B, J, F, T)` input shape as
`SyntheticMLPClassifier` you can copy the Gaussian branch verbatim.

### 4b. Benchmark loader in `scripts/evaluate_shap_synthetic_gaussian.py`

The script reads `config.json["data_mode"]` and imports the matching
benchmark class. Today there is a single hard-coded import of
`GaussianMotionBenchmark`. Extract that into a small dispatch (or just add
another branch) that imports `YourBenchmark` for your `data_mode`.

Everything else in this script — the zero/mean/marginal imputers, the flow
imputer loader, the VAEAC imputer loader, the EC1/EC2/EC3 computation —
operates purely through the interface declared in §2 and does not need
changes.

### 4c. Orchestration shell script (optional but recommended)

Copy `scripts/run_flow_matching_synthetic.sh` to
`scripts/run_flow_matching_<your_benchmark>.sh` and adjust `TAG`, `DATA_DIR`,
and the `BUILD_ARGS` block to call your build script.

### 4d. Flow config

Create `configs/flow_matching/<your_benchmark>.json` using
`configs/flow_matching/synthetic_gaussian.json` as a template. Adjust only:

- `"dataset"`: your `data_mode` string
- `"seq_len"`: your `T`
- `"cache_dir"` and `"checkpoint_dir"`

You almost never need to touch the transformer hyperparameters (d_model,
nhead, num_layers, ff_dim) unless you deliberately want an ablation.

VAEAC config is analogous: copy `configs/vaeac/synthetic_gaussian.json`.

---

## 5. Checklist for a new benchmark

Copy this into your PR description.

- `synthetic/<benchmark>.py` — benchmark class (§2) + classifier (or reuse)
- `scripts/build_<benchmark>_data.py` — mirrors `build_synthetic_gaussian_data.py`
- `configs/flow_matching/<benchmark>.json`
- `configs/vaeac/<benchmark>.json` (if running VAEAC baselines)
- `scripts/compute_flow_shap_synthetic.py` — new `elif data_mode == ...` branch
- `scripts/evaluate_shap_synthetic_gaussian.py` — benchmark-loader branch
- `scripts/run_flow_matching_<benchmark>.sh` (optional one-shot)
- Smoke test (§6) passes
- README.md snippet linking from `synthetic/README.md` pointing at your files

---

## 6. How to verify end-to-end

Before running the full flow-training (~~10 minutes) + EC sweep (~~1 h for K=8)
you can sanity-check the wiring in under two minutes:

### 6a. Class-level smoke test

Put a `__main_`_ block at the bottom of `synthetic/<benchmark>.py` that
exercises every interface method with a tiny config (`J=5, T=16, K=4, n=10`), the way `gaussian_motion.py` does. Run:

```bash
python synthetic/<benchmark>.py
```

This should print `Smoke test PASSED` and exit 0 without allocating a GPU.

### 6b. Full-pipeline smoke test

```bash
GPU=0 TAG=<benchmark>_smoke K=4 \
    N_TEST=5 K_MC_TRUE=50 N_COMP=5 \
    FLOW_CKPT_DIR=/tmp/flow_smoke_<benchmark> \
    bash scripts/run_flow_matching_<benchmark>.sh
```

Runtime budget on a single GPU: build+cache+train+EC ≈ 5 minutes with these
reduced counts. The EC table printed at the end should show

- `gaussian_oracle` (or your analogue): lowest EC1/EC2/EC3 of all methods
(it's the upper bound by construction)
- `zero` / `mean` / `marginal`: worse than the oracle
- `flow_matching`: between the baselines and the oracle (with only 5 min of
training you shouldn't expect much — this is a plumbing check, not a
quality check)

If `flow_matching` returns NaNs or is dramatically worse than `zero`,
something is wrong in your classifier adapter (§4a) or your conditional
sampler (§2). The most common bug is returning `(n_samples, J, F, T)`
instead of preserving observed frames exactly — check with:

```python
samps = bench.conditional_sample(x0, s_obs=(0,), s_hid=(1, 2, 3), n_samples=10)
for k in s_obs:
    for t in bench.window_assignments[k]:
        assert np.allclose(samps[:, :, :, t], x0[:, :, t][None])
```

### 6c. Full-scale run

Once smoke passes:

```bash
GPU=0 TAG=<benchmark>_k4 K=4 TRAIN_VAEAC=1 \
    bash scripts/run_flow_matching_<benchmark>.sh
```

Then for ranking metrics (after OTFlow-SHAP has been merged in):

```bash
python scripts/compute_ranking_metrics.py \
    --ec_dir experiment_outs/flow_matching_synthetic/ec_<benchmark>_k4
```

---

## 7. Common gotchas

- **Permutation ordering.** `(N, J, F, T)` for training, `(N, T, J, F)` for
test tensors and for the flow model. If your EC sweep prints `v` values
that are constant across coalitions you almost certainly have a layout
bug between these two conventions.
- **Seed propagation.** Always accept `seed` / `rng` and never call
`np.random.seed()` globally. The EC evaluator runs 100+ sequences in
series and needs per-sequence reproducibility.
- **Pickle safety.** `synthetic_benchmark.pkl` must round-trip through
`pickle.dump` / `pickle.load` without losing any field the EC evaluator
reads. Guard lazy caches (`self._cond_cache`) with `__getstate__` /
`__setstate__` if they grow too large, or clear them before `save()`.
- **Label monotonicity.** The default label function in `gaussian_motion.py`
quantile-bins a scalar score into 3 classes. If your label function
produces constant labels (e.g. because of a scale mismatch), the
classifier will memorise one class and every SHAP method will report
`phi ≈ 0`. Verify class balance is roughly 1/3 — 1/3 — 1/3 after
`canonical_label_fn`.
- `**K` and `K // 4`.** The Olsen-style label sums `K // 4` interaction
terms. If you want `K = 4` use one term; `K = 8` uses two; `K = 12` uses
three. Odd `K` or non-multiples-of-4 are explicitly rejected.
- **Spatial mode is optional.** If your benchmark does not naturally admit
joint-wise conditioning, raise `NotImplementedError` from
`conditional_sample_spatial` / `compute_true_shapley_spatial` and only
register the `temporal` player mode in §4.

---

## 8. Reference

- Implementation: `synthetic/gaussian_motion.py`
- Build script: `scripts/build_synthetic_gaussian_data.py`
- Flow config: `configs/flow_matching/synthetic_gaussian.json`
- VAEAC config: `configs/vaeac/synthetic_gaussian.json`
- One-shot pipeline: `scripts/run_flow_matching_synthetic.sh`
- EC evaluator: `scripts/evaluate_shap_synthetic_gaussian.py`
- Ranking metrics: `scripts/compute_ranking_metrics.py`
- Motivating paper: Olsen et al. (JMLR 2022), *A comparative study of methods
for estimating conditional Shapley values and when to use them*, Section 4.2.

