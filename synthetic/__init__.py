"""synthetic — controlled-ground-truth benchmarks for SHAP imputation.

Contents
--------

gaussian_motion
    Temporal-or-spatial motion drawn from a Gaussian with equicorrelation
    across joints and AR(1) across time. Analytically-tractable conditional
    distribution lets us compute ground-truth Shapley values (exact by 2^K
    enumeration in temporal mode; KernelSHAP with the oracle Gaussian
    conditional in spatial mode). This is the benchmark the flow-matching /
    VAEAC / OTFlow-SHAP evaluations in this repo target.

Adding a new synthetic benchmark
--------------------------------

See ``docs/CONTRIBUTING-DATASETS.md`` for a step-by-step walkthrough of the
interface a new benchmark must implement (sampling, conditional sampling,
label function, oracle Shapley, classifier) and where it plugs into the
build / flow-cache / EC-evaluation pipeline.
"""
