"""OTFlow-SHAP attribution pipeline for CARE-PD classifiers.

Public API:
    - ``compute_flow_shap``: core path-integral attribution using a trained
      flow-matching velocity field and a differentiable classifier.
    - ``build_classifier_fn``: factory that wraps any supported backbone
      (POTR, PoseFormerV2, ...) into a differentiable ``classifier_fn``
      operating in pelvis-centered z-score flow space.
    - Diagnostics helpers: completeness residual, flow-consistency error,
      pelvis-leak metric.
"""

from .attribution import compute_flow_shap
from .classifier_adapter import build_classifier_fn
from .diagnostics import (
    completeness_residual,
    flow_consistency_error,
    pelvis_leak,
)

__all__ = [
    "compute_flow_shap",
    "build_classifier_fn",
    "completeness_residual",
    "flow_consistency_error",
    "pelvis_leak",
]
