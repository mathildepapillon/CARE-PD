"""Shared SHAP infrastructure.

Historically this package hosted the ActorSHAP CVAE stack. The
flow-matching cleanup branch retired that stack; the modules that remain
(``shap_compute``, ``shap_masking``, ``shap_metrics``, ``shap_eval_shared``,
``cvae_data``, ``motion_utils``, ``backbone_projection``) are the generic
KernelSHAP engine, masking helpers, faithfulness metrics, and data/IO
plumbing shared by every SHAP method in this repo. See
``README_FLOW_MATCHING_SHAP.md`` for the new narrative.
"""
