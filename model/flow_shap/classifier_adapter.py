"""classifier_adapter.py — differentiable flow-space → classifier-logit bridge.

The flow-matching velocity field in CARE-PD operates on pelvis-centered,
per-(joint, coord) z-scored 3D clips of shape ``(B, T, 17, 3)``. Backbone
classifiers (POTR, PoseFormerV2, ...) each want a different input
representation and assume the world pelvis trajectory is present. This module
closes that gap with a single factory that:

1. Loads the pretrained ``MotionEncoder`` for a given backbone checkpoint.
2. Builds a differentiable chain
   ``x_flow (B, T, 17, 3)  →  logit_c (B,)``
   that:

   * de-z-scores into metres (pelvis-centered);
   * glues the original sample's world pelvis trajectory back into joint 0;
   * un-roots into world 3D via ``unroot_to_global``;
   * dispatches to ``project_for_backbone`` (handles POTR's z-score +
     mirroring, PoseFormerV2's camera projection + screen normalisation, etc.);
   * calls ``motion_encoder`` and selects the requested class logit.

The "if a backbone is selected that requires it, auto-process" requirement is
met because ``project_for_backbone`` already dispatches on ``backbone_name``.
Adding a new backbone later only requires registering it in
``_BACKBONE_CONFIG_MODULE`` (in ``shap_eval_shared``) and in
``project_for_backbone``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from model.actor.backbone_projection import project_for_backbone
from model.actor.motion_utils import unroot_to_global
from model.actor.shap_eval_shared import (
    _load_backbone_params,
    build_zscore_stats_for_potr,
    load_motion_encoder,
)
from model.motion_encoder import MotionEncoder


# ---------------------------------------------------------------------------
# Internal: differentiable flow-space → classifier-logit adapter
# ---------------------------------------------------------------------------

def _build_adapter(
    *,
    motion_encoder: MotionEncoder,
    backbone_name: str,
    flow_stats_mean: Tensor,   # (J, 3) metres — pelvis-centered stats
    flow_stats_std:  Tensor,   # (J, 3) metres
    class_idx: int,
    potr_zscore_mean: Optional[Tensor],   # (J, 3) for POTR only
    potr_zscore_std:  Optional[Tensor],
    device: torch.device,
) -> Tuple[Callable[[Tensor, Dict[str, Any]], Tensor], Callable[[Tensor, Dict[str, Any]], Tensor]]:
    """Return a differentiable ``classifier_fn(x_flow, ctx) -> (B,)`` callable.

    All persistent tensors are cached on ``device``; the returned callable
    only moves per-call ``ctx`` entries if they happen to be on a different
    device.
    """
    mean = flow_stats_mean.to(device=device, dtype=torch.float32)
    std  = flow_stats_std.to(device=device,  dtype=torch.float32)
    pmean = potr_zscore_mean.to(device=device, dtype=torch.float32) if potr_zscore_mean is not None else None
    pstd  = potr_zscore_std.to(device=device,  dtype=torch.float32) if potr_zscore_std  is not None else None

    def _forward_logits(x_flow: Tensor, ctx: Dict[str, Any]) -> Tensor:
        """Shared flow-space → all-class-logits path (B, C)."""
        if x_flow.ndim != 4 or x_flow.shape[-2:] != (17, 3):
            raise ValueError(
                f"x_flow must be (B, T, 17, 3); got {tuple(x_flow.shape)}"
            )
        B, T, J, C = x_flow.shape
        pelvis_world = ctx["pelvis_world"].to(device=x_flow.device, dtype=x_flow.dtype)
        if pelvis_world.shape != (B, T, 3):
            raise ValueError(
                f"ctx['pelvis_world'] must be (B, T, 3) = ({B}, {T}, 3); "
                f"got {tuple(pelvis_world.shape)}"
            )
        mask = ctx["mask"].to(device=x_flow.device)
        if mask.shape != (B, T):
            raise ValueError(
                f"ctx['mask'] must be (B, T) = ({B}, {T}); got {tuple(mask.shape)}"
            )

        # 1. de-z-score to metres (still pelvis-centered; joint-0 ≈ 0).
        x_m = x_flow * std.to(x_flow.dtype) + mean.to(x_flow.dtype)     # (B, T, J, 3)

        # 2. Replace joint-0 with the original world pelvis trajectory so the
        #    classifier sees the full walking path rather than a clip frozen
        #    at the origin.  This is the "fix_original" policy agreed with
        #    the user: the pelvis trajectory is held constant along the flow,
        #    which (combined with the pelvis-quotient construction) makes the
        #    pelvis attribution identically zero by the Dummy axiom.
        pelvis_bt1 = pelvis_world.unsqueeze(-2)                           # (B, T, 1, 3)
        non_pelvis = x_m[:, :, 1:, :]                                     # (B, T, J-1, 3)
        x_gp = torch.cat([pelvis_bt1, non_pelvis], dim=-2)                # (B, T, J, 3)

        # 3. un-root to global world 3D (joint k = pelvis_world + relative).
        x_world = unroot_to_global(x_gp)                                   # (B, T, J, 3)

        # 4. permute to (B, J, 3, T) for project_for_backbone and dispatch.
        x_jft = x_world.permute(0, 2, 3, 1).contiguous()                   # (B, J, 3, T)
        x_proj = project_for_backbone(
            x_jft, backbone_name,
            zscore_mean=pmean, zscore_std=pstd, pad_mask=mask,
        )

        # 5. classifier forward.
        metadata = ctx.get("metadata", None)
        if metadata is None:
            metadata = torch.zeros(B, 0, device=x_proj.device, dtype=x_proj.dtype)
        else:
            metadata = metadata.to(device=x_proj.device, dtype=x_proj.dtype)
        logits = motion_encoder(x_proj, metadata, valid_mask=mask.bool())  # (B, C)
        return logits

    def classifier_fn(x_flow: Tensor, ctx: Dict[str, Any]) -> Tensor:
        """Differentiable flow-space → logit at ``class_idx`` (B,).

        Per-sample class selection: if the caller passes a ``(B,)`` long tensor
        under ``ctx['class_idx']`` we gather one logit per sample, otherwise
        fall back to the fixed ``class_idx`` set at adapter-build time.
        """
        logits = _forward_logits(x_flow, ctx)           # (B, C)
        B = logits.shape[0]
        per_sample_cls = ctx.get("class_idx", None)
        if per_sample_cls is not None:
            per_sample_cls = torch.as_tensor(
                per_sample_cls, device=logits.device, dtype=torch.long,
            )
            if per_sample_cls.shape != (B,):
                raise ValueError(
                    f"ctx['class_idx'] must be (B,) long; got {tuple(per_sample_cls.shape)}"
                )
            return logits.gather(1, per_sample_cls.view(-1, 1)).squeeze(-1)
        return logits[:, class_idx]

    def full_logits_fn(x_flow: Tensor, ctx: Dict[str, Any]) -> Tensor:
        """Return full ``(B, C)`` logits — useful for class-policy resolution."""
        return _forward_logits(x_flow, ctx)

    return classifier_fn, full_logits_fn


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_classifier_fn(
    *,
    backbone_name: str,
    classifier_ckpt: str,
    flow_stats_mean: Tensor,
    flow_stats_std:  Tensor,
    class_idx: int,
    device: torch.device,
    config_file: str = "BMCLab.json",
    num_folds: int = 23,
    fold: int = 1,
) -> Tuple[Callable[[Tensor, Dict[str, Any]], Tensor],
           Callable[[Tensor, Dict[str, Any]], Tensor],
           MotionEncoder,
           Dict[str, Any]]:
    """Assemble a flow-space-aware classifier for OTFlow-SHAP attribution.

    Args:
        backbone_name: ``'potr'``, ``'poseformerv2'``, ``'motionbert'``,
            ``'motionagformer'``, or ``'mixste'``. Anything registered in
            ``project_for_backbone`` works automatically.
        classifier_ckpt: path to the ``run.py`` checkpoint (``latest_epoch.pth.tr``).
        flow_stats_mean, flow_stats_std: ``(17, 3)`` per-joint mean / std the
            flow was trained against; these are **pelvis-centered** stats so
            joint-0 row has mean ≈ 0 and std clamped by ``std_floor``.
        class_idx: which class logit to attribute.
        device: ``torch.device`` for classifier + stats.
        config_file: config-name passed to ``generate_config_<backbone>`` —
            ``'BMCLab.json'`` for all CARE-PD BMCLab backbones.
        num_folds, fold: fold identifiers matching the classifier's training
            fold — only used to compute POTR's z-score stats (the stats are
            identical across all folds because they're taken over the whole
            dataset, but we still need the dataset loader).

    Returns:
        ``(classifier_fn, full_logits_fn, motion_encoder, backbone_params)``.

        - ``classifier_fn(x_flow, ctx) -> (B,)`` is the differentiable bridge
          used by :func:`attribution.compute_flow_shap`.
        - ``full_logits_fn(x_flow, ctx) -> (B, C)`` returns all-class logits
          (for class-policy resolution / p_full checks).
        - ``motion_encoder`` is the loaded classifier.
        - ``backbone_params`` is the full backbone-config dict.
    """
    backbone_name = backbone_name.lower()
    params = _load_backbone_params(backbone_name, config_file, num_folds)

    motion_encoder = load_motion_encoder(classifier_ckpt, params, device)
    motion_encoder.eval()
    for p in motion_encoder.parameters():
        p.requires_grad_(False)

    potr_mean = potr_std = None
    if backbone_name == "potr":
        potr_mean, potr_std = build_zscore_stats_for_potr(
            params, fold, device, batch_size=64, root_centered=False,
        )

    classifier_fn, full_logits_fn = _build_adapter(
        motion_encoder=motion_encoder,
        backbone_name=backbone_name,
        flow_stats_mean=flow_stats_mean,
        flow_stats_std=flow_stats_std,
        class_idx=class_idx,
        potr_zscore_mean=potr_mean,
        potr_zscore_std=potr_std,
        device=device,
    )
    return classifier_fn, full_logits_fn, motion_encoder, params
