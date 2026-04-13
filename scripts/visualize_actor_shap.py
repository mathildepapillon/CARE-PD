#!/usr/bin/env python3
"""
visualize_actor_shap.py — Render ActorSHAP masked completions as animated GIFs.

Loads a trained ActorSHAP checkpoint (``config.json`` next to it or via ``--config``),
samples sequences from the CARE-PD H36M loader, applies spatial **or** temporal
coalition masks (one axis per forward, matching ``MaskedActorEncoder``), and saves
GT vs completion GIFs (same style as ``visualize_actor_cvae``).

Usage::

    python scripts/visualize_actor_shap.py \\
        --checkpoint experiment_outs/actor_shap/<run>/actor_shap_last.ckpt \\
        --out_dir artifacts/actor_shap_gifs

    python scripts/visualize_actor_shap.py \\
        --checkpoint ... --spatial single --spatial_joint 3 \\
        --temporal stride_mask_one --stride_windows 0,1,2,3

``stride_observe_one`` / ``stride_mask_one`` emit one GIF per stride window (default
all four phases). Use ``--stride_windows 2`` to render a single phase only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from typing import Iterator

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _SCRIPTS)

import viz_utils  # noqa: E402
from data.dataloaders import collate_fn  # noqa: E402
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets  # noqa: E402
from model.actor.motion_utils import unroot_to_global  # noqa: E402
from model.actor.shap_masking import (  # noqa: E402
    H36M_GROUPS,
    H36M_JOINT_NAMES,
    build_spatial_shap_mask,
    build_temporal_shap_mask,
    build_temporal_windows,
    detect_stride_period,
)

NJ = 17

# Gait-phase labels for K=4 stride windows (see evaluate_shap / shap_compute temporal SHAP).
STRIDE_WINDOW_LABELS = (
    "IC_loading",
    "midstance_terminal",
    "preswing_initswing",
    "midswing_terminal",
)


def parse_stride_window_list(s: str) -> list[int]:
    """Parse comma-separated indices in 0..3 for stride temporal modes."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--stride_windows must list at least one index in 0..3")
    out = [int(p) for p in parts]
    for k in out:
        if not 0 <= k < 4:
            raise ValueError(f"stride window index must be in [0, 3], got {k}")
    return out


# ---------------------------------------------------------------------------
# Coalition mask builders
# ---------------------------------------------------------------------------

def spatial_mask_mask_group(group_name: str, device: torch.device) -> torch.Tensor:
    """(J,) True = observed; entire anatomical group is masked (held out)."""
    if group_name not in H36M_GROUPS:
        raise ValueError(f"Unknown group {group_name!r}. Choose from {sorted(H36M_GROUPS)}.")
    m = torch.ones(NJ, dtype=torch.bool, device=device)
    m[H36M_GROUPS[group_name]] = False
    if m.sum() == 0:
        m[0] = True
    return m


def spatial_mask_single_joint(joint_idx: int, device: torch.device) -> torch.Tensor:
    m = torch.ones(NJ, dtype=torch.bool, device=device)
    if not 0 <= joint_idx < NJ:
        raise ValueError(f"joint_idx must be in [0, {NJ - 1}], got {joint_idx}")
    m[joint_idx] = False
    return m


def parse_observed_csv(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = [int(p) for p in parts]
    for j in out:
        if not 0 <= j < NJ:
            raise ValueError(f"Joint index out of range: {j}")
    if len(set(out)) != len(out):
        raise ValueError("Duplicate joint indices in --spatial_observed")
    if len(out) == 0:
        raise ValueError("At least one observed joint required")
    return out


def quarter_frame_splits(real_len: int, k_parts: int = 4) -> list[list[int]]:
    """Equal-length quarters (training-style fallback), indices 0..real_len-1."""
    if real_len < k_parts:
        raise ValueError(f"Sequence too short for {k_parts} quarters: len={real_len}")
    q = real_len // k_parts
    windows: list[list[int]] = []
    for k in range(k_parts):
        start = k * q
        end = (k + 1) * q if k < k_parts - 1 else real_len
        windows.append(list(range(start, end)))
    return windows


def temporal_coalition_stride_windows(
    x_bjft: torch.Tensor,
    mask_bt: torch.Tensor,
    b: int,
    device: torch.device,
    *,
    observe_only: int | None = None,
    mask_only: int | None = None,
) -> torch.Tensor:
    """Stride-aligned gait windows; coalition length ``T_pad``, padding unobserved."""
    real_len = int(mask_bt[b].sum().item())
    if real_len < 2:
        raise ValueError("empty sequence")
    t_pad = mask_bt.shape[1]
    # (J, F, T_valid) → (T, J, F) for detect_stride_period (same as evaluate_shap.py).
    x_np = x_bjft[b, :, :, :real_len].permute(2, 0, 1).detach().cpu().numpy()
    period, _ = detect_stride_period(x_np)
    windows = build_temporal_windows(real_len, period, K=4)
    if observe_only is not None:
        c = build_temporal_shap_mask([observe_only], windows, real_len, device)
    elif mask_only is not None:
        c = build_temporal_shap_mask([k for k in range(4) if k != mask_only], windows, real_len, device)
    else:
        raise RuntimeError("specify observe_only or mask_only")
    out = torch.zeros(t_pad, dtype=torch.bool, device=device)
    out[:real_len] = c
    return out & mask_bt[b]


def temporal_coalition_quarters(
    real_len: int,
    t_pad: int,
    mask_bt: torch.Tensor,
    b: int,
    device: torch.device,
    *,
    quarter_idx: int,
    observe_only_quarter: bool,
) -> torch.Tensor:
    windows = quarter_frame_splits(real_len, 4)
    if observe_only_quarter:
        c = build_temporal_shap_mask([quarter_idx], windows, real_len, device)
    else:
        c = build_temporal_shap_mask([k for k in range(4) if k != quarter_idx], windows, real_len, device)
    out = torch.zeros(t_pad, dtype=torch.bool, device=device)
    out[:real_len] = c
    return out & mask_bt[b]


def iter_scenarios(args: argparse.Namespace) -> Iterator[tuple[str, str]]:
    """Yield (scenario_id, human_label). Each scenario uses one coalition axis."""
    spatial_specs: list[tuple[str, str]] = []
    if args.spatial == "none":
        spatial_specs.append(("spatial_none", "spatial: full obs"))
    elif args.spatial == "groups":
        for name in sorted(H36M_GROUPS.keys()):
            spatial_specs.append((f"spatial_group_{name}", f"spatial: mask {name}"))
    elif args.spatial == "single":
        j = args.spatial_joint
        jname = H36M_JOINT_NAMES[j] if j < len(H36M_JOINT_NAMES) else str(j)
        spatial_specs.append((f"spatial_joint{j}", f"spatial: mask joint {j} ({jname})"))
    elif args.spatial == "custom":
        spatial_specs.append(("spatial_custom", f"spatial: observed {args.spatial_observed}"))
    else:
        raise ValueError(args.spatial)

    temporal_specs: list[tuple[str, str]] = []
    if args.temporal == "none":
        temporal_specs.append(("temporal_none", "temporal: full obs"))
    elif args.temporal == "per_quarter_observe":
        for k in range(4):
            temporal_specs.append((f"temporal_q{k}_observe", f"temporal: observe Q{k} only"))
    elif args.temporal == "per_quarter_mask":
        for k in range(4):
            temporal_specs.append((f"temporal_q{k}_mask", f"temporal: mask Q{k} only"))
    elif args.temporal == "stride_observe_one":
        for k in parse_stride_window_list(args.stride_windows):
            label = STRIDE_WINDOW_LABELS[k]
            temporal_specs.append(
                (
                    f"temporal_stride_obs_k{k}",
                    f"temporal: observe {label} only (mask other stride parts)",
                ),
            )
    elif args.temporal == "stride_mask_one":
        for k in parse_stride_window_list(args.stride_windows):
            label = STRIDE_WINDOW_LABELS[k]
            temporal_specs.append(
                (
                    f"temporal_stride_mask_k{k}",
                    f"temporal: mask {label} only",
                ),
            )
    else:
        raise ValueError(args.temporal)

    has_spatial = any(s != "spatial_none" for s, _ in spatial_specs)
    has_temporal = any(t != "temporal_none" for t, _ in temporal_specs)
    if not has_spatial and not has_temporal:
        yield "spatial_none", "full obs baseline (all joints observed)"
        return

    for sid_s, lab_s in spatial_specs:
        if sid_s != "spatial_none":
            yield sid_s, f"{lab_s}; temporal: full obs"
    for sid_t, lab_t in temporal_specs:
        if sid_t != "temporal_none":
            yield sid_t, f"spatial: full obs; {lab_t}"


def build_coalition_for_scenario(
    scenario_id: str,
    args: argparse.Namespace,
    batch: dict,
    b: int,
    device: torch.device,
) -> torch.Tensor:
    """Coalition for sequence ``b``: shape ``(J,)`` or ``(T,)``."""
    x = batch["x"]
    mask_bt = batch["mask"]
    t_pad = mask_bt.shape[1]
    real_len = int(mask_bt[b].sum().item())

    if scenario_id.startswith("spatial_group_"):
        name = scenario_id[len("spatial_group_") :]
        return spatial_mask_mask_group(name, device)

    if scenario_id.startswith("spatial_joint"):
        j = int(scenario_id[len("spatial_joint") :])
        return spatial_mask_single_joint(j, device)

    if scenario_id == "spatial_custom":
        obs = parse_observed_csv(args.spatial_observed)
        return build_spatial_shap_mask(obs, device, NJ)

    if scenario_id == "spatial_none":
        return torch.ones(NJ, dtype=torch.bool, device=device)

    m_q = re.match(r"temporal_q(\d+)_(observe|mask)", scenario_id)
    if m_q:
        k = int(m_q.group(1))
        observe_only = m_q.group(2) == "observe"
        return temporal_coalition_quarters(
            real_len, t_pad, mask_bt, b, device,
            quarter_idx=k, observe_only_quarter=observe_only,
        )

    m_stride_obs = re.match(r"^temporal_stride_obs_k(\d+)$", scenario_id)
    if m_stride_obs:
        k = int(m_stride_obs.group(1))
        return temporal_coalition_stride_windows(
            x, mask_bt, b, device, observe_only=k, mask_only=None,
        )

    m_stride_mask = re.match(r"^temporal_stride_mask_k(\d+)$", scenario_id)
    if m_stride_mask:
        k = int(m_stride_mask.group(1))
        return temporal_coalition_stride_windows(
            x, mask_bt, b, device, observe_only=None, mask_only=k,
        )

    raise ValueError(f"unknown scenario {scenario_id}")


def run_completion_gif(
    model: torch.nn.Module,
    batch: dict,
    b: int,
    coalition_mask_1: torch.Tensor,
    out_path: str,
    fps: int,
    title_prefix: str,
    *,
    seed: int,
    paste_observed: bool,
    verbose: bool,
    legacy_world_coords: bool = False,
) -> None:
    torch.manual_seed(seed)
    x = batch["x"][b : b + 1]
    y = batch["y"][b : b + 1]
    mask = batch["mask"][b : b + 1]
    lengths = batch["lengths"][b : b + 1]
    cm = coalition_mask_1.unsqueeze(0)

    comps = model.sample_completions(
        x, y, mask, lengths, cm, n_samples=1, paste_observed=paste_observed,
    )
    x_hat = comps[0]

    gt_bjft = batch.get("x_xyz", batch["x"])[b : b + 1]
    real_len = int(mask[0].sum().item())
    gt_btj3_t  = gt_bjft[:, :, :, :real_len].permute(0, 3, 1, 2)
    out_btj3_t = x_hat[:, :, :, :real_len].permute(0, 3, 1, 2)
    if legacy_world_coords:
        # Old checkpoints output raw world-space coords — no unrooting needed.
        gt_btj3  = gt_btj3_t.cpu().numpy()
        out_btj3 = out_btj3_t.cpu().numpy()
    else:
        # Recover full global 3D from global-pelvis representation so the whole
        # skeleton moves coherently in world space during rendering.
        gt_btj3  = unroot_to_global(gt_btj3_t).cpu().numpy()
        out_btj3 = unroot_to_global(out_btj3_t).cpu().numpy()

    edges = viz_utils.edges_for_njoints(gt_btj3.shape[2])
    viz_utils.save_motion_comparison_gif(
        gt_btj3[0],
        out_btj3[0],
        edges,
        out_path,
        fps,
        title_prefix=title_prefix,
        legend_pred_label="completion",
        verbose=verbose,
    )


def print_presets() -> None:
    print("Anatomical groups (for --spatial groups):", ", ".join(sorted(H36M_GROUPS.keys())))
    print("Joint names:")
    for i, n in enumerate(H36M_JOINT_NAMES):
        print(f"  {i:2d}  {n}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize ActorSHAP masked completions as animated GIFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="actor_shap_last.ckpt (config.json alongside or --config). Not required with --list_presets.",
    )
    p.add_argument("--config", default=None, help="Override path to training config.json.")
    p.add_argument("--out_dir", default="artifacts/actor_shap_gifs")
    p.add_argument("--n_examples", type=int, default=5, help="Sequences per batch.")
    p.add_argument("--n_batches", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--device", default=None)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (incremented per GIF).")
    p.add_argument("--no_paste", action="store_true", help="Do not paste observed inputs into the output.")
    p.add_argument(
        "--legacy_world_coords", action="store_true",
        help="Use raw world-space coordinates (no global-pelvis transform, no unrooting). "
             "Required for checkpoints trained before the global-pelvis representation was introduced.",
    )
    p.add_argument(
        "--spatial",
        choices=("none", "groups", "single", "custom"),
        default="groups",
        help="Spatial coalition sweep; 'none' skips spatial GIFs.",
    )
    p.add_argument("--spatial_joint", type=int, default=0, help="Joint index for --spatial single.")
    p.add_argument("--spatial_observed", type=str, default="0,7,8,9", help="Comma-separated observed joints for --spatial custom.")
    p.add_argument(
        "--temporal",
        choices=(
            "none",
            "per_quarter_observe",
            "per_quarter_mask",
            "stride_observe_one",
            "stride_mask_one",
        ),
        default="none",
        help="Temporal sweep; 'none' skips temporal-only GIFs.",
    )
    p.add_argument(
        "--stride_windows",
        type=str,
        default="0,1,2,3",
        help="Comma-separated stride window indices 0–3 for stride_observe_one / stride_mask_one "
        "(default: all four gait phases).",
    )
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main process only; avoids CUDA fork deadlocks).")
    p.add_argument("--list_presets", action="store_true", help="Print joint groups/names and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        print_presets()
        return

    if not args.checkpoint:
        sys.exit("--checkpoint is required unless using --list_presets.")

    from evaluate_shap import load_actor_shap  # noqa: E402

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    cfg_path = args.config or os.path.join(os.path.dirname(ckpt_path), "config.json")
    if not os.path.isfile(cfg_path):
        sys.exit(f"config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"Loaded config from {cfg_path}")

    model = load_actor_shap(ckpt_path, device, config_path=args.config)
    print(f"ActorSHAP loaded (eval). pose_rep={model.pose_rep}")

    ns = types.SimpleNamespace(**{k: v for k, v in cfg.items() if not str(k).startswith("_")})
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 16))

    train_ds, val_ds = get_carepd_datasets(ns)
    ds = train_ds if args.split == "train" else val_ds
    shuffle = (args.split == "train") or args.shuffle
    print(f"Using {args.split} split ({len(ds)} sequences, shuffle={shuffle}).")

    scenarios = list(iter_scenarios(args))
    if not scenarios:
        sys.exit("No scenarios selected; use --spatial and/or --temporal.")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    gif_count = 0
    seed = args.seed

    for batch_idx, batch_raw in enumerate(loader):
        if batch_idx >= args.n_batches:
            break
        x, lab, _vidx, _meta, pad_mask = batch_raw
        bdict = actor_batch_from_carepd(
            x.to(device), pad_mask.to(device), model.num_classes, device, y=lab.to(device),
            world_coords=args.legacy_world_coords,
        )
        if model.pose_rep == "xyz":
            bdict["x_xyz"] = bdict["x"]

        n_seq = min(args.n_examples, x.shape[0])
        print(f"Batch {batch_idx} shape={tuple(x.shape)} scenarios={len(scenarios)}")

        for b in range(n_seq):
            for sid, lab_human in scenarios:
                coalition = build_coalition_for_scenario(sid, args, bdict, b, device)
                slug = re.sub(r"[^\w.\-]+", "_", sid)
                out_path = os.path.join(
                    args.out_dir,
                    f"shap_b{batch_idx:02d}_s{b:02d}_{slug}.gif",
                )
                title = f"{lab_human}  batch {batch_idx:02d} seq {b}"
                run_completion_gif(
                    model,
                    bdict,
                    b,
                    coalition,
                    out_path,
                    args.fps,
                    title_prefix=title,
                    seed=seed,
                    paste_observed=not args.no_paste,
                    verbose=True,
                    legacy_world_coords=args.legacy_world_coords,
                )
                seed += 1
                gif_count += 1

    print(f"Done. {gif_count} GIF(s) → {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
