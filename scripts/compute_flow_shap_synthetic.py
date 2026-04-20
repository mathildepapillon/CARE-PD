"""
compute_flow_shap_synthetic.py — OTFlow-SHAP path-integral (IG) attribution
for the synthetic SHAP benchmark.

This is the synthetic analogue of ``scripts/compute_flow_shap.py``. It reuses
``model.flow_shap.attribution.compute_flow_shap`` (unchanged) but skips the
CARE-PD-specific machinery:

* no pelvis recovery / un-rooting;
* no z-score denormalization (synthetic caches write ``stats_mean=0``,
  ``stats_std=1``);
* ``zero_pelvis=False`` — joint 0 carries real signal in synthetic data;
* classifier is ``SyntheticMLPClassifier`` (Gaussian benchmark) which takes
  ``(B, J, F, T)`` input.

Output: ``psi.npz`` + ``summary.json`` under ``--output_dir`` (default:
``<flow_ckpt_dir>/ig_synthetic``).

Usage
-----

    python scripts/compute_flow_shap_synthetic.py \\
        --ckpt_dir       experiment_outs/actor_shap_synthetic/GAUSS \\
        --flow_ckpt_dir  experiment_outs/flow_matching_synthetic/GAUSS \\
        --flow_config    configs/flow_matching/synthetic_gaussian.json \\
        --data_mode      synthetic_gaussian \\
        --n_clips        100 --class_idx 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.flow_matching import VelocityNet
from model.flow_shap import compute_flow_shap, completeness_residual


# ---------------------------------------------------------------------------
# VelocityNet loading (mirrors scripts/compute_flow_shap.py)
# ---------------------------------------------------------------------------

def _strip_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _find_flow_ckpt(flow_ckpt_dir: Path) -> Path:
    """Prefer ``last.ckpt``; fall back to the newest ``*.ckpt``."""
    last = flow_ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted(flow_ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {flow_ckpt_dir}")
    return ckpts[-1]


def _load_velocity_net(flow_cfg: dict, ckpt_path: Path, device: torch.device) -> VelocityNet:
    net = VelocityNet(
        n_joints=17, n_coords=3,
        d_model=int(flow_cfg["d_model"]),
        nhead=int(flow_cfg["nhead"]),
        num_layers=int(flow_cfg["num_layers"]),
        ff_dim=int(flow_cfg["ff_dim"]),
        dropout=float(flow_cfg.get("dropout", 0.0)),
        time_emb_dim=int(flow_cfg["time_emb_dim"]),
        max_len=max(int(flow_cfg["seq_len"]) + 16, 256),
        tokenization=str(flow_cfg.get("tokenization", "frame")),
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    clean_state = _strip_prefix(raw_state, "model.")
    if not clean_state:
        raise RuntimeError(
            f"No 'model.*' keys in {ckpt_path}; keys sampled: {list(raw_state.keys())[:5]}"
        )
    missing, unexpected = net.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[load_velocity_net] missing {len(missing)} keys (first: {missing[:3]})")
    if unexpected:
        print(f"[load_velocity_net] unexpected {len(unexpected)} keys (first: {unexpected[:3]})")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------

def _load_gaussian_classifier(ckpt_dir: Path, device: torch.device) -> nn.Module:
    from synthetic.gaussian_motion import SyntheticMLPClassifier

    meta_path = ckpt_dir / "synthetic_clf_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"J": 17, "F": 3, "T": 81, "K": 4, "num_classes": 3}
    clf = SyntheticMLPClassifier(
        J=meta["J"], F=meta["F"], T=meta["T"],
        K=meta.get("K", 4), num_classes=meta["num_classes"],
        player_mode=meta.get("player_mode", "temporal"),
    )
    clf.load_state_dict(torch.load(str(ckpt_dir / "synthetic_clf.pt"),
                                   map_location="cpu", weights_only=False))
    clf.to(device).eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


# ---------------------------------------------------------------------------
# Classifier adapters: bridge (B, T, J, C) flow-space ↔ classifier input
# ---------------------------------------------------------------------------

def _make_classifier_fn(
    clf: nn.Module,
    data_mode: str,
    num_classes: int,
):
    """Return a callable ``classifier_fn(x_flow, ctx) -> (B,)``.

    The flow input is ``(B, T, 17, 3)``; the synthetic classifiers expect
    ``(B, J, F, T)``. We permute and gather ``ctx["class_idx"]``.
    """
    if data_mode == "synthetic_gaussian":
        def classifier_fn(x_flow: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
            x_bjft = x_flow.permute(0, 2, 3, 1).contiguous()  # (B, J, F, T)
            logits = clf(x_bjft)                               # (B, C)
            probs = torch.softmax(logits, dim=-1)
            cls = ctx["class_idx"]
            if cls.ndim == 0:
                cls = cls.expand(x_flow.shape[0])
            return probs.gather(1, cls.view(-1, 1)).squeeze(-1)
        return classifier_fn

    raise ValueError(f"Unknown data_mode: {data_mode!r}")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _summarise(arr: np.ndarray, name: str) -> Dict[str, float]:
    if arr.size == 0:
        return {f"{name}_mean": float("nan"), f"{name}_median": float("nan"),
                f"{name}_p95":  float("nan"), f"{name}_max":    float("nan")}
    return {
        f"{name}_mean":   float(arr.mean()),
        f"{name}_median": float(np.median(arr)),
        f"{name}_p95":    float(np.percentile(arr, 95.0)),
        f"{name}_max":    float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory produced by scripts/build_synthetic_gaussian_data.py.")
    p.add_argument("--flow_ckpt_dir", required=True,
                   help="Directory produced by train_flow_matching.py "
                        "(must contain last.ckpt or flow_matching_*_best.ckpt).")
    p.add_argument("--flow_config", required=True,
                   help="Path to the flow-matching JSON config used to train "
                        "the checkpoint in --flow_ckpt_dir.")
    p.add_argument("--data_mode", choices=("synthetic_gaussian",),
                   default="synthetic_gaussian",
                   help="Synthetic benchmark. Only 'synthetic_gaussian' is supported.")
    p.add_argument("--n_clips", type=int, default=100,
                   help="Cap on test clips to attribute (0 = all).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_ode_steps", type=int, default=100)
    p.add_argument("--solver", default="midpoint")
    p.add_argument("--class_idx", type=int, default=0,
                   help="Class index for probability evaluation (gaussian only).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir).resolve()
    flow_ckpt_dir = Path(args.flow_ckpt_dir).resolve()
    flow_cfg_path = Path(args.flow_config)
    if not flow_cfg_path.is_absolute():
        flow_cfg_path = PROJECT_ROOT / flow_cfg_path
    with open(flow_cfg_path) as f:
        flow_cfg = json.load(f)

    data_mode = args.data_mode
    if data_mode is None:
        cfg_json = ckpt_dir / "config.json"
        if not cfg_json.exists():
            raise ValueError(
                f"--data_mode not set and {cfg_json} missing; cannot infer."
            )
        with open(cfg_json) as f:
            data_mode = json.load(f)["data_mode"]

    out_dir = Path(args.output_dir) if args.output_dir else flow_ckpt_dir / "ig_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load checkpoints ----------------------------------------------------
    ckpt_path = _find_flow_ckpt(flow_ckpt_dir)
    print(f"[ig_synth] velocity net ← {ckpt_path}")
    velocity_net = _load_velocity_net(flow_cfg, ckpt_path, device)

    print(f"[ig_synth] classifier   ← {ckpt_dir} (mode={data_mode})")
    clf = _load_gaussian_classifier(ckpt_dir, device)
    num_classes = int(getattr(clf, "net", nn.Identity())[-1].out_features) \
                  if hasattr(clf, "net") else 3
    classifier_fn = _make_classifier_fn(clf, data_mode, num_classes)

    # --- load test clips ----------------------------------------------------
    test_pt = torch.load(str(ckpt_dir / "synthetic_test.pt"), map_location="cpu",
                         weights_only=False)
    x_test_btjf = test_pt["x"]                      # (N, T, J, F)
    y_test       = test_pt["y"]
    pm_test      = test_pt["pad_mask"]

    N_all = x_test_btjf.shape[0]
    n_clips = N_all if args.n_clips <= 0 else min(N_all, int(args.n_clips))
    x_test_btjf = x_test_btjf[:n_clips].float()
    pm_test     = pm_test[:n_clips].bool()
    print(f"[ig_synth] attributing {n_clips} test clips (of {N_all})")

    # --- iterate in batches -------------------------------------------------
    psi_all, x0_all, fstar_all, fx0_all, comp_all = [], [], [], [], []
    cls_idx_all = []
    t0 = time.time()

    for i in range(0, n_clips, args.batch_size):
        xb = x_test_btjf[i : i + args.batch_size].to(device)   # (B, T, J, F)
        mb = pm_test[i : i + args.batch_size].to(device)        # (B, T)
        B = xb.shape[0]
        cls = torch.full((B,), int(args.class_idx), dtype=torch.long, device=device)

        ctx: Dict[str, Any] = {"mask": mb, "class_idx": cls}
        out = compute_flow_shap(
            velocity_net=velocity_net,
            classifier_fn=classifier_fn,
            x_star_flow=xb,
            ctx=ctx,
            num_steps=args.num_ode_steps,
            solver_method=args.solver,
            zero_pelvis=False,  # pelvis is a real joint in synthetic data
            return_trajectory=False,
        )
        comp = completeness_residual(out["psi"], out["f_xstar"], out["f_x0"], mask=mb)

        psi_all.append(out["psi"].detach().cpu().numpy())
        x0_all.append(out["x0_hat"].detach().cpu().numpy())
        fstar_all.append(out["f_xstar"].detach().cpu().numpy())
        fx0_all.append(out["f_x0"].detach().cpu().numpy())
        comp_all.append(comp["rel"].numpy())
        cls_idx_all.append(cls.detach().cpu().numpy())

        done = i + B
        if done % max(args.batch_size * 4, 1) == 0 or done == n_clips:
            print(f"[ig_synth] {done}/{n_clips}  t={time.time() - t0:6.1f}s",
                  flush=True)

    psi_arr     = np.concatenate(psi_all, axis=0)
    x0_arr      = np.concatenate(x0_all, axis=0)
    fstar_arr   = np.concatenate(fstar_all, axis=0)
    fx0_arr     = np.concatenate(fx0_all, axis=0)
    comp_arr    = np.concatenate(comp_all, axis=0)
    cls_idx_arr = np.concatenate(cls_idx_all, axis=0)

    psi_path     = out_dir / "psi.npz"
    summary_path = out_dir / "summary.json"
    np.savez_compressed(
        psi_path,
        psi=psi_arr.astype(np.float32),
        x_star=x_test_btjf.numpy().astype(np.float32),
        x0_hat=x0_arr.astype(np.float32),
        mask=pm_test.numpy().astype(bool),
        f_xstar=fstar_arr.astype(np.float32),
        f_x0=fx0_arr.astype(np.float32),
        completeness_rel=comp_arr.astype(np.float32),
        class_idx=cls_idx_arr.astype(np.int32),
        labels=y_test[:n_clips].numpy().astype(np.int64),
    )
    print(f"[ig_synth] wrote {psi_path}")

    summary = {
        "source_ckpt_dir":   str(ckpt_dir),
        "flow_ckpt_dir":     str(flow_ckpt_dir),
        "flow_checkpoint":   str(ckpt_path),
        "flow_config":       str(flow_cfg_path),
        "data_mode":         data_mode,
        "n_clips":           int(psi_arr.shape[0]),
        "num_ode_steps":     int(args.num_ode_steps),
        "solver":            args.solver,
        "class_idx":         int(args.class_idx),
        **_summarise(comp_arr, "completeness_rel"),
        **_summarise(np.abs(fstar_arr - fx0_arr), "delta_f_abs"),
        "elapsed_seconds":   float(time.time() - t0),
        "output_psi_path":   str(psi_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ig_synth] wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
