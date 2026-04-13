"""sandbox_vaeac_diversity_eval.py

Evaluate VaeacMotion completion diversity vs. reconstruction fidelity across
all spatial coalition types considered in evaluate_shap.py.

For each coalition type (anatomical group masks + random Bernoulli masks at
several sparsity levels) and a set of test sequences we measure:

  - diversity_raw    : mean pairwise RMSE across K completions, computed over
                       the masked joints only (same space as masked_rmse).
  - diversity_enc    : mean pairwise L2 of full-encoder mu vectors, same as
                       compute_completion_diversity() in shap_metrics.py.
  - masked_rmse      : RMSE of completions vs GT on held-out joints.
  - obs_rmse         : RMSE on observed joints (should be ~0 with paste_observed).
  - div_over_rmse    : diversity_raw / max(masked_rmse, 1e-6) — fidelity-normalised
                       diversity.  Higher is better: model is diverse AND accurate.

Usage::

    python scripts/sandbox_vaeac_diversity_eval.py \\
        --ckpt experiment_outs/vaeac_motion/vaeac_motion_BMCLab_fold1_20260413_102809/vaeac_motion_last.ckpt \\
        --fold 1 --n_seq 15 --n_samples 20 --device cuda:0

    # auto-find the latest checkpoint:
    python scripts/sandbox_vaeac_diversity_eval.py --fold 1 --n_seq 15 --n_samples 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make sure the project root is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataloaders import collate_fn
from model.actor.cvae_data import actor_batch_from_carepd, get_carepd_datasets
from model.actor.shap_masking import H36M_GROUPS, H36M_JOINT_NAMES
from model.actor.transformer_arch import Decoder_TRANSFORMER
from model.actor.vaeac_motion import VaeacFullEncoder, VaeacMaskedEncoder, VaeacMotion


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_latest_ckpt(base_dir: str, fold: int) -> str | None:
    pattern = os.path.join(
        base_dir, f"vaeac_motion_BMCLab_fold{fold}_*", "vaeac_motion_last.ckpt"
    )
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def load_vaeac_from_ckpt(ckpt_path: str, device: torch.device) -> VaeacMotion:
    """Rebuild VaeacMotion from a Lightning checkpoint saved by train_vaeac_motion.py.

    Auto-detects whether the checkpoint pre-dates the binary coalition indicator
    (observed_projection input = J*F=51) or uses the current architecture
    (input = J*F+J=68) by inspecting the saved weight shape.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, "config.json")
    cfg: dict = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        print(f"[WARNING] no config.json found next to {ckpt_path}; using defaults.")

    latent_dim  = int(cfg.get("latent_dim",  256))
    ff_size     = int(cfg.get("ff_size",     1024))
    num_layers  = int(cfg.get("num_layers",  8))
    num_heads   = int(cfg.get("num_heads",   4))
    dropout     = float(cfg.get("dropout",   0.1))
    num_classes = int(cfg.get("num_classes", 3))

    # Auto-detect whether this is a legacy checkpoint (no binary indicator in
    # observed_projection).  J*F = 17*3 = 51 → legacy; J*F+J = 68 → current.
    raw = torch.load(ckpt_path, map_location=device)
    sd  = raw.get("state_dict", raw)
    sd  = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    proj_in = sd["observed_projection.weight"].shape[1]  # 51 (legacy) or 68 (current)
    use_obs_indicator = (proj_in != 17 * 3)
    if not use_obs_indicator:
        print("[load] legacy checkpoint detected (observed_projection input=51, no indicator)")

    common = dict(
        modeltype="cvae",
        njoints=17, nfeats=3,
        num_frames=0, num_classes=num_classes,
        translation=True, pose_rep="xyz",
        glob=True, glob_rot=[3.141592653589793, 0, 0],
        latent_dim=latent_dim, ff_size=ff_size,
        num_layers=num_layers, num_heads=num_heads,
        dropout=dropout, ablation=None, activation="gelu",
    )

    model = VaeacMotion(
        VaeacFullEncoder(**common),
        VaeacMaskedEncoder(**common),
        Decoder_TRANSFORMER(**common),
        latent_dim=latent_dim,
        njoints=17, nfeats=3,
        device=device,
        pose_rep="xyz",
        num_classes=num_classes,
        use_obs_indicator=use_obs_indicator,
    )

    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"[load] loaded VaeacMotion from {ckpt_path}  "
          f"(use_obs_indicator={use_obs_indicator})")
    return model


# ---------------------------------------------------------------------------
# Coalition catalogue — matches what evaluate_shap.py generates at test time
# ---------------------------------------------------------------------------

def build_coalition_catalogue(device: torch.device) -> list[dict]:
    """Return a list of {name, coalition_mask (1, 17)} dicts covering:

    1. Each of the 7 anatomical groups masked individually.
    2. Random Bernoulli coalitions at 5 sparsity levels (20 samples each).
    3. Single-joint hold-outs for every joint (17 total).
    """
    entries: list[dict] = []

    # ---- 1. Anatomical group masks ------------------------------------------
    for group_name, joint_ids in H36M_GROUPS.items():
        cm = torch.ones(1, 17, dtype=torch.bool, device=device)
        cm[0, joint_ids] = False
        n_held = len(joint_ids)
        entries.append({"name": f"group:{group_name}", "n_held": n_held,
                        "coalition_mask": cm})

    # ---- 2. Bernoulli masks at varied densities (10 per density) ------------
    rng = np.random.default_rng(42)
    for p_mask in (0.12, 0.25, 0.50, 0.75, 0.88):
        for i in range(10):
            keep = rng.random(17) >= p_mask
            if keep.sum() == 0:
                keep[0] = True            # guarantee at least one observed
            cm = torch.tensor(keep, dtype=torch.bool, device=device).unsqueeze(0)
            n_held = int((~cm[0]).sum().item())
            entries.append({
                "name": f"bernoulli_p{int(p_mask*100):02d}_{i:02d}",
                "n_held": n_held,
                "p_mask": p_mask,
                "coalition_mask": cm,
            })

    # ---- 3. Single-joint hold-outs ------------------------------------------
    for j in range(17):
        cm = torch.ones(1, 17, dtype=torch.bool, device=device)
        cm[0, j] = False
        entries.append({"name": f"single:{H36M_JOINT_NAMES[j]}", "n_held": 1,
                        "coalition_mask": cm})

    return entries


# ---------------------------------------------------------------------------
# Per-coalition metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_coalition(
    model: VaeacMotion,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    coalition_mask: torch.Tensor,
    n_samples: int = 20,
) -> dict[str, float]:
    """Compute diversity + reconstruction metrics for one coalition on one sequence.

    Returns
    -------
    dict with keys: masked_rmse, obs_rmse, diversity_raw, diversity_enc.
    """
    B, J, F, T = x.shape
    device = x.device

    # ---- draw K stochastic completions (paste_observed=True) ----------------
    completions = model.sample_completions(
        x, y, mask, lengths, coalition_mask,
        n_samples=n_samples, paste_observed=True,
    )
    K = len(completions)

    # ---- build masks ---------------------------------------------------------
    # held-out: shape (B, J, F, T)
    held_out = ~coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
    obs_mask = coalition_mask.unsqueeze(2).unsqueeze(3).expand(B, J, F, T)
    # apply sequence length mask
    real = mask.unsqueeze(1).unsqueeze(1).expand(B, J, F, T)

    held_real = (held_out & real).float()
    obs_real  = (obs_mask  & real).float()
    n_held    = held_real.sum().clamp(min=1.0)
    n_obs     = obs_real.sum().clamp(min=1.0)

    # ---- masked and observed RMSE (averaged over K completions) -------------
    masked_mse_list, obs_mse_list = [], []
    for x_hat in completions:
        diff_sq = (x_hat - x).pow(2)
        masked_mse_list.append(float((diff_sq * held_real).sum() / n_held))
        obs_mse_list.append(float((diff_sq * obs_real).sum() / n_obs))
    masked_rmse = float(np.sqrt(np.mean(masked_mse_list)))
    obs_rmse    = float(np.sqrt(np.mean(obs_mse_list)))

    # ---- diversity_raw: mean pairwise RMSE over masked joints ---------------
    # Stack completions → (K, J, F, T), then compute pairwise distances over
    # masked+real entries only.
    if K < 2:
        diversity_raw = 0.0
    else:
        stacked = torch.stack(completions, dim=0).squeeze(1)  # (K, J, F, T)
        mask_expand = held_real[0]  # (J, F, T)
        n_elem = mask_expand.sum().clamp(min=1.0)
        pairwise: list[float] = []
        for i in range(K):
            for j in range(i + 1, K):
                diff = ((stacked[i] - stacked[j]).pow(2) * mask_expand).sum() / n_elem
                pairwise.append(float(diff.sqrt()))
        diversity_raw = float(np.mean(pairwise)) if pairwise else 0.0

    # ---- diversity_enc: mean pairwise L2 in full-encoder latent space --------
    # The full encoder (q_ϕ) acts as a fixed feature extractor here, exactly as
    # compute_completion_diversity() uses model.encoder in the ActorSHAP version.
    full_cm = torch.ones(1, J, dtype=torch.bool, device=device)
    enc_mus: list[np.ndarray] = []
    for x_hat in completions:
        out = model.full_encoder({"x": x_hat, "y": y, "mask": mask,
                                  "coalition_mask": full_cm})
        enc_mus.append(out["mu_full"].squeeze(0).cpu().numpy())
    if K < 2:
        diversity_enc = 0.0
    else:
        feats = np.stack(enc_mus, axis=0)  # (K, D)
        pairwise_enc = [
            float(np.linalg.norm(feats[i] - feats[j]))
            for i in range(K)
            for j in range(i + 1, K)
        ]
        diversity_enc = float(np.mean(pairwise_enc))

    return {
        "masked_rmse":    masked_rmse,
        "obs_rmse":       obs_rmse,
        "diversity_raw":  diversity_raw,
        "diversity_enc":  diversity_enc,
        "div_over_rmse":  diversity_raw / max(masked_rmse, 1e-6),
    }


# ---------------------------------------------------------------------------
# Printing helpers (shared between single-process and merge paths)
# ---------------------------------------------------------------------------

def _print_table(
    results: dict[str, list[dict]],
    coa_meta: dict[str, dict],
    coalitions: list[dict],
    n_seq: int,
    n_samples: int,
    ckpt_path: str,
) -> None:
    print("\n" + "=" * 110)
    print(f"VaeacMotion — diversity vs fidelity  ({n_seq} seqs, K={n_samples} completions)")
    print(f"checkpoint: {ckpt_path}")
    print("=" * 110)

    print("\n--- Anatomical group masks (1 group held out) ---")
    hdr = (
        f"{'Coalition':<28}  {'n_held':>6}  "
        f"{'div_raw':>9}  {'div_enc':>9}  "
        f"{'mask_rmse':>9}  {'obs_rmse':>8}  "
        f"{'div/rmse':>9}"
    )
    print(hdr)
    print("-" * 110)
    for c in coalitions:
        if not c["name"].startswith("group:"):
            continue
        name = c["name"]
        seqm = results[name]
        if not seqm:
            continue
        div_r  = float(np.mean([m["diversity_raw"] for m in seqm]))
        div_e  = float(np.mean([m["diversity_enc"] for m in seqm]))
        mrmse  = float(np.mean([m["masked_rmse"]   for m in seqm]))
        ormse  = float(np.mean([m["obs_rmse"]       for m in seqm]))
        dor    = float(np.mean([m["div_over_rmse"]  for m in seqm]))
        n_held = coa_meta[name]["n_held"]
        print(f"  {name:<26}  {n_held:>6}  "
              f"{div_r:>9.5f}  {div_e:>9.4f}  "
              f"{mrmse:>9.5f}  {ormse:>8.5f}  "
              f"{dor:>9.4f}")

    print("\n--- Bernoulli masks by p_mask (10 random instances each) ---")
    print(f"{'p_mask':>8}  {'avg n_held':>10}  "
          f"{'div_raw':>9}  {'div_enc':>9}  "
          f"{'mask_rmse':>9}  {'obs_rmse':>8}  "
          f"{'div/rmse':>9}")
    print("-" * 110)
    for p_mask in (0.12, 0.25, 0.50, 0.75, 0.88):
        group_c  = [c for c in coalitions
                    if c["name"].startswith(f"bernoulli_p{int(p_mask*100):02d}")]
        all_seqm = [m for c in group_c for m in results[c["name"]]]
        if not all_seqm:
            continue
        avg_nheld = float(np.mean([coa_meta[c["name"]]["n_held"] for c in group_c]))
        div_r = float(np.mean([m["diversity_raw"] for m in all_seqm]))
        div_e = float(np.mean([m["diversity_enc"] for m in all_seqm]))
        mrmse = float(np.mean([m["masked_rmse"]   for m in all_seqm]))
        ormse = float(np.mean([m["obs_rmse"]       for m in all_seqm]))
        dor   = float(np.mean([m["div_over_rmse"]  for m in all_seqm]))
        print(f"  {p_mask:>6.0%}  {avg_nheld:>10.1f}  "
              f"{div_r:>9.5f}  {div_e:>9.4f}  "
              f"{mrmse:>9.5f}  {ormse:>8.5f}  "
              f"{dor:>9.4f}")

    single_c = [c for c in coalitions if c["name"].startswith("single:")]
    all_single = [m for c in single_c for m in results[c["name"]]]
    if all_single:
        div_r = float(np.mean([m["diversity_raw"] for m in all_single]))
        div_e = float(np.mean([m["diversity_enc"] for m in all_single]))
        mrmse = float(np.mean([m["masked_rmse"]   for m in all_single]))
        ormse = float(np.mean([m["obs_rmse"]       for m in all_single]))
        dor   = float(np.mean([m["div_over_rmse"]  for m in all_single]))
        print(f"\n--- Single-joint hold-outs (all 17 joints, averaged) ---")
        print(f"  n_held=1   div_raw={div_r:.5f}  div_enc={div_e:.4f}  "
              f"mask_rmse={mrmse:.5f}  obs_rmse={ormse:.5f}  div/rmse={dor:.4f}")

    print("\n--- Diversity vs masked_rmse by n_held (all coalition types pooled) ---")
    bins = [(1, 1), (2, 3), (4, 6), (7, 12), (13, 17)]
    print(f"{'n_held range':>14}  {'#evals':>7}  "
          f"{'div_raw':>9}  {'div_enc':>9}  "
          f"{'mask_rmse':>9}  {'div/rmse':>9}")
    print("-" * 80)
    for lo, hi in bins:
        relevant = [
            m
            for c in coalitions
            if lo <= coa_meta[c["name"]]["n_held"] <= hi
            for m in results[c["name"]]
        ]
        if not relevant:
            continue
        div_r = float(np.mean([m["diversity_raw"] for m in relevant]))
        div_e = float(np.mean([m["diversity_enc"] for m in relevant]))
        mrmse = float(np.mean([m["masked_rmse"]   for m in relevant]))
        dor   = float(np.mean([m["div_over_rmse"]  for m in relevant]))
        print(f"  [{lo:2d}-{hi:2d}]          {len(relevant):>7}  "
              f"{div_r:>9.5f}  {div_e:>9.4f}  "
              f"{mrmse:>9.5f}  {dor:>9.4f}")

    print("\nInterpretation guide:")
    print("  diversity_raw  — mean pairwise RMSE over masked joints across K completions.")
    print("                   Captures spatial spread directly in pose space.")
    print("  diversity_enc  — mean pairwise L2 in full-encoder latent space (like shap_metrics).")
    print("  masked_rmse    — average reconstruction error on held-out joints (lower = faithful).")
    print("  obs_rmse       — error on observed joints (~0 expected with paste_observed=True).")
    print("  div/rmse       — diversity relative to reconstruction error.")
    print("                   >>1: model is diverse AND accurate → good SHAP perturbations.")
    print("                   ~1 : diversity ≈ error floor  → just noise, not real uncertainty.")
    print("                   <1 : model is very accurate but mode-collapses → bad diversity.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate VaeacMotion diversity vs fidelity across spatial coalitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", default=None,
                    help="Path to vaeac_motion_last.ckpt (auto-detected if omitted).")
    ap.add_argument("--ckpt_base_dir", default="./experiment_outs/vaeac_motion",
                    help="Base dir to search for checkpoints when --ckpt is omitted.")
    ap.add_argument("--fold",      type=int, default=1)
    ap.add_argument("--num_folds", type=int, default=23)
    ap.add_argument("--dataset",   default="BMCLab")
    ap.add_argument("--n_seq",     type=int, default=15,
                    help="Total number of test sequences to evaluate (across all workers).")
    ap.add_argument("--n_samples", type=int, default=20,
                    help="Stochastic completions per coalition per sequence.")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",      type=int, default=0)
    # ---- parallel sharding --------------------------------------------------
    ap.add_argument("--num_workers", type=int, default=1,
                    help="Total number of parallel worker processes.")
    ap.add_argument("--worker_id", type=int, default=0,
                    help="Index of this worker (0-indexed, 0 .. num_workers-1).")
    ap.add_argument("--results_dir", default=None,
                    help="Directory for partial-results JSON files when num_workers > 1. "
                         "Defaults to <ckpt_dir>/sandbox_results/.")
    ap.add_argument("--merge", action="store_true",
                    help="Instead of running evaluation, merge all partial worker JSONs "
                         "from --results_dir and print the final table.")
    return ap.parse_args()


def _resolve_ckpt(args) -> str:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = _find_latest_ckpt(args.ckpt_base_dir, args.fold)
        if ckpt_path is None:
            raise SystemExit(
                f"No checkpoint found under {args.ckpt_base_dir}. "
                "Pass --ckpt explicitly."
            )
        print(f"[auto] using checkpoint: {ckpt_path}")
    return ckpt_path


def _results_dir_for(ckpt_path: str, results_dir_arg) -> str:
    if results_dir_arg:
        return results_dir_arg
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "sandbox_results")


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path   = _resolve_ckpt(args)
    results_dir = _results_dir_for(ckpt_path, args.results_dir)

    # ------------------------------------------------------------------ merge
    if args.merge:
        import glob as _glob
        pattern = os.path.join(results_dir, "worker_*.json")
        files   = sorted(_glob.glob(pattern))
        if not files:
            raise SystemExit(f"No worker JSON files found in {results_dir}")
        print(f"[merge] loading {len(files)} partial result files …")

        # Load a reference coalition catalogue (no GPU needed for merge).
        ref_device = torch.device("cpu")
        coalitions = build_coalition_catalogue(ref_device)
        coa_meta   = {c["name"]: {"n_held": c["n_held"]} for c in coalitions}
        merged: dict[str, list[dict]] = {c["name"]: [] for c in coalitions}
        total_seqs = 0
        for fpath in files:
            with open(fpath) as f:
                partial = json.load(f)
            for cname, rows in partial["results"].items():
                if cname in merged:
                    merged[cname].extend(rows)
            total_seqs += partial.get("n_seq_this_worker", 0)

        _print_table(merged, coa_meta, coalitions,
                     n_seq=total_seqs, n_samples=args.n_samples,
                     ckpt_path=ckpt_path)
        return

    # --------------------------------------------------------- worker / single
    device = torch.device(args.device)
    model  = load_vaeac_from_ckpt(ckpt_path, device)

    class _DataArgs:
        dataset          = args.dataset
        num_folds        = args.num_folds
        fold             = args.fold
        batch_size       = 1
        source_seq_len   = 81
        experiment_name  = "VaeacMotion"
        carepd_pose_npz  = None
        carepd_labels_pkl = None

    _, test_ds = get_carepd_datasets(_DataArgs())

    # Compute this worker's sequence slice.
    n_seq_total  = min(args.n_seq, len(test_ds))
    num_workers  = max(1, args.num_workers)
    worker_id    = args.worker_id
    # Distribute n_seq_total as evenly as possible.
    base, extra  = divmod(n_seq_total, num_workers)
    starts       = []
    s = 0
    for w in range(num_workers):
        starts.append(s)
        s += base + (1 if w < extra else 0)
    seq_start    = starts[worker_id]
    seq_end      = starts[worker_id + 1] if worker_id + 1 < num_workers else n_seq_total
    n_seq_worker = seq_end - seq_start

    tag = (f"worker {worker_id}/{num_workers-1} seqs [{seq_start},{seq_end})"
           if num_workers > 1 else f"seqs [0,{n_seq_total})")
    print(f"[data] test set: {len(test_ds)} sequences | this run: {tag}")

    loader = DataLoader(
        torch.utils.data.Subset(test_ds, list(range(seq_start, seq_end))),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    coalitions = build_coalition_catalogue(device)
    coa_meta   = {c["name"]: {"n_held": c["n_held"]} for c in coalitions}
    results: dict[str, list[dict]] = {c["name"]: [] for c in coalitions}

    for local_i, raw_batch in enumerate(loader):
        x_raw, labels, _, _, pad_mask = raw_batch
        b = actor_batch_from_carepd(
            x_raw.float().to(device), pad_mask.to(device),
            model.num_classes, device, y=labels,
        )
        x, y_t, mask_t, lengths = b["x"], b["y"], b["mask"], b["lengths"]

        for coa in coalitions:
            m = eval_coalition(model, x, y_t, mask_t, lengths,
                               coa["coalition_mask"], n_samples=args.n_samples)
            results[coa["name"]].append(m)

        print(f"  seq {seq_start + local_i + 1}/{n_seq_total} done", flush=True)

    # ---- save partial results or print directly ----------------------------
    if num_workers > 1:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"worker_{worker_id:04d}.json")
        payload = {
            "worker_id": worker_id,
            "seq_start": seq_start,
            "seq_end": seq_end,
            "n_seq_this_worker": n_seq_worker,
            "ckpt_path": ckpt_path,
            "n_samples": args.n_samples,
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f"[worker {worker_id}] saved {out_path}")
        print(f"  When all workers finish, run:\n"
              f"  python {os.path.relpath(__file__)} "
              f"--ckpt {ckpt_path} --results_dir {results_dir} --merge")
    else:

        _print_table(results, coa_meta, coalitions,
                     n_seq=n_seq_worker, n_samples=args.n_samples,
                     ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
