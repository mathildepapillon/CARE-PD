"""data_loading.py — recover flow-space val clips + world pelvis trajectories.

The flow cache produced by :mod:`scripts.generate_velocity` stores:

- ``x1_val``   — (N, T, 17, 3) pelvis-centered z-scored clips.
- ``mask_val`` — (N, T) bool.
- ``meta_seq_key_val`` — (N,) the NPZ key each clip came from.
- ``meta_clip_idx_val`` — (N,) the ``make_clips`` index.

It does NOT store the world pelvis trajectory per clip (joint 0 was zeroed
before caching). OTFlow-SHAP's classifier adapter needs that trajectory to
re-glue the walking path back into the flow output (so the classifier sees
in-distribution data).

This module rebuilds the pelvis trajectory by replaying the exact clipping
logic from :mod:`scripts.generate_velocity`:

1. Load the raw dataset NPZ.
2. For every unique ``seq_key`` in ``meta_seq_key_val``, run ``make_clips``
   with the cached ``clip_stride_val`` and cache the full ``(n_clips, T,
   17, 3)`` array in RAM.
3. Index ``seq_key, clip_idx`` to pull the correct per-clip ``(T, 17, 3)``
   sequence, and keep only ``seq[:, 0:1, :]`` as the pelvis trajectory.

Matches the ordering / count of ``x1_val`` by construction (same NPZ iteration
order, same stride, same clip index semantics).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from scripts.generate_velocity import (
    DATASET_NPZ,
    make_clips,
    pelvis_centre,
    parse_seq_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pelvis trajectory recovery
# ---------------------------------------------------------------------------

def load_pelvis_world_for_val(
    dataset: str,
    meta_seq_key_val: np.ndarray,   # (N,) object array of NPZ keys
    meta_clip_idx_val: np.ndarray,  # (N,) int32
    seq_len: int,
    clip_stride_val: int,
) -> np.ndarray:
    """Return ``(N, T, 3)`` world pelvis trajectory per val clip.

    Replays ``make_clips`` on the raw dataset NPZ using the same
    ``clip_stride_val`` the flow cache used. Joint-0 of each clip is extracted
    (still in world coordinates because ``pelvis_centre`` is applied AFTER we
    extract the clip; see the original cached clips in ``generate_velocity``).

    NB: ``generate_velocity`` calls ``pelvis_centre`` BEFORE ``make_clips``,
    which means the cached flow-space clips have a pelvis at 0. To recover the
    world pelvis, we call ``make_clips`` on the *un-centred* raw sequence and
    take ``clip[:, 0, :]``.
    """
    npz_path = PROJECT_ROOT / DATASET_NPZ[dataset]
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {npz_path}")
    npz = np.load(str(npz_path), allow_pickle=False)

    unique_keys = list(dict.fromkeys(meta_seq_key_val.tolist()))
    per_seq_cache: Dict[str, np.ndarray] = {}
    for key in unique_keys:
        seq_world = npz[key].astype(np.float32)                 # (T_raw, 17, 3)
        if seq_world.ndim != 3 or seq_world.shape[1] != 17 or seq_world.shape[2] != 3:
            raise ValueError(
                f"Unexpected shape for NPZ key {key!r}: {seq_world.shape}"
            )
        clips_world, _ = make_clips(seq_world, seq_len, clip_stride_val)
        per_seq_cache[key] = clips_world[:, :, 0:1, :].squeeze(-2)  # (n_clips, T, 3)

    N = int(meta_seq_key_val.shape[0])
    pelvis = np.zeros((N, seq_len, 3), dtype=np.float32)
    for i in range(N):
        key = str(meta_seq_key_val[i])
        idx = int(meta_clip_idx_val[i])
        arr = per_seq_cache[key]
        if idx >= arr.shape[0]:
            raise IndexError(
                f"clip_idx {idx} out of range for seq_key {key!r} "
                f"(only {arr.shape[0]} clips)"
            )
        pelvis[i] = arr[idx]
    return pelvis


# ---------------------------------------------------------------------------
# Flow-cache loader
# ---------------------------------------------------------------------------

def load_flow_cache(
    cache_path: str | Path,
    split: str = "val",
) -> Dict[str, np.ndarray]:
    """Load a flow cache NPZ and slice out the requested split.

    Returns a dict with the same naming convention as the cache but without the
    ``_train`` / ``_val`` suffix, for convenience:

    ``{"x1", "mask", "meta_pid", "meta_walk_id", "meta_seq_key",
       "meta_clip_idx", "meta_updrs_gait", "meta_medication",
       "stats_mean", "stats_std", "seq_len", "clip_stride_val",
       "dataset", ...}``
    """
    cache = np.load(str(cache_path), allow_pickle=True)
    sfx = "_" + split

    # Split-suffixed keys we want to strip and expose as un-suffixed:
    # x1, mask, and any ``meta_*``.
    def _is_split_data_key(k: str) -> bool:
        return (
            k == f"x1{sfx}"
            or k == f"mask{sfx}"
            or k.startswith("meta_")
        )

    out: Dict[str, np.ndarray] = {}
    for k in cache.files:
        if _is_split_data_key(k) and k.endswith(sfx):
            out[k[: -len(sfx)]] = cache[k]
        elif k.startswith("meta_"):
            # Other-split meta — skip.
            continue
        elif k == f"x1{sfx}" or k == f"mask{sfx}":
            # Already handled above; no-op.
            continue
        elif k.startswith("x1_") or k.startswith("mask_"):
            # Other-split data — skip.
            continue
        else:
            out[k] = cache[k]
    required = {"x1", "mask", "meta_seq_key", "meta_clip_idx",
                "clip_stride_val", "stats_mean", "stats_std", "seq_len"}
    missing = required - set(out.keys())
    if missing:
        raise KeyError(
            f"Flow cache {cache_path} missing keys for split={split!r}: {sorted(missing)}"
        )
    return out


# ---------------------------------------------------------------------------
# Batched iterator
# ---------------------------------------------------------------------------

def iter_flow_shap_batches(
    *,
    x1: np.ndarray,             # (N, T, 17, 3)
    mask: np.ndarray,           # (N, T) bool
    pelvis_world: np.ndarray,   # (N, T, 3) metres
    batch_size: int,
    indices: List[int] | None = None,
) -> Iterator[Tuple[List[int], Dict[str, np.ndarray]]]:
    """Yield ``(indices, batch_dict)`` tuples for the driver.

    Each batch_dict contains numpy arrays ready to be moved to device:
    ``{"x_star_flow", "mask", "pelvis_world"}``.
    """
    N = x1.shape[0]
    if indices is None:
        indices = list(range(N))
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start:start + batch_size]
        yield batch_ids, {
            "x_star_flow": x1[batch_ids],
            "mask":        mask[batch_ids],
            "pelvis_world": pelvis_world[batch_ids],
        }
