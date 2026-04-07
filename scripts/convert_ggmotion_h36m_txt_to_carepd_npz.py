#!/usr/bin/env python3
"""
Convert GGMotion-style Human3.6M ``.txt`` sequences (99-D exp maps per frame) to a
single compressed ``.npz`` in **CARE-PD major-joint order**: ``(T, 17, 3)`` float32 per sequence.

**Why not raw slicing?** Each line is **99 exponential-map parameters**, not XYZ. This script
runs the same forward kinematics as GGMotion (``utils.kinematics.fkl``) to obtain **32×3** joint
positions in **millimeters**, optionally scales to **meters**, then:

1. **MPI 32 → 17** — keep the standard “moving” joints (3d-pose-baseline / VideoPose3D convention)::

       [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

2. **Permute to CARE-PD order** — matches ``utility/Visualize_reconst3d.H36M_FULL`` (pelvis, left leg,
   right leg, spine, arms as used in this repo)::

       carepd[i] = mpi17[[0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13][i]]

**Requires:** a local checkout of **GGMotion** (or equivalent) whose ``utils/kinematics.py`` implements
``fkl`` — same as used for ``manifoldshap/data/h36m/h36m``.

Example::

    python scripts/convert_ggmotion_h36m_txt_to_carepd_npz.py \\
        --src /home/papillon/code/manifoldshap/data/h36m/h36m \\
        --out-npz assets/datasets/h36m/Human36M_public/carepd_order17.npz \\
        --ggmotion-root /home/papillon/code/GGMotion

Optional: ``--mirror-copy`` duplicates ``--src`` into ``--copy-dest`` before conversion (raw backup).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Joint topology (see conversation / 3d-pose-baseline H36M_NAMES, VideoPose3D remove_joints)
# ---------------------------------------------------------------------------
MPI_17_FROM_32 = np.array(
    [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27],
    dtype=np.int64,
)
# Permute MPI-17 (baseline listing order) → CARE-PD ``H36M_FULL`` / dataloader major-joint order
CAREPD_PERM = np.array(
    [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13],
    dtype=np.int64,
)


def _read_txt_sequence(path: Path) -> np.ndarray:
    """(T, D) float32, comma- or space-separated."""
    lines = path.read_text().splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",") if "," in line else line.split()
        rows.append([np.float32(x) for x in parts])
    if not rows:
        raise ValueError(f"Empty file: {path}")
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[1] != 99:
        raise ValueError(
            f"{path}: expected 99 coefficients per line (exp-map H36M), got {arr.shape[1]}"
        )
    return arr


def _ensure_ggmotion(ggmotion_root: Path):
    root = ggmotion_root.resolve()
    kin = root / "utils" / "kinematics.py"
    if not kin.is_file():
        raise FileNotFoundError(
            f"GGMotion not found at {root} (missing {kin}). "
            "Clone GGMotion or pass --ggmotion-root."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils.kinematics import fkl, _some_variables  # type: ignore

    return fkl, _some_variables()


def _expmap_to_xyz32(
    frames99: np.ndarray,
    fkl,
    parent,
    offset,
    rotInd,
    expmapInd,
    zero_global: bool,
) -> np.ndarray:
    """(T, 32, 3) float32, millimeters (FK internal units)."""
    t = frames99.shape[0]
    out = np.empty((t, 32, 3), dtype=np.float32)
    for i in range(t):
        a = frames99[i].copy()
        if zero_global:
            a[0:6] = 0.0
        xyz = fkl(a, parent, offset, rotInd, expmapInd)
        out[i] = np.asarray(xyz, dtype=np.float32)
    return out


def _to_carepd17(xyz32_mm: np.ndarray, scale_to_meters: float) -> np.ndarray:
    """(T, 17, 3) CARE-PD order, meters if scale_to_meters==1e-3."""
    xyz = xyz32_mm * np.float32(scale_to_meters)
    mpi17 = xyz[:, MPI_17_FROM_32, :]
    return mpi17[:, CAREPD_PERM, :].copy()


def _iter_sequence_files(src: Path, only_subjects: set[str] | None):
    for subdir in sorted(src.glob("S*")):
        if not subdir.is_dir():
            continue
        if only_subjects is not None and subdir.name not in only_subjects:
            continue
        for fp in sorted(subdir.glob("*.txt")):
            yield subdir.name, fp.stem, fp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--src",
        type=Path,
        default=Path("/home/papillon/code/manifoldshap/data/h36m/h36m"),
        help="Root folder containing S1/, S5/, … with *.txt sequences.",
    )
    p.add_argument(
        "--out-npz",
        type=Path,
        required=True,
        help="Output path, e.g. assets/datasets/h36m/Human36M_public/carepd17.npz",
    )
    p.add_argument(
        "--ggmotion-root",
        type=Path,
        default=Path(os.environ.get("GGMOTION_ROOT", "/home/papillon/code/GGMotion")),
    )
    p.add_argument(
        "--mirror-copy",
        action="store_true",
        help="If set, copy --src to --copy-dest first (full tree backup).",
    )
    p.add_argument(
        "--copy-dest",
        type=Path,
        default=None,
        help="Destination for mirror copy (required if --mirror-copy).",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="Skip sequences shorter than this (same spirit as smpl2h36m).",
    )
    p.add_argument(
        "--no-zero-global",
        action="store_true",
        help="Do not zero the first 6 exp-map coeffs (GGMotion training zeros these).",
    )
    p.add_argument(
        "--mm",
        action="store_true",
        help="Keep positions in millimeters instead of converting to meters.",
    )
    p.add_argument(
        "--only-subjects",
        type=str,
        default=None,
        help="Comma-separated subject folders to include, e.g. ``S1,S5``. Default: all.",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Stop after this many successfully converted sequences (smoke test).",
    )
    args = p.parse_args()

    if args.mirror_copy:
        if args.copy_dest is None:
            p.error("--mirror-copy requires --copy-dest")
        shutil.copytree(args.src, args.copy_dest, dirs_exist_ok=True)
        print(f"[info] Mirrored {args.src} → {args.copy_dest}")

    fkl, pack = _ensure_ggmotion(args.ggmotion_root)
    parent, offset, rotInd, expmapInd = pack

    scale = 1.0 if args.mm else 1e-3
    unit = "mm" if args.mm else "m"

    only = None
    if args.only_subjects:
        only = {s.strip() for s in args.only_subjects.split(",") if s.strip()}

    out: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    zero_global = not args.no_zero_global

    n_ok = 0
    for subject, stem, fp in _iter_sequence_files(args.src, only):
        key = f"{subject}__{stem}"
        try:
            raw = _read_txt_sequence(fp)
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")
            continue
        if raw.shape[0] < args.min_frames:
            print(f"[warn] skip {key}: only {raw.shape[0]} frames (< min_frames)")
            continue
        xyz32 = _expmap_to_xyz32(raw, fkl, parent, offset, rotInd, expmapInd, zero_global)
        seq = _to_carepd17(xyz32, scale)
        out[key] = seq
        meta.append(
            {
                "key": key,
                "source_file": str(fp.resolve()),
                "num_frames": int(seq.shape[0]),
                "shape": list(seq.shape),
                "units": unit,
            }
        )
        print(f"[ok] {key}  T={seq.shape[0]}  {unit}  (17,3) CARE-PD order")
        n_ok += 1
        if args.max_sequences is not None and n_ok >= args.max_sequences:
            break

    if not out:
        raise SystemExit("No sequences written; check --src and errors above.")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **out)
    manifest = args.out_npz.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "joint_mapping": {
                    "mpi_17_indices_into_32": MPI_17_FROM_32.tolist(),
                    "carepd_permute_from_mpi17": CAREPD_PERM.tolist(),
                    "reference": "3d-pose-baseline H36M_NAMES; CARE-PD H36M_FULL order",
                },
                "fk": {"ggmotion_root": str(args.ggmotion_root.resolve()), "zero_global_first6": zero_global},
                "sequences": meta,
            },
            indent=2,
        )
    )
    print(f"[done] wrote {args.out_npz} ({len(out)} sequences)")
    print(f"[done] manifest {manifest}")


if __name__ == "__main__":
    main()
