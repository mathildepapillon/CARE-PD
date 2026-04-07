#!/usr/bin/env python3
"""
Build a small or large synthetic H36M-layout .npz for testing ACTOR pre-training.

Does not require a Human3.6M license. Sequences are smooth random walks in joint
space (root-centered). Use only to verify ``train_actor_cvae.py``; for real
experiments, export real mocap into the same format (see data/datasets/h36m_actor_npz.py).

Usage::

    python scripts/build_synthetic_h36m_actor_npz.py \\
        --out artifacts/synthetic_h36m_actor.npz \\
        --num_sequences 5000 \\
        --frames 60
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="artifacts/synthetic_h36m_actor.npz")
    p.add_argument("--num_sequences", type=int, default=2000)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    n, t, j = args.num_sequences, args.frames, 17
    poses = np.zeros((n, t, j, 3), dtype=np.float32)
    for i in range(n):
        p0 = rng.randn(j, 3).astype(np.float32) * 0.15
        vel = rng.randn(t, j, 3).astype(np.float32) * 0.02
        vel = np.cumsum(vel, axis=0)
        poses[i] = p0[None, :, :] + vel
        poses[i] -= poses[i][:, :1, :]

    mask = np.ones((n, t), dtype=bool)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, poses=poses, mask=mask)
    print(f"Wrote {args.out}  poses={poses.shape}  mask={mask.shape}")


if __name__ == "__main__":
    main()
