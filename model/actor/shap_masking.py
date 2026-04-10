"""shap_masking.py — Coalition mask sampling for ActorSHAP (spatial and temporal axes).

Implements SHAP player definitions and training mask distributions described in the
ActorSHAP plan. Spatial players are 17 individual H36M joints; temporal players are
K=4 stride-aligned gait-phase windows detected via foot-pelvis autocorrelation.

No model imports — pure NumPy/PyTorch utilities.

Dependencies: numpy, torch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# -------------------------------------------------------------------
# H36M joint registry
# -------------------------------------------------------------------

H36M_JOINT_NAMES: list[str] = [
    "Pelvis",                               # 0
    "RHip", "RKnee", "RAnkle",             # 1-3
    "LHip", "LKnee", "LAnkle",             # 4-6
    "Spine", "Thorax", "Neck",              # 7-9
    "Head",                                 # 10
    "LShoulder", "LElbow", "LWrist",        # 11-13
    "RShoulder", "RElbow", "RWrist",        # 14-16
]

# Used only post-hoc to aggregate individual Shapley values into clinical groups.
# SHAP players are always individual joints (17), never these groups.
H36M_GROUPS: dict[str, list[int]] = {
    "root":      [0],
    "right_leg": [1, 2, 3],
    "left_leg":  [4, 5, 6],
    "spine":     [7, 8, 9],
    "head":      [10],
    "left_arm":  [11, 12, 13],
    "right_arm": [14, 15, 16],
}


# -------------------------------------------------------------------
# Spatial masks — 17 individual H36M joints
# -------------------------------------------------------------------

def sample_spatial_training_mask(
    B: int,
    device: torch.device,
    n_joints: int = 17,
) -> torch.BoolTensor:
    """Sample random joint coalition masks for a training batch.

    Mixes two distributions so the masked encoder sees both independent-joint
    and structured coalitions:
    - ~67%: independent Bernoulli per joint, p_mask ~ Uniform(0.1, 0.9).
    - ~33%: one anatomical group fully masked (covers inference-time structure).

    Args:
        B:        batch size.
        device:   target device.
        n_joints: number of joints (17 for H36M).

    Returns:
        BoolTensor (B, n_joints) — True = joint is *observed* (unmasked).
    """
    masks = torch.ones(B, n_joints, dtype=torch.bool, device=device)
    group_list = list(H36M_GROUPS.values())
    for i in range(B):
        if torch.rand(1).item() < 0.33:
            # Mask one anatomical group entirely.
            idx = int(torch.randint(len(group_list), (1,)).item())
            masks[i, group_list[idx]] = False
        else:
            p_mask = 0.1 + 0.8 * torch.rand(1).item()
            keep = torch.bernoulli(torch.full((n_joints,), 1.0 - p_mask)).bool()
            masks[i] = keep
            # Guarantee at least one observed joint to prevent degenerate inputs.
            if masks[i].sum() == 0:
                masks[i, 0] = True
    return masks


def build_spatial_shap_mask(
    observed_joints: list[int] | np.ndarray,
    device: torch.device,
    n_joints: int = 17,
) -> torch.BoolTensor:
    """Build a (n_joints,) coalition mask from observed joint indices.

    Args:
        observed_joints: indices of joints that are observed (not replaced).
        device:          target device.
        n_joints:        total joint count.

    Returns:
        BoolTensor (n_joints,) — True = observed.
    """
    m = torch.zeros(n_joints, dtype=torch.bool, device=device)
    m[list(observed_joints)] = True
    return m


# -------------------------------------------------------------------
# Temporal masks — stride-aligned windows
# -------------------------------------------------------------------

def _find_first_peak(acf: np.ndarray, min_lag: int = 5) -> int:
    """Return the lag of the first local maximum > 0.3 in a normalised ACF."""
    for i in range(min_lag, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.3:
            return i
    return len(acf) // 2


def detect_stride_period(
    x_xyz: np.ndarray,
    fps: int = 30,
) -> tuple[int, bool]:
    """Detect stride period from foot anterior-posterior distance to pelvis.

    Uses the maximum of the two feet's forward projection relative to the pelvis
    along the principal walking direction. Robust for PD gait because even a
    severe shuffler steps forward relative to the pelvis, unlike pelvis vertical
    displacement which may be nearly flat.

    H36M joints used: 0=Pelvis, 3=RAnkle, 6=LAnkle.
    Walking direction = first PC of the pelvis trajectory (SVD), so the method
    is invariant to arbitrary facing direction in the capture volume.

    Args:
        x_xyz: (T, J=17, 3) xyz positions.
        fps:   capture frame rate (not used in computation; reserved for docs).

    Returns:
        (stride_period_frames, stride_fallback)
        stride_fallback is True when autocorrelation found no clear peak and
        T // 2 was returned as a fallback.
    """
    T = x_xyz.shape[0]
    pelvis = x_xyz[:, 0, :]  # (T, 3)
    r_foot = x_xyz[:, 3, :]  # (T, 3) RAnkle
    l_foot = x_xyz[:, 6, :]  # (T, 3) LAnkle

    # Walking direction = first PC of the pelvis trajectory.
    pelvis_centred = pelvis - pelvis.mean(axis=0)
    _, _, Vt = np.linalg.svd(pelvis_centred, full_matrices=False)
    walk_dir = Vt[0]  # (3,)

    r_fwd = (r_foot - pelvis) @ walk_dir  # (T,)
    l_fwd = (l_foot - pelvis) @ walk_dir  # (T,)

    # max(feet): one peak per stride at heel strike of the leading foot,
    # avoiding the double-peak that arises when treating each foot separately.
    foot_signal = np.maximum(r_fwd, l_fwd)
    foot_signal -= foot_signal.mean()

    acf = np.correlate(foot_signal, foot_signal, mode="full")
    acf = acf[T - 1:]  # positive lags only
    if acf[0] > 0:
        acf = acf / acf[0]  # normalise to [-1, 1]

    fallback_period = T // 2
    period = _find_first_peak(acf, min_lag=5)
    stride_fallback = period == fallback_period

    return int(period), stride_fallback


def build_temporal_windows(T: int, stride_period: int, K: int = 4) -> list[list[int]]:
    """Partition T frames into K gait-phase-aligned windows.

    The four windows correspond to the canonical gait phases:
      0: initial contact + loading response  (0–30% of stride)
      1: mid-stance + terminal stance        (30–60%)
      2: pre-swing + initial swing           (60–80%)
      3: mid-swing + terminal swing          (80–100%)

    The pattern repeats for sequences covering more than one stride.

    Args:
        T:             total number of frames.
        stride_period: stride period in frames (from detect_stride_period).
        K:             number of windows (must be 4).

    Returns:
        List of K lists; each inner list contains the frame indices for that window.
    """
    assert K == 4, "Only K=4 gait-phase windows are supported."
    boundaries = [0.0, 0.30, 0.60, 0.80, 1.00]
    windows: list[list[int]] = [[] for _ in range(K)]
    for t in range(T):
        phase = (t % stride_period) / stride_period
        for k in range(K):
            if boundaries[k] <= phase < boundaries[k + 1]:
                windows[k].append(t)
                break
        else:
            windows[K - 1].append(t)  # safety: place overflow in last window
    return windows


def sample_temporal_training_mask(
    B: int,
    T: int,
    device: torch.device,
    K: int = 4,
) -> torch.BoolTensor:
    """Sample random temporal coalition masks at training time.

    Uses K equal-length quarters (fallback mode) to avoid per-batch stride
    detection. Each window is independently masked with p=0.5.

    Args:
        B:      batch size.
        T:      number of frames.
        device: target device.
        K:      number of windows.

    Returns:
        BoolTensor (B, T) — True = frame is *observed*.
    """
    quarter = T // K
    window_assignments: list[list[int]] = []
    for k in range(K):
        start = k * quarter
        end = (k + 1) * quarter if k < K - 1 else T
        window_assignments.append(list(range(start, end)))

    masks = torch.ones(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        for frames in window_assignments:
            if torch.rand(1).item() < 0.5:
                masks[i, frames] = False
        # Guarantee at least one observed frame.
        if masks[i].sum() == 0:
            masks[i, window_assignments[0]] = True
    return masks


def build_temporal_shap_mask(
    observed_windows: list[int],
    window_assignments: list[list[int]],
    T: int,
    device: torch.device,
) -> torch.BoolTensor:
    """Build a (T,) coalition mask from a list of observed window indices.

    Args:
        observed_windows:   indices of windows that are observed (not replaced).
        window_assignments: K lists of frame indices (from build_temporal_windows).
        T:                  total number of frames.
        device:             target device.

    Returns:
        BoolTensor (T,) — True = frame is observed.
    """
    m = torch.zeros(T, dtype=torch.bool, device=device)
    for k in observed_windows:
        m[window_assignments[k]] = True
    return m


# -------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------

if __name__ == "__main__":
    B, J, T = 8, 17, 60
    dev = torch.device("cpu")

    # Spatial masks
    smask = sample_spatial_training_mask(B, dev, n_joints=J)
    assert smask.shape == (B, J) and smask.dtype == torch.bool

    obs = [0, 1, 3, 7, 14]
    smask1 = build_spatial_shap_mask(obs, dev)
    assert smask1.shape == (J,)
    assert smask1[obs].all() and not smask1[2].item()
    print("Spatial masks OK")

    # Temporal training mask
    tmask = sample_temporal_training_mask(B, T, dev, K=4)
    assert tmask.shape == (B, T) and tmask.dtype == torch.bool
    print("Temporal training mask OK")

    # detect_stride_period on synthetic sinusoidal foot signal
    stride_gt = 28
    t_arr = np.arange(T)
    x_syn = np.zeros((T, 17, 3))
    x_syn[:, 0, 0] = t_arr * 0.02  # pelvis moves forward along x
    r_sin = np.sin(2 * np.pi * t_arr / stride_gt)
    l_sin = np.sin(2 * np.pi * t_arr / stride_gt + np.pi)
    x_syn[:, 3, 0] = x_syn[:, 0, 0] + r_sin * 0.4  # RAnkle
    x_syn[:, 6, 0] = x_syn[:, 0, 0] + l_sin * 0.4  # LAnkle

    period, fallback = detect_stride_period(x_syn)
    assert not fallback, f"Unexpected stride fallback; period={period}"
    assert 10 <= period <= 59, f"Stride period out of range: {period}"
    print(f"detect_stride_period OK: period={period} (ground truth {stride_gt})")

    # build_temporal_windows covers all frames
    windows = build_temporal_windows(T, period)
    all_frames = sorted(f for w in windows for f in w)
    assert all_frames == list(range(T)), "Windows do not cover all frames"
    print("build_temporal_windows OK")

    # build_temporal_shap_mask
    tshap = build_temporal_shap_mask([0, 2], windows, T, dev)
    assert tshap.shape == (T,) and tshap.dtype == torch.bool
    print("build_temporal_shap_mask OK")

    print("OK")
