"""diagnostic_motion.py — Synthetic gait benchmark with known diagnostic joints.

PURPOSE
-------
Show that ActorSHAP assigns the highest Shapley values to the joints that
actually carry label-relevant signal — and recovers them more accurately than
zero / mean / marginal imputation baselines.

DATA GENERATION
---------------
Each sequence is:

    x[j, f, t] = base_gait[j, f, t]  +  signal_strength[j] * label * sin(2π t/T)  +  ε

where:
    base_gait     — realistic Fourier-series gait cycle (joint-specific amplitudes
                    and phase offsets, same for all sequences)
    signal_strength[j]  — non-zero only for diagnostic joints; zero for others
    label         — scalar in [-1, +1] drawn from N(0, 1) (continuous severity)
    ε             — i.i.d. Gaussian noise (same std σ_noise for all joints)

DIAGNOSTIC JOINTS (H36M)
------------------------
    group "right_leg": joints 1 (RHip), 2 (RKnee), 3 (RAnkle)
    group "spine":     joint  7 (Spine)

Clinically motivated: PD primarily affects gait symmetry (right vs left leg)
and spine posture.  All other joints carry only noise + base gait.

CLASSIFIER
----------
    f(x) = Σ_j  w_j * mean_velocity(x_j)

where mean_velocity(x_j) = mean_over_t( |x[j,:,t+1] - x[j,:,t]| ).

Because f is linear in the velocity statistics, E[f(x) | x_S] reduces to a
linear function of the observed joint velocities — computable analytically via
the Gaussian conditional mean formula.

TRUE SHAPLEY VALUES (linear + Gaussian)
-----------------------------------------
For a linear f(x) = Σ_j w_j * h_j(x_j) where h_j is a deterministic function
of joint j's trajectory:

    v(S)  = Σ_{j ∈ S} w_j * h_j(x_j*)
           + Σ_{j ∉ S} w_j * E[h_j(x_j) | x_S]

For Gaussian data with equicorrelation (rho) and independent joints if rho=0,
E[h_j(x_j) | x_S] reduces to the marginal mean (zero-centered data) when rho=0,
or a linear combination of observed joint statistics when rho > 0.

For simplicity the diagnostic benchmark uses INDEPENDENT JOINTS (rho=0).  Then:

    E[h_j(x_j) | x_S] = E[h_j(x_j)]   (unconditional mean, = 0 for zero-mean data)

So:
    v(S) = Σ_{j ∈ S} w_j * h_j(x_j*)

And the true Shapley value of joint j is:

    phi_j = w_j * (h_j(x_j*) - E[h_j(x_j)])

This is EXACT — no Monte Carlo needed.  The baselines (zero, mean, marginal)
recover phi_j well only if their imputed h_j(x̂_j) is close to E[h_j(x_j)].
Zero imputation gives h_j = 0 (correct for zero-mean data on average but
off-manifold — the reconstructed velocity is wrong because the motion is clipped).
Mean imputation gives the training-set mean velocity — correct on average.
Marginal gives random draws — noisy.
ActorSHAP generates motion-consistent completions — produces correct velocity
statistics even for held-out joints, leading to more accurate phi estimates.

USAGE
-----
    from synthetic.diagnostic_motion import (
        generate_diagnostic_gait,
        LinearDiagnosticClassifier,
    )

    x, labels, w_true = generate_diagnostic_gait(N=500, seed=0)
    clf = LinearDiagnosticClassifier(w_true)
    phi_true = clf.compute_true_shapley(x[0])  # (17,)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# H36M joint indices (must stay consistent with model/actor/shap_masking.py)
J_TOTAL = 17
H36M_JOINT_NAMES = [
    "Pelvis",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck",
    "Head",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
]

# Diagnostic joints: right leg + spine.
DIAGNOSTIC_JOINTS = [1, 2, 3, 7]      # RHip, RKnee, RAnkle, Spine
NON_DIAGNOSTIC_JOINTS = [j for j in range(J_TOTAL) if j not in DIAGNOSTIC_JOINTS]

# Base gait amplitudes (in metres, roughly realistic for H36M scale).
# Right leg joints: hip ≈ 0.10, knee ≈ 0.18, ankle ≈ 0.12
# Left leg joints:  symmetric
# Spine: ≈ 0.04 (subtle lateral sway)
_GAIT_AMPLITUDE = {
    0: 0.00,   # Pelvis: root stays near origin (handled separately)
    1: 0.10, 2: 0.18, 3: 0.12,   # Right leg
    4: 0.10, 5: 0.18, 6: 0.12,   # Left leg
    7: 0.04, 8: 0.02, 9: 0.01,   # Spine, Thorax, Neck
    10: 0.01,                      # Head
    11: 0.08, 12: 0.14, 13: 0.10, # Left arm
    14: 0.08, 15: 0.14, 16: 0.10, # Right arm
}

# Phase offsets between joints (radians); right/left symmetric with π shift.
_GAIT_PHASE = {
    0: 0.0,
    1: 0.0,   2: 0.4,   3: 0.8,    # Right leg (hip→knee→ankle progression)
    4: np.pi, 5: np.pi + 0.4, 6: np.pi + 0.8,  # Left leg (opposite phase)
    7: 0.2,   8: 0.3,   9: 0.4,    # Spine sway
    10: 0.4,
    11: np.pi, 12: np.pi + 0.3, 13: np.pi + 0.6,
    14: 0.0,  15: 0.3,  16: 0.6,
}

# Noise std (metres).
_NOISE_STD = 0.015


def _base_gait(T: int, n_cycles: float = 2.0) -> np.ndarray:
    """Return (J, 3, T) base gait trajectory (no diagnostic signal, no noise).

    Each joint oscillates in its dominant plane with a sinusoidal pattern.
    The three coordinates (x, y, z) use slightly different phase offsets to
    create a 3-D trajectory.
    """
    t = np.linspace(0.0, 2.0 * np.pi * n_cycles, T)
    gait = np.zeros((J_TOTAL, 3, T), dtype=np.float32)
    for j in range(J_TOTAL):
        amp = _GAIT_AMPLITUDE[j]
        phi = _GAIT_PHASE[j]
        # x (sagittal), y (vertical, half amplitude), z (lateral, quarter amplitude)
        gait[j, 0, :] = amp * np.sin(t + phi).astype(np.float32)
        gait[j, 1, :] = (amp * 0.5) * np.cos(t + phi).astype(np.float32)
        gait[j, 2, :] = (amp * 0.25) * np.sin(2.0 * (t + phi)).astype(np.float32)
    return gait


def _mean_velocity(x: np.ndarray) -> np.ndarray:
    """Compute mean L2 velocity per joint.

    Parameters
    ----------
    x : (J, F, T) or (N, J, F, T)

    Returns
    -------
    v : (J,) or (N, J)
    """
    if x.ndim == 3:
        diffs = np.diff(x, axis=-1)          # (J, F, T-1)
        return np.linalg.norm(diffs, axis=1).mean(axis=-1)  # (J,)
    elif x.ndim == 4:
        diffs = np.diff(x, axis=-1)          # (N, J, F, T-1)
        return np.linalg.norm(diffs, axis=2).mean(axis=-1)  # (N, J)
    else:
        raise ValueError(f"Expected 3- or 4-D array, got {x.ndim}")


def _diagnostic_signal_strength(
    diagnostic_joints: list[int] = DIAGNOSTIC_JOINTS,
    signal_scale: float = 0.12,
) -> np.ndarray:
    """Return (J,) array of signal strengths; zero for non-diagnostic joints."""
    w = np.zeros(J_TOTAL, dtype=np.float32)
    for j in diagnostic_joints:
        w[j] = signal_scale
    return w


def generate_diagnostic_gait(
    N: int,
    J: int = J_TOTAL,
    F: int = 3,
    T: int = 81,
    n_cycles: float = 2.0,
    signal_scale: float = 0.12,
    noise_std: float = _NOISE_STD,
    seed: int = 0,
    diagnostic_joints: list[int] = DIAGNOSTIC_JOINTS,
    num_classes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate N synthetic diagnostic gait sequences.

    Each sequence:
        x[j, f, t] = base_gait[j, f, t]
                   + signal_strength[j] * label_scalar * sin(2π n_cycles t/T)
                   + ε[j, f, t]

    where label_scalar ~ N(0, 1) independently per sequence.

    Parameters
    ----------
    N : int — number of sequences.
    J, F, T : int — joints, features, frames.
    n_cycles : float — gait cycles per sequence.
    signal_scale : float — amplitude of diagnostic signal (metres).
    noise_std : float — per-frame i.i.d. noise std.
    seed : int — random seed.
    diagnostic_joints : list[int] — joints carrying the label signal.
    num_classes : int — number of label classes (1 = regression, >1 = classification).

    Returns
    -------
    x      : (N, J, F, T) float32 — motion sequences.
    labels : (N,) int64 — class labels in {0, ..., num_classes-1}.
    w_true : (J,) float32 — true linear-classifier weights (signal strength per joint).
    """
    rng = np.random.default_rng(seed)
    base = _base_gait(T, n_cycles=n_cycles)  # (J, 3, T)
    signal_strength = _diagnostic_signal_strength(diagnostic_joints, signal_scale)

    # Diagnostic modulation signal (same shape for all joints, scaled per-joint)
    t = np.linspace(0.0, 2.0 * np.pi * n_cycles, T)
    modulation = np.sin(t).astype(np.float32)  # (T,)

    label_scalars = rng.standard_normal(N).astype(np.float32)  # N(0,1)
    noise = rng.standard_normal((N, J, F, T)).astype(np.float32) * noise_std

    x = np.zeros((N, J, F, T), dtype=np.float32)
    for n in range(N):
        x[n] = base.copy()
        # Add diagnostic signal for each diagnostic joint.
        for j in diagnostic_joints:
            x[n, j, :, :] += signal_strength[j] * label_scalars[n] * modulation[None, :]
        x[n] += noise[n]

    # Convert continuous severity label → class indices.
    q_bounds = np.percentile(label_scalars, np.linspace(0, 100, num_classes + 1)[1:-1])
    labels = np.searchsorted(q_bounds, label_scalars).astype(np.int64)

    # True classifier weight = signal_strength per joint (drives velocity differences).
    w_true = signal_strength.copy()

    return x, labels, w_true


# ---------------------------------------------------------------------------
# Linear diagnostic classifier
# ---------------------------------------------------------------------------

class LinearDiagnosticClassifier(nn.Module):
    """f(x) = Σ_j w_j * mean_velocity(x_j) — known linear function.

    The weights w_j are set to the signal strength per joint so that diagnostic
    joints have higher weight than non-diagnostic joints.

    TRUE SHAPLEY VALUES (independent joints, zero-mean data)
    ---------------------------------------------------------
    For rho = 0 (independent joints) and zero-mean data:

        E[h_j(x_j) | x_S] = E[h_j(x_j)]   (marginal expectation)

    The marginal expectation of mean_velocity(x_j) is positive (velocity is
    non-negative) and equals the training-set mean velocity for joint j.

    Therefore:
        v(S)   = Σ_{j ∈ S} w_j * h_j(x_j*)
                + Σ_{j ∉ S} w_j * μ_j        where μ_j = E[h_j(x_j)]
        v(∅)   = Σ_j w_j * μ_j
        v(M)   = Σ_j w_j * h_j(x_j*)  = f(x*)

    Shapley value of joint j:
        phi_j = w_j * (h_j(x_j*) - μ_j)

    This is exactly the "dummy player" property of Shapley values for a linear
    additive model — each feature contributes proportionally to its deviation
    from the marginal mean.
    """

    def __init__(self, w: np.ndarray):
        """
        Parameters
        ----------
        w : (J,) array of per-joint weights.
        """
        super().__init__()
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))

    def _velocities(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, J, F, T) → per-joint mean velocity (B, J)."""
        diffs = x[..., 1:] - x[..., :-1]          # (B, J, F, T-1)
        return diffs.norm(dim=2).mean(dim=-1)       # (B, J)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B,) scalar predictions."""
        vels = self._velocities(x)                  # (B, J)
        return (vels * self.w[None, :]).sum(dim=-1) # (B,)

    def compute_true_shapley(
        self,
        x_single: np.ndarray,
        mu_j: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute exact per-joint Shapley values for a single sequence.

        Parameters
        ----------
        x_single : (J, F, T) numpy array.
        mu_j     : (J,) marginal mean velocities.  If None, assumes zero (true
                   for zero-mean data; call fit_marginal_means first).

        Returns
        -------
        phi : (J,) Shapley values.
        """
        if mu_j is None:
            mu_j = np.zeros(J_TOTAL, dtype=np.float32)
        w = self.w.cpu().numpy()
        h = _mean_velocity(x_single)    # (J,)
        return w * (h - mu_j)

    def fit_marginal_means(self, x_train: np.ndarray) -> np.ndarray:
        """Compute and store the training-set marginal mean velocity per joint.

        Parameters
        ----------
        x_train : (N, J, F, T)

        Returns
        -------
        mu_j : (J,) stored in self.mu_j buffer.
        """
        mu_j = _mean_velocity(x_train).mean(axis=0).astype(np.float32)  # (J,)
        self.register_buffer("mu_j", torch.tensor(mu_j))
        return mu_j

    def true_shapley_batch(
        self, x_batch: np.ndarray, mu_j: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute exact Shapley values for a batch.

        Parameters
        ----------
        x_batch : (N, J, F, T)
        mu_j    : (J,) marginal mean velocities.

        Returns
        -------
        phi : (N, J)
        """
        if mu_j is None:
            if hasattr(self, "mu_j"):
                mu_j = self.mu_j.cpu().numpy()
            else:
                mu_j = np.zeros(J_TOTAL, dtype=np.float32)
        w = self.w.cpu().numpy()
        h = _mean_velocity(x_batch)    # (N, J)
        return w[None, :] * (h - mu_j[None, :])

    def prob_fn(self) -> object:
        """Return a callable compatible with compute_spatial_shap.

        The spatial SHAP infrastructure expects
            classifier(x: Tensor(B,J,F,T)) → Tensor(B, C) or Tensor(B,)
        We return the scalar f(x) directly (treated as regression / SHAP
        approximates the continuous output rather than a class probability).
        """
        def fn(x: torch.Tensor) -> torch.Tensor:
            return self(x)
        return fn


# ---------------------------------------------------------------------------
# PyTorch dataset builder
# ---------------------------------------------------------------------------

def build_diagnostic_dataset(
    N: int = 500,
    T: int = 81,
    signal_scale: float = 0.12,
    seed: int = 0,
    num_classes: int = 3,
) -> tuple[
    "LinearDiagnosticClassifier",
    np.ndarray,
    np.ndarray,
    TensorDataset,
    TensorDataset,
    TensorDataset,
]:
    """Generate data, build classifier, split into train/val/test.

    Returns
    -------
    clf          : LinearDiagnosticClassifier (weights pre-set, marginal means fit on train)
    x_test       : (n_test, J, F, T) numpy array for evaluation
    phi_true_test: (n_test, J) true Shapley values for each test sequence
    train_ds, val_ds, test_ds : TensorDatasets in ACTOR (T-first) format
    """
    n_test = max(50, N // 10)
    n_val  = max(50, N // 10)
    n_tr   = N - n_test - n_val

    x_all, y_all, w_true = generate_diagnostic_gait(
        N, T=T, signal_scale=signal_scale, seed=seed, num_classes=num_classes
    )
    x_tr, y_tr = x_all[:n_tr],  y_all[:n_tr]
    x_va, y_va = x_all[n_tr:n_tr+n_val], y_all[n_tr:n_tr+n_val]
    x_te, y_te = x_all[n_tr+n_val:],     y_all[n_tr+n_val:]

    clf = LinearDiagnosticClassifier(w_true)
    mu_j = clf.fit_marginal_means(x_tr)
    phi_true_test = clf.true_shapley_batch(x_te, mu_j=mu_j)  # (n_test, J)

    def _to_ds(x_arr, y_arr):
        xt = torch.tensor(x_arr).permute(0, 3, 1, 2).contiguous()  # (N, T, J, F)
        yt = torch.tensor(y_arr, dtype=torch.long)
        pm = torch.ones(len(x_arr), T, dtype=torch.bool)
        return TensorDataset(xt, yt, pm)

    return (
        clf,
        x_te,
        phi_true_test,
        _to_ds(x_tr, y_tr),
        _to_ds(x_va, y_va),
        _to_ds(x_te, y_te),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x, labels, w_true = generate_diagnostic_gait(N=50, T=32, seed=0)
    assert x.shape == (50, 17, 3, 32), x.shape
    assert labels.shape == (50,), labels.shape
    assert w_true.shape == (17,), w_true.shape
    assert w_true[1] > 0 and w_true[0] == 0, "diagnostic/non-diagnostic weights"
    print("generate_diagnostic_gait OK")

    clf = LinearDiagnosticClassifier(w_true)
    clf.fit_marginal_means(x)

    phi = clf.compute_true_shapley(x[0])
    assert phi.shape == (17,), phi.shape
    # Diagnostic joints should have non-zero Shapley values; non-diagnostic ≈ 0.
    diag_phi = np.abs(phi[DIAGNOSTIC_JOINTS]).mean()
    nondiag_phi = np.abs(phi[NON_DIAGNOSTIC_JOINTS]).mean()
    print(f"  Diag phi mean abs:     {diag_phi:.4f}")
    print(f"  Non-diag phi mean abs: {nondiag_phi:.4f}")
    assert diag_phi > nondiag_phi * 2, "diagnostic joints should dominate"
    print("LinearDiagnosticClassifier OK")

    # Check forward pass.
    x_t = torch.tensor(x[:4])
    out = clf(x_t)
    assert out.shape == (4,), out.shape
    print(f"  Classifier output range: [{out.min():.3f}, {out.max():.3f}]")

    # Check efficiency (v(S) = Σ_{j} phi_j + v(∅))
    phi_batch = clf.true_shapley_batch(x[:5], mu_j=clf.mu_j.numpy())
    vM = clf(torch.tensor(x[:5])).numpy()         # f(x*)
    v0 = (clf.w.numpy() * clf.mu_j.numpy()).sum()  # Σ w_j * μ_j
    recon = phi_batch.sum(axis=1) + v0
    assert np.allclose(recon, vM, atol=1e-4), f"Efficiency property failed: {np.abs(recon - vM).max()}"
    print("Efficiency property of true Shapley values: OK")
    print("Smoke test PASSED")
