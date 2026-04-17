"""synthetic — Synthetic benchmark datasets for ActorSHAP validation.

Two benchmarks are provided:

gaussian_motion
    Temporal motion drawn from a known Gaussian distribution
    (equicorrelation × AR(1) covariance). True conditional distributions are
    analytically Gaussian, enabling exact v_true(S) computation for EC1/EC2/EC3
    comparison (Olsen et al. JMLR 2022, Section 4.2 analogue).

diagnostic_motion
    Fourier-series synthetic gait with a known subset of diagnostic joints
    that carry label-relevant amplitude signal.  The black-box classifier is a
    linear function of joint velocities with analytically exact true Shapley
    values — enabling top-k joint recovery comparisons.
"""
