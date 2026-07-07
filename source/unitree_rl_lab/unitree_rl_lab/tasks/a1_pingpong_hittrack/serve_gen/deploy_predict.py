"""Reproduce the deployment/tuning KF_pred prediction method (from
kalman_filter_pingpong/dataprocess/tune_kalman_params.py) so synthetic KF_pred is computed the SAME
way the recordings' ``KF_pred_*`` column was, removing any "prediction-method" confound.

Method (per tune_kalman_params.evaluate + tuned_params_0617.json):
  * run the KF over the measurement stream (predict+update per frame), keep the filtered state;
  * at a trigger frame, OPTIONALLY replace the state velocity with a local polynomial fit over the
    last ``window`` FILTERED positions (``open_loop_velocity_fit``; tuned json = FALSE => plain KF
    velocity, ``blend=1.0`` would fully replace it if enabled);
  * predict-only roll to the target plane; report the crossing (y,z,vx,vy,vz).

``fit_local_velocity_mm_s`` is copied verbatim from tune_kalman_params.py.
"""

from __future__ import annotations

import numpy as np

try:
    from .ball_physics import predict_to_plane
    from .kalman import KalmanConfig
except ImportError:
    from ball_physics import predict_to_plane
    from kalman import KalmanConfig


def fit_local_velocity_mm_s(t, pos_mm, idx, window, degree, min_samples):
    """Verbatim from tune_kalman_params.py: velocity at ``idx`` from a degree-``degree`` polyfit of
    the last ``window`` positions. Returns ``[3]`` mm/s or ``None``."""
    if idx < 0 or idx >= len(t):
        return None
    n = max(int(window), 2)
    start = max(0, idx - n + 1)
    tt = t[start: idx + 1] - t[idx]
    pp = pos_mm[start: idx + 1]
    if len(tt) < max(int(min_samples), 2):
        return None
    if not (np.isfinite(tt).all() and np.isfinite(pp).all()):
        return None
    if float(np.ptp(tt)) < 1e-6:
        return None
    deg = min(max(int(degree), 1), len(tt) - 1)
    vel = np.empty(3, dtype=float)
    try:
        for axis in range(3):
            coeff = np.polyfit(tt, pp[:, axis], deg)
            vel[axis] = np.polyval(np.polyder(coeff), 0.0)
    except np.linalg.LinAlgError:
        return None
    return vel if np.isfinite(vel).all() else None


def deploy_predict_to_plane(
    state6_mm: np.ndarray,
    times: np.ndarray,
    filt_pos_mm: np.ndarray,
    idx: int,
    plane_x_mm: float,
    cfg: KalmanConfig,
    *,
    velocity_fit: bool = False,
    window: int = 8,
    degree: int = 2,
    min_samples: int = 5,
    blend: float = 1.0,
):
    """Deployment-style hit-plane prediction from filtered state at frame ``idx``.

    ``state6_mm`` is the KF filtered [x,y,z,vx,vy,vz] at ``idx`` (mm, mm/s). ``filt_pos_mm`` is the
    full filtered-position history (mm) for the velocity fit. Returns ``[y,z,vx,vy,vz]`` (mm) at the
    plane crossing, or ``None``.
    """
    s = np.asarray(state6_mm, dtype=float).copy()
    if velocity_fit:
        vel = fit_local_velocity_mm_s(times, filt_pos_mm, idx, window, degree, min_samples)
        if vel is not None:
            a = float(np.clip(blend, 0.0, 1.0))
            s[3:6] = (1.0 - a) * s[3:6] + a * vel
    return predict_to_plane(s, plane_x_mm, cfg)
