"""Ball-flight forward rollout, reusing the vendored Kalman physics VERBATIM.

The generator MUST propagate a ball state with the exact same dynamics the KF uses to
predict (gravity + quadratic air drag + table bounce with restitution/friction + spin-coupling;
Magnus disabled by ``magnus_coeff=0``). Instead of re-deriving that physics we drive the vendored
``Kalman`` in *predict-only* mode: set its state, then call ``predict(t)`` over a time grid. This
guarantees ``rollout`` and the KF's own forward roll-out are bit-identical, so a trajectory the
generator produces is exactly the trajectory the KF would have predicted -- the property the
sim2real-gap check depends on.

Units follow the vendored KF: position mm, velocity mm/s, timestamp s.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

try:  # support both package-relative and flat-dir (script) imports
    from .kalman import Kalman, KalmanConfig
except ImportError:
    from kalman import Kalman, KalmanConfig


def default_config() -> KalmanConfig:
    """The vendored KF's default config (the tuned constants that match the C++ recordings).

    Kept as a single factory so the generator, the offline KF runner and the validation all
    share ONE physics parameterization -- changing a drag/bounce constant here changes every stage
    consistently.
    """
    return KalmanConfig()


def gen_config(magnus_coeff: float = 0.003604) -> KalmanConfig:
    """Config for generating balls WITH spin/Magnus on (legacy; the drag/bounce model is preferred).

    Kept for the spin-vs-drag/bounce discriminating experiment. Experiment 3 showed symmetric
    per-serve drag/bounce reproduces real trajectories as well as spin, so generation uses
    :func:`ball_config` (spin off, magnus off, same model family as the KF) instead.
    """
    cfg = KalmanConfig()
    cfg.magnus_coeff = float(magnus_coeff)
    return cfg


def ball_config(air_drag_coeff: float, bounce_alpha_z: float, bounce_alpha_xy: float) -> KalmanConfig:
    """Per-serve ball config: the KF's own tuned model with per-serve drag + bounce parameters and
    Magnus OFF. The real ball's drag/bounce vary serve-to-serve around the KF's single tuned value;
    the KF (using the fixed tuned value) cannot track that variation, which is what makes its
    forward prediction realistically wrong. This stays entirely inside the KF's model family -- no
    Magnus term (which the deployment KF deliberately disables) and no assumed coefficient."""
    return replace(
        KalmanConfig(),
        magnus_coeff=0.0,
        air_drag_coeff=float(air_drag_coeff),
        bounce_alpha_z=float(bounce_alpha_z),
        bounce_alpha_xy=float(bounce_alpha_xy),
    )


def _seed_filter(pos_mm, vel_mm, t0_s: float, cfg: KalmanConfig, spin_rad_s=None) -> Kalman:
    kf = Kalman(cfg)
    x = np.zeros((kf.dim, 1), dtype=float)
    x[0:3, 0] = np.asarray(pos_mm, dtype=float).reshape(3)
    x[3:6, 0] = np.asarray(vel_mm, dtype=float).reshape(3)
    if kf.dim == 9:
        # spin (rad/s). Zero for the drag/bounce ball model; nonzero only for the legacy spin model.
        if spin_rad_s is not None:
            x[6:9, 0] = np.asarray(spin_rad_s, dtype=float).reshape(3)
    kf.x = x
    kf.initialized = True
    kf.last_timestamp = float(t0_s)
    return kf


def rollout_positions(pos_mm, vel_mm, t0_s: float, query_times_s, cfg: KalmanConfig | None = None,
                      spin_rad_s=None) -> np.ndarray:
    """Predict-only ball positions at ``query_times_s`` given the state ``(pos_mm, vel_mm)`` at
    ``t0_s``. ``query_times_s`` must be non-decreasing and >= ``t0_s``. Returns ``[len,3]`` mm."""
    cfg = cfg or default_config()
    kf = _seed_filter(pos_mm, vel_mm, t0_s, cfg, spin_rad_s)
    q = np.asarray(query_times_s, dtype=float).reshape(-1)
    out = np.empty((q.shape[0], 3), dtype=float)
    for i, t in enumerate(q):
        kf.predict(float(t))
        out[i] = kf.get_position()
    return out


def rollout_state_stream(pos_mm, vel_mm, t0_s: float, query_times_s, cfg: KalmanConfig | None = None,
                         spin_rad_s=None) -> np.ndarray:
    """Like :func:`rollout_positions` but returns the full ``[len,6]`` ``[x,y,z,vx,vy,vz]`` state
    stream (mm, mm/s). Used by the generator to emit clean ball trajectories with velocities."""
    cfg = cfg or default_config()
    kf = _seed_filter(pos_mm, vel_mm, t0_s, cfg, spin_rad_s)
    q = np.asarray(query_times_s, dtype=float).reshape(-1)
    out = np.empty((q.shape[0], 6), dtype=float)
    for i, t in enumerate(q):
        kf.predict(float(t))
        out[i, 0:3] = kf.get_position()
        out[i, 3:6] = kf.get_velocity()
    return out


def predict_to_plane(state6_mm, plane_x_mm: float, cfg: KalmanConfig | None = None,
                     horizon_s: float = 1.6, dt_s: float = 0.005):
    """Predict-only roll a state ``[x,y,z,vx,vy,vz]`` (mm, mm/s) forward until x crosses
    ``plane_x_mm`` (x decreasing), returning the interpolated ``[y,z,vx,vy,vz]`` there (mm, mm/s),
    or ``None`` if the plane is not reached within ``horizon_s``. This mirrors how the deployment KF
    produces its ``KF_pred`` column (forward roll-out to the hit plane)."""
    cfg = cfg or default_config()
    q = np.arange(0.0, horizon_s + dt_s, dt_s)
    stream = rollout_state_stream(state6_mm[0:3], state6_mm[3:6], 0.0, q, cfg)  # [T,6]
    x = stream[:, 0]
    d = x - float(plane_x_mm)
    for i in range(len(d) - 1):
        if d[i] > 0.0 and d[i + 1] <= 0.0:
            a = d[i] / (d[i] - d[i + 1])
            s = stream[i] + a * (stream[i + 1] - stream[i])
            return np.array([s[1], s[2], s[3], s[4], s[5]], dtype=float)
    return None


def first_downward_crossing(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray, plane_x: float):
    """First linear-interpolated crossing of ``x == plane_x`` (x decreasing). Returns
    ``(y_c, z_c, t_c)`` or ``None``. Mirrors ``bake_hittrack_references.true_crossing`` semantics."""
    d = np.asarray(x, dtype=float) - float(plane_x)
    for i in range(len(d) - 1):
        if d[i] == 0.0:
            return float(y[i]), float(z[i]), float(t[i])
        if d[i] > 0.0 and d[i + 1] <= 0.0:
            a = d[i] / (d[i] - d[i + 1])
            return (
                float(y[i] + a * (y[i + 1] - y[i])),
                float(z[i] + a * (z[i + 1] - z[i])),
                float(t[i] + a * (t[i + 1] - t[i])),
            )
    return None
