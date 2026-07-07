"""Serve-onset velocity reset gating around the position-only Kalman filter.

Problem this solves
--------------------
``kalman.Kalman`` observes position only; velocity is inferred from the position
sequence. While the ball is *held* before a serve, repeated near-static
measurements collapse the velocity covariance block ``P[3:6,3:6]`` -> the filter
becomes over-confident that velocity ~ 0. At the serve the true velocity jumps to
4-5 m/s, but the (now tiny) velocity gain lets the estimate ramp up only over
~7 frames (~58 ms @120Hz), during which the forward hit-point prediction is
garbage.

Fix
---
Detect the release frame (single-frame finite-difference speed crossing) and, on
release, fully re-``initialize()`` the filter at the current position. A fresh
filter has a large ``P_vv`` again, so it re-acquires the correct velocity in ~1
frame (verified on data/0617_traj_data: lag median 7->1 frame, peak velocity
error 3.2->0.4 m/s).

Composition with deployment rally segmentation
----------------------------------------------
This wrapper does NOT re-arm itself (instantaneous speed dips at the apex / bounce
would false-trigger a re-arm). Rally segmentation stays with the deployment's
own region-based reset: call :meth:`reset_to_hold` whenever the ball re-enters the
serve zone, and the gate will detect the next release from there.

See ``docs/serve_onset_reset_gating.md`` for the full design + data evidence.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:  # vendored into PPO-pingpong: support both package-relative and flat-dir imports
    from .kalman import Kalman, KalmanConfig
except ImportError:  # running a script from inside serve_gen/ (dir on sys.path)
    from kalman import Kalman, KalmanConfig


class ServeGatedKalman:
    """Wraps :class:`kalman.Kalman` with serve-onset velocity reset gating.

    Units follow ``Kalman``: position in mm, velocity in mm/s, timestamp in s.

    Parameters
    ----------
    config:
        Kalman configuration (defaults to ``KalmanConfig()``).
    v_release_m_s:
        Single-frame |vx| finite-difference threshold (m/s) that marks release.
        2.0 sits cleanly between hold drift (~0.8) and the serve plateau (3.5-5.9).
    n_persist:
        Consecutive frames above ``v_release`` required to confirm release.
        2 eliminates single-frame-noise false triggers while keeping ~1-frame lag.
    v_max_m_s:
        Glitch rejection: a measurement implying a speed above this (vs. the last
        accepted point) is treated as a mocap outlier and skipped (predict-only).
    release_axis:
        Index of the velocity component used for release detection (0=x). The serve
        is x-dominated; set to ``None`` to use the full 3D speed instead.
    """

    def __init__(
        self,
        config: Optional[KalmanConfig] = None,
        v_release_m_s: float = 2.0,
        n_persist: int = 2,
        v_max_m_s: float = 15.0,
        release_axis: Optional[int] = 0,
    ) -> None:
        self.cfg = config or KalmanConfig()
        self.v_release_mm = float(v_release_m_s) * 1000.0
        self.n_persist = int(n_persist)
        self.v_max_mm = float(v_max_m_s) * 1000.0
        self.release_axis = release_axis
        self.kf = Kalman(self.cfg)
        self.state = "HOLD"  # "HOLD" -> "TRACKING"
        self._prev: Optional[tuple[float, np.ndarray]] = None        # last frame
        self._last_good: Optional[tuple[float, np.ndarray]] = None    # last accepted
        self._persist = 0

    # ------------------------------------------------------------------ API
    @property
    def tracking(self) -> bool:
        """True once a serve release has been detected (predictions are usable)."""
        return self.state == "TRACKING"

    def reset_to_hold(self) -> None:
        """Re-arm for the next serve (call from the deployment region reset).

        Clears the filter and returns to HOLD; the next :meth:`step` re-initializes
        the filter from that measurement.
        """
        self.kf = Kalman(self.cfg)
        self.state = "HOLD"
        self._prev = None
        self._last_good = None
        self._persist = 0

    def step(self, position_xyz_mm: Iterable[float], timestamp_s: float):
        """Feed one mocap measurement. Returns ``(state_vector, gate_state)``.

        ``state_vector`` is ``Kalman.get_state()`` ([x,y,z,vx,vy,vz(,spin)]);
        ``gate_state`` is ``"HOLD"`` or ``"TRACKING"``. Only trust hit-point
        predictions when ``gate_state == "TRACKING"``.
        """
        pos = np.asarray(list(position_xyz_mm), dtype=float).reshape(3)
        t = float(timestamp_s)

        # --- glitch rejection: drop implausible jumps, keep time moving ---
        if self._last_good is not None:
            dt = t - self._last_good[0]
            if dt > 1e-6 and np.linalg.norm(pos - self._last_good[1]) / dt > self.v_max_mm:
                if self.kf.initialized:
                    self.kf.predict(t)
                return self.kf.get_state(), self.state

        # --- single-frame finite-difference velocity (for release detection) ---
        v_fd = None
        if self._prev is not None:
            dt = t - self._prev[0]
            if dt > 1e-6:
                v_fd = (pos - self._prev[1]) / dt
        self._last_good = (t, pos.copy())
        self._prev = (t, pos.copy())

        # --- first measurement: initialize ---
        if not self.kf.initialized:
            self.kf.initialize(pos, t)
            return self.kf.get_state(), self.state

        # --- release detection while holding ---
        if self.state == "HOLD" and v_fd is not None:
            metric = abs(v_fd[self.release_axis]) if self.release_axis is not None \
                else float(np.linalg.norm(v_fd))
            self._persist = self._persist + 1 if metric > self.v_release_mm else 0
            if self._persist >= self.n_persist:
                self.kf.initialize(pos, t)        # full re-init -> P_vv reinflated
                self.state = "TRACKING"
                return self.kf.get_state(), self.state

        # --- normal predict + update ---
        self.kf.predict(t)
        self.kf.update(pos)
        return self.kf.get_state(), self.state

    # --------------------------------------------------- passthrough helpers
    def get_state(self) -> np.ndarray:
        return self.kf.get_state()

    def get_position(self) -> np.ndarray:
        return self.kf.get_position()

    def get_velocity(self) -> np.ndarray:
        return self.kf.get_velocity()

    def get_spin(self) -> np.ndarray:
        return self.kf.get_spin()

    def predict(self, timestamp_s: float) -> np.ndarray:
        """Predict-only to a future timestamp (for forward hit-point roll-out)."""
        return self.kf.predict(timestamp_s)
