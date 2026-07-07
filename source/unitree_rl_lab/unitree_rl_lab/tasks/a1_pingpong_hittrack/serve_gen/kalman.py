from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class KalmanConfig:
    """Configuration for 3D constant-velocity Kalman filter."""

    # Process acceleration noise std (mm/s^2)
    process_accel_std: float = 5747.941426551285
    # Measurement noise std for xyz (mm)
    measurement_std: float = 5.0
    # Initial velocity std (mm/s)
    initial_velocity_std: float = 5000.0
    # Keep covariance numerically stable
    min_covariance: float = 1e-9
    # Cap prediction step and split long gaps into sub-steps
    max_predict_step_s: float = 0.02
    # Skip update when measurement contains NaN/Inf
    skip_invalid_measurement: bool = True
    # Enable constant gravity constraint on z axis
    enable_gravity: bool = True
    # Gravity acceleration magnitude (mm/s^2), applied as az = -gravity_mm_s2
    gravity_mm_s2: float = 9810.0
    # Enable bounce constraint on z axis
    enable_bounce: bool = True
    # Split a predict sub-step at the estimated table-impact time instead of
    # reflecting only after the state has already penetrated the bounce plane.
    enable_event_based_bounce: bool = False
    # Enable quadratic air drag on velocity: F = -k * |v| * v
    enable_air_drag: bool = True
    # Quadratic drag coefficient k (1/mm).  k = 0.5 * rho * Cd * A / m.
    # Standard ping-pong ball (m≈2.7g, d=40mm, Cd≈0.45): k ≈ 0.000126 1/mm.
    air_drag_coeff: float = 0.0001048181017752461
    # Bounce plane height (mm), e.g. table/ground height in your world frame
    # bounce_plane_z_mm: float = 19.935158411891802
    bounce_plane_z_mm: float = 27.884971761308254
    # Coefficient of restitution in [0, 1], 1 means perfectly elastic
    bounce_restitution: float = 0.90
    # Horizontal velocity reduction at bounce due to table friction.
    # vx_after = vx * (1 - coeff), vy_after = vy * (1 - coeff).
    # Ping-pong on table: typically 5-15% horizontal speed lost per bounce.
    bounce_friction_coeff: float = 0.15
    # Small threshold to avoid micro-jitter around bounce plane
    bounce_min_speed_mm_s: float = 20.0
    # Enable limiting bounce to table top XY range
    bounce_only_within_table: bool = True
    # Table dimensions (standard ping-pong table: 2740 x 1525 mm)
    table_length_mm: float = 2740.0
    table_width_mm: float = 1525.0
    # Extra tolerance margin for table range check (both +x/-x and +y/-y)
    table_range_margin_mm: float = 0.0

    # ---- Magnus force (spin-induced lift) ------------------------------------
    # When True the state vector expands from 6-D to 9-D:
    #   [x, y, z, vx, vy, vz, ωx, ωy, ωz]
    # and the Magnus acceleration  a_M = k_M · (ω × v)  is applied each
    # predict sub-step.
    enable_magnus: bool = True
    # Magnus coefficient k_M = ½·C_M·ρ·A·r / m  (dimensionless).
    # Paper: C_M=0.6, ρ=1.29 kg/m³, A=π·r², r=0.02 m, m=0.0027 kg → 0.003604
    # 2026-06-17 tuning selected 0.0 for the current default.
    magnus_coeff: float = 0.0
    # Initial angular-velocity std (rad/s) for the spin part of P₀.
    initial_spin_std: float = 200.0
    # Process noise std for angular-velocity (rad/s per √s).
    # Spin barely changes in flight; a small value keeps it stiff.
    spin_process_std: float = 50.0

    # ---- Bounce spin-coupling parameters (paper Eq.13-18 / ref [31]) --------
    # v⁺_x = bounce_alpha_xy·v⁻_x + bounce_beta_xy·ω⁻_y
    # v⁺_y = bounce_alpha_xy·v⁻_y + bounce_beta_xy·ω⁻_x
    # v⁺_z handled by bounce_alpha_z (restitution)
    # ω⁺_x = bounce_gamma_x·ω⁻_x + bounce_delta_x·v⁻_y
    # ω⁺_y = bounce_gamma_y·ω⁻_y + bounce_delta_y·v⁻_x
    # ω⁺_z = bounce_gamma_z·ω⁻_z
    # Note: β has units [mm], δ has units [1/mm] (converted from SI in paper).
    bounce_alpha_xy: float = 0.7299465301633188
    bounce_alpha_z: float = 0.8787306911173297
    bounce_beta_xy: float = 2.9582864029656184  # mm  (paper: 0.0015 m)
    bounce_gamma_x: float = 0.2528471726648266
    bounce_gamma_y: float = 0.2528471726648266
    bounce_gamma_z: float = 0.90
    bounce_delta_x: float = -0.04155312393251931  # 1/mm  (paper: -26  1/m)
    bounce_delta_y: float = 0.02414416993343777   # 1/mm  (paper:  25  1/m)


class Kalman:
    """
    Kalman filter for 3D ping-pong trajectory prediction.

    State vector:
        6-D (enable_magnus=False):  [x, y, z, vx, vy, vz]^T
        9-D (enable_magnus=True):   [x, y, z, vx, vy, vz, ωx, ωy, ωz]^T

    Measurement vector (3x1):
        [x, y, z]^T

    Physical model (per sub-step):
        1. Linear transition  (constant-velocity)
        2. Gravity             az  = −g
        3. Quadratic air drag  a_D = −k_D |v| v
        4. Magnus force        a_M = k_M  (ω × v)   [when enabled]
        5. Bounce (table)      velocity & spin coupling

    Units:
        position: mm
        velocity: mm/s
        angular velocity: rad/s
        timestamp: seconds (float)
    """

    def __init__(self, config: Optional[KalmanConfig] = None) -> None:
        self.config = config or KalmanConfig()

        # State dimension: 9 when Magnus/spin enabled, 6 otherwise
        self.dim = 9 if self.config.enable_magnus else 6

        # State estimate and covariance
        self.x = np.zeros((self.dim, 1), dtype=float)
        self.P = np.eye(self.dim, dtype=float)

        # Measurement model: observe only position [x, y, z]
        self.H = np.zeros((3, self.dim), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        r = self.config.measurement_std**2
        self.R = np.eye(3, dtype=float) * r

        self.initialized = False
        self.last_timestamp: Optional[float] = None

    def initialize(self, position_xyz_mm: Iterable[float], timestamp_s: float) -> None:
        """Initialize the filter with first position measurement."""
        z = self._to_measurement(position_xyz_mm)
        self.x = np.zeros((self.dim, 1), dtype=float)
        self.x[0:3, 0] = z[:, 0]

        pos_var = self.config.measurement_std**2
        vel_var = self.config.initial_velocity_std**2
        diag_vals = [pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]
        if self.dim == 9:
            spin_var = self.config.initial_spin_std**2
            diag_vals += [spin_var, spin_var, spin_var]
        self.P = np.diag(diag_vals).astype(float)

        self.initialized = True
        self.last_timestamp = float(timestamp_s)

    def predict(self, timestamp_s: float) -> np.ndarray:
        """
        Predict state to given timestamp.
        If uninitialized, raises RuntimeError.
        """
        self._check_initialized()
        ts = float(timestamp_s)
        if not np.isfinite(ts):
            # Keep previous state if timestamp is invalid.
            return self.get_state()

        dt = ts - float(self.last_timestamp)
        if dt < 0:
            # Tolerate out-of-order timestamps to keep pipeline running.
            dt = 0.0
        if dt == 0:
            return self.get_state()

        max_step = max(float(self.config.max_predict_step_s), 1e-6)
        n_steps = max(1, int(np.ceil(dt / max_step)))
        dt_step = dt / n_steps
        for _ in range(n_steps):
            F = self._build_transition(dt_step)
            Q = self._build_process_noise(dt_step, self.config.process_accel_std)
            self._predict_state_substep(dt_step)
            self.P = F @ self.P @ F.T + Q
        self._stabilize_covariance()
        self.last_timestamp = ts
        return self.get_state()

    def update(self, position_xyz_mm: Iterable[float]) -> np.ndarray:
        """
        Update filter with current position measurement.
        Call predict(timestamp) first if timestamp changed.
        """
        self._check_initialized()
        try:
            z = self._to_measurement(position_xyz_mm)
        except ValueError:
            if self.config.skip_invalid_measurement:
                return self.get_state()
            raise
        if not np.isfinite(z).all():
            if self.config.skip_invalid_measurement:
                return self.get_state()
            raise ValueError("position_xyz_mm contains NaN/Inf.")

        y = z - (self.H @ self.x)  # residual
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = np.linalg.solve(S.T, (self.H @ self.P.T)).T
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        self.x = self.x + (K @ y)
        if self.config.enable_bounce:
            self._apply_bounce_constraint()

        # Joseph form for better numerical stability
        I = np.eye(self.dim, dtype=float)
        I_KH = I - (K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self._stabilize_covariance()
        return self.get_state()

    def step(self, position_xyz_mm: Iterable[float], timestamp_s: float) -> np.ndarray:
        """
        Convenience method: initialize (if needed), then predict + update.
        Returns current estimated state [x,y,z,vx,vy,vz].
        """
        if not self.initialized:
            self.initialize(position_xyz_mm, timestamp_s)
            return self.get_state()

        self.predict(timestamp_s)
        self.update(position_xyz_mm)
        return self.get_state()

    def predict_to(self, timestamp_s: float) -> np.ndarray:
        """
        Predict only (no measurement update), often used for future trajectory.
        Returns predicted [x,y,z,vx,vy,vz] at timestamp_s.
        """
        return self.predict(timestamp_s)

    def get_state(self) -> np.ndarray:
        """Return flattened state [x, y, z, vx, vy, vz]."""
        return self.x[:, 0].copy()

    def get_position(self) -> np.ndarray:
        """Return position estimate [x, y, z] in mm."""
        return self.x[0:3, 0].copy()

    def get_velocity(self) -> np.ndarray:
        """Return velocity estimate [vx, vy, vz] in mm/s."""
        return self.x[3:6, 0].copy()

    def _build_transition(self, dt: float) -> np.ndarray:
        F = np.eye(self.dim, dtype=float)
        F[0, 3] = dt  # x += vx·dt
        F[1, 4] = dt  # y += vy·dt
        F[2, 5] = dt  # z += vz·dt
        # spin states (indices 6-8) stay constant → identity block
        return F

    def _build_process_noise(self, dt: float, accel_std: float) -> np.ndarray:
        """
        Constant-velocity model driven by white acceleration noise.
        For each axis:
            [x, v]
            Q_1d = q * [[dt^4/4, dt^3/2],
                        [dt^3/2, dt^2]]
        When spin is tracked (dim=9), angular-velocity gets an
        independent random-walk noise block.
        """
        q = accel_std**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        q11 = 0.25 * dt4 * q
        q12 = 0.5 * dt3 * q
        q22 = dt2 * q

        Q = np.zeros((self.dim, self.dim), dtype=float)
        # position-velocity block (same for x, y, z)
        for i in range(3):
            Q[i, i] = q11
            Q[i, i + 3] = q12
            Q[i + 3, i] = q12
            Q[i + 3, i + 3] = q22
        # spin block (random walk: σ_ω² · dt²)
        if self.dim == 9:
            q_spin = self.config.spin_process_std**2 * dt2
            for i in range(6, 9):
                Q[i, i] = q_spin
        return Q

    @staticmethod
    def _to_measurement(position_xyz_mm: Iterable[float]) -> np.ndarray:
        values = np.array(list(position_xyz_mm), dtype=float).reshape(-1)
        if values.size != 3:
            raise ValueError("position_xyz_mm must contain exactly 3 values: [x, y, z].")
        return values.reshape(3, 1)

    def _check_initialized(self) -> None:
        if not self.initialized or self.last_timestamp is None:
            raise RuntimeError("Filter is not initialized. Call initialize() or step() first.")

    def _stabilize_covariance(self) -> None:
        # Force symmetry and keep diagonal non-negative.
        self.P = 0.5 * (self.P + self.P.T)
        diag = np.diag(self.P).copy()
        diag[diag < self.config.min_covariance] = self.config.min_covariance
        self.P[np.diag_indices_from(self.P)] = diag

    def _predict_state_substep(self, dt: float) -> None:
        if dt <= 0.0:
            return

        if self.config.enable_bounce and self.config.enable_event_based_bounce:
            t_hit = self._time_to_bounce_plane(dt)
            if t_hit is not None:
                if t_hit > 0.0:
                    self._advance_state_no_bounce(t_hit)
                self.x[2, 0] = float(self.config.bounce_plane_z_mm)
                self._apply_bounce_constraint(force_response=True)

                dt_remain = dt - t_hit
                if dt_remain > 1e-9:
                    self._advance_state_no_bounce(dt_remain)
                    # A normal constraint pass catches numerical penetration and
                    # rare double-contact cases within one sub-step.
                    self._apply_bounce_constraint()
                return

        self._advance_state_no_bounce(dt)
        if self.config.enable_bounce:
            self._apply_bounce_constraint()

    def _advance_state_no_bounce(self, dt: float) -> None:
        F = self._build_transition(dt)
        self.x = F @ self.x
        if self.config.enable_gravity:
            # Add deterministic gravity term on z axis:
            # z <- z + 0.5 * az * dt^2
            # vz <- vz + az * dt
            az = -float(self.config.gravity_mm_s2)
            self.x[2, 0] += 0.5 * az * dt * dt
            self.x[5, 0] += az * dt
        if self.config.enable_air_drag:
            self._apply_air_drag(dt)
        if self.config.enable_magnus:
            self._apply_magnus_force(dt)

    def _time_to_bounce_plane(self, dt: float) -> Optional[float]:
        plane_z = float(self.config.bounce_plane_z_mm)
        z0 = float(self.x[2, 0])
        vz0 = float(self.x[5, 0])
        min_speed = max(float(self.config.bounce_min_speed_mm_s), 0.0)

        if z0 < plane_z:
            if vz0 < -min_speed and self._is_xy_inside_bounce_table(
                float(self.x[0, 0]), float(self.x[1, 0])
            ):
                return 0.0
            return None
        if (
            z0 == plane_z
            and vz0 < -min_speed
            and self._is_xy_inside_bounce_table(float(self.x[0, 0]), float(self.x[1, 0]))
        ):
            return 0.0

        az = -float(self.config.gravity_mm_s2) if self.config.enable_gravity else 0.0
        c = z0 - plane_z
        roots = []
        if abs(az) < 1e-12:
            if vz0 < -min_speed:
                roots.append(-c / vz0)
        else:
            a = 0.5 * az
            disc = vz0 * vz0 - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = float(np.sqrt(disc))
                roots.append((-vz0 - sqrt_disc) / (2.0 * a))
                roots.append((-vz0 + sqrt_disc) / (2.0 * a))

        candidates = []
        for t in roots:
            if not np.isfinite(t) or t < -1e-12 or t > dt + 1e-12:
                continue
            t = float(np.clip(t, 0.0, dt))
            vz_hit = vz0 + az * t
            if vz_hit >= -min_speed:
                continue
            x_hit = float(self.x[0, 0]) + float(self.x[3, 0]) * t
            y_hit = float(self.x[1, 0]) + float(self.x[4, 0]) * t
            if not self._is_xy_inside_bounce_table(x_hit, y_hit):
                continue
            candidates.append(t)

        return min(candidates) if candidates else None

    def _is_xy_inside_bounce_table(self, x_mm: float, y_mm: float) -> bool:
        if not self.config.bounce_only_within_table:
            return True
        half_len = 0.5 * max(float(self.config.table_length_mm), 0.0)
        half_wid = 0.5 * max(float(self.config.table_width_mm), 0.0)
        margin = max(float(self.config.table_range_margin_mm), 0.0)
        return (
            abs(float(x_mm)) <= (half_len + margin)
            and abs(float(y_mm)) <= (half_wid + margin)
        )

    def _apply_bounce_constraint(self, force_response: bool = False) -> None:
        plane_z = float(self.config.bounce_plane_z_mm)
        z_now = float(self.x[2, 0])
        if z_now >= plane_z and not force_response:
            return

        if not self._is_xy_inside_bounce_table(float(self.x[0, 0]), float(self.x[1, 0])):
            # Out of table top area: do not apply table-bounce constraint.
            return

        restitution = float(np.clip(self.config.bounce_restitution, 0.0, 1.0))
        friction = float(np.clip(self.config.bounce_friction_coeff, 0.0, 1.0))
        min_speed = max(float(self.config.bounce_min_speed_mm_s), 0.0)

        if force_response:
            self.x[2, 0] = plane_z
        else:
            # Mirror position back above the plane to handle penetration robustly.
            penetration = plane_z - z_now
            self.x[2, 0] = plane_z + penetration

        vz_now = float(self.x[5, 0])
        if vz_now < -min_speed:
            if self.config.enable_magnus and self.dim == 9:
                # Paper Eq.13-18: coupled spin-velocity bounce model
                vx_pre = float(self.x[3, 0])
                vy_pre = float(self.x[4, 0])
                wx_pre = float(self.x[6, 0])
                wy_pre = float(self.x[7, 0])
                cfg = self.config
                self.x[3, 0] = cfg.bounce_alpha_xy * vx_pre + cfg.bounce_beta_xy * wy_pre
                self.x[4, 0] = cfg.bounce_alpha_xy * vy_pre + cfg.bounce_beta_xy * wx_pre
                self.x[5, 0] = cfg.bounce_alpha_z * abs(vz_now)  # reverse upward
                self.x[6, 0] = cfg.bounce_gamma_x * wx_pre + cfg.bounce_delta_x * vy_pre
                self.x[7, 0] = cfg.bounce_gamma_y * wy_pre + cfg.bounce_delta_y * vx_pre
                self.x[8, 0] = cfg.bounce_gamma_z * float(self.x[8, 0])
            else:
                self.x[5, 0] = -restitution * vz_now
                if friction > 0.0:
                    self.x[3, 0] *= (1.0 - friction)
                    self.x[4, 0] *= (1.0 - friction)
        elif abs(vz_now) <= min_speed:
            # Near the plane with tiny speed -> clamp to avoid repeated jitter.
            self.x[2, 0] = plane_z
            self.x[5, 0] = 0.0
        else:
            # Already moving upward but penetrated due to numeric effects.
            self.x[5, 0] = max(vz_now, 0.0)

    def _apply_air_drag(self, dt: float) -> None:
        """Apply quadratic air drag: dv/dt = -k * |v| * v (Euler step)."""
        k = max(float(self.config.air_drag_coeff), 0.0)
        if k <= 0.0 or dt <= 0.0:
            return
        v = self.x[3:6, 0]  # (3,) view
        speed = float(np.linalg.norm(v))
        if speed < 1e-9:
            return
        # Clamp the drag impulse so speed cannot reverse direction.
        # max fractional speed loss per sub-step = 1.0 (full stop).
        drag_decel = k * speed          # |dv/dt| / |v|  (1/s)
        factor = drag_decel * dt         # dimensionless fraction of speed lost
        if factor > 1.0:
            factor = 1.0                 # clamp to prevent sign flip
        self.x[3:6, 0] = v * (1.0 - factor)

    def _apply_magnus_force(self, dt: float) -> None:
        """Apply Magnus (spin-induced lift) acceleration: a_M = k_M·(ω×v).

        Paper Eq.11:  F_m = ½·C_M·ρ·A·r·(ω×v)
        k_M = ½·C_M·ρ·A·r / m  (dimensionless).
        """
        if self.dim < 9 or dt <= 0.0:
            return
        k_m = float(self.config.magnus_coeff)
        if k_m <= 0.0:
            return
        vx = float(self.x[3, 0])
        vy = float(self.x[4, 0])
        vz = float(self.x[5, 0])
        wx = float(self.x[6, 0])
        wy = float(self.x[7, 0])
        wz = float(self.x[8, 0])
        # ω × v
        ax = k_m * (wy * vz - wz * vy)
        ay = k_m * (wz * vx - wx * vz)
        az = k_m * (wx * vy - wy * vx)
        self.x[3, 0] += ax * dt
        self.x[4, 0] += ay * dt
        self.x[5, 0] += az * dt

    # ----- public helpers ----------------------------------------------------

    def get_spin(self) -> np.ndarray:
        """Return angular velocity [ωx, ωy, ωz] in rad/s (zeros if spin not tracked)."""
        if self.dim >= 9:
            return self.x[6:9, 0].copy()
        return np.zeros(3, dtype=float)
