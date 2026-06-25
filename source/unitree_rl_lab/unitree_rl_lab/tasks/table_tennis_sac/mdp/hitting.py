"""Analytic table-tennis hitting model: the racket-tip velocity that returns a given
incoming ball to a target landing point.

This module is deliberately **pure torch** (no isaaclab import) so the physics can be
unit-tested standalone, without a simulator. Two stages:

  1. ``solve_launch_velocity`` -- a drag-aware boundary-value solve. Given the contact
     point, the desired landing target, and a fixed launch elevation ``theta`` (the
     "arc" knob; neutral by default, later a return-strategy curriculum), find the
     outgoing ball velocity ``v_out`` whose flight -- integrated with the SAME discrete
     dynamics the sim applies (gravity + per-substep linear damping + one quadratic air
     -drag patch per control step, mirroring ``events.apply_air_drag``) -- passes through
     the target. The legacy ``_desired_launch_dir`` and ``ball_predicted_hit_point`` are
     gravity-only; over a ~2 m / ~0.7 s outgoing flight that drops ~0.5 m/s and lands the
     ball short, so the drag term is not optional here.

  2. ``contact_inverse`` -- a closed-form rigid, frictionless ball/paddle collision solve.
     For a plane of restitution ``e`` moving with velocity ``v_p`` and unit normal ``n``:
         v_out = v_in - (1 + e) * ((v_in - v_p) . n) * n
     so ``v_out - v_in`` is parallel to ``n``. Hence
         n      = normalize(v_out - v_in)
         v_p.n  = v_in . n + ||v_out - v_in|| / (1 + e)
     and the minimal paddle velocity (normal component only; tangential is free and does
     not affect ``v_out``) is ``(v_p.n) * n``.

``ideal_racket_velocity`` chains the two. All inputs/outputs are world-frame; ``origin``
and ``target`` need only share a frame (the env-origin offset cancels in the displacement).
"""

from __future__ import annotations

import math

import torch

# Neutral launch elevation (rad). theta IS the return-strategy knob: low -> flat/fast
# drive, high -> loopy/safe. Neutral ~28 deg for the first pass; later sweep as curriculum.
NEUTRAL_THETA = math.radians(28.0)

# Ball/paddle restitution from the a1.usd paddle material.
PADDLE_RESTITUTION = 0.75


def _step_dynamics(
    vr: torch.Tensor,
    vz: torch.Tensor,
    *,
    drag_k: float,
    lin_damp: float,
    control_dt: float,
    substeps: int,
    gravity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the (horizontal, vertical) velocity one control step, replicating the sim:
    ``substeps`` gravity + linear-damping sub-updates, then ONE quadratic air-drag patch.

    Position is integrated by the caller using the *pre-step* velocity (semi-implicit).
    Returns the post-step (vr, vz).
    """
    sub_dt = control_dt / substeps
    damp = 1.0 / (1.0 + lin_damp * sub_dt)
    # gravity + PhysX linear_damping over the substeps, applied analytically:
    vz = vz - gravity * control_dt
    damp_total = damp ** substeps
    vr = vr * damp_total
    vz = vz * damp_total
    # one quadratic-drag velocity patch per control step (events.apply_air_drag: a=-k|v|v)
    speed = torch.sqrt(vr * vr + vz * vz).clamp(min=1.0e-9)
    factor = (1.0 - drag_k * speed * control_dt).clamp(min=0.0)
    vr = vr * factor
    vz = vz * factor
    return vr, vz


def _height_at_range(
    z0: torch.Tensor,
    speed: torch.Tensor,
    cos_t: float,
    sin_t: float,
    horiz_dist: torch.Tensor,
    *,
    drag_k: float,
    lin_damp: float,
    control_dt: float,
    substeps: int,
    gravity: float,
    n_steps: int,
) -> torch.Tensor:
    """Forward-integrate a projectile launched from height ``z0`` at ``speed`` along
    elevation (cos_t, sin_t), and return the height ``z`` at the moment its cumulative
    horizontal travel first reaches ``horiz_dist``. If it never reaches the range within
    ``n_steps`` (undershoot), returns a large negative sentinel so the caller raises speed.
    """
    n = z0.shape[0]
    device = z0.device
    vr = speed * cos_t
    vz = speed * sin_t
    r = torch.zeros(n, device=device, dtype=z0.dtype)
    z = z0.clone()
    z_at = torch.full((n,), -1.0e6, device=device, dtype=z0.dtype)
    reached = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(n_steps):
        r_next = r + vr * control_dt
        z_next = z + vz * control_dt
        newly = (~reached) & (r < horiz_dist) & (r_next >= horiz_dist)
        if torch.any(newly):
            denom = (r_next - r).clamp(min=1.0e-9)
            frac = ((horiz_dist - r) / denom).clamp(0.0, 1.0)
            z_cross = z + frac * (z_next - z)
            z_at = torch.where(newly, z_cross, z_at)
            reached = reached | newly
        r, z = r_next, z_next
        vr, vz = _step_dynamics(
            vr, vz, drag_k=drag_k, lin_damp=lin_damp, control_dt=control_dt,
            substeps=substeps, gravity=gravity,
        )
        if bool(reached.all()):
            break
    return z_at


def solve_launch_velocity(
    origin: torch.Tensor,
    target: torch.Tensor,
    *,
    theta: float = NEUTRAL_THETA,
    drag_k: float = 0.08,
    lin_damp: float = 0.05,
    control_dt: float = 0.02,
    substeps: int = 4,
    gravity: float = 9.81,
    speed_lo: float = 0.5,
    speed_hi: float = 12.0,
    bisection_iters: int = 18,
    max_flight_s: float = 1.4,
) -> torch.Tensor:
    """Outgoing ball velocity ``v_out`` [N,3] that lands a drag-decelerated projectile at
    ``target`` when launched from ``origin`` at fixed elevation ``theta``.

    Heading (xy) is fixed by origin->target geometry, elevation by ``theta``; the only free
    scalar is launch speed, found by monotone bisection on "height at the target horizontal
    range minus target height".
    """
    d = target - origin
    dx, dy, dz_unused = d[:, 0], d[:, 1], d[:, 2]
    horiz = torch.sqrt(dx * dx + dy * dy).clamp(min=1.0e-4)
    hx, hy = dx / horiz, dy / horiz
    target_z = target[:, 2]
    z0 = origin[:, 2]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    n_steps = int(round(max_flight_s / control_dt))

    lo = torch.full_like(z0, speed_lo)
    hi = torch.full_like(z0, speed_hi)
    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        z_at = _height_at_range(
            z0, mid, cos_t, sin_t, horiz,
            drag_k=drag_k, lin_damp=lin_damp, control_dt=control_dt,
            substeps=substeps, gravity=gravity, n_steps=n_steps,
        )
        overshoot = (z_at - target_z) > 0.0  # too high at the range -> reduce speed
        hi = torch.where(overshoot, mid, hi)
        lo = torch.where(overshoot, lo, mid)
    speed = 0.5 * (lo + hi)

    v_out = torch.stack([speed * cos_t * hx, speed * cos_t * hy, speed * sin_t], dim=-1)
    return v_out


def contact_inverse(
    v_in: torch.Tensor,
    v_out: torch.Tensor,
    restitution: float = PADDLE_RESTITUTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal paddle velocity (and unit face normal) producing ``v_out`` from ``v_in`` in
    a rigid, frictionless collision of restitution ``e``. Returns (v_paddle [N,3], n [N,3])."""
    w = v_out - v_in
    w_norm = torch.norm(w, dim=-1, keepdim=True).clamp(min=1.0e-9)
    n = w / w_norm
    v_in_n = torch.sum(v_in * n, dim=-1, keepdim=True)
    v_p_n = v_in_n + w_norm / (1.0 + restitution)
    v_paddle = v_p_n * n
    return v_paddle, n


def ideal_racket_velocity(
    origin: torch.Tensor,
    v_in: torch.Tensor,
    target: torch.Tensor,
    *,
    theta: float = NEUTRAL_THETA,
    restitution: float = PADDLE_RESTITUTION,
    drag_k: float = 0.08,
    lin_damp: float = 0.05,
    control_dt: float = 0.02,
    substeps: int = 4,
    gravity: float = 9.81,
    speed_lo: float = 0.5,
    speed_hi: float = 12.0,
    bisection_iters: int = 18,
    max_flight_s: float = 1.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Racket-tip velocity that returns ``v_in`` (incoming ball velocity at ``origin``) to
    land at ``target``, via a neutral-arc drag-aware launch solve + closed-form contact
    inversion. Returns (v_paddle_ideal [N,3], v_out [N,3], face_normal [N,3])."""
    v_out = solve_launch_velocity(
        origin, target, theta=theta, drag_k=drag_k, lin_damp=lin_damp,
        control_dt=control_dt, substeps=substeps, gravity=gravity,
        speed_lo=speed_lo, speed_hi=speed_hi, bisection_iters=bisection_iters,
        max_flight_s=max_flight_s,
    )
    v_paddle, n = contact_inverse(v_in, v_out, restitution=restitution)
    return v_paddle, v_out, n


def predict_landing_xy(
    origin: torch.Tensor,
    v_out: torch.Tensor,
    *,
    table_z: float = 0.76,
    drag_k: float = 0.08,
    lin_damp: float = 0.05,
    control_dt: float = 0.02,
    substeps: int = 4,
    gravity: float = 9.81,
    max_flight_s: float = 1.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-integrate a launched ball ``[N,3]`` and return the (x, y) where it first
    descends back through ``table_z``, plus a boolean ``valid`` mask.

    Replicates the SAME discrete dynamics the sim applies -- ``substeps`` gravity + linear
    -damping sub-updates then ONE quadratic air-drag patch per control step (mirroring
    ``events.apply_air_drag`` / ``_step_dynamics``) -- in full 3D, so the predicted landing
    matches the simulator's draggy flight rather than a gravity-only (optimistic) solve.
    Position is integrated semi-implicitly with the pre-step velocity, exactly like
    ``_height_at_range``. Pure torch, world-frame.

    ``valid`` is False where the ball never crosses ``table_z`` going down within
    ``max_flight_s`` (launched downward, stalls, or still rising at the horizon) -- the
    caller should treat those as "no landing" (zero reward), not as a landing at (0, 0)."""
    n = origin.shape[0]
    device = origin.device
    dtype = origin.dtype
    pos = origin.clone()
    vx = v_out[:, 0].clone()
    vy = v_out[:, 1].clone()
    vz = v_out[:, 2].clone()
    land_x = torch.zeros(n, device=device, dtype=dtype)
    land_y = torch.zeros(n, device=device, dtype=dtype)
    valid = torch.zeros(n, dtype=torch.bool, device=device)
    passed_apex = vz <= 0.0

    n_steps = int(round(max_flight_s / control_dt))
    sub_dt = control_dt / substeps
    damp_total = (1.0 / (1.0 + lin_damp * sub_dt)) ** substeps
    for _ in range(n_steps):
        z = pos[:, 2]
        x_next = pos[:, 0] + vx * control_dt
        y_next = pos[:, 1] + vy * control_dt
        z_next = z + vz * control_dt
        passed_apex = passed_apex | (vz <= 0.0)
        crossing = (~valid) & passed_apex & (z >= table_z) & (z_next < table_z)
        if torch.any(crossing):
            denom = (z - z_next).clamp(min=1.0e-9)
            frac = ((z - table_z) / denom).clamp(0.0, 1.0)
            land_x = torch.where(crossing, pos[:, 0] + frac * (x_next - pos[:, 0]), land_x)
            land_y = torch.where(crossing, pos[:, 1] + frac * (y_next - pos[:, 1]), land_y)
            valid = valid | crossing
        pos = torch.stack([x_next, y_next, z_next], dim=-1)
        # advance velocity: gravity + linear damping (analytic over substeps) + one quad drag patch
        vz = vz - gravity * control_dt
        vx = vx * damp_total
        vy = vy * damp_total
        vz = vz * damp_total
        speed = torch.sqrt(vx * vx + vy * vy + vz * vz).clamp(min=1.0e-9)
        factor = (1.0 - drag_k * speed * control_dt).clamp(min=0.0)
        vx = vx * factor
        vy = vy * factor
        vz = vz * factor
        if bool(valid.all()):
            break
    return land_x, land_y, valid
