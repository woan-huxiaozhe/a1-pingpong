from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_rotate

from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TO_BIT
from unitree_rl_lab.tasks.table_tennis_sac.mdp.events import _ensure_tracker
from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_lin_vel, _racket_body_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def racket_ball_proximity_dense(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    sigma: float = 10.0,
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    racket_pos, _, _ = _racket_body_state(env, racket_body_name)
    dist_sq = torch.sum((racket_pos - ball.data.root_pos_w[:, :3]) ** 2, dim=-1)
    return torch.exp(-sigma * dist_sq)


def racket_approach_ball(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    target_vel: float = 3.0,
) -> torch.Tensor:
    """Reward racket speed toward the ball, monotonically up to ``target_vel``.

    The previous Gaussian-around-``optimal_vel`` form peaked at ~1 m/s and actively
    penalized faster swings, which capped hit strength. This version never punishes a
    faster approach; it grows linearly with closing speed and only saturates once the
    racket is closing fast enough to drive a real return.
    """
    ball: RigidObject = env.scene[ball_name]
    racket_pos, racket_vel, _ = _racket_body_state(env, racket_body_name)
    direction = ball.data.root_pos_w[:, :3] - racket_pos
    direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp(min=1.0e-6)
    approach_vel = torch.sum(racket_vel * direction, dim=-1)
    return (approach_vel / max(target_vel, 1.0e-6)).clamp(min=0.0, max=1.0)


def _desired_launch_dir(
    racket_pos: torch.Tensor,
    target: torch.Tensor,
    launch_speed: float,
    u_min: float = 0.176,  # tan(10 deg): never aim the face down at the table
    u_max: float = 3.732,  # tan(75 deg): never aim near-vertical
    gravity: float = 9.81,
) -> torch.Tensor:
    """Unit launch-velocity direction that lands a projectile at ``target``.

    Instead of pointing the paddle normal *at* a fixed point (which, for a target near
    net height, asks for a nearly-horizontal/slightly-downward face -- the wrong launch
    direction, since a ball must be thrown *up-and-forward* to arc onto the table), this
    solves the projectile equations for a fixed ``launch_speed`` and returns the
    up-forward direction that actually reaches ``target``.

    Vertical-plane solve (theta above horizontal, u = tan(theta), R = horizontal range,
    dz = target_z - launch_z, k = g R^2 / (2 v0^2)):
        k u^2 - R u + (dz + k) = 0  ->  u = (R - sqrt(R^2 - 4 k (dz + k))) / (2 k)
    The minus root is the *low* (flatter, drive-like) arc. When the target is out of reach
    at ``launch_speed`` the discriminant goes negative; clamping it to 0 degrades u to the
    max-range angle R/(2k), so the direction stays smooth and always up-forward.
    """
    h = target - racket_pos
    hx = h[:, 0]
    hy = h[:, 1]
    dz = h[:, 2]
    rng = torch.sqrt(hx * hx + hy * hy).clamp(min=1.0e-4)
    k = gravity * rng * rng / (2.0 * max(launch_speed, 1.0e-3) ** 2)
    disc = (rng * rng - 4.0 * k * (dz + k)).clamp(min=0.0)
    u = (rng - torch.sqrt(disc)) / (2.0 * k.clamp(min=1.0e-6))
    u = u.clamp(min=u_min, max=u_max)
    d = torch.stack([hx / rng, hy / rng, u], dim=-1)
    return d / torch.norm(d, dim=-1, keepdim=True).clamp(min=1.0e-6)


def _launch_target(env: ManagerBasedRLEnv, racket_pos: torch.Tensor, target_x, target_y, target_z) -> torch.Tensor:
    target = torch.zeros_like(racket_pos)
    target[:, 0] = env.scene.env_origins[:, 0] + target_x
    target[:, 1] = env.scene.env_origins[:, 1] + target_y
    target[:, 2] = env.scene.env_origins[:, 2] + target_z
    return target


def racket_face_toward_target(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    target_x: float,
    target_y: float = 0.0,
    target_z: float = 0.76,
    launch_speed: float = 4.5,
    proximity_gate: float = 0.55,
) -> torch.Tensor:
    """Reward aligning the paddle normal with the up-forward launch direction that lands
    the ball at ``(target_x, target_y, target_z)`` (opponent table center, table height)."""
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    racket_pos, _, racket_quat = _racket_body_state(env, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    target = _launch_target(env, racket_pos, target_x, target_y, target_z)
    desired_dir = _desired_launch_dir(racket_pos, target, launch_speed)

    local_normal = torch.zeros_like(racket_pos)
    local_normal[:, 1] = 1.0
    racket_normal = quat_rotate(racket_quat, local_normal)
    alignment = torch.sum(racket_normal * desired_dir, dim=-1).clamp(min=0.0, max=1.0)

    incoming = ball.data.root_lin_vel_w[:, 0].abs() > 0.1
    active = (dist < proximity_gate) & incoming & ~env._sac_hit & ~env._sac_miss
    return torch.where(active, alignment, torch.zeros_like(alignment))


def racket_normal_swing_velocity(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    target_x: float,
    target_y: float = 0.0,
    target_z: float = 0.76,
    launch_speed: float = 4.5,
    target_speed: float = 1.0,
    proximity_gate: float = 0.45,
) -> torch.Tensor:
    """Reward translational paddle speed along the face normal while the normal points in
    the up-forward launch direction. Uses body-origin (translational) velocity, *not* the
    blade-center velocity, so a wrist flick through the 0.045 m offset cannot farm it."""
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    racket_pos, _, racket_quat = _racket_body_state(env, racket_body_name)
    racket_lin_vel = _racket_body_lin_vel(env, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    target = _launch_target(env, racket_pos, target_x, target_y, target_z)
    desired_dir = _desired_launch_dir(racket_pos, target, launch_speed)

    local_normal = torch.zeros_like(racket_pos)
    local_normal[:, 1] = 1.0
    racket_normal = quat_rotate(racket_quat, local_normal)
    alignment = torch.sum(racket_normal * desired_dir, dim=-1).clamp(min=0.0, max=1.0)
    normal_speed = torch.sum(racket_lin_vel * racket_normal, dim=-1)
    speed_reward = (normal_speed / max(target_speed, 1.0e-6)).clamp(min=0.0, max=1.0)

    incoming = ball.data.root_lin_vel_w[:, 0].abs() > 0.1
    active = (dist < proximity_gate) & incoming & ~env._sac_hit & ~env._sac_miss
    return torch.where(active, alignment * speed_reward, torch.zeros_like(speed_reward))


def racket_forward_push_velocity(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    robot_side: int,
    target_speed: float = 1.0,
    distance_threshold: float = 0.55,
) -> torch.Tensor:
    """Encourage a simple forward push before contact instead of passive blocking."""
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    racket_pos, racket_vel, _ = _racket_body_state(env, racket_body_name)
    dist = torch.norm(racket_pos - ball.data.root_pos_w[:, :3], dim=-1)
    incoming = ball.data.root_lin_vel_w[:, 0] * float(robot_side) > 0.0
    forward_speed = -float(robot_side) * racket_vel[:, 0]
    scaled = (forward_speed / max(target_speed, 1.0e-6)).clamp(-1.0, 1.0)
    active = (dist < distance_threshold) & incoming & ~env._sac_hit & ~env._sac_miss
    return torch.where(active, scaled, torch.zeros_like(scaled))


def sac_event_reward(env: ManagerBasedRLEnv, event: str) -> torch.Tensor:
    _ensure_tracker(env)
    bit = EVENT_TO_BIT[event]
    return ((env._sac_step_event_mask & bit) != 0).float()


def _center_gate_factor(
    env: ManagerBasedRLEnv,
    center_sigma: float,
    center_gate_floor: float,
) -> torch.Tensor:
    """Per-hit multiplicative centering gate ``floor + (1 - floor) * exp(-sigma * offset^2)``.

    Uses the in-plane ball-to-blade-center offset cached at first contact
    (``_sac_hit_center_offset``), which persists for the rest of the episode, so the factor is
    valid both on the hit step and on a later outcome step (e.g. ``valid_return``). It is 1.0 at
    perfect center and decays toward ``center_gate_floor`` at the blade edge -- turning an
    additive "center bonus" into a price-of-admission multiplier on whichever reward applies it.
    """
    center_offset = torch.nan_to_num(env._sac_hit_center_offset, nan=1.0, posinf=1.0, neginf=1.0)
    factor = torch.exp(-center_sigma * center_offset * center_offset)
    return center_gate_floor + (1.0 - center_gate_floor) * factor


def sac_quality_hit_reward(
    env: ManagerBasedRLEnv,
    min_outgoing_speed: float = 1.5,
    good_outgoing_speed: float = 3.5,
    min_up_speed: float = -0.2,
    up_tolerance: float = 0.4,
    center_sigma: float = 0.0,
    center_gate_floor: float = 0.0,
) -> torch.Tensor:
    """Reward first contact for basic outgoing quality, optionally gated by contact centering.

    The term asks for enough outgoing x speed to be returnable and uses a shallow vertical
    gate to reject downward hits without prescribing a fixed upward launch speed. It does not
    score final placement.

    When ``center_sigma > 0`` the quality is *multiplied* by a centering factor
    ``floor + (1 - floor) * exp(-center_sigma * offset^2)``, so an off-center (blade-edge)
    contact cannot collect the full quality reward. This is the multiplicative-gate
    counterpart to the additive ``hit_centered`` bonus: an edge hit simply forgoes the small
    additive bonus while still farming the much larger ``valid_return`` / ``landing`` rewards,
    so additive weight has weak leverage on a policy that already returns reliably; gating
    instead makes near-center contact the *price of admission* to the quality reward. Edge
    restitution is highly sim-specific and will not transfer, so this also pushes the policy
    toward a sim-to-real-robust contact. ``center_gate_floor`` keeps a reward fraction at any
    offset so the gate stays a gradient rather than a cliff (which the converged-but-oscillating
    policy is sensitive to). The offset is the per-hit value recorded at first contact, so it is
    valid on the hit step where this term fires.
    """
    _ensure_tracker(env)
    hit_event = (env._sac_step_event_mask & EVENT_TO_BIT["hit"]) != 0
    outgoing_speed = torch.nan_to_num(env._sac_hit_outgoing_speed, nan=0.0, posinf=0.0, neginf=0.0)
    up_speed = torch.nan_to_num(env._sac_hit_up_speed, nan=0.0, posinf=0.0, neginf=0.0)

    outgoing_range = max(good_outgoing_speed - min_outgoing_speed, 1.0e-6)
    outgoing_score = ((outgoing_speed - min_outgoing_speed) / outgoing_range).clamp(min=0.0, max=1.0)
    vertical_gate = ((up_speed - min_up_speed) / max(up_tolerance, 1.0e-6)).clamp(min=0.0, max=1.0)
    quality = outgoing_score * vertical_gate
    if center_sigma > 0.0:
        quality = quality * _center_gate_factor(env, center_sigma, center_gate_floor)
    return torch.where(hit_event, quality, torch.zeros_like(quality))


def sac_centered_hit_reward(
    env: ManagerBasedRLEnv,
    sigma: float = 220.0,
) -> torch.Tensor:
    """Reward first contact by how close it is to the blade center in the racket plane."""
    _ensure_tracker(env)
    hit_event = (env._sac_step_event_mask & EVENT_TO_BIT["hit"]) != 0
    center_offset = torch.nan_to_num(env._sac_hit_center_offset, nan=1.0, posinf=1.0, neginf=1.0)
    score = torch.exp(-sigma * center_offset * center_offset)
    return torch.where(hit_event, score, torch.zeros_like(score))


def sac_miss_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return sac_event_reward(env, "miss")


def sac_bad_hit_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return sac_event_reward(env, "bad_hit")


def sac_landing_placement(
    env: ManagerBasedRLEnv,
    target_x: float,
    target_y: float = 0.0,
    sigma_x: float = 0.35,
    sigma_y: float = 0.4,
    center_sigma: float = 0.0,
    center_gate_floor: float = 0.0,
) -> torch.Tensor:
    """Reward how close a *valid return* lands to the opponent-table target (its center).

    Fires only on the step the ``valid_return`` event is recorded, using the landing
    position captured at that step (env-local frame, same as ``target_x``/``target_y``).
    This is the depth-aware companion to the launch-direction shaping: the orientation
    reward aims the swing at this target, and this term grades whether the ball *actually*
    arrives there -- turning the otherwise binary valid_return into a placement gradient so
    the policy stops settling for a shallow ball that just clears the net.

    When ``center_sigma > 0`` the placement score is multiplied by the per-hit centering gate
    (see :func:`_center_gate_factor`). This is the high-leverage half of the center-contact
    fix: ``landing_placement`` is the largest reward an edge hit can farm while still ignoring
    where the ball was struck on the blade, so gating it -- not just ``quality_hit`` -- is what
    actually puts enough reward mass behind center contact to move the policy off the blade
    edge. Semantically it reads as "a good return = lands near target *and* was struck cleanly".
    """
    _ensure_tracker(env)
    fired = (env._sac_step_event_mask & EVENT_TO_BIT["valid_return"]) != 0
    dx = env._sac_landing_x - target_x
    dy = env._sac_landing_y - target_y
    score = torch.exp(-(dx * dx / (2.0 * sigma_x**2) + dy * dy / (2.0 * sigma_y**2)))
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if center_sigma > 0.0:
        score = score * _center_gate_factor(env, center_sigma, center_gate_floor)
    return torch.where(fired, score, torch.zeros_like(score))


def joint_limit_margin_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.20,
) -> torch.Tensor:
    """Penalize joints before they reach the hard position limits."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lower_clearance = joint_pos - limits[..., 0]
    upper_clearance = limits[..., 1] - joint_pos
    clearance = torch.minimum(lower_clearance, upper_clearance)
    normalized = ((margin - clearance) / max(margin, 1.0e-6)).clamp(min=0.0)
    return torch.sum(normalized.square(), dim=-1)


def joint_effort_margin_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin_frac: float = 0.85,
) -> torch.Tensor:
    """Penalize joint torques as they approach the actuator effort limit.

    A1's wrist joints (yb_4..7) cap at 8 Nm versus 28 Nm for the proximal joints, and the
    catch policy drives joint_4 into that 8 Nm ceiling on a few percent of steps. The
    saturation is a quasi-static holding-torque deficit -- the joint is nearly still yet the
    PD demands more than 8 Nm to hold the commanded pose -- so it cannot be relieved by a
    faster control rate; it degrades tracking ~6x and will be worse on the real arm. This
    term gives a smooth gradient that steers the policy away from the ceiling: zero below
    ``margin_frac`` of each joint's *own* effort limit, ramping quadratically to 1.0 at the
    limit. Normalizing by each joint's own limit puts the 8 Nm and 28 Nm joints on equal
    fractional footing. ``applied_torque`` is the PhysX-clamped torque, so the penalty
    saturates at the limit; the gradient lives in the ``[margin_frac, 1.0]`` band, which is
    where the policy must learn to back off before it clips.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    tau = robot.data.applied_torque[:, asset_cfg.joint_ids].abs()
    limit = robot.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp(min=1.0e-6)
    over = ((tau / limit - margin_frac) / max(1.0 - margin_frac, 1.0e-6)).clamp(min=0.0, max=1.0)
    return torch.sum(over.square(), dim=-1)


def post_hit_outgoing_velocity(
    env: ManagerBasedRLEnv,
    ball_name: str,
    robot_side: int,
    target_speed: float = 2.0,
) -> torch.Tensor:
    """Shape hit outcomes toward the opponent half before sparse return events exist."""
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    outgoing_speed = -float(robot_side) * ball.data.root_lin_vel_w[:, 0]
    scaled = (outgoing_speed / max(target_speed, 1.0e-6)).clamp(0.0, 1.0)
    active = env._sac_hit & ~env._sac_valid_return & ~env._sac_bad_hit
    return torch.where(active, scaled, torch.zeros_like(scaled))


def post_hit_net_clearance(
    env: ManagerBasedRLEnv,
    ball_name: str,
    robot_side: int,
    net_x: float = 0.0,
    net_z: float = 0.9125,
    ramp_low: float = -0.6,
    ramp_high: float = 0.1,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Dense bridge toward an actual return: reward the *predicted* ball height at the net.

    The sparse return/valid_return events only fire once the ball is already above the net
    at ``x = net_x``; before the policy can ever produce such a hit they give zero gradient,
    so training settles into a steep lob that farms ``post_hit_outgoing`` /
    ``post_hit_net_progress`` (both height-blind) and then times out as a bad hit. This term
    closes that gap. The scene has no air drag, so a gravity-only projectile solve predicts
    the ball's height when it reaches the net plane *exactly*; rewarding that predicted
    clearance gives a smooth, every-step gradient that rises as the hit gets flatter /
    faster / struck from a higher contact point -- i.e. toward a real return.

    ``ramp_low`` starts the slope *below* the current lob's clearance so there is a non-zero
    gradient at the policy's operating point; the score saturates at ``ramp_high`` (ball
    passing ~10 cm above the net). Over-lofting can be handled by the hit-time quality term
    or the optional ``post_hit_lob_penalty`` ablation. Active only while the ball is still on
    the robot's side and travelling toward the net, so it shapes the approach to the net
    rather than re-scoring a ball that has already crossed.
    """
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    ball_x = ball.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    outgoing_speed = -float(robot_side) * ball.data.root_lin_vel_w[:, 0]
    vz = ball.data.root_lin_vel_w[:, 2]
    time_to_net = (net_x - ball_x).abs() / outgoing_speed.clamp(min=0.2)
    z_at_net = ball_z + vz * time_to_net - 0.5 * gravity * time_to_net * time_to_net
    clearance = z_at_net - net_z
    score = ((clearance - ramp_low) / max(ramp_high - ramp_low, 1.0e-6)).clamp(min=0.0, max=1.0)
    before_net = (ball_x - net_x) * float(robot_side) > 0.0
    active = env._sac_hit & ~env._sac_valid_return & ~env._sac_bad_hit & (outgoing_speed > 0.2) & before_net
    return torch.where(active, score, torch.zeros_like(score))


def post_hit_net_progress(
    env: ManagerBasedRLEnv,
    ball_name: str,
    robot_side: int,
    robot_x: float,
    net_x: float = 0.0,
    overshoot: float = 1.2,
) -> torch.Tensor:
    """Reward how far the post-hit ball travels toward (and past) the net.

    The bad_hit outcome currently fires on every episode, so its constant penalty gives
    no gradient that distinguishes "ball driven almost to the net" from "ball lobbed
    straight up and fell short near the robot". This term grades the post-hit ball by its
    forward progress from the robot toward the net (0 at the racket, 1.0 at the net plane,
    up to ``overshoot`` past it), so the policy gets a smooth gradient toward an actual
    return even before the sparse return/valid_return events are ever sampled.
    """
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    ball_x = ball.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress = -float(robot_side) * (ball_x - robot_x)
    span = abs(net_x - robot_x)
    scaled = (progress / max(span, 1.0e-6)).clamp(min=0.0, max=overshoot)
    active = env._sac_hit & ~env._sac_valid_return & ~env._sac_bad_hit
    return torch.where(active, scaled, torch.zeros_like(scaled))


def post_hit_landing_prediction(
    env: ManagerBasedRLEnv,
    ball_name: str,
    robot_side: int,
    target_x: float,
    target_y: float = 0.0,
    table_x_min: float = 0.0,
    table_x_max: float = 1.37,
    table_y_half: float = 0.7625,
    table_z: float = 0.76,
    sigma_x: float = 0.5,
    sigma_y: float = 0.35,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Dense bridge toward a *valid* return: reward the predicted post-hit landing.

    ``post_hit_net_clearance`` shapes whether the ball gets *over* the net; this is its
    companion, shaping where the arc comes *down*. The scene has no air drag, so a gravity-only
    free-flight solve gives the time to return to ``table_z`` exactly:
        t = (vz + sqrt(vz^2 + 2 g (z - table_z))) / g
    and the predicted landing is ``(x = ball_x + vx t, y = ball_y + vy t)``. Rewarding an
    in-court predicted landing near the opponent-table center gives a smooth every-step
    gradient toward a real return *before* the sparse valid_return event is ever sampled.
    Out-of-court predictions earn zero, so the gradient pulls landings into the table instead
    of merely forward.
    """
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    ball_x = ball.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    vx = ball.data.root_lin_vel_w[:, 0]
    vy = ball.data.root_lin_vel_w[:, 1]
    vz = ball.data.root_lin_vel_w[:, 2]
    outgoing_speed = -float(robot_side) * vx

    disc = (vz * vz + 2.0 * gravity * (ball_z - table_z)).clamp(min=0.0)
    time_to_land = (vz + torch.sqrt(disc)) / gravity
    landing_x = ball_x + vx * time_to_land
    landing_y = (ball.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]) + vy * time_to_land

    in_bounds = (landing_x > table_x_min) & (landing_x < table_x_max) & (landing_y.abs() < table_y_half)
    score = torch.exp(
        -(
            (landing_x - target_x) ** 2 / (2.0 * sigma_x**2)
            + (landing_y - target_y) ** 2 / (2.0 * sigma_y**2)
        )
    )
    score = torch.where(in_bounds, score, torch.zeros_like(score))
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    above_table = ball_z > table_z
    active = (
        env._sac_hit
        & ~env._sac_valid_return
        & ~env._sac_bad_hit
        & (outgoing_speed > 0.2)
        & above_table
    )
    return torch.where(active, score, torch.zeros_like(score))


def post_hit_lob_penalty(
    env: ManagerBasedRLEnv,
    ball_name: str,
    robot_side: int,
    max_height: float = 1.25,
    height_band: float = 0.35,
    max_up_speed: float = 1.6,
    up_speed_band: float = 1.2,
) -> torch.Tensor:
    """Penalize post-hit lobs so the policy prefers a flatter drive over a high arc."""
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    outgoing_speed = -float(robot_side) * ball.data.root_lin_vel_w[:, 0]
    vz = ball.data.root_lin_vel_w[:, 2]
    height_excess = ((ball_z - max_height) / max(height_band, 1.0e-6)).clamp(min=0.0, max=1.0)
    up_excess = ((vz - max_up_speed) / max(up_speed_band, 1.0e-6)).clamp(min=0.0, max=1.0)
    score = torch.maximum(height_excess, up_excess)
    active = env._sac_hit & ~env._sac_valid_return & ~env._sac_bad_hit & (outgoing_speed > 0.2)
    return torch.where(active, score, torch.zeros_like(score))


class joint_jerk_l2(ManagerTermBase):
    """Penalize joint jerk (finite-difference rate of change of joint acceleration).

    ``jerk_t = (acc_t - acc_{t-1}) / step_dt`` is computed per control step from the
    simulated joint acceleration. Unlike ``action_rate_l2`` (a first-order penalty on the
    policy output), an L2 jerk penalty directly punishes the single-control-step velocity
    spikes / acceleration reversals (the "bang-bang" flicks) that a real PD-controlled arm
    cannot reproduce -- without globally penalizing the legitimate swing acceleration the way
    a large ``joint_acc_l2`` weight does.

    The previous acceleration is cached per environment and zeroed on reset so the first step
    of a new episode does not register a spurious jerk against the prior episode's motion.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Lazily sized on the first call once the joint selection is known.
        self._prev_acc: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._prev_acc is None:
            return
        if env_ids is None:
            self._prev_acc[:] = 0.0
        else:
            self._prev_acc[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
        if self._prev_acc is None or self._prev_acc.shape != acc.shape:
            self._prev_acc = torch.zeros_like(acc)
        jerk = (acc - self._prev_acc) / env.step_dt
        self._prev_acc = acc.detach().clone()
        return torch.sum(torch.square(jerk), dim=1)
