from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate

from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TO_BIT
from unitree_rl_lab.tasks.table_tennis_sac.mdp.events import _ensure_tracker
from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_lin_vel, _racket_body_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def sac_quality_hit_reward(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    target_outgoing_speed: float = 3.5,
    min_up_speed: float = 0.2,
    target_up_speed: float = 1.0,
    over_up_speed: float = 2.0,
    outgoing_weight: float = 0.85,
    up_weight: float = 0.15,
    center_sigma: float = 25.0,
    center_floor: float = 0.4,
) -> torch.Tensor:
    """Reward first contact only when it sends the ball out and slightly upward,
    scaled by how close the contact is to the paddle blade center.

    ``target_outgoing_speed`` controls where the outgoing-speed score saturates: set it
    near the forward speed physically needed to clear the net (~3.5 m/s) so the gradient
    stays steep in the reachable band; too high a target flattens the gradient and the
    policy settles for a gentle block. The up-speed score is a *band* peaking at
    ``target_up_speed`` and decaying back to zero by ``over_up_speed`` -- an excessive
    loft (ball lobbed steeply upward) earns no up credit, so the policy must convert that
    energy into forward speed instead of bunting the ball straight up. The centeredness
    factor (``center_floor`` .. 1.0) makes a blade-center contact pay more than an
    edge/handle contact so the policy stops catching the ball with the handle.
    """
    _ensure_tracker(env)
    hit_event = (env._sac_step_event_mask & EVENT_TO_BIT["hit"]) != 0
    outgoing_speed = torch.nan_to_num(env._sac_hit_outgoing_speed, nan=0.0, posinf=0.0, neginf=0.0)
    up_speed = torch.nan_to_num(env._sac_hit_up_speed, nan=0.0, posinf=0.0, neginf=0.0)

    outgoing_score = (outgoing_speed / max(target_outgoing_speed, 1.0e-6)).clamp(min=0.0, max=1.0)
    up_range = max(target_up_speed - min_up_speed, 1.0e-6)
    up_rising = ((up_speed - min_up_speed) / up_range).clamp(min=0.0, max=1.0)
    over_range = max(over_up_speed - target_up_speed, 1.0e-6)
    up_decay = (1.0 - (up_speed - target_up_speed) / over_range).clamp(min=0.0, max=1.0)
    up_score = torch.minimum(up_rising, up_decay)
    quality = (outgoing_weight * outgoing_score + up_weight * up_score).clamp(min=0.0, max=1.0)

    ball: RigidObject = env.scene[ball_name]
    racket_center, _, _ = _racket_body_state(env, racket_body_name)
    center_dist_sq = torch.sum((racket_center - ball.data.root_pos_w[:, :3]) ** 2, dim=-1)
    centeredness = center_floor + (1.0 - center_floor) * torch.exp(-center_sigma * center_dist_sq)
    quality = (quality * centeredness).clamp(min=0.0, max=1.0)
    return torch.where(hit_event, quality, torch.zeros_like(quality))


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
) -> torch.Tensor:
    """Reward how close a *valid return* lands to the opponent-table target (its center).

    Fires only on the step the ``valid_return`` event is recorded, using the landing
    position captured at that step (env-local frame, same as ``target_x``/``target_y``).
    This is the depth-aware companion to the launch-direction shaping: the orientation
    reward aims the swing at this target, and this term grades whether the ball *actually*
    arrives there -- turning the otherwise binary valid_return into a placement gradient so
    the policy stops settling for a shallow ball that just clears the net.
    """
    _ensure_tracker(env)
    fired = (env._sac_step_event_mask & EVENT_TO_BIT["valid_return"]) != 0
    dx = env._sac_landing_x - target_x
    dy = env._sac_landing_y - target_y
    score = torch.exp(-(dx * dx / (2.0 * sigma_x**2) + dy * dy / (2.0 * sigma_y**2)))
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
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
    passing ~10 cm above the net). Over-lofting is left to ``sac_quality_hit_reward`` (the
    up-speed band) and ``sac_landing_placement``. Active only while the ball is still on the
    robot's side and travelling toward the net, so it shapes the approach to the net rather
    than re-scoring a ball that has already crossed.
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
    table_x_min: float = 0.0,
    table_x_max: float = 1.37,
    table_z: float = 0.76,
    sigma_x: float = 0.5,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Dense bridge toward a *valid* return: reward the *predicted* landing-x of the post-hit
    ball when it falls back to table height, scored by how close it lands to the opponent-court
    target and whether it lands in bounds at all.

    ``post_hit_net_clearance`` shapes whether the ball gets *over* the net; this is its
    companion, shaping where the arc comes *down*. The scene has no air drag, so a gravity-only
    free-flight solve gives the time to return to ``table_z`` exactly:
        t = (vz + sqrt(vz^2 + 2 g (z - table_z))) / g
    and the predicted landing ``x = ball_x + vx t``. Rewarding an in-court predicted landing
    gives a smooth every-step gradient toward a real return *before* the sparse valid_return
    event is ever sampled, so the policy stops settling for a hard forward hit that flies out
    or drops short (the bad-hit local optimum). Out-of-court predictions earn zero, so the
    gradient pulls landings *into* ``(table_x_min, table_x_max)`` rather than merely forward.
    """
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    ball_x = ball.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    vx = ball.data.root_lin_vel_w[:, 0]
    vz = ball.data.root_lin_vel_w[:, 2]
    outgoing_speed = -float(robot_side) * vx

    disc = (vz * vz + 2.0 * gravity * (ball_z - table_z)).clamp(min=0.0)
    time_to_land = (vz + torch.sqrt(disc)) / gravity
    landing_x = ball_x + vx * time_to_land

    in_bounds = (landing_x > table_x_min) & (landing_x < table_x_max)
    score = torch.exp(-((landing_x - target_x) ** 2) / (2.0 * sigma_x**2))
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
