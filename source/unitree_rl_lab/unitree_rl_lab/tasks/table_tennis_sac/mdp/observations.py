from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_rotate

from unitree_rl_lab.tasks.table_tennis.mdp.observations import ball_predicted_hit_point

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Paddle blade geometric center, measured from the Link_yb_paddle body origin in the
# link-local frame (a1.usd: blade plate 0.15 x 0.02 x 0.25, subtree center at local
# +Z 0.045). The body origin sits ~0.045 m below the blade center (toward the lower
# edge / handle side), which is why a zero offset rewards "handle" contact. Shifting
# the racket reference point to +0.045 makes proximity / approach / hit all radiate
# from the true paddle center. Face normal is local +Y, so the in-plane offset is +Z.
RACKET_OFFSET_Z = 0.045


def _racket_body_state(env: ManagerBasedEnv, racket_body_name: str):
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    pos = robot.data.body_pos_w[:, body_idx]
    quat = robot.data.body_quat_w[:, body_idx]
    lin_vel = robot.data.body_lin_vel_w[:, body_idx]
    ang_vel = robot.data.body_ang_vel_w[:, body_idx]
    offset_l = torch.zeros_like(pos)
    offset_l[:, 2] = RACKET_OFFSET_Z
    offset_w = quat_rotate(quat, offset_l)
    center = pos + offset_w
    center_vel = lin_vel + torch.cross(ang_vel, offset_w, dim=-1)
    return center, center_vel, quat


def _racket_body_lin_vel(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    """Translational (body-origin) linear velocity of the paddle, *without* the
    ``ang_vel x offset`` term that ``_racket_body_state`` adds for the blade center.

    Used by the swing-speed reward so a fast wrist spin (which produces large blade-center
    velocity through the 0.045 m offset moment arm) cannot farm the reward without the arm
    actually translating the paddle forward.
    """
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    return robot.data.body_lin_vel_w[:, body_idx]


def racket_pos(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    pos, _, _ = _racket_body_state(env, racket_body_name)
    return pos - env.scene.env_origins


def racket_vel(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    _, vel, _ = _racket_body_state(env, racket_body_name)
    return vel.clamp(-10.0, 10.0)


def racket_ang_vel(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    return robot.data.body_ang_vel_w[:, body_idx].clamp(-20.0, 20.0)


def racket_normal(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    _, _, quat = _racket_body_state(env, racket_body_name)
    local_normal = torch.zeros(quat.shape[0], 3, device=quat.device)
    local_normal[:, 1] = 1.0
    return quat_rotate(quat, local_normal)


def racket_axes(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    _, _, quat = _racket_body_state(env, racket_body_name)
    local_x = torch.zeros(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    local_y = torch.zeros_like(local_x)
    local_z = torch.zeros_like(local_x)
    local_x[:, 0] = 1.0
    local_y[:, 1] = 1.0
    local_z[:, 2] = 1.0
    axis_x = quat_rotate(quat, local_x)
    axis_y = quat_rotate(quat, local_y)
    axis_z = quat_rotate(quat, local_z)
    return torch.cat([axis_x, axis_y, axis_z], dim=-1)


def ball_vel_w(env: ManagerBasedEnv, ball_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    return ball.data.root_lin_vel_w.clamp(-15.0, 15.0)


def ball_pos_relative_to_racket(env: ManagerBasedEnv, ball_name: str, racket_body_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    racket_center, _, _ = _racket_body_state(env, racket_body_name)
    return (ball.data.root_pos_w[:, :3] - racket_center).clamp(-5.0, 5.0)


def ball_vel_relative_to_racket(env: ManagerBasedEnv, ball_name: str, racket_body_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    _, racket_center_vel, _ = _racket_body_state(env, racket_body_name)
    return (ball.data.root_lin_vel_w - racket_center_vel).clamp(-15.0, 15.0)


def joint_pos_delta_history(
    env: ManagerBasedEnv,
    asset_cfg,
    history_length: int = 3,
    clip: float = 1.0,
) -> torch.Tensor:
    """Recent joint-position deltas ``[q_t-q_{t-1}, q_t-q_{t-2}, ...]``.

    This gives the actor deployable velocity/trend information without depending on a
    noisy joint-velocity estimator. The buffer updates once per policy step and resets to
    zero deltas at episode start.
    """
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    attr = f"_sac_joint_pos_history_{history_length}_{asset_cfg.name}"
    step_attr = f"_sac_joint_pos_history_step_{history_length}_{asset_cfg.name}"

    if not hasattr(env, attr):
        history = joint_pos.unsqueeze(1).repeat(1, history_length + 1, 1)
        setattr(env, attr, history)
        setattr(env, step_attr, torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device))

    history = getattr(env, attr)
    last_step = getattr(env, step_attr)
    step = env.episode_length_buf.to(torch.long)

    reset = (last_step < 0) | (step == 0) | (step < last_step)
    if torch.any(reset):
        history[reset] = joint_pos[reset].unsqueeze(1).repeat(1, history_length + 1, 1)
        last_step[reset] = step[reset]

    push = step != last_step
    if torch.any(push):
        history[push] = torch.roll(history[push], shifts=1, dims=1)
        history[push, 0] = joint_pos[push]
        last_step[push] = step[push]

    deltas = history[:, 0:1] - history[:, 1:]
    return deltas.reshape(env.num_envs, history_length * joint_pos.shape[-1]).clamp(-clip, clip)


def ball_pos_history(
    env: ManagerBasedEnv,
    ball_name: str,
    history_length: int = 4,
) -> torch.Tensor:
    """Ball position history in table/world frame, newest sample first."""
    ball: RigidObject = env.scene[ball_name]
    ball_pos = (ball.data.root_pos_w[:, :3] - env.scene.env_origins).clamp(-5.0, 5.0)
    attr = f"_sac_ball_pos_history_{history_length}"
    step_attr = f"_sac_ball_pos_history_step_{history_length}"

    if not hasattr(env, attr):
        history = ball_pos.unsqueeze(1).repeat(1, history_length, 1)
        setattr(env, attr, history)
        setattr(env, step_attr, torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device))

    history = getattr(env, attr)
    last_step = getattr(env, step_attr)
    step = env.episode_length_buf.to(torch.long)

    reset = (last_step < 0) | (step == 0) | (step < last_step)
    if torch.any(reset):
        history[reset] = ball_pos[reset].unsqueeze(1).repeat(1, history_length, 1)
        last_step[reset] = step[reset]

    push = step != last_step
    if torch.any(push):
        history[push] = torch.roll(history[push], shifts=1, dims=1)
        history[push, 0] = ball_pos[push]
        last_step[push] = step[push]

    return history.reshape(env.num_envs, history_length * 3)


def hit_command_at_robot_x(
    env: ManagerBasedEnv,
    ball_name: str,
    robot_x: float,
    robot_side: int,
) -> torch.Tensor:
    """Clean simulator hit command: ``[p_hit_x, p_hit_y, p_hit_z, tau]``."""
    hit = ball_predicted_hit_point(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)
    x = torch.full((hit.shape[0], 1), robot_x, device=hit.device, dtype=hit.dtype)
    return torch.cat([x, hit[:, 0:2], hit[:, 2:3]], dim=-1)


def estimated_hit_command_at_robot_x(
    env: ManagerBasedEnv,
    ball_name: str,
    robot_x: float,
    robot_side: int,
    position_noise_std_near: float = 0.01,
    position_noise_std_far: float = 0.03,
    tau_noise_std_near: float = 0.002,
    tau_noise_std_far: float = 0.015,
    far_tau: float = 0.8,
    y_abs_limit: float = 2.0,
    z_min: float = 0.45,
    z_max: float = 2.0,
) -> torch.Tensor:
    """Deployment-style hit command with KF-like prediction error.

    The clean simulator prediction is cached with phase-dependent Gaussian error so the
    actor and critic see the same deployed command when both groups are evaluated in the
    same policy step. The strike-plane x coordinate stays fixed at ``robot_x``; noise is
    applied to predicted y/z and time-to-strike.
    """
    clean = hit_command_at_robot_x(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)
    attr = "_sac_estimated_hit_command"
    step_attr = "_sac_estimated_hit_command_step"

    if not hasattr(env, attr):
        setattr(env, attr, clean.clone())
        setattr(env, step_attr, torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device))

    estimate = getattr(env, attr)
    last_step = getattr(env, step_attr)
    step = env.episode_length_buf.to(torch.long)
    refresh = step != last_step
    if torch.any(refresh):
        next_estimate = clean.clone()
        tau = clean[:, 3:4].clamp(min=0.0)
        phase = (tau / max(far_tau, 1.0e-6)).clamp(min=0.0, max=1.0)
        position_std = position_noise_std_near + (position_noise_std_far - position_noise_std_near) * phase
        tau_std = tau_noise_std_near + (tau_noise_std_far - tau_noise_std_near) * phase

        next_estimate[:, 1:3] += torch.randn_like(next_estimate[:, 1:3]) * position_std
        next_estimate[:, 3:4] += torch.randn_like(next_estimate[:, 3:4]) * tau_std
        next_estimate[:, 1:2] = next_estimate[:, 1:2].clamp(-y_abs_limit, y_abs_limit)
        next_estimate[:, 2:3] = next_estimate[:, 2:3].clamp(z_min, z_max)
        next_estimate[:, 3:4] = next_estimate[:, 3:4].clamp(0.0, 3.0)

        estimate[refresh] = next_estimate[refresh]
        last_step[refresh] = step[refresh]

    return estimate


def predicted_hit_point_at_robot_x(
    env: ManagerBasedEnv,
    ball_name: str,
    robot_x: float,
    robot_side: int,
) -> torch.Tensor:
    return hit_command_at_robot_x(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)[:, :3]


def time_to_predicted_intercept(
    env: ManagerBasedEnv,
    ball_name: str,
    robot_x: float,
    robot_side: int,
) -> torch.Tensor:
    return hit_command_at_robot_x(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)[:, 3:4]
