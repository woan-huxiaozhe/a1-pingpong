from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_rotate

from unitree_rl_lab.tasks.table_tennis.mdp.observations import ball_predicted_hit_point

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

RACKET_OFFSET_Z = 0.0


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


def racket_pos(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    pos, _, _ = _racket_body_state(env, racket_body_name)
    return pos - env.scene.env_origins


def racket_vel(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    _, vel, _ = _racket_body_state(env, racket_body_name)
    return vel.clamp(-10.0, 10.0)


def racket_normal(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    _, _, quat = _racket_body_state(env, racket_body_name)
    local_normal = torch.zeros(quat.shape[0], 3, device=quat.device)
    local_normal[:, 1] = 1.0
    return quat_rotate(quat, local_normal)


def ball_pos_relative_to_racket(env: ManagerBasedEnv, ball_name: str, racket_body_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    racket_center, _, _ = _racket_body_state(env, racket_body_name)
    return (ball.data.root_pos_w[:, :3] - racket_center).clamp(-5.0, 5.0)


def ball_vel_relative_to_racket(env: ManagerBasedEnv, ball_name: str, racket_body_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    _, racket_center_vel, _ = _racket_body_state(env, racket_body_name)
    return (ball.data.root_lin_vel_w - racket_center_vel).clamp(-15.0, 15.0)


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


def predicted_hit_point_at_robot_x(
    env: ManagerBasedEnv,
    ball_name: str,
    robot_x: float,
    robot_side: int,
) -> torch.Tensor:
    hit = ball_predicted_hit_point(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)
    x = torch.full((hit.shape[0], 1), robot_x, device=hit.device, dtype=hit.dtype)
    return torch.cat([x, hit[:, 0:2]], dim=-1)


def time_to_predicted_intercept(
    env: ManagerBasedEnv,
    ball_name: str,
    robot_x: float,
    robot_side: int,
) -> torch.Tensor:
    hit = ball_predicted_hit_point(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)
    return hit[:, 2:3]
