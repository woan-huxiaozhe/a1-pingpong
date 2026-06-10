from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject

from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TO_BIT
from unitree_rl_lab.tasks.table_tennis_sac.mdp.events import _ensure_tracker
from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import _racket_body_state

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
    optimal_vel: float = 1.0,
    sigma: float = 0.8,
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    racket_pos, racket_vel, _ = _racket_body_state(env, racket_body_name)
    direction = ball.data.root_pos_w[:, :3] - racket_pos
    direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp(min=1.0e-6)
    approach_vel = torch.sum(racket_vel * direction, dim=-1)
    reward = torch.exp(-((approach_vel - optimal_vel) ** 2) / (2.0 * sigma**2))
    return torch.where(approach_vel > 0.0, reward, torch.zeros_like(reward))


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


def sac_miss_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return sac_event_reward(env, "miss")


def sac_bad_hit_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return sac_event_reward(env, "bad_hit")


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
    scaled = (outgoing_speed / max(target_speed, 1.0e-6)).clamp(-1.0, 1.0)
    active = env._sac_hit & ~env._sac_valid_return & ~env._sac_bad_hit
    return torch.where(active, scaled, torch.zeros_like(scaled))


def post_hit_lift_velocity(
    env: ManagerBasedRLEnv,
    ball_name: str,
    target_up_speed: float = 1.0,
) -> torch.Tensor:
    """Encourage a hit that keeps the ball high enough to clear the net."""
    _ensure_tracker(env)
    ball: RigidObject = env.scene[ball_name]
    scaled = (ball.data.root_lin_vel_w[:, 2] / max(target_up_speed, 1.0e-6)).clamp(-1.0, 1.0)
    active = env._sac_hit & ~env._sac_return & ~env._sac_valid_return & ~env._sac_bad_hit
    return torch.where(active, scaled, torch.zeros_like(scaled))
