from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate

from unitree_rl_lab.tasks.table_tennis.mdp.events import launch_ball
from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TO_BIT
from unitree_rl_lab.tasks.table_tennis_sac.mdp.observations import RACKET_OFFSET_Z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _ensure_tracker(env: ManagerBasedEnv):
    n = env.num_envs
    device = env.device
    specs = {
        "_sac_min_dist": (torch.float32, float("inf")),
        "_sac_final_min_dist": (torch.float32, float("inf")),
        "_sac_landing_y": (torch.float32, float("nan")),
        "_sac_final_landing_y": (torch.float32, float("nan")),
        "_sac_landing_x": (torch.float32, float("nan")),
        "_sac_final_landing_x": (torch.float32, float("nan")),
        "_sac_hit_center_offset": (torch.float32, float("nan")),
        "_sac_final_hit_center_offset": (torch.float32, float("nan")),
        # Racket body angular-velocity magnitude at first contact, used by the R_omega
        # (sac_racket_spin_penalty) clean-contact reward that replaces the center gate.
        "_sac_hit_racket_ang_vel": (torch.float32, float("nan")),
        "_sac_final_hit_racket_ang_vel": (torch.float32, float("nan")),
        # Ball xy at the first post-hit descent through table height, used by the tier-1
        # sac_table_proximity bootstrap (records once per episode).
        "_sac_table_cross_x": (torch.float32, float("nan")),
        "_sac_final_table_cross_x": (torch.float32, float("nan")),
        "_sac_table_cross_y": (torch.float32, float("nan")),
        "_sac_final_table_cross_y": (torch.float32, float("nan")),
        "_sac_hit_outgoing_speed": (torch.float32, float("nan")),
        "_sac_hit_up_speed": (torch.float32, float("nan")),
        "_sac_post_hit_max_outgoing_speed": (torch.float32, float("-inf")),
        "_sac_post_hit_max_height": (torch.float32, float("-inf")),
        "_sac_final_hit_outgoing_speed": (torch.float32, float("nan")),
        "_sac_final_hit_up_speed": (torch.float32, float("nan")),
        "_sac_final_post_hit_max_outgoing_speed": (torch.float32, float("-inf")),
        "_sac_final_post_hit_max_height": (torch.float32, float("-inf")),
        "_sac_closest_step": (torch.long, -1),
        "_sac_hit_step": (torch.long, -1),
        "_sac_return_step": (torch.long, -1),
        "_sac_valid_return_step": (torch.long, -1),
        "_sac_bad_hit_step": (torch.long, -1),
        "_sac_miss_step": (torch.long, -1),
        "_sac_final_closest_step": (torch.long, -1),
        "_sac_final_hit_step": (torch.long, -1),
        "_sac_final_return_step": (torch.long, -1),
        "_sac_final_valid_return_step": (torch.long, -1),
        "_sac_final_bad_hit_step": (torch.long, -1),
        "_sac_final_miss_step": (torch.long, -1),
        "_sac_step_event_mask": (torch.long, 0),
        "_sac_final_event_mask": (torch.long, 0),
    }
    for name, (dtype, fill) in specs.items():
        if not hasattr(env, name):
            setattr(env, name, torch.full((n,), fill, dtype=dtype, device=device))
    for name in (
        "_sac_hit",
        "_sac_return",
        "_sac_valid_return",
        "_sac_bad_hit",
        "_sac_miss",
        "_sac_near_miss",
    ):
        if not hasattr(env, name):
            setattr(env, name, torch.zeros(n, dtype=torch.bool, device=device))


def reset_sac_episode_state(env: ManagerBasedEnv, env_ids: torch.Tensor | None = None):
    _ensure_tracker(env)
    ids = _env_ids(env, env_ids)
    for name in ("_sac_hit", "_sac_return", "_sac_valid_return", "_sac_bad_hit", "_sac_miss", "_sac_near_miss"):
        getattr(env, name)[ids] = False
    env._sac_min_dist[ids] = float("inf")
    env._sac_landing_y[ids] = float("nan")
    env._sac_landing_x[ids] = float("nan")
    env._sac_hit_center_offset[ids] = float("nan")
    env._sac_hit_racket_ang_vel[ids] = float("nan")
    env._sac_table_cross_x[ids] = float("nan")
    env._sac_table_cross_y[ids] = float("nan")
    env._sac_hit_outgoing_speed[ids] = float("nan")
    env._sac_hit_up_speed[ids] = float("nan")
    env._sac_post_hit_max_outgoing_speed[ids] = float("-inf")
    env._sac_post_hit_max_height[ids] = float("-inf")
    for name in (
        "_sac_closest_step",
        "_sac_hit_step",
        "_sac_return_step",
        "_sac_valid_return_step",
        "_sac_bad_hit_step",
        "_sac_miss_step",
    ):
        getattr(env, name)[ids] = -1
    env._sac_step_event_mask[ids] = 0


def reset_robot_to_ready_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
    joint_pos: list[float] | tuple[float, ...] | None = None,
    action_name: str = "right_arm",
    reset_root: bool = True,
):
    """Reset the fixed-base robot and controlled arm before every ball episode."""
    ids = _env_ids(env, env_ids)
    robot: Articulation = env.scene[asset_cfg.name]

    if reset_root:
        root_state = robot.data.default_root_state[ids].clone()
        root_state[:, :3] += env.scene.env_origins[ids]
        root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_state, env_ids=ids)

    full_joint_pos = robot.data.default_joint_pos[ids].clone()
    joint_ids, _ = robot.find_joints(joint_names)
    if joint_pos is not None:
        ready = torch.tensor(joint_pos, dtype=full_joint_pos.dtype, device=env.device).reshape(1, -1)
        if ready.shape[1] != len(joint_ids):
            raise ValueError(f"Expected {len(joint_ids)} ready joints, got {ready.shape[1]}.")
        full_joint_pos[:, joint_ids] = ready

    full_joint_vel = torch.zeros_like(full_joint_pos)
    robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=ids)

    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return
    try:
        action = action_manager.get_term(action_name)
    except Exception:
        return
    if hasattr(action, "_raw_actions"):
        action._raw_actions[ids] = 0.0
    if hasattr(action, "_processed_actions") and hasattr(action, "_joint_ids"):
        action._processed_actions[ids] = full_joint_pos[:, action._joint_ids]
    if hasattr(action, "_limit_violation"):
        action._limit_violation[ids] = False


def _racket_pose_from_sensor(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    racket_body_name = getattr(sensor_cfg, "_cached_racket_body_name", None)
    if racket_body_name is None:
        racket_body_name = contact_sensor.body_names[sensor_cfg.body_ids[0]]
        sensor_cfg._cached_racket_body_name = racket_body_name
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    pos = robot.data.body_pos_w[:, body_idx]
    # Shift to the paddle blade center so the hit/near-miss distance gate is measured
    # from the same sweet spot as the reward shaping (see observations.RACKET_OFFSET_Z).
    quat = robot.data.body_quat_w[:, body_idx]
    offset_l = torch.zeros_like(pos)
    offset_l[:, 2] = RACKET_OFFSET_Z
    return pos + quat_rotate(quat, offset_l), quat


def _racket_pos_from_sensor(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return _racket_pose_from_sensor(env, sensor_cfg)[0]


def update_sac_episode_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    ball_name: str,
    robot_x: float,
    robot_side: int,
    table_z: float = 0.76,
    net_x: float = 0.0,
    net_z: float = 0.9125,
    own_table_x_min: float = -1.37,
    own_table_x_max: float = 0.0,
    opponent_table_x_min: float = 0.0,
    opponent_table_x_max: float = 1.37,
    table_y_half: float = 0.7625,
    hit_distance_threshold: float = 0.15,
    near_miss_threshold: float = 0.25,
    miss_margin: float = 0.15,
    out_x_limit: float = 3.0,
    out_y_limit: float = 1.4,
    z_min: float = 0.45,
    hit_return_timeout_s: float = 0.40,
):
    _ensure_tracker(env)
    ids = _env_ids(env, env_ids)
    env._sac_step_event_mask[ids] = 0

    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w
    outgoing_speed = -float(robot_side) * ball_vel[:, 0]
    racket_pos_w, racket_quat = _racket_pose_from_sensor(env, sensor_cfg)
    racket_pos = racket_pos_w - env.scene.env_origins
    # Racket-body angular-velocity magnitude (same source as mdp.racket_ang_vel), cached at
    # first contact for the R_omega clean-contact reward.
    robot = env.scene["robot"]
    racket_body_name = getattr(sensor_cfg, "_cached_racket_body_name", None)
    racket_body_idx = robot.body_names.index(racket_body_name)
    racket_ang_vel_mag = torch.norm(robot.data.body_ang_vel_w[:, racket_body_idx], dim=-1)
    dist = torch.norm(racket_pos - ball_pos, dim=-1)
    rel_w = ball.data.root_pos_w[:, :3] - racket_pos_w
    local_x = torch.zeros_like(rel_w)
    local_z = torch.zeros_like(rel_w)
    local_x[:, 0] = 1.0
    local_z[:, 2] = 1.0
    axis_x = quat_rotate(racket_quat, local_x)
    axis_z = quat_rotate(racket_quat, local_z)
    hit_center_offset = torch.sqrt(
        torch.sum(rel_w * axis_x, dim=-1).square() + torch.sum(rel_w * axis_z, dim=-1).square()
    )
    step = env.episode_length_buf.to(torch.long)

    closer = dist < env._sac_min_dist
    env._sac_min_dist[closer] = dist[closer]
    env._sac_closest_step[closer] = step[closer]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_magnitude = torch.norm(net_forces, dim=-1).max(dim=1).values.squeeze(-1)
    hit_now = (force_magnitude > 0.1) & (dist < hit_distance_threshold)
    new_hit = hit_now & ~env._sac_hit
    env._sac_hit[new_hit] = True
    env._sac_hit_step[new_hit] = step[new_hit]
    env._sac_hit_center_offset[new_hit] = hit_center_offset[new_hit]
    env._sac_hit_racket_ang_vel[new_hit] = racket_ang_vel_mag[new_hit]
    env._sac_hit_outgoing_speed[new_hit] = outgoing_speed[new_hit]
    env._sac_hit_up_speed[new_hit] = ball_vel[new_hit, 2]
    env._sac_step_event_mask[new_hit] |= EVENT_TO_BIT["hit"]

    post_hit_active = env._sac_hit & ~env._sac_valid_return & ~env._sac_bad_hit
    if torch.any(post_hit_active):
        env._sac_post_hit_max_outgoing_speed[post_hit_active] = torch.maximum(
            env._sac_post_hit_max_outgoing_speed[post_hit_active],
            outgoing_speed[post_hit_active],
        )
        env._sac_post_hit_max_height[post_hit_active] = torch.maximum(
            env._sac_post_hit_max_height[post_hit_active],
            ball_pos[post_hit_active, 2],
        )

    crossed_net = (
        env._sac_hit
        & ~env._sac_return
        & ((ball_pos[:, 0] - net_x) * robot_side < 0)
        & (ball_vel[:, 0] * robot_side < -0.5)
        & (ball_pos[:, 2] > net_z)
    )
    env._sac_return[crossed_net] = True
    env._sac_return_step[crossed_net] = step[crossed_net]
    env._sac_step_event_mask[crossed_net] |= EVENT_TO_BIT["return"]

    near_table = (ball_pos[:, 2] > table_z) & (ball_pos[:, 2] < table_z + 0.08)
    in_bounds_y = ball_pos[:, 1].abs() < table_y_half
    going_down = ball_vel[:, 2] < 0
    on_opponent = (ball_pos[:, 0] > opponent_table_x_min) & (ball_pos[:, 0] < opponent_table_x_max)
    on_own = (ball_pos[:, 0] > own_table_x_min) & (ball_pos[:, 0] < own_table_x_max)
    opponent_landing = env._sac_hit & ~env._sac_valid_return & near_table & on_opponent & in_bounds_y & going_down
    own_landing = env._sac_hit & ~env._sac_bad_hit & near_table & on_own & in_bounds_y & going_down

    env._sac_valid_return[opponent_landing] = True
    env._sac_valid_return_step[opponent_landing] = step[opponent_landing]
    env._sac_landing_y[opponent_landing] = ball_pos[opponent_landing, 1]
    env._sac_landing_x[opponent_landing] = ball_pos[opponent_landing, 0]
    env._sac_step_event_mask[opponent_landing] |= EVENT_TO_BIT["valid_return"]

    env._sac_bad_hit[own_landing] = True
    env._sac_bad_hit_step[own_landing] = step[own_landing]
    env._sac_landing_y[own_landing] = ball_pos[own_landing, 1]
    env._sac_landing_x[own_landing] = ball_pos[own_landing, 0]
    env._sac_step_event_mask[own_landing] |= EVENT_TO_BIT["bad_hit"]

    # First post-hit descent through table height: cache ball xy once (NaN until then). This
    # feeds the tier-1 sac_table_proximity bootstrap, which grades how close a hit that did
    # NOT become a valid_return (own-table / out / timeout) came to the opponent table.
    table_cross = (
        env._sac_hit
        & torch.isnan(env._sac_table_cross_x)
        & near_table
        & going_down
    )
    env._sac_table_cross_x[table_cross] = ball_pos[table_cross, 0]
    env._sac_table_cross_y[table_cross] = ball_pos[table_cross, 1]

    out_after_hit = (
        env._sac_hit
        & ~env._sac_valid_return
        & ~env._sac_bad_hit
        & ((ball_pos[:, 2] < z_min) | (ball_pos[:, 0].abs() > out_x_limit) | (ball_pos[:, 1].abs() > out_y_limit))
    )
    env._sac_bad_hit[out_after_hit] = True
    env._sac_bad_hit_step[out_after_hit] = step[out_after_hit]
    env._sac_step_event_mask[out_after_hit] |= EVENT_TO_BIT["bad_hit"]

    timeout_steps = max(1, int(round(hit_return_timeout_s / env.step_dt)))
    hit_without_return_timeout = (
        env._sac_hit
        & ~env._sac_return
        & ~env._sac_valid_return
        & ~env._sac_bad_hit
        & (env._sac_hit_step >= 0)
        & ((step - env._sac_hit_step) >= timeout_steps)
    )
    env._sac_bad_hit[hit_without_return_timeout] = True
    env._sac_bad_hit_step[hit_without_return_timeout] = step[hit_without_return_timeout]
    env._sac_step_event_mask[hit_without_return_timeout] |= EVENT_TO_BIT["bad_hit"]

    missed_robot = (ball_pos[:, 0] - robot_x) * robot_side > miss_margin
    missed = ~env._sac_hit & ~env._sac_miss & (missed_robot | (ball_pos[:, 2] < z_min))
    env._sac_miss[missed] = True
    env._sac_miss_step[missed] = step[missed]
    env._sac_step_event_mask[missed] |= EVENT_TO_BIT["miss"]

    near = ~env._sac_hit & (env._sac_min_dist <= near_miss_threshold)
    env._sac_near_miss[near] = True


def capture_sac_final_info(env: ManagerBasedEnv, done: torch.Tensor):
    _ensure_tracker(env)
    if not torch.any(done):
        return
    mask = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    mask = torch.where(env._sac_near_miss, mask | EVENT_TO_BIT["near_miss"], mask)
    mask = torch.where(env._sac_hit, mask | EVENT_TO_BIT["hit"], mask)
    mask = torch.where(env._sac_bad_hit, mask | EVENT_TO_BIT["bad_hit"], mask)
    mask = torch.where(env._sac_return, mask | EVENT_TO_BIT["return"], mask)
    mask = torch.where(env._sac_valid_return, mask | EVENT_TO_BIT["valid_return"], mask)
    mask = torch.where(env._sac_miss, mask | EVENT_TO_BIT["miss"], mask)

    env._sac_final_event_mask[done] = mask[done]
    env._sac_final_min_dist[done] = env._sac_min_dist[done]
    env._sac_final_landing_y[done] = env._sac_landing_y[done]
    env._sac_final_landing_x[done] = env._sac_landing_x[done]
    env._sac_final_hit_center_offset[done] = env._sac_hit_center_offset[done]
    env._sac_final_hit_racket_ang_vel[done] = env._sac_hit_racket_ang_vel[done]
    env._sac_final_table_cross_x[done] = env._sac_table_cross_x[done]
    env._sac_final_table_cross_y[done] = env._sac_table_cross_y[done]
    env._sac_final_hit_outgoing_speed[done] = env._sac_hit_outgoing_speed[done]
    env._sac_final_hit_up_speed[done] = env._sac_hit_up_speed[done]
    env._sac_final_post_hit_max_outgoing_speed[done] = env._sac_post_hit_max_outgoing_speed[done]
    env._sac_final_post_hit_max_height[done] = env._sac_post_hit_max_height[done]
    for src, dst in (
        ("_sac_closest_step", "_sac_final_closest_step"),
        ("_sac_hit_step", "_sac_final_hit_step"),
        ("_sac_return_step", "_sac_final_return_step"),
        ("_sac_valid_return_step", "_sac_final_valid_return_step"),
        ("_sac_bad_hit_step", "_sac_final_bad_hit_step"),
        ("_sac_miss_step", "_sac_final_miss_step"),
    ):
        getattr(env, dst)[done] = getattr(env, src)[done]
