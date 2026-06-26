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
    bias_std_far: tuple[float, float, float] = (0.024, 0.022, 0.016),
    jitter_std: tuple[float, float, float] = (0.003, 0.008, 0.003),
    fixed_offset_far: tuple[float, float, float] = (0.0, -0.010, -0.005),
    far_tau: tuple[float, float, float] = (0.65, 0.40, 0.55),
    y_abs_limit: float = 2.0,
    z_min: float = 0.45,
    z_max: float = 2.0,
) -> torch.Tensor:
    """Deployment-style hit command with KF-like prediction error.

    Calibrated from the deployment Kalman (``kalman_filter_pingpong``) open-loop hit-point
    residuals on the 0617 mocap serves. The real KF error is dominated by a *per-serve
    consistent bias* (random direction each serve, set by that ball's spin/launch that the
    non-Magnus KF mispredicts), NOT per-step white noise -- so a per-step Gaussian (the old
    model) would be averaged out by the policy. The three components, each scaled by the
    per-axis horizon phase ``tau / far_tau`` so the command converges to truth as the ball
    arrives:

      - ``bias``  : per-episode unit normal (y, z, tau) drawn once per ball reset, times
                    ``bias_std_far`` -- consistent in direction over the whole flight.
      - ``jitter``: per-step white noise, ``jitter_std`` (small).
      - ``offset``: fixed systematic offset ``fixed_offset_far`` shared by all serves
                    (mainly z ~ -1cm: the KF predicts the strike height slightly low).

    ``far_tau`` is per-axis because the three error profiles differ in horizon: z saturates
    early (~0.40 s), y grows ~linearly out to ~0.65 s, tau plateaus mid-flight (~0.55 s).
    The strike-plane x stays fixed at ``robot_x``; error is applied to predicted y/z/tau.
    Cached per policy step so the actor and critic groups see the same command in one step.
    """
    clean = hit_command_at_robot_x(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)
    num = clean.shape[0]
    device = clean.device
    step = env.episode_length_buf.to(torch.long)

    # --- per-episode bias unit vector (y, z, tau), redrawn at episode reset ---
    bias_attr = "_sac_hit_bias_unit"
    bias_step_attr = "_sac_hit_bias_step"
    if not hasattr(env, bias_attr):
        setattr(env, bias_attr, torch.randn(num, 3, device=device))
        setattr(env, bias_step_attr, torch.full((num,), -1, dtype=torch.long, device=device))
    bias_unit = getattr(env, bias_attr)
    bias_last = getattr(env, bias_step_attr)
    reset = (bias_last < 0) | (step == 0) | (step < bias_last)
    if torch.any(reset):
        bias_unit[reset] = torch.randn(int(reset.sum()), 3, device=device)
        bias_last[reset] = step[reset]

    # --- per-step cached estimate (actor/critic consistency within a step) ---
    attr = "_sac_estimated_hit_command"
    step_attr = "_sac_estimated_hit_command_step"
    if not hasattr(env, attr):
        setattr(env, attr, clean.clone())
        setattr(env, step_attr, torch.full((num,), -1, dtype=torch.long, device=device))

    estimate = getattr(env, attr)
    last_step = getattr(env, step_attr)
    refresh = step != last_step
    if torch.any(refresh):
        next_estimate = clean.clone()
        tau = clean[:, 3:4].clamp(min=0.0)  # [N, 1]

        far = torch.tensor(far_tau, device=device, dtype=clean.dtype).clamp(min=1.0e-6)  # [3]
        phase = (tau / far.unsqueeze(0)).clamp(min=0.0, max=1.0)  # [N, 3], per-axis horizon scaling

        bias_std = torch.tensor(bias_std_far, device=device, dtype=clean.dtype).unsqueeze(0)  # [1, 3]
        jit_std = torch.tensor(jitter_std, device=device, dtype=clean.dtype).unsqueeze(0)
        offset = torch.tensor(fixed_offset_far, device=device, dtype=clean.dtype).unsqueeze(0)

        # err on (y, z, tau): bias keeps its per-episode direction, magnitude scales with phase
        err = bias_unit * (bias_std * phase) + torch.randn(num, 3, device=device) * jit_std + offset * phase

        next_estimate[:, 1:3] += err[:, 0:2]  # y, z
        next_estimate[:, 3:4] += err[:, 2:3]  # tau
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


# --- HitTrack: model-derived end-effector reference observations (additive) ---
# These read the per-step ``_ht_*`` reference buffers maintained by
# ``mdp.reference_commands.update_hit_track_state``. They are used only by the
# A1-TableTennis-SAC-HitTrack task; the Catch task never wires them.


def hit_reference_command(env) -> torch.Tensor:
    """Actor reference command ``[p_ref, v_ref, n_ref, tau]`` (noisy, deployable). 10-dim."""
    return torch.cat(
        [env._ht_p_ref_noisy, env._ht_v_ref_noisy, env._ht_n_ref_noisy, env._ht_tau_noisy.unsqueeze(-1)],
        dim=-1,
    )


def hit_reference_command_clean(env) -> torch.Tensor:
    """Critic reference command ``[p_ref, v_ref, n_ref, tau_true]`` (clean/privileged). 10-dim."""
    return torch.cat(
        [env._ht_p_ref_clean, env._ht_v_ref_clean, env._ht_n_ref_clean, env._ht_tau_true.unsqueeze(-1)],
        dim=-1,
    )


def hit_ref_pos_error(env, racket_body_name: str) -> torch.Tensor:
    """racket blade-center (env-local) minus noisy p_ref. Deployable (FK) -> actor."""
    center, _, _ = _racket_body_state(env, racket_body_name)
    return (center - env.scene.env_origins) - env._ht_p_ref_noisy


def hit_ref_vel_error(env, racket_body_name: str) -> torch.Tensor:
    """racket blade-center velocity minus noisy v_ref. Privileged (sim vel) -> critic only."""
    _, center_vel, _ = _racket_body_state(env, racket_body_name)
    return center_vel - env._ht_v_ref_noisy
