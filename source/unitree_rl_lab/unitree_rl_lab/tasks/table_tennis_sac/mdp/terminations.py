from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.table_tennis_sac.mdp.events import _ensure_tracker, capture_sac_final_info

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sac_episode_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    _ensure_tracker(env)
    done = env._sac_valid_return | env._sac_bad_hit | env._sac_miss
    capture_sac_final_info(env, done)
    return done


def sac_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    done = env.episode_length_buf >= env.max_episode_length
    capture_sac_final_info(env, done)
    return done


def joint_target_limit_violation(env: ManagerBasedRLEnv, action_name: str = "right_arm") -> torch.Tensor:
    action = env.action_manager.get_term(action_name)
    if hasattr(action, "limit_violation"):
        done = action.limit_violation
        capture_sac_final_info(env, done)
        return done
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return done


def joint_position_limit_violation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.02,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    violation = (joint_pos < limits[..., 0] + margin) | (joint_pos > limits[..., 1] - margin)
    done = torch.any(violation, dim=-1)
    if torch.any(done):
        num_envs, num_joints = joint_pos.shape
        if not hasattr(env, "_sac_final_joint_limit_q") or env._sac_final_joint_limit_q.shape != joint_pos.shape:
            env._sac_final_joint_limit_q = torch.full(
                (num_envs, num_joints), float("nan"), dtype=joint_pos.dtype, device=env.device
            )
            env._sac_final_joint_limit_low = torch.full_like(env._sac_final_joint_limit_q, float("nan"))
            env._sac_final_joint_limit_high = torch.full_like(env._sac_final_joint_limit_q, float("nan"))
            env._sac_final_joint_limit_mask = torch.zeros(
                (num_envs, num_joints), dtype=torch.bool, device=env.device
            )
            joint_ids = asset_cfg.joint_ids
            if isinstance(joint_ids, slice):
                resolved_joint_ids = list(range(len(robot.joint_names)))[joint_ids]
            else:
                resolved_joint_ids = [int(idx) for idx in joint_ids]
            env._sac_joint_limit_joint_names = [robot.joint_names[idx] for idx in resolved_joint_ids]

        env._sac_final_joint_limit_q[done] = joint_pos[done]
        env._sac_final_joint_limit_low[done] = limits[..., 0][done]
        env._sac_final_joint_limit_high[done] = limits[..., 1][done]
        env._sac_final_joint_limit_mask[done] = violation[done]
    capture_sac_final_info(env, done)
    return done


def joint_state_nan(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    done = torch.any(torch.isnan(robot.data.joint_pos), dim=-1) | torch.any(torch.isnan(robot.data.joint_vel), dim=-1)
    capture_sac_final_info(env, done)
    return done
