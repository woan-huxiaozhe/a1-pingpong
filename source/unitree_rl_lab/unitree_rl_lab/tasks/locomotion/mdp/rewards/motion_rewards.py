from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.utils.math import quat_error_magnitude

from unitree_rl_lab.tasks.locomotion.mdp.commands.motion_command import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_joint_pos_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.joint_pos - command.robot_joint_pos
    return error.pow(2).mean(-1).mul(-1 / std**2).exp()


def motion_joint_vel_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.joint_vel - command.robot_joint_vel
    return error.pow(2).mean(-1).mul(-1 / std**2).exp()


def motion_ref_pos_w_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.ref_pos_w - command.robot_ref_pos_w
    return error.pow(2).sum().mul(-1 / std**2).exp()


def motion_ref_ori_w_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.ref_quat_w, command.robot_ref_quat_w)
    return error.mul(-1 / std**2).exp()


def motion_ref_lin_vel_w_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.ref_lin_vel_w - command.robot_ref_lin_vel_w
    return error.pow(2).sum(-1).mean(-1).mul(-1 / std**2).exp()


def motion_ref_ang_vel_w_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.ref_ang_vel_w, command.robot_ref_ang_vel_w)
    return error.pow(2).sum(-1).mean(-1).mul(-1 / std**2).exp()


def motion_body_pos_relative_error_exp(
    env: ManagerBasedRLEnv, std: float, body_names: list[str] | None = None, command_name: str = "motion"
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = string_utils.resolve_matching_names(body_names, command.cfg.body_names, preserve_order=True)[0]
    error = command.body_pos_relative_w[:, ids] - command.robot_body_pos_w[:, ids]
    return error.pow(2).sum(-1).mean(-1).mul(-1 / std**2).exp()


def motion_body_ori_relative_error_exp(
    env: ManagerBasedRLEnv, std: float, body_names: list[str] | None = None, command_name: str = "motion"
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = string_utils.resolve_matching_names(body_names, command.cfg.body_names, preserve_order=True)[0]
    error = quat_error_magnitude(command.body_quat_relative_w[:, ids], command.robot_body_quat_w[:, ids])
    return error.pow(2).mean(-1).mul(-1 / std**2).exp()
