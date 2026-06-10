from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.table_tennis_sac.control import compute_joint_delta_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointDeltaTargetAction(ActionTerm):
    """Joint position action with target = current q + scaled normalized delta."""

    cfg: JointDeltaTargetActionCfg

    def __init__(self, cfg: JointDeltaTargetActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._robot.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._processed_actions = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._limit_violation = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        if isinstance(cfg.action_scale, (list, tuple)):
            self._action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32, device=env.device)
        else:
            self._action_scale = float(cfg.action_scale)
        if cfg.max_joint_velocity is None:
            self._max_joint_velocity = None
        elif isinstance(cfg.max_joint_velocity, (list, tuple)):
            self._max_joint_velocity = torch.tensor(cfg.max_joint_velocity, dtype=torch.float32, device=env.device)
        else:
            self._max_joint_velocity = float(cfg.max_joint_velocity)

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def limit_violation(self) -> torch.Tensor:
        return self._limit_violation

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]
        lower = limits[..., 0] + self.cfg.limit_margin
        upper = limits[..., 1] - self.cfg.limit_margin

        raw_target = joint_pos + actions.clamp(-1.0, 1.0) * self._action_scale
        self._limit_violation[:] = torch.any((raw_target < lower) | (raw_target > upper), dim=-1)

        max_delta = None
        if self._max_joint_velocity is not None:
            max_delta = self._max_joint_velocity * self._env.step_dt

        self._processed_actions = compute_joint_delta_target(
            joint_pos,
            actions,
            lower,
            upper,
            action_scale=self._action_scale,
            previous_target=self._processed_actions,
            smoothing=self.cfg.smoothing,
            max_delta_per_step=max_delta,
        )

    def apply_actions(self):
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._robot.data.joint_pos[env_ids][:, self._joint_ids]
        self._limit_violation[env_ids] = False


@configclass
class JointDeltaTargetActionCfg(ActionTermCfg):
    class_type: type = JointDeltaTargetAction
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    action_scale: float | list[float] = 0.12
    smoothing: float = 0.5
    max_joint_velocity: float | list[float] | None = 6.0
    limit_margin: float = 1.0e-3
