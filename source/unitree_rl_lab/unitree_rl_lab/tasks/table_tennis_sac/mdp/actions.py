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

        # Sub-step (physics-dt) transport delay of the position target. apply_actions() runs once per
        # physics sub-step, so a sub-step ring buffer gives sim.dt-granularity delay (5ms @200Hz). Models
        # the real deploy comms/actuation dead-time (~63ms measured). 0 = disabled (default; SAC unaffected).
        self._sub_max = int(getattr(cfg, "action_delay_substeps_max", 0))
        if self._sub_max > 0:
            self._sub_buffer = self._processed_actions.unsqueeze(1).repeat(1, self._sub_max + 1, 1).clone()
            self._sub_delay = torch.full(
                (env.num_envs,), int(cfg.action_delay_substeps_min), dtype=torch.long, device=env.device
            )

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
        if self._sub_max > 0:
            # push the current position target into the sub-step ring, apply the one delayed by
            # self._sub_delay sub-steps (per-env, resampled each reset) -> transport dead-time.
            self._sub_buffer = torch.roll(self._sub_buffer, 1, dims=1)
            self._sub_buffer[:, 0] = self._processed_actions
            idx = self._sub_delay.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self._num_joints)
            delayed = self._sub_buffer.gather(1, idx).squeeze(1)
            self._robot.set_joint_position_target(delayed, joint_ids=self._joint_ids)
        else:
            self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._robot.data.joint_pos[env_ids][:, self._joint_ids]
        self._limit_violation[env_ids] = False
        if self._sub_max > 0:
            cur = self._robot.data.joint_pos[env_ids][:, self._joint_ids]
            self._sub_buffer[env_ids] = cur.unsqueeze(1)   # fill history with the reset pose (no stale target)
            self._sub_delay[env_ids] = torch.randint(
                int(self.cfg.action_delay_substeps_min), self._sub_max + 1,
                (len(env_ids),), device=self._sub_delay.device,
            )


@configclass
class JointDeltaTargetActionCfg(ActionTermCfg):
    class_type: type = JointDeltaTargetAction
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    action_scale: float | list[float] = 0.12
    smoothing: float = 0.5
    max_joint_velocity: float | list[float] | None = 6.0
    limit_margin: float = 1.0e-3
    # Physics-substep (sim.dt) transport delay of the position target. 0 = disabled. Each reset samples
    # a per-env integer delay in [min, max] sub-steps (5ms @200Hz) for domain randomization.
    action_delay_substeps_min: int = 0
    action_delay_substeps_max: int = 0
