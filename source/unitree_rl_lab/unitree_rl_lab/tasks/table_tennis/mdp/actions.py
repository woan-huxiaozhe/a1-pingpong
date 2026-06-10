from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from unitree_rl_lab.tasks.table_tennis.mdp.commands import UpperBodyMotionCommand


class ReferenceResidualJointAction(ActionTerm):
    """Joint position action where target = reference_pose + scale * action.

    When action=0, the robot perfectly tracks the motion reference.
    The RL agent only learns small residual corrections.
    """

    cfg: ReferenceResidualJointActionCfg

    def __init__(self, cfg: ReferenceResidualJointActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._robot.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        if isinstance(cfg.residual_scale, (list, tuple)):
            self._scale = torch.tensor(cfg.residual_scale, dtype=torch.float32, device=env.device)
        else:
            self._scale = cfg.residual_scale

        command: UpperBodyMotionCommand = env.command_manager.get_term(cfg.command_name)
        ref_joint_ids = command.upper_body_joint_ids
        self._ref_indices = []
        for jid in self._joint_ids:
            idx = (ref_joint_ids == jid).nonzero(as_tuple=True)[0].item()
            self._ref_indices.append(idx)
        self._ref_indices = torch.tensor(self._ref_indices, dtype=torch.long, device=env.device)

        # Action delay buffer (control-step granularity, legacy / forehand)
        self._max_delay = cfg.action_delay_steps_max
        if self._max_delay > 0:
            self._action_buffer = torch.zeros(
                env.num_envs, self._max_delay + 1, self._num_joints, device=env.device
            )
            self._delay_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self._step_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        # Sub-step (physics-dt) action delay (plan A3). apply_actions() is called once
        # per physics sub-step, so a sub-step ring buffer represents 5ms-granularity
        # actuation latency. delay in {min..max} sub-steps; with sim.dt=0.005 that is
        # {5,10,15}ms for {1,2,3}. Opt-in (max>0); forehand keeps the legacy path above.
        self._sub_max = cfg.action_delay_substeps_max
        if self._sub_max > 0:
            self._sub_buffer = torch.zeros(
                env.num_envs, self._sub_max + 1, self._num_joints, device=env.device
            )
            self._sub_delay = torch.full(
                (env.num_envs,), cfg.action_delay_substeps_min, dtype=torch.long, device=env.device
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

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        command: UpperBodyMotionCommand = self._env.command_manager.get_term(self.cfg.command_name)

        if self._sub_max > 0:
            # Sub-step delay: the delayed target is computed per sub-step in apply_actions.
            # Keep _processed_actions current (undelayed) for any property consumers.
            self._processed_actions = command.ref_dof[:, self._ref_indices] + actions * self._scale
        elif self._max_delay > 0:
            # Shift buffer and store new action
            self._action_buffer = torch.roll(self._action_buffer, 1, dims=1)
            self._action_buffer[:, 0] = actions
            # Read delayed action
            idx = self._delay_steps.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self._num_joints)
            delayed_actions = self._action_buffer.gather(1, idx).squeeze(1)
            self._processed_actions = command.ref_dof[:, self._ref_indices] + delayed_actions * self._scale
        else:
            self._processed_actions = command.ref_dof[:, self._ref_indices] + self._raw_actions * self._scale

    def apply_actions(self):
        if self._sub_max > 0:
            # Push current raw action into the sub-step ring (called every physics sub-step),
            # read the action delayed by self._sub_delay sub-steps, build target = ref + delayed*scale.
            command: UpperBodyMotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
            self._sub_buffer = torch.roll(self._sub_buffer, 1, dims=1)
            self._sub_buffer[:, 0] = self._raw_actions
            idx = self._sub_delay.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self._num_joints)
            delayed = self._sub_buffer.gather(1, idx).squeeze(1)
            self._processed_actions = command.ref_dof[:, self._ref_indices] + delayed * self._scale
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        if self._max_delay > 0:
            self._action_buffer[env_ids] = 0.0
            # Randomize delay for each env: [min, max] steps
            min_delay = self.cfg.action_delay_steps_min
            self._delay_steps[env_ids] = torch.randint(
                min_delay, self._max_delay + 1, (len(env_ids),), device=self._delay_steps.device
            )
        if self._sub_max > 0:
            self._sub_buffer[env_ids] = 0.0
            self._sub_delay[env_ids] = torch.randint(
                self.cfg.action_delay_substeps_min, self._sub_max + 1,
                (len(env_ids),), device=self._sub_delay.device
            )


class BaseYSliderAction(ActionTerm):
    cfg: BaseYSliderActionCfg

    def __init__(self, cfg: BaseYSliderActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(env.num_envs, 1, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 1, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self.cfg.scale

    def apply_actions(self):
        desired_vy = self._processed_actions.squeeze(-1)

        root_state = self._robot.data.root_state_w.clone()
        env_origins = self._env.scene.env_origins

        root_state[:, 0] = env_origins[:, 0] + self.cfg.fixed_x
        root_state[:, 2] = env_origins[:, 2] + self.cfg.fixed_z

        y_local = root_state[:, 1] - env_origins[:, 1]
        y_local = y_local.clamp(self.cfg.y_min, self.cfg.y_max)
        root_state[:, 1] = env_origins[:, 1] + y_local

        root_state[:, 3] = 0.0
        root_state[:, 4:6] = 0.0
        root_state[:, 6] = 1.0

        root_state[:, 7] = 0.0
        root_state[:, 8] = desired_vy
        root_state[:, 9] = 0.0

        root_state[:, 10:13] = 0.0

        self._robot.write_root_state_to_sim(root_state)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


class ReferenceTrackingJointAction(ActionTerm):
    """Non-RL action that tracks motion reference exactly. action_dim=0."""

    cfg: ReferenceTrackingJointActionCfg

    def __init__(self, cfg: ReferenceTrackingJointActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._robot.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(env.num_envs, 0, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)

        command: UpperBodyMotionCommand = env.command_manager.get_term(cfg.command_name)
        ref_joint_ids = command.upper_body_joint_ids
        self._ref_indices = []
        for jid in self._joint_ids:
            idx = (ref_joint_ids == jid).nonzero(as_tuple=True)[0].item()
            self._ref_indices.append(idx)
        self._ref_indices = torch.tensor(self._ref_indices, dtype=torch.long, device=env.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        command: UpperBodyMotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
        self._processed_actions = command.ref_dof[:, self._ref_indices]

    def apply_actions(self):
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


class PhaseSpeedAction(ActionTerm):
    """Controls motion reference playback speed. action_dim=1.

    Maps action in [-1, 1] to phase speed in [speed_min, speed_max],
    with action=0 mapping to speed=1.0 (normal playback).
    """

    cfg: PhaseSpeedActionCfg

    def __init__(self, cfg: PhaseSpeedActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 1, device=env.device)
        self._processed_actions = torch.ones(env.num_envs, 1, device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = actions.squeeze(-1)
        speed = torch.where(
            a >= 0,
            1.0 + a * (self.cfg.speed_max - 1.0),
            1.0 + a * (1.0 - self.cfg.speed_min),
        )
        self._processed_actions[:] = speed.unsqueeze(-1)

    def apply_actions(self):
        command: UpperBodyMotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
        command.phase_speed[:] = self._processed_actions.squeeze(-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 1.0


@configclass
class BaseYSliderActionCfg(ActionTermCfg):
    class_type: type = BaseYSliderAction
    asset_name: str = MISSING
    scale: float = 0.5
    fixed_x: float = 1.5
    fixed_z: float = 0.76
    y_min: float = -1.0
    y_max: float = 1.0


@configclass
class ReferenceResidualJointActionCfg(ActionTermCfg):
    class_type: type = ReferenceResidualJointAction
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    command_name: str = "motion"
    residual_scale: float | list[float] = 0.1
    action_delay_steps_min: int = 0
    action_delay_steps_max: int = 0
    # Sub-step (physics-dt) action delay (plan A3). Opt-in: when *_substeps_max > 0 the
    # legacy control-step delay above is bypassed. delay ~ uniform{min..max} sub-steps;
    # with sim.dt=0.005, {1,2,3} == {5,10,15}ms (mean 10ms).
    action_delay_substeps_min: int = 1
    action_delay_substeps_max: int = 0


@configclass
class ReferenceTrackingJointActionCfg(ActionTermCfg):
    class_type: type = ReferenceTrackingJointAction
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    command_name: str = "motion"


@configclass
class PhaseSpeedActionCfg(ActionTermCfg):
    class_type: type = PhaseSpeedAction
    asset_name: str = MISSING
    command_name: str = "motion"
    speed_min: float = 0.2
    speed_max: float = 3.0


class BallObsDelayAction(ActionTerm):
    """Zero-dim action term that delays only the ball-derived observations (plan A4).

    apply_actions() runs once per physics sub-step (the only per-sub-step hook in the env
    loop), so it caches the ball world pos/vel into a ring buffer and writes a delayed
    snapshot onto the env (env._delayed_ball_pos / _delayed_ball_vel). The ball-derived
    observation functions read that snapshot; joint / phase / racket / last_action stay live.

    delay ~ uniform{min..max} sub-steps; with sim.dt=0.005 that is {5,10,15}ms (mean 10ms).
    action_dim=0 -> consumes no policy output. Forehand simply omits this term, so the obs
    functions fall back to the live ball state (backward compatible).
    """

    cfg: BallObsDelayActionCfg

    def __init__(self, cfg: BallObsDelayActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._ball: RigidObject = env.scene[cfg.ball_name]
        self._raw_actions = torch.zeros(env.num_envs, 0, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 0, device=env.device)
        self._L = cfg.ball_delay_substeps_max
        # ring: index 0 = most recently cached sub-step state
        self._ring_pos = torch.zeros(env.num_envs, self._L, 3, device=env.device)
        self._ring_vel = torch.zeros(env.num_envs, self._L, 3, device=env.device)
        self._delay = torch.full(
            (env.num_envs,), cfg.ball_delay_substeps_min, dtype=torch.long, device=env.device
        )
        env._delayed_ball_pos = None
        env._delayed_ball_vel = None

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        self._ring_pos = torch.roll(self._ring_pos, 1, dims=1)
        self._ring_vel = torch.roll(self._ring_vel, 1, dims=1)
        self._ring_pos[:, 0] = self._ball.data.root_pos_w
        self._ring_vel[:, 0] = self._ball.data.root_lin_vel_w
        # delay d sub-steps -> ring[d-1] (d>=1 => never 0ms; matches {5,10,15}ms)
        ridx = (self._delay - 1).clamp(0, self._L - 1)
        idx = ridx.view(-1, 1, 1).expand(-1, 1, 3)
        self._env._delayed_ball_pos = self._ring_pos.gather(1, idx).squeeze(1)
        self._env._delayed_ball_vel = self._ring_vel.gather(1, idx).squeeze(1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # fill ring with the current ball state so early reads are sane (no garbage delay)
        self._ring_pos[env_ids] = self._ball.data.root_pos_w[env_ids].unsqueeze(1)
        self._ring_vel[env_ids] = self._ball.data.root_lin_vel_w[env_ids].unsqueeze(1)
        n = self._delay[env_ids].shape[0]
        self._delay[env_ids] = torch.randint(
            self.cfg.ball_delay_substeps_min, self._L + 1, (n,), device=self._delay.device
        )


@configclass
class BallObsDelayActionCfg(ActionTermCfg):
    class_type: type = BallObsDelayAction
    asset_name: str = MISSING
    ball_name: str = "ball"
    ball_delay_substeps_min: int = 1
    ball_delay_substeps_max: int = 3
