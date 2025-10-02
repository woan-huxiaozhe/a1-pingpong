from __future__ import annotations

import numpy as np
import os
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device | str) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.device = device

        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / data["fps"]
        self.num_frames = torch.tensor(self.dof_positions.shape[0], dtype=torch.long, device=self.device)
        self.duration = self.dt * (self.num_frames - 1)

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: torch.Tensor | None = None,
        end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times):
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        phase = torch.clip(times / self.duration, 0.0, 1.0)
        self.index_0 = (phase * (self.num_frames - 1)).round(decimals=0).to(torch.long)
        self.index_1 = torch.minimum(self.index_0 + 1, self.num_frames - 1).to(torch.long)
        self.blend = ((times - self.index_0 * self.dt) / self.dt).round(decimals=5).to(torch.float32)

    def sample_times(self, num_samples: int, duration: float | None = None) -> torch.Tensor:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert (
            duration <= self.duration
        ), f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        return duration * sample_uniform(0.0, 1.0, size=num_samples, device=self.device)

    def sample(self, num_samples: int, times: torch.Tensor | None = None, duration: float | None = None):
        """Update motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        self._compute_frame_blend(times)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._interpolate(self.dof_positions, blend=self.blend, start=self.index_0, end=self.index_1)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._interpolate(self.dof_velocities, blend=self.blend, start=self.index_0, end=self.index_1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._interpolate(self.body_positions, blend=self.blend, start=self.index_0, end=self.index_1)

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._slerp(self.body_rotations, blend=self.blend, start=self.index_0, end=self.index_1)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._interpolate(self.body_linear_velocities, blend=self.blend, start=self.index_0, end=self.index_1)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._interpolate(self.body_angular_velocities, blend=self.blend, start=self.index_0, end=self.index_1)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- motion
        self.motion = MotionLoader(cfg.motion_file, device=env.device)
        self.motion.sample(self.num_envs)  # update current times

        self.robot_ref_index = self.robot.body_names.index(self.cfg.reference_body)
        self.motion_ref_index = self.motion.body_names.index(self.cfg.reference_body)
        self.robot_body_ids = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion_body_ids = torch.tensor(
            string_utils.resolve_matching_names(self.cfg.body_names, self.motion.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # -- metrics
        self.metrics["error_ref_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.ref_pos_w, self.ref_quat_w], dim=1)

    # ----- motion -----
    # -- motion joint data
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel

    # -- motion body data
    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[:, self.motion_body_ids] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[:, self.motion_body_ids]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[:, self.motion_body_ids]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[:, self.motion_body_ids]

    # -- motion reference body data
    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[:, self.motion_ref_index] + self._env.scene.env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[:, self.motion_ref_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[:, self.motion_ref_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[:, self.motion_ref_index]

    # ----- robot -----
    # -- robot joint data
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    # -- robot body data
    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_body_ids]

    # -- robot reference body data
    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_ref_index]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_ref_index]

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_ref_index]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_ref_index]

    def _resample_command(self, env_ids):
        self.times[env_ids] = self.motion.sample_times(len(env_ids))

        root_pos = self.body_pos_w[:, 0].clone()
        root_quat = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_quat[env_ids] = quat_mul(orientations_delta, root_quat[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_quat[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.times += self._env.step_dt
        env_ids = torch.where(self.times >= self.motion.duration)[0]
        self._resample_command(env_ids)
        self.motion.sample(self.num_envs, times=self.times)

        ref_pos_w_repeat = self.ref_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        ref_quat_w_repeat = self.ref_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_ref_pos_w_repeat = self.robot_ref_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_ref_quat_w_repeat = self.robot_ref_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w[..., :2] = 0.0
        delta_ori_w = yaw_quat(quat_mul(robot_ref_quat_w_repeat, quat_inv(ref_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat + delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - ref_pos_w_repeat)
        )

    def _update_metrics(self):
        self.metrics["error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(
                    self.cfg.target_body_visualizer_cfg,
                )
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.target_visualizer.visualize(self.body_pos_w.reshape(-1, 3))


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command generator."""

    class_type: type = MotionCommand

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    motion_file: str = MISSING
    """Path to the motion file (NumPy format)."""

    reference_body: str = MISSING
    """Name of the body to use as reference for the position and orientation commands."""

    body_names: list[str] = MISSING
    """List of body names to track for position and orientation commands."""

    pose_range: dict[str, tuple[float, float]] = {}
    """Distribution ranges for the position (x, y, z in meters) and orientation (roll, pitch, yaw in radians) offsets."""

    velocity_range: dict[str, tuple[float, float]] = {}
    """Distribution ranges for the linear (x, y, z in m/s) and angular (roll, pitch, yaw in rad/s) velocity offsets."""

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Distribution range for the joint position offsets (in radians)."""

    target_body_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
        prim_path="/Visuals/Target/pose",
    )
