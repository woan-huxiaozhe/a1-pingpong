"""A1-TableTennis-SAC-HitTrack environment config (100 Hz, non-end-to-end).

A new, additive task that does NOT touch the Catch ``env_cfg.py``. The arm tracks a
model-derived end-effector hit reference ``(p_ref, v_ref, n_ref)`` at the predicted ball hit
time; the ball is out of the MDP. The reference is produced at runtime by
``mdp.reset_reference_command`` / ``mdp.update_hit_track_state`` (which wrap the pure-torch
``reference_planner`` + ``reference_source``), and tracked by the time-gated Gaussian rewards
``mdp.hit_ref_pos`` / ``mdp.hit_ref_vel``.

Control runs at 100 Hz (``decimation=2``, ``sim.dt=0.005``, ``step_dt=0.01``) to match the real
control loop; Catch stays at 50 Hz. The scene, ready pose, robot placement and physics are
reused verbatim from the Catch config; only the action scale (halved for 100 Hz) differs.
"""

from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.table_tennis_sac.mdp as mdp
from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import (
    OPP_TABLE_CENTER_X,
    RACKET_BODY_NAME,
    RIGHT_ARM_JOINT_NAMES,
    ROBOT_SIDE,
    TABLE_Z,
)
from unitree_rl_lab.tasks.table_tennis_sac.env_cfg import (
    A1TableTennisSacSceneCfg,
    SAC_BALL_LINEAR_DAMPING,
    SAC_MAX_JOINT_VELOCITY,
    SAC_READY_JOINT_POS,
    SAC_READY_LIFT_POS,
    SAC_ROBOT_BASE_X,
    SAC_ROBOT_X,
)

# --- HitTrack constants (v1, 100 Hz) ---
HIT_PLANE_X = SAC_ROBOT_X  # = -1.37
HITTRACK_TARGET_XYZ = (OPP_TABLE_CENTER_X, 0.0, TABLE_Z)
HITTRACK_BOX = {
    "y": (-0.2, 0.3),
    "z": (0.9, 1.25),
    "vx": (-4.5, -3.0),
    "vy": (-0.3, 0.3),
    "vz": (-1.0, 0.5),
}
MAX_PREP_S = 0.6
POST_MARGIN_S = 0.12
STEP_DT = 0.01  # 100 Hz control (decimation=2 * sim.dt=0.005)
SIGMA_T = 0.03
SIGMA_P = 0.03
SIGMA_V = 0.3
W_POS = 20.0
W_VEL = 20.0
SUCCESS_POS = 0.05
SUCCESS_VEL = 0.2
REACH_Y = (-0.6, 0.6)
REACH_Z = (0.7, 1.5)
JOINT_POS_DELTA_HISTORY_LENGTH = 5

# Curriculum (3): baked real-serve source. Off by default (curriculum (1)/(2) train on the
# synthetic box). When on, the reset loads `HITTRACK_BAKED_PATH` (produced by
# bake_hittrack_references.py) and samples a recorded serve per env instead of sampling the box.
HITTRACK_USE_BAKED = False
HITTRACK_BAKED_PATH = os.path.join(os.path.dirname(__file__), "hittrack_references.npz")


@configclass
class ActionsCfg:
    right_arm = mdp.JointDeltaTargetActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        action_scale=0.06,  # 100 Hz: halved from Catch's 0.12
        smoothing=0.5,
        max_joint_velocity=SAC_MAX_JOINT_VELOCITY,
    )


@configclass
class ObservationsCfg:
    @configclass
    class ActorCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
        )
        joint_pos_delta_history = ObsTerm(
            func=mdp.joint_pos_delta_history,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES),
                "history_length": JOINT_POS_DELTA_HISTORY_LENGTH,
            },
        )
        # noisy model-derived end-effector reference command [p_ref, v_ref, n_ref, tau] (deployable)
        hit_reference_command = ObsTerm(func=mdp.hit_reference_command)
        # FK blade-center pose (encoder-exact, deployable)
        racket_pos = ObsTerm(func=mdp.racket_pos, params={"racket_body_name": RACKET_BODY_NAME})
        racket_normal = ObsTerm(func=mdp.racket_normal, params={"racket_body_name": RACKET_BODY_NAME})
        hit_ref_pos_error = ObsTerm(func=mdp.hit_ref_pos_error, params={"racket_body_name": RACKET_BODY_NAME})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ActorCfg):
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
        )
        racket_vel = ObsTerm(func=mdp.racket_vel, params={"racket_body_name": RACKET_BODY_NAME})
        racket_ang_vel = ObsTerm(func=mdp.racket_ang_vel, params={"racket_body_name": RACKET_BODY_NAME})
        racket_axes = ObsTerm(func=mdp.racket_axes, params={"racket_body_name": RACKET_BODY_NAME})
        # clean (privileged) reference command + tau_true
        hit_reference_command_clean = ObsTerm(func=mdp.hit_reference_command_clean)
        hit_ref_vel_error = ObsTerm(func=mdp.hit_ref_vel_error, params={"racket_body_name": RACKET_BODY_NAME})

    policy: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # --- core: model-derived hit-time reference tracking (time-gated Gaussian) ---
    hit_ref_pos = RewTerm(
        func=mdp.hit_ref_pos,
        weight=W_POS,
        params={"racket_body_name": RACKET_BODY_NAME, "sigma_t": SIGMA_T, "sigma_p": SIGMA_P},
    )
    hit_ref_vel = RewTerm(
        func=mdp.hit_ref_vel,
        weight=W_VEL,
        params={"racket_body_name": RACKET_BODY_NAME, "sigma_t": SIGMA_T, "sigma_v": SIGMA_V},
    )

    # --- sim-to-real smoothing regularizers (copied verbatim from Catch env_cfg) ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    joint_jerk = RewTerm(
        func=mdp.joint_jerk_l2,
        weight=-2.0e-10,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    joint_limit = RewTerm(
        func=mdp.joint_limit_margin_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES), "margin": 0.05},
    )
    joint_effort_margin = RewTerm(
        func=mdp.joint_effort_margin_penalty,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES), "margin_frac": 0.85},
    )


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_ready_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": RIGHT_ARM_JOINT_NAMES,
            "joint_pos": SAC_READY_JOINT_POS,
        },
    )
    reset_reference = EventTerm(
        func=mdp.reset_reference_command,
        mode="reset",
        params={
            "hit_plane_x": HIT_PLANE_X,
            "target_xyz": HITTRACK_TARGET_XYZ,
            "box": HITTRACK_BOX,
            "max_prep_s": MAX_PREP_S,
            "post_margin_s": POST_MARGIN_S,
            "step_dt": STEP_DT,
            "reach_y_range": REACH_Y,
            "reach_z_range": REACH_Z,
        },
    )
    update_ref = EventTerm(
        func=mdp.update_hit_track_state,
        mode="interval",
        interval_range_s=(STEP_DT, STEP_DT),
        params={"success_pos_thresh": SUCCESS_POS, "success_vel_thresh": SUCCESS_VEL},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.sac_time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.joint_state_nan)
    hit_done = DoneTerm(func=mdp.hit_window_elapsed)


@configclass
class HitTrackEnvCfg(ManagerBasedRLEnvCfg):
    scene: A1TableTennisSacSceneCfg = A1TableTennisSacSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2  # 100 Hz control over 200 Hz physics
        self.episode_length_s = 0.72  # ~= max_prep + post_margin (~72 steps @100 Hz)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.enable_ccd = True
        self.scene.ball.spawn.rigid_props.linear_damping = SAC_BALL_LINEAR_DAMPING
        self.scene.robot.init_state.pos = (SAC_ROBOT_BASE_X, 0.0, 0.0)
        if ROBOT_SIDE < 0:
            self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos["joint_lift"] = SAC_READY_LIFT_POS
        self.scene.robot.init_state.joint_pos = joint_pos
        # Curriculum (3): switch the reference source to the baked real serves (lazy-loaded
        # on the first reset onto env.device).
        if HITTRACK_USE_BAKED:
            self.events.reset_reference.params["baked_path"] = HITTRACK_BAKED_PATH


class HitTrackPlayEnvCfg(HitTrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.viewer.eye = (-2.5, -2.0, 1.4)
        self.viewer.lookat = (-1.0, 0.0, 1.0)
