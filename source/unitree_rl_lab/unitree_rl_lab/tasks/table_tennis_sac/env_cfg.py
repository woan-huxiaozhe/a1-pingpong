from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.table_tennis_sac.mdp as mdp
from unitree_rl_lab.assets.robots.a1 import A1_ARM_VELOCITY
from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import (
    OPP_TABLE_CENTER_X,
    OPP_TABLE_X,
    OWN_TABLE_X,
    RACKET_BODY_NAME,
    RIGHT_ARM_JOINT_NAMES,
    ROBOT_BASE_X,
    ROBOT_SIDE,
    ROBOT_X,
    TABLE_Z,
    X1TableTennisSceneCfg,
)

BALL_HISTORY_LENGTH = 4

SAC_READY_LIFT_POS = -0.22
SAC_READY_JOINT_POS = [1.53, -0.39, 1.60, -1.32, 0.0, 1.0, -1.845288]
SAC_MAX_JOINT_VELOCITY = [A1_ARM_VELOCITY[name] for name in RIGHT_ARM_JOINT_NAMES]

SAC_FIXED_MIDDLE_BALL = {
    "x_range": (-1.25 * ROBOT_SIDE, -1.25 * ROBOT_SIDE),
    "y_range": (0.0, 0.0),
    "z_range": (1.05, 1.05),
    "vx_range": (3.4 * ROBOT_SIDE, 3.4 * ROBOT_SIDE),
    "vy_range": (0.0, 0.0),
    "vz_range": (0.0, 0.0),
}


@configclass
class A1TableTennisSacSceneCfg(X1TableTennisSceneCfg):
    pass


@configclass
class ActionsCfg:
    right_arm = mdp.JointDeltaTargetActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        action_scale=0.12,
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
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
        )
        racket_pos = ObsTerm(func=mdp.racket_pos, params={"racket_body_name": RACKET_BODY_NAME})
        racket_normal = ObsTerm(func=mdp.racket_normal, params={"racket_body_name": RACKET_BODY_NAME})
        ball_pos_history = ObsTerm(
            func=mdp.ball_pos_history,
            params={"ball_name": "ball", "history_length": BALL_HISTORY_LENGTH},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ActorCfg):
        ball_vel = ObsTerm(func=mdp.ball_vel_w, params={"ball_name": "ball"})
        racket_vel = ObsTerm(func=mdp.racket_vel, params={"racket_body_name": RACKET_BODY_NAME})
        racket_ang_vel = ObsTerm(func=mdp.racket_ang_vel, params={"racket_body_name": RACKET_BODY_NAME})
        ball_pos_rel_racket = ObsTerm(
            func=mdp.ball_pos_relative_to_racket,
            params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME},
        )
        ball_vel_rel_racket = ObsTerm(
            func=mdp.ball_vel_relative_to_racket,
            params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME},
        )
        racket_axes = ObsTerm(func=mdp.racket_axes, params={"racket_body_name": RACKET_BODY_NAME})
        predicted_hit_point = ObsTerm(
            func=mdp.predicted_hit_point_at_robot_x,
            params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE},
        )
        time_to_predicted_intercept = ObsTerm(
            func=mdp.time_to_predicted_intercept,
            params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE},
        )

    policy: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    racket_ball_proximity = RewTerm(
        func=mdp.racket_ball_proximity_dense,
        weight=0.8,
        params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME, "sigma": 12.0},
    )
    racket_approach = RewTerm(
        func=mdp.racket_approach_ball,
        weight=1.2,
        params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME, "target_vel": 2.0},
    )
    racket_face_target = RewTerm(
        func=mdp.racket_face_toward_target,
        weight=0.3,
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "target_x": OPP_TABLE_CENTER_X,
            "target_z": TABLE_Z,
            "launch_speed": 4.5,
            "proximity_gate": 0.55,
        },
    )
    racket_normal_swing = RewTerm(
        func=mdp.racket_normal_swing_velocity,
        weight=1.0,
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "target_x": OPP_TABLE_CENTER_X,
            "target_z": TABLE_Z,
            "launch_speed": 4.5,
            "target_speed": 2.5,
            "proximity_gate": 0.45,
        },
    )
    hit = RewTerm(func=mdp.sac_event_reward, weight=10.0, params={"event": "hit"})
    quality_hit = RewTerm(
        func=mdp.sac_quality_hit_reward,
        weight=60.0,
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "target_outgoing_speed": 3.5,
            "min_up_speed": 0.2,
            "target_up_speed": 1.0,
            "over_up_speed": 2.0,
            "outgoing_weight": 0.85,
            "up_weight": 0.15,
            "center_sigma": 60.0,
            "center_floor": 0.15,
        },
    )
    return_cross_net = RewTerm(func=mdp.sac_event_reward, weight=200.0, params={"event": "return"})
    valid_return = RewTerm(func=mdp.sac_event_reward, weight=200.0, params={"event": "valid_return"})
    landing_placement = RewTerm(
        func=mdp.sac_landing_placement,
        weight=200.0,
        params={"target_x": OPP_TABLE_CENTER_X, "target_y": 0.0, "sigma_x": 0.35, "sigma_y": 0.4},
    )
    miss = RewTerm(func=mdp.sac_miss_penalty, weight=-50.0)
    bad_hit = RewTerm(func=mdp.sac_bad_hit_penalty, weight=-40.0)
    post_hit_outgoing = RewTerm(
        func=mdp.post_hit_outgoing_velocity,
        weight=3.0,
        params={"ball_name": "ball", "robot_side": ROBOT_SIDE, "target_speed": 3.5},
    )
    post_hit_net_progress = RewTerm(
        func=mdp.post_hit_net_progress,
        weight=2.0,
        params={"ball_name": "ball", "robot_side": ROBOT_SIDE, "robot_x": ROBOT_X},
    )
    post_hit_lift = RewTerm(
        func=mdp.post_hit_lift_velocity,
        weight=1.0,
        params={"ball_name": "ball", "target_up_speed": 1.0},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    joint_limit = RewTerm(
        func=mdp.joint_limit_margin_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES), "margin": 0.20},
    )


@configclass
class EventCfg:
    reset_tracker = EventTerm(func=mdp.reset_sac_episode_state, mode="reset")
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_ready_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": RIGHT_ARM_JOINT_NAMES,
            "joint_pos": SAC_READY_JOINT_POS,
        },
    )
    reset_ball = EventTerm(
        func=mdp.launch_ball,
        mode="reset",
        params={"ball_cfg": SceneEntityCfg("ball"), **SAC_FIXED_MIDDLE_BALL},
    )
    track_episode = EventTerm(
        func=mdp.update_sac_episode_state,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "robot_x": ROBOT_X,
            "robot_side": ROBOT_SIDE,
            "own_table_x_min": OWN_TABLE_X[0],
            "own_table_x_max": OWN_TABLE_X[1],
            "opponent_table_x_min": OPP_TABLE_X[0],
            "opponent_table_x_max": OPP_TABLE_X[1],
            "hit_return_timeout_s": 0.90,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.sac_time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.joint_state_nan)
    episode_outcome = DoneTerm(func=mdp.sac_episode_done)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: A1TableTennisSacSceneCfg = A1TableTennisSacSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 2.5
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.enable_ccd = True
        self.scene.robot.init_state.pos = (ROBOT_BASE_X, 0.0, 0.0)
        if ROBOT_SIDE < 0:
            self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos["joint_lift"] = SAC_READY_LIFT_POS
        self.scene.robot.init_state.joint_pos = joint_pos


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.viewer.eye = (-2.5, -2.0, 1.4)
        self.viewer.lookat = (-1.0, 0.0, 1.0)
