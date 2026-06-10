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
    OPP_TABLE_X,
    OWN_TABLE_X,
    RACKET_BODY_NAME,
    RIGHT_ARM_JOINT_NAMES,
    ROBOT_BASE_X,
    ROBOT_SIDE,
    ROBOT_X,
    X1TableTennisSceneCfg,
)

BALL_HISTORY_LENGTH = 4

SAC_READY_LIFT_POS = -0.22
SAC_READY_JOINT_POS = [1.533406, -0.523925, 1.60474, -1.183103, -0.007649, 1.042375, -1.845288]
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
        pass

    policy: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    racket_ball_proximity = RewTerm(
        func=mdp.racket_ball_proximity_dense,
        weight=0.4,
        params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME, "sigma": 12.0},
    )
    racket_approach = RewTerm(
        func=mdp.racket_approach_ball,
        weight=0.1,
        params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME, "optimal_vel": 1.0, "sigma": 0.8},
    )
    racket_forward_push = RewTerm(
        func=mdp.racket_forward_push_velocity,
        weight=2.0,
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "robot_side": ROBOT_SIDE,
            "target_speed": 1.0,
            "distance_threshold": 0.55,
        },
    )
    hit = RewTerm(func=mdp.sac_event_reward, weight=50.0, params={"event": "hit"})
    return_cross_net = RewTerm(func=mdp.sac_event_reward, weight=200.0, params={"event": "return"})
    valid_return = RewTerm(func=mdp.sac_event_reward, weight=400.0, params={"event": "valid_return"})
    miss = RewTerm(func=mdp.sac_miss_penalty, weight=-50.0)
    bad_hit = RewTerm(func=mdp.sac_bad_hit_penalty, weight=-100.0)
    post_hit_outgoing = RewTerm(
        func=mdp.post_hit_outgoing_velocity,
        weight=3.0,
        params={"ball_name": "ball", "robot_side": ROBOT_SIDE, "target_speed": 2.0},
    )
    post_hit_lift = RewTerm(
        func=mdp.post_hit_lift_velocity,
        weight=1.0,
        params={"ball_name": "ball", "target_up_speed": 1.0},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
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
            "hit_return_timeout_s": 0.40,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.sac_time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.joint_state_nan)
    episode_outcome = DoneTerm(func=mdp.sac_episode_done)
    joint_limit = DoneTerm(
        func=mdp.joint_position_limit_violation,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES), "margin": 0.005},
    )


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
