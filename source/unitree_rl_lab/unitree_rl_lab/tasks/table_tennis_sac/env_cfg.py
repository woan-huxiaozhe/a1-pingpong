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
from unitree_rl_lab.assets.robots.a1 import A1_ARM_VELOCITY
from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import (
    OPP_TABLE_CENTER_X,
    OPP_TABLE_X,
    OWN_TABLE_X,
    RACKET_BODY_NAME,
    RIGHT_ARM_JOINT_NAMES,
    ROBOT_SIDE,
    TABLE_Z,
    X1TableTennisSceneCfg,
)

BALL_HISTORY_LENGTH = 4
JOINT_POS_DELTA_HISTORY_LENGTH = 3
SAC_ROBOT_BASE_X = (1.37 + 0.45) * ROBOT_SIDE
SAC_ROBOT_X = -1.37
# For ROBOT_SIDE=-1, an unhit ball is missed once it passes SAC_ROBOT_X - margin.
SAC_MISS_MARGIN = 0.10

SAC_READY_LIFT_POS = -0.22
# SAC_READY_JOINT_POS = [1.13, -0.39, 2.00, -1.32, 0.0, 1.0, -1.845288]
SAC_READY_JOINT_POS = [1.13, -0.39, 1.80, -1.4, 0.0, 0.8, -1.845288]
SAC_MAX_JOINT_VELOCITY = [A1_ARM_VELOCITY[name] for name in RIGHT_ARM_JOINT_NAMES]
# Deployment KF hit-point prediction error model, calibrated from the real Kalman
# (kalman_filter_pingpong) open-loop residuals on the 0617 mocap serves. The real error is
# dominated by a per-serve consistent bias (random direction each serve, set by that ball's
# spin the non-Magnus KF mispredicts), not per-step white noise -- hence the per-episode
# bias term. Each component scales with the per-axis horizon phase tau/far_tau so the command
# converges to truth as the ball arrives. far_tau is per-axis: z saturates early (~0.40s),
# y grows ~linearly to ~0.65s, tau plateaus mid-flight (~0.55s). Tuples are (y, z, tau);
# units m / m / s.
SAC_HIT_COMMAND_NOISE = {
    "bias_std_far": (0.024, 0.022, 0.016),
    "jitter_std": (0.003, 0.008, 0.003),
    "fixed_offset_far": (0.0, -0.010, -0.005),
    "far_tau": (0.65, 0.40, 0.55),
}

# Centering reward calibration. As of the Ace three-tier refactor (feat/sony-ace) NO active
# reward references these -- the multiplicative center gate was replaced by the physics-based
# racket_spin_penalty (R_omega). Kept only as ablation hooks for the disabled center terms
# (quality_hit, hit_centered, center-gated landing_placement). The paddle blade radius is
# ~7.5 cm; sigma concentrates reward in the inner ~2.5 cm (offset 7.5 cm -> factor ~0.24).
SAC_CENTER_SIGMA = 400.0
SAC_CENTER_GATE_FLOOR = 0.15

SAC_FIXED_MIDDLE_BALL = {
    "x_range": (-1.0 * ROBOT_SIDE, -1.0 * ROBOT_SIDE),
    "y_range": (0.0, 0.20),
    "z_range": (1.15, 1.25),
    "vx_range": (3.8 * ROBOT_SIDE, 4.3 * ROBOT_SIDE),
    "vy_range": (0.0, 0.0),
    "vz_range": (1.05, 1.35),
}

# slower
# SAC_FIXED_MIDDLE_BALL = {
#     "x_range": (-1.0 * ROBOT_SIDE, -1.0 * ROBOT_SIDE),
#     "y_range": (-0.1, 0.3),
#     "z_range": (0.8, 1.0),
#     "vx_range": (3.0 * ROBOT_SIDE, 3.5 * ROBOT_SIDE),
#     "vy_range": (0.0, 0.0),
#     "vz_range": (2.5, 3.0),
# }

SAC_SERVE_STATES_PATH = os.path.join(os.path.dirname(__file__), "serve_states_x1.npz")
SAC_USE_SERVE_STATES = False
SAC_BALL_LINEAR_DAMPING = 0.05
SAC_BALL_DRAG_K = 0.08
SAC_BALL_RESET_PARAMS = (
    {"ball_cfg": SceneEntityCfg("ball"), "serve_states_path": SAC_SERVE_STATES_PATH}
    if SAC_USE_SERVE_STATES
    else {"ball_cfg": SceneEntityCfg("ball"), **SAC_FIXED_MIDDLE_BALL}
)

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
        joint_pos_delta_history = ObsTerm(
            func=mdp.joint_pos_delta_history,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES),
                "history_length": JOINT_POS_DELTA_HISTORY_LENGTH,
            },
        )
        ball_pos_history = ObsTerm(
            func=mdp.ball_pos_history,
            params={"ball_name": "ball", "history_length": BALL_HISTORY_LENGTH},
        )
        estimated_hit_command = ObsTerm(
            func=mdp.estimated_hit_command_at_robot_x,
            params={
                "ball_name": "ball",
                "robot_x": SAC_ROBOT_X,
                "robot_side": ROBOT_SIDE,
                **SAC_HIT_COMMAND_NOISE,
            },
        )
        # FK three-piece: blade-center pose + ball-relative geometry. These are FK-derived
        # quantities the real robot can compute exactly, so they are deployable -> actor.
        racket_pos = ObsTerm(func=mdp.racket_pos, params={"racket_body_name": RACKET_BODY_NAME})
        racket_normal = ObsTerm(func=mdp.racket_normal, params={"racket_body_name": RACKET_BODY_NAME})
        # TODO: ball_pos_rel_racket currently uses the CLEAN ball state (ball history is not yet
        # noised, so it is equivalent). Once ball noise is injected, recompute this from the
        # actor's noised ball signal rather than the clean simulator ball.
        ball_pos_rel_racket = ObsTerm(
            func=mdp.ball_pos_relative_to_racket,
            params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME},
        )
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
        ball_vel = ObsTerm(func=mdp.ball_vel_w, params={"ball_name": "ball"})
        racket_vel = ObsTerm(func=mdp.racket_vel, params={"racket_body_name": RACKET_BODY_NAME})
        racket_ang_vel = ObsTerm(func=mdp.racket_ang_vel, params={"racket_body_name": RACKET_BODY_NAME})
        ball_vel_rel_racket = ObsTerm(
            func=mdp.ball_vel_relative_to_racket,
            params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME},
        )
        racket_axes = ObsTerm(func=mdp.racket_axes, params={"racket_body_name": RACKET_BODY_NAME})
        groundtruth_hit_command = ObsTerm(
            func=mdp.hit_command_at_robot_x,
            params={"ball_name": "ball", "robot_x": SAC_ROBOT_X, "robot_side": ROBOT_SIDE},
        )

    policy: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # racket_ball_proximity = RewTerm(
    #     func=mdp.racket_ball_proximity_dense,
    #     weight=0.0,
    #     params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME, "sigma": 12.0},
    # )
    # racket_approach = RewTerm(
    #     func=mdp.racket_approach_ball,
    #     weight=0.0,
    #     params={"ball_name": "ball", "racket_body_name": RACKET_BODY_NAME, "target_vel": 2.0},
    # )
    # racket_face_target = RewTerm(
    #     func=mdp.racket_face_toward_target,
    #     weight=0.0,
    #     params={
    #         "ball_name": "ball",
    #         "racket_body_name": RACKET_BODY_NAME,
    #         "target_x": OPP_TABLE_CENTER_X,
    #         "target_z": TABLE_Z,
    #         "launch_speed": 4.5,
    #         "proximity_gate": 0.55,
    #     },
    # )
    # racket_normal_swing = RewTerm(
    #     func=mdp.racket_normal_swing_velocity,
    #     weight=0.0,
    #     params={
    #         "ball_name": "ball",
    #         "racket_body_name": RACKET_BODY_NAME,
    #         "target_x": OPP_TABLE_CENTER_X,
    #         "target_z": TABLE_Z,
    #         "launch_speed": 4.5,
    #         "target_speed": 2.5,
    #         "proximity_gate": 0.45,
    #     },
    # )
    # === Sparse/event-heavy reward ladder (effective = weight * step_dt(0.02)) ===
    #
    # 0625 lob-plateau diagnosis: the approach-window dense terms
    # (racket_ideal_velocity_match, racket_ideal_normal_match, racket_predicted_landing)
    # were learnable proxies, but their optimum was "hold an ideal-looking paddle in the
    # approach window", not "commit a flat drive through the contact frame". They are
    # therefore unwired by default; re-enable them only in an explicit ablation.
    #
    # miss_approach is also unwired: it fired only on the mutually-exclusive miss event and
    # paid the policy for near-missing, which encouraged hover/chase behavior instead of
    # contact. Exploration now comes from random warmup, entropy, event replay, and the event
    # ladder below.

    # --- Tier 1: make contact, then cross the net ---
    hit_bonus = RewTerm(func=mdp.sac_event_reward, weight=20.0, params={"event": "hit"})  # +0.40
    return_cross_net = RewTerm(func=mdp.sac_event_reward, weight=40.0, params={"event": "return"})  # +0.80
    # quality_hit DISABLED: hit speed/up-speed proxies are too easy to satisfy with the same
    # stereotyped lob; the active ladder scores real return events instead.
    # quality_hit = RewTerm(
    #     func=mdp.sac_quality_hit_reward,
    #     weight=10.0,
    #     params={"min_outgoing_speed": 1.0, "good_outgoing_speed": 3.0,
    #             "min_up_speed": -0.2, "up_tolerance": 0.4},
    # )
    table_proximity = RewTerm(  # signed bad-hit bridge, effective [-0.10, +0.10]
        func=mdp.sac_table_proximity,
        weight=5.0,
        params={
            "target_x": OPP_TABLE_CENTER_X,
            "table_x_min": OPP_TABLE_X[0],
            "table_x_max": OPP_TABLE_X[1],
            "table_y_half": 0.7625,
            "scale": 1.0,
            "floor": -1.0,
        },
    )
    # R_omega clean-contact penalty is kept small while the sparse/event-heavy run is still
    # searching for a committed swing; too much early wrist-spin pressure can suppress useful
    # exploration before valid returns are common.
    racket_spin_penalty = RewTerm(func=mdp.sac_racket_spin_penalty, weight=-2.0, params={"scale": 12.0})

    # --- Tier 2: valid return ---
    return_bonus = RewTerm(func=mdp.sac_event_reward, weight=100.0, params={"event": "valid_return"})  # +2.00 const
    landing_placement = RewTerm(  # +0.50 cap; NO center gate (center_sigma/floor default 0)
        func=mdp.sac_landing_placement,
        weight=25.0,
        params={
            "target_x": OPP_TABLE_CENTER_X,
            "target_y": 0.0,
            "sigma_x": 0.25,
            "sigma_y": 0.3,
        },
    )
    # Flat-return bonus (replaces disabled lob penalty). ref_height ~ 1.4 m, band 0.4 m: a flat
    # apex (<=1.0 m) -> +0.20, a 1.4 m+ lob -> 0.
    flat_return = RewTerm(func=mdp.sac_flat_return, weight=10.0, params={"ref_height": 1.4, "band": 0.4})

    # === Disabled terminal/dense terms (ablation hooks; not referenced as active rewards) ===
    # Replaced by the three-tier ladder above. Old small hit const, center-gating subsystem,
    # per-step post_hit dense terms, and the flat miss/bad_hit penalties.
    # hit = RewTerm(func=mdp.sac_event_reward, weight=2.0, params={"event": "hit"})
    # hit_centered = RewTerm(func=mdp.sac_centered_hit_reward, weight=8.0, params={"sigma": SAC_CENTER_SIGMA})
    # return_cross_net old wiring used weight=15.0; active sparse/event run uses 40.0.
    # valid_return = RewTerm(func=mdp.sac_event_reward, weight=25.0, params={"event": "valid_return"})
    # landing_placement (center-gated) = RewTerm(
    #     func=mdp.sac_landing_placement,
    #     weight=25.0,
    #     params={
    #         "target_x": OPP_TABLE_CENTER_X,
    #         "target_y": 0.0,
    #         "sigma_x": 0.25,
    #         "sigma_y": 0.3,
    #         "center_sigma": SAC_CENTER_SIGMA,
    #         "center_gate_floor": SAC_CENTER_GATE_FLOOR,
    #     },
    # )
    # miss = RewTerm(func=mdp.sac_miss_penalty, weight=-5.0)
    # bad_hit = RewTerm(func=mdp.sac_bad_hit_penalty, weight=-3.0)
    # Dense bridge hit->return (re-enabled): rewards outgoing x-speed toward the opponent on
    # hit & ~valid_return & ~bad_hit, [0,1] scaled by target_speed. Height-blind (x only), so it
    # must be paired with a height/flatness term to avoid farming lobs. Starting weight 1.0 --
    # this is the main knob: raise it if returns still fail to emerge against joint_limit.
    post_hit_outgoing = RewTerm(
        func=mdp.post_hit_outgoing_velocity,
        weight=1.0,
        params={"ball_name": "ball", "robot_side": ROBOT_SIDE, "target_speed": 3.5},
    )
    # post_hit_net_progress = RewTerm(
    #     func=mdp.post_hit_net_progress,
    #     weight=0.0,
    #     params={"ball_name": "ball", "robot_side": ROBOT_SIDE, "robot_x": SAC_ROBOT_X},
    # )
    # post_hit_net_clearance / post_hit_landing_prediction DISABLED (feat/sony-ace ideal-
    # velocity refactor): both use a GRAVITY-ONLY projectile solve, but the scene now has
    # quadratic air drag (k=0.08) + linear_damping (0.05), so their landing prediction is
    # systematically optimistic -- they paid partial credit to a high lob that never reaches
    # the opponent table, cementing the touch-lob local optimum. The active sparse/event run
    # relies on return_cross_net, valid_return, landing_placement, and flat_return instead.
    # Height-aware partner to post_hit_outgoing (re-enabled with a drag-correct predictor):
    # rewards the predicted ball height at the net plane via mdp.predict_z_at_x, which mirrors
    # the sim's quadratic-drag (k=0.08) + linear-damping (0.05) dynamics -- so it no longer
    # over-credits high lobs the way the old gravity-only solve did. This is the term that gives
    # the policy a "flatter/faster = better" gradient and discourages the touch-lob optimum.
    post_hit_net_clearance = RewTerm(
        func=mdp.post_hit_net_clearance,
        weight=1.0,
        params={"ball_name": "ball", "robot_side": ROBOT_SIDE},
    )
    # post_hit_landing_prediction = RewTerm(
    #     func=mdp.post_hit_landing_prediction, weight=5.0,
    #     params={"ball_name": "ball", "robot_side": ROBOT_SIDE, "target_x": OPP_TABLE_CENTER_X,
    #             "target_y": 0.0, "table_x_min": OPP_TABLE_X[0], "table_x_max": OPP_TABLE_X[1],
    #             "table_y_half": 0.7625, "table_z": TABLE_Z, "sigma_x": 0.25, "sigma_y": 0.3},
    # )
    # post_hit_lob_penalty = RewTerm(
    #     func=mdp.post_hit_lob_penalty,
    #     weight=0.0,
    #     params={
    #         "ball_name": "ball",
    #         "robot_side": ROBOT_SIDE,
    #         "max_height": 1.25,
    #         "height_band": 0.35,
    #         "max_up_speed": 1.6,
    #         "up_speed_band": 1.2,
    #     },
    # )

    # === Sim-to-real smoothing regularizers (kept at current weights) ===
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,  # halved (-1.0e-6->-5.0e-7) to let the swing accelerate while learning the motion
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    joint_jerk = RewTerm(
        func=mdp.joint_jerk_l2,
        weight=-2.0e-10,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    # margin 0.20->0.05 (rad) + weight -3.0->-1.0: the 0.20 rad (~11.5 deg) band over all 7
    # arm joints fired a dense per-step penalty across the entire return swing, so producing a
    # return cratered total reward (joint_limit hit -0.46 the moment returns appeared) while the
    # sparse return payoff was discounted away -- returns were net-negative and got suppressed.
    # The narrowed band only bites within ~2.9 deg of a hard limit; revisit/anneal up later.
    joint_limit = RewTerm(
        func=mdp.joint_limit_margin_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES), "margin": 0.05},
    )
    joint_effort_margin = RewTerm(
        func=mdp.joint_effort_margin_penalty,
        weight=-0.25,  # halved (-0.5->-0.25) to unblock the swing while learning the motion
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES), "margin_frac": 0.85},
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
        params=SAC_BALL_RESET_PARAMS,
    )
    ball_air_drag = EventTerm(
        func=mdp.apply_air_drag,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"ball_cfg": SceneEntityCfg("ball"), "k": SAC_BALL_DRAG_K},
    )
    track_episode = EventTerm(
        func=mdp.update_sac_episode_state,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "robot_x": SAC_ROBOT_X,
            "robot_side": ROBOT_SIDE,
            "own_table_x_min": OWN_TABLE_X[0],
            "own_table_x_max": OWN_TABLE_X[1],
            "opponent_table_x_min": OPP_TABLE_X[0],
            "opponent_table_x_max": OPP_TABLE_X[1],
            "miss_margin": SAC_MISS_MARGIN,
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
        self.scene.ball.spawn.rigid_props.linear_damping = SAC_BALL_LINEAR_DAMPING
        self.scene.robot.init_state.pos = (SAC_ROBOT_BASE_X, 0.0, 0.0)
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
