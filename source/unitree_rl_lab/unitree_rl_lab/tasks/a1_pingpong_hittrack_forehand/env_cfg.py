"""A1-Pingpong-HitTrack-FOREHAND environment config (100 Hz, non-end-to-end).

FORK of the backhand HitTrack env_cfg -- ONLY the stance constants (ready pose + base) differ; all
mdp / analytic reference / baked npz are reused unchanged. Do NOT edit the backhand task from here.

A new, additive task that does NOT touch the Catch ``env_cfg.py``. The arm tracks a
model-derived end-effector hit reference ``(p_ref, v_ref, n_ref)`` at the predicted ball hit
time; the ball is out of the MDP. The reference is produced at runtime by
``mdp.reset_reference_command`` / ``mdp.update_hit_track_state`` (which wrap the pure-torch
``reference_planner`` + ``reference_source``), and tracked by the time-gated Gaussian rewards
``mdp.hit_ref_pos`` / ``mdp.hit_ref_vel``.

Control runs at 100 Hz (``decimation=2``, ``sim.dt=0.005``, ``step_dt=0.01``) to match the real
control loop; Catch stays at 50 Hz. The scene / robot placement come from the forehand base task
with the ball DROPPED (it is out of the MDP); the ready pose, physics and action scale (halved for
100 Hz) are owned here directly.
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

import unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp as mdp
from unitree_rl_lab.assets.robots.a1 import A1_ARM_VELOCITY
from unitree_rl_lab.tasks.table_tennis.robots.a1.forehand.env_cfg import (
    X1TableTennisSceneCfg,
)

# --- Robot / scene facts (inlined; HitTrack no longer inherits constants from the Catch SAC
# env cfg). These are physical facts of the A1 + table + ball scene, owned here directly. The
# scene class itself comes from the forehand base task -- the SAC scene was a no-op `pass`
# subclass of it, so this drops the table_tennis_sac.env_cfg dependency entirely. ---
ROBOT_SIDE = -1  # +1 = robot at +X, -1 = robot at -X (flipped); A1 runs flipped
RACKET_BODY_NAME = "Link_yb_paddle"
RIGHT_ARM_JOINT_NAMES = [
    "joint_yb_1",
    "joint_yb_2",
    "joint_yb_3",
    "joint_yb_4",
    "joint_yb_5",
    "joint_yb_6",
    "joint_yb_7",
]
TABLE_Z = 0.76  # table surface height (m)
OPP_TABLE_CENTER_X = 0.685  # opponent half-table center x (landing target); = 0.5*(0+1.37), ROBOT_SIDE=-1
# Forehand standoff 0.20 (base = -1.57): moved +0.25 m toward the opponent vs the inherited backhand
# 0.45 (-1.82). FK-verified 2026-07-06 at the forehand ready pose (joint7 flipped): ready blade center
# = (-1.544, -0.058, +1.003) => 0.104 m BEHIND HIT_PLANE_X (-1.44), i.e. positive forward runway
# (tighter than the same pose's 0.354 m at -1.82, but still behind the plane). Trainable in a smoke run.
ROBOT_BASE_X = (1.37 + 0.20) * ROBOT_SIDE  # = -1.57; robot base placement
ROBOT_BASE_Y = 0.8  # forehand lateral base offset (tunable; 0.0 = same as backhand). Paired with the
# forehand ready pose; adjust together with READY_JOINT_POS if FK shows the blade off-center laterally.
MAX_JOINT_VELOCITY = [A1_ARM_VELOCITY[name] for name in RIGHT_ARM_JOINT_NAMES]  # = [8,8,8,20,20,20,20]

# --- Tuning carried over from the Catch SAC config, now OWNED (forked) by HitTrack. Changes to
# the Catch ready pose / lift no longer propagate here. ---
# READY_JOINT_POS = [1.13, -0.39, 1.80, -1.4, 0.0, 0.8, -1.845288]  # old
# READY_JOINT_POS = [1.6, -0.7, 1.6, -1.7, 0.0, 0.6, -1.8]
# Forehand ready pose (user viewer-tuned, 2026-07-06). Replaces the backhand ready pose. joint_yb_1..7.
# joint7=-1.57 rolls the wrist ~180 deg so the paddle's local +Y face points toward the opponent (+X):
# FK rn.nref -0.89 -> +0.95 (the signed-cosine normal reward needs the +Y face forward, not backward).
# The proximal J1/J2 drive swing speed while the wrist holds loft (the forehand decoupling hypothesis).
READY_JOINT_POS = [-0.25, -0.96, 1.32, 0.98, -1.27, 0.47, -1.57]
# READY_JOINT_POS = [0.69, -0.81, 0.28, 0.979, -1.52, 0.47, -1.57]

READY_LIFT_POS = -0.22

# --- HitTrack constants (v1, 100 Hz) ---
# HIT_PLANE_X = -1.37  # old plane = the Catch robot_x (was SAC_ROBOT_X)
HIT_PLANE_X = -1.44 # = -1.44; move the hit plane slightly forward to avoid clipping the racket
HITTRACK_TARGET_XYZ = (OPP_TABLE_CENTER_X, 0.0, TABLE_Z)
# Paddle restitution used by the v_ref planner. HitTrack-LOCAL on purpose: the ball is out of the
# MDP, so v_ref is the paddle velocity that returns the ball on the REAL robot -> use the real
# rubber's normal coefficient (~0.9), NOT the sim a1.usd material 0.75 that the Catch task's
# in-sim physics reward must match. Higher e => less paddle speed needed (0.75 -> 0.9 lowers the
# commanded |v_ref| from ~1.54 to ~1.23 m/s, matching the traditional controller).
HIT_RESTITUTION = 0.9
# Synthetic-serve sampling box: the (y,z,vx,vy,vz) ranges of a virtual ball state AT the hit plane.
# ONLY used by the synthetic source (curriculum (1)/(2)) -- i.e. when ``HITTRACK_USE_BAKED=False``.
# Under the default baked mode the reset reads whole recorded trajectories from the npz and this box
# drives nothing (the bake script derives its own serve range from the real data + bake-time gates,
# not from here). Kept as the synthetic fallback / ablation toggle.
HITTRACK_BOX = {
    "y": (-0.15, 0.05),
    "z": (0.9, 1.25),
    "vx": (-4.5, -3.0),
    "vy": (-0.3, 0.3),
    "vz": (-1.0, 0.5),
}
MAX_PREP_S = 0.98  # baked real serves are seen from ball-x in [0.6,1.5]; flight to the hit plane
# takes up to ~0.97 s, so the prep horizon must cover it (was 0.6 for the synthetic box).
POST_MARGIN_S = 0.12
STEP_DT = 0.01  # 100 Hz control (decimation=2 * sim.dt=0.005)
SIGMA_T = 0.03
SIGMA_P = 0.05
# Velocity retune after lowering A1 PD gains: keep the Gaussian broad enough that the resumed policy
# still sees gradient at ~0.6 m/s error, but make the term more valuable than the old 20 * sigma=0.5
# setup once it starts closing the x-velocity gap.
SIGMA_V = 0.25
SIGMA_NORMAL_DEG = 10.0  # angular Gaussian width for blade-normal alignment at the hit instant
# 20.0 -> 15.0: the time gate (SIGMA_T_NORMAL) was already narrowed once (0.03->0.015) yet normal
# error stayed ~20deg/92% of hits >10deg, identical between success and failed hits (a systematic
# tracking floor, not timing noise) -- narrowing the window further has little room left and mostly
# adds sparsity. Sharpen the angular tolerance instead so the reward gradient pushes harder toward
# small errors from wherever the policy's swing dynamics actually land.
# Narrower than SIGMA_T: the blade normal sweeps continuously through the swing (unlike pos/vel,
# it is never "held"), so gating it over the same +/-2*SIGMA_T window as pos/vel demanded alignment
# across a span wide enough for the racket to rotate ~20-30 deg, fighting swing speed. Keep pos/vel
# timing untouched; only tighten the window the normal term actually pays out over.
SIGMA_T_NORMAL = 0.010
# 12.0 -> 30.0 (2026-07-04): normal_err_deg does NOT converge -- it bottoms ~4-5deg early (step ~100,
# while the swing is still slow) then DIVERGES back to ~19deg in lockstep with vel_err falling. The
# policy is TRADING a square paddle for swing speed because W_VEL(40) >> W_NORMAL(12) and normal only
# pays in a narrow +/-2-step gate. A J4 big-motor preview (torque 8->28) did NOT fix it (diverged
# earlier) -> torque is not the binding cause; the reward balance is. Raise W_NORMAL toward W_POS(20)/
# W_VEL(40) so holding loft is worth the swing-speed it costs. Isolated change (gate/sigma untouched).
W_NORMAL = 30.0
W_POS = 25.0
W_VEL = 30.0
SUCCESS_POS = 0.05
SUCCESS_VEL = 0.2
REACH_Y = (-0.20, 0.10)  # baked-mode reset draws each serve's y target uniformly from this range minus
# a 0.05 margin each end via sample_lateral_shift() (mdp/reference_source.py), so it IS the trained
# lateral serve range -- HITTRACK_BOX["y"] is INERT under baked mode. History: -0.2,0.2 -> -0.10,0.30
# -> -0.20,0.10. The -0.10,0.30 range put targets at U(-0.05,+0.25) (mean +0.10), offset ~+0.16 from the
# FK ready blade-center y (-0.058); -0.20,0.10 gives U(-0.15,+0.05) (mean -0.05) centered on the blade so
# both reach directions match (also realizes the lateral intent of the earlier, baked-inert HITTRACK_BOX).
REACH_Z = (0.7, 1.5)
JOINT_POS_DELTA_HISTORY_LENGTH = 5

# Curriculum (3): baked real-serve source. Off by default (curriculum (1)/(2) train on the
# synthetic box). When on, the reset loads `HITTRACK_BAKED_PATH` (produced by
# bake_hittrack_references.py) and samples a recorded serve per env instead of sampling the box.
HITTRACK_USE_BAKED = True
# Baked real-serve references are POSTURE-AGNOSTIC (defined at the hit plane in the env frame), so the
# forehand task REUSES the backhand package's npz rather than duplicating it (spec: do not copy). NOTE:
# os.path.dirname(__file__) resolves to THIS forehand dir, which has NO npz -- hop to the sibling
# backhand package dir explicitly.
HITTRACK_BAKED_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "a1_pingpong_hittrack", "hittrack_references.npz"
)


@configclass
class ActionsCfg:
    right_arm = mdp.JointDeltaTargetActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        # 0.06 -> 0.10: the 0.06 (halved-from-Catch) scale was THE swing-speed bottleneck. Play-CSV
        # telemetry (model_1500) showed the policy SATURATING the action (raw |act|~1.85, clamped to
        # 1) on the proximal joints while joint-vel utilization was only 16-36% and torque/limits were
        # slack -> the arm reached just ~56% of the commanded v_ref_x (verr_x~0.54). A zero-cost
        # inference sweep on model_1500 confirmed the direction: scale 0.06->0.10 lifted x-reach to
        # ~72% and cut verr_x to ~0.33.
        # 2026-07-04 (model_10000 telemetry, new ready pose): at the hit instant J1-3 use only
        # 8-16% of their 8 rad/s velocity limit and 33-41% of their 28 N*m torque -- huge headroom on
        # BOTH -- yet the raw action is clamped 57-83% of steps (J2 net output ~4.7). The throttle is
        # NOT the rate cap and NOT the actuators: raw_target = q + action*scale re-anchors to the
        # CURRENT q every step, so the commanded lead is permanently capped at action_scale and the
        # PD steady velocity ceiling is ~K*scale/D (=200*0.10/5=4 rad/s, realized ~1.5 in the short
        # swing). Lever = raise the lead on the joints WITH headroom. J1-3 -> 0.16 (paired with
        # damping 5.0->3.5 in a1.py so K*scale/D~=9 > 8 rad/s hw, uncapping within the hw limit);
        # J4-6 stay 0.10 (already torque-saturated at their 8 N*m ceiling, more scale just clips); J7 unused.
        action_scale=[0.20, 0.20, 0.20, 0.10, 0.10, 0.10, 0.10],
        # 0.5 -> 0.7: `smoothing` is the EMA blend toward the raw target, so HIGHER = more responsive
        # (less lag), not smoother-slower. The same sweep showed smoothing 0.8 beat 0.5 on swing speed
        # (0.3 was worse); 0.7 pairs with the larger scale for a faster target ramp without going fully
        # unfiltered. Retrained (resume from model_1500) so the policy re-calibrates position precision
        # to the higher authority (inference-only overrides raised perr past 0.05 -- an expected
        # train/test mismatch, recovered by retraining).
        smoothing=0.7,
        max_joint_velocity=MAX_JOINT_VELOCITY,
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
    # blade-normal alignment at the hit instant (previously UNREWARDED -> normal drifted to ~100 deg
    # error). Cosine kernel, gated identically to pos/vel via _ht_tau_true.
    hit_ref_normal = RewTerm(
        func=mdp.hit_ref_normal,
        weight=W_NORMAL,
        params={
            "racket_body_name": RACKET_BODY_NAME,
            "sigma_t": SIGMA_T_NORMAL,
            "sigma_normal_deg": SIGMA_NORMAL_DEG,
        },
    )

    # --- sim-to-real smoothing regularizers (copied verbatim from Catch env_cfg) ---
    # -0.005 -> -0.0025: J1-3 already saturate raw action 74-79% of the time (bottlenecked by the
    # max_joint_velocity rate cap, not this penalty) so they're unaffected; J4-6 saturate only
    # 27-56% with torque occasionally at the 8N*m ceiling, i.e. there is some headroom the penalty
    # may be suppressing. Halved rather than dropped further to keep most of the smoothness margin
    # for sim-to-real transfer.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0025)
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
            "joint_pos": READY_JOINT_POS,
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
            "restitution": HIT_RESTITUTION,
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
class ForehandHitTrackEnvCfg(ManagerBasedRLEnvCfg):
    scene: X1TableTennisSceneCfg = X1TableTennisSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2  # 100 Hz control over 200 Hz physics
        self.episode_length_s = 1.10  # ~= max_prep(0.98) + post_margin(0.12) (~110 steps @100 Hz)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.enable_ccd = False
        # HitTrack does not model the ball (it is out of the MDP); drop the inert rigid body so
        # PhysX never simulates it. IsaacLab's InteractiveScene skips ``asset_cfg is None`` entities.
        self.scene.ball = None
        self.scene.robot.init_state.pos = (ROBOT_BASE_X, ROBOT_BASE_Y, 0.0)
        if ROBOT_SIDE < 0:
            self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos["joint_lift"] = READY_LIFT_POS
        self.scene.robot.init_state.joint_pos = joint_pos
        # Curriculum (3): switch the reference source to the baked real serves (lazy-loaded
        # on the first reset onto env.device).
        if HITTRACK_USE_BAKED:
            self.events.reset_reference.params["baked_path"] = HITTRACK_BAKED_PATH


class ForehandHitTrackPlayEnvCfg(ForehandHitTrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.viewer.eye = (-0.5, -2.4, 1.4)  # closer play/record framing (synced from hitter@4da9e2b)
        self.viewer.lookat = (-1.0, 0.0, 1.0)
