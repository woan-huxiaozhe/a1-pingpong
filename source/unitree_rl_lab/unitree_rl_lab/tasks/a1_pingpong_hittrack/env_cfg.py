"""A1-Pingpong-HitTrack environment config (100 Hz, non-end-to-end).

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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
RACKET_BODY_NAME = "right_paddle"
RIGHT_ARM_JOINT_NAMES = [
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "r7",
]
TABLE_Z = 0.76  # table surface height (m)
OPP_TABLE_CENTER_X = 0.685  # opponent half-table center x (landing target); = 0.5*(0+1.37), ROBOT_SIDE=-1
# 0.45 standoff (reverted 2026-07-04 from the 0.47 back-move). model_10000 telemetry showed the
# ready-pose blade already sits behind HIT_PLANE_X, so runway was never the swing-speed bottleneck
# (the action interface is); the extra 2 cm only shortened far-+y lateral reach (perr_y up to 0.12 on
# the +y serves) for no velocity gain. FK-verified at base=-1.82: ready blade center = (-1.486, +0.034,
# +1.032) => 0.046 m BEHIND the plane (positive forward runway; not past it). Back to original standoff.
ROBOT_BASE_X = (1.37 + 0.45) * ROBOT_SIDE  # = -1.82; robot base placement
MAX_JOINT_VELOCITY = [A1_ARM_VELOCITY[name] for name in RIGHT_ARM_JOINT_NAMES]  # = [8,8,8,20,20,20,20]

# --- Tuning carried over from the Catch SAC config, now OWNED (forked) by HitTrack. Changes to
# the Catch ready pose / lift no longer propagate here. ---
# READY_JOINT_POS = [1.13, -0.39, 1.80, -1.4, 0.0, 0.8, -1.845288]  # old
# READY_JOINT_POS = [1.6, -0.7, 1.6, -1.7, 0.0, 0.6, -1.8]
# 2026-07-06 (model_9500 telemetry): re-optimized as a WIND-UP + normal PRE-TILT pose. model_9500
# play showed success gated purely by velocity (pos 100% pass, perr 0.019; verr median 0.192 just at
# the 0.2 gate) and the x-speed deficit is a TIMING problem: peak |ee_vx| reaches ~94% of v_ref but
# lands ~54 ms AFTER the hit (at tau=0 only 87%). Cause = torque-limited ramp still rising at contact;
# with 28/8 N*m now HW-matched (peak torque), the fix is more acceleration RUNWAY, not more torque.
# Solved from a CSV-fit FK Jacobian at ready (pos+normal, R2>=0.99, env frame), damped-LS small Δq,
# validated to ~1cm/2.4deg against the nearest real sim poses:
#   * WIND-UP: J4 pre-cocked -1.366 -> -1.738 (+0.37 rad more +x swing range); blade retreats ~2.4cm
#     so the runway to HIT_PLANE grows 5.7 -> 8.1 cm -> the torque-limited swing can reach v_ref BY
#     tau=0 without leaning on the peak-torque ceiling. y/z held (Δ<=1.6cm), base stays -1.82 (retreat
#     via pose, NOT a base-move -- avoids the 0.47 back-move's far-+y lateral-reach loss).
#   * PRE-TILT: ready blade normal pre-tilted toward the MEAN hit normal (0.937,-0.055,0.344); the
#     ready->hit sweep the wrist must do drops 21.1 -> ~10 deg (per-serve residual floor ~7.5deg), so
#     the torque-saturated (88-92%) wrist is freed to HOLD loft -> should shrink the normal-err tail.
# Requires a FRESH retrain (not a resume: ready pose changed substantially, max|Δq|=0.37).
# 2026-07-08: EXTRA pre-tilt (incremental on the 07-06 wind-up pose). model_13000 (run A) FK analysis
# showed the ready->mean-desired-hit-normal sweep still = 10.70deg vs a per-serve floor of ~6.7deg, and
# the wrist was torque-saturated (J4-6 95-98% of 8 N*m) HOLDING the blade square through the swing --
# the real vx<->normal coupling. Damped-LS on the FK Jacobian at ready (deploy fk.py, frame residual
# 0.024deg vs the env rn) rotated the ready normal ~30% further toward the mean desired hit normal
# [0.934,-0.022,0.356], cutting the sweep 10.70 -> 8.68deg (max|Δq|=0.059 rad, blade center held to
# 0.5 mm) so the wrist spends less torque re-aiming and can hold loft while the proximal chain swings
# faster. Tilt lives in J4-7 (proximal J1-3 move <0.016 rad). max|Δq|=0.059 (<< the 0.37 that forced a
# fresh retrain), so a RESUME from A's checkpoint should re-adapt quickly.
# READY_JOINT_POS = [1.377, -0.639, 1.660, -1.738, 0.118, 0.721, -2.094]  # 07-06 wind-up pose (pre-tilt)
READY_JOINT_POS = [1.369, -0.651, 1.656, -1.767, 0.145, 0.684, -2.153]

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
    "y": (-0.15, 0.25),
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
SIGMA_NORMAL_DEG = 10.0  # angular Gaussian width for blade-normal alignment at the hit instant.
# Tolerance is set in ANGLE, not time: the ~20deg normal error was a systematic tracking floor
# (identical on success/failed hits), so a narrower time gate only adds sparsity -- sharpen the
# angular gradient instead.
# SIGMA_T_NORMAL: time-gate width for BOTH orientation terms, narrower than SIGMA_T (0.03). The blade
# normal sweeps continuously (unlike pos/vel it is never "held"), so gating it over the full pos/vel
# window would demand alignment across a ~20-30deg rotation span and fight swing speed. 0.015 (~3.75
# gated steps) rewards "arrive square and HOLD through contact" -- reachable since the wind-up pose
# cut the ready->hit sweep to ~10deg.
SIGMA_T_NORMAL = 0.015
# Blade-face *turn-rate* Gaussian width (rad/s), for the still-face term (mdp.hit_ref_normal_rate).
# 1.0 rad/s ~= 57 deg/s: the knee sits exactly at the traditional controller's demonstrated cruise
# face-turn rate (~58 deg/s). Measured RL play (model_5500) tumbles the face at ~189 deg/s (~3.3 rad/s)
# at contact -> scores exp(-0.5*3.3^2) ~= 0.004 (heavily pressured); <=1 rad/s is nearly free. The
# term forces swing speed to be sourced from the proximal sweep (steady face) rather than a wrist
# snap (which tumbles the face -- the root of the ~20deg normal floor at peak speed). See rewards.py.
SIGMA_FACE_RATE = 1.0
# Blade-normal ALIGNMENT weight (WHERE the face points at contact). 60.0 -> 25.0 (2026-07-07): at 60
# the orientation cluster (alignment + still-face) over-solved normal_err (1.85deg median on run
# 2026-07-07_20-43-08) by trading away velocity (verr 0.33, ee_vx to 76%, success 5%) -- "slow but
# square" became a local optimum. The still-face term now attacks the wrist-snap tumble at its root,
# so alignment no longer has to be heavy; drop it to 25 to let pos/vel win the swing back. Nudge back
# toward ~45 if normal_err creeps >12deg.
# 2026-07-08 revert to run A's balance (W_NORMAL 35->40, W_POS 25->15) for a clean pose+damping test:
# A (07-07_15-09, W_NORMAL=40, W_POS=15, NO still-face) reached pos 2.1cm / vel_x 0.20 / normal 9.75deg
# -- the best 3-way balance of the four runs -- so we resume A into ITS OWN reward landscape + the new
# pre-tilt/damping, isolating the kinematic change. W_POS back to 15: D's bump to 25 barely moved pos
# (0.035->0.033) yet cost the swing, confirming pos is training-length/reach limited, not weight limited.
W_NORMAL = 40.0
W_POS = 15.0
W_VEL = 30.0
# Still-face term weight (rewards a non-tumbling blade face while swinging; mdp.hit_ref_normal_rate).
# It reshapes HOW speed is sourced (proximal sweep vs distal wrist snap), so it is a helper alongside
# the alignment weight, not the primary orientation driver. The kernel multiplies the still-face score
# by the velocity-matching Gaussian, so the old "escape by slowing the wrist" route (run
# 2026-07-07_10-30-18 collapsed to verr~1.16) is structurally CLOSED: a slow wrist zeroes the bonus.
# It therefore only refines a face that is ALREADY swinging fast -> best paired with a RESUME from a
# velocity-solved checkpoint, not from-scratch. Watch normal_rate_at_hit (target ~1 rad/s, down from
# ~3.3) AND that verr stays <=0.2.
# 2026-07-08 -> 0.0: DISABLE still-face for the run-A resume. It contributes ~0.01 reward in every run
# (the velocity coupling structurally zeroes it) yet the runs carrying it (B/C/D) had ~3.5x worse pos_z
# and no vel<->normal frontier gain vs A (which never had the term). The new ready pre-tilt attacks the
# wrist-tumble at its kinematic root instead. Re-enable (~10-20) only if normal_rate_at_hit creeps back
# toward ~3 rad/s after training.
W_NORMAL_RATE = 0.0
# --- Tolerance-plateau ("box") tracking (2026-07-09) ---
# The pos/vel/normal terms were unbounded Gaussians: vel keeps rewarding verr->0, and closing verr
# physically raises normal (wrist tumble / J4-6 torque saturation), so the optimizer perpetually
# trades normal for speed it no longer needs -- run 2026-07-08_20-42-36 showed normal 5.3->8.5deg
# WHILE pos/vel kept falling. A flat top inside each tolerance zeroes that marginal gradient once a
# term is "good enough", so the three stop fighting and gradient concentrates on the errors still
# OUTSIDE tolerance (the hard tail serves). Tolerances sit at / just inside the acceptance targets
# (pos<2cm, vel<0.2, normal<10deg); SIGMA_* stay as the beyond-box decay widths (dense pull-in for
# the tail). tol=0 anywhere recovers the old pure Gaussian.
POS_TOL = 0.02
VEL_TOL = 0.18  # a hair inside the 0.20 success gate for margin
NORMAL_TOL_DEG = 9.0  # a hair inside the 10 deg target

# --- Wide-gate constant-v_ref approach guidance (2026-07-09) ---
# Root cause of the late wrist snap: the core hit terms are gated within ~2*SIGMA_T (~60 ms) of the
# hit, so PPO gets NO gradient for the 70-200 ms wind-up where a proximal cruise must be initiated;
# the only actuator that can fix the hit-instant state inside the narrow gate is the light wrist -> it
# learns a late snap (measured play: peak |ee_v| ~55 ms AFTER the hit, blade face tumbling ~160 deg/s,
# which is the root of the normal-error tail). This term rewards the racket for already streaming at
# v_ref (full vector, CONSTANT target) through the wind-up, so cruising-through pays and the snap loses
# its reason to exist. Demo-free: v_ref is the model reference extended in time -- the traditional log
# only PROVES the flat push is feasible (vx~0.74 with face-turn 5-9 deg/s), it is NOT an imitation
# target, so RL still finds its own (possibly better) joint coordination. The gate is a wide Gaussian
# with the core hit-gate carved out of its center (mdp.approach_gate) -> ZERO at tau=0, so it cannot
# fight the tuned sharp velocity term at the hit instant (which solely owns tau=0).
SIGMA_T_APPROACH = 0.09   # wide-gate width (s): product gate peaks ~tau 0.05-0.08, tapers to ~0.2 s
SIGMA_V_APPROACH = 0.70   # BROAD (vs core SIGMA_V=0.25) so there IS gradient while still ramping from rest
W_APPROACH_VEL = 10.0     # ~1/3 of W_VEL(30); primary knob -- raise if the wind-up stays flat, lower if it swamps hit precision

SUCCESS_POS = 0.05
SUCCESS_VEL = 0.2
REACH_Y = (-0.15, 0.25)  # -0.2,0.2 -> -0.10,0.30: baked-mode reset uses sample_lateral_shift() to
# draw each serve's y target uniformly from this range (mdp/reference_source.py), so it IS the
# trained lateral serve range, not just a synthetic-curriculum box. Was symmetric about 0; recenter
# on the new ready pose's resting blade-center y (~+0.10) so both reach directions are comparable.
REACH_Z = (0.7, 1.5)
JOINT_POS_DELTA_HISTORY_LENGTH = 5

# --- Sim-to-real domain randomization (2026-07-10) ---
# The real A1 joint encoders are noisy; a policy trained on CLEAN sim joint state learns a high
# local gain w.r.t. joint_pos and AMPLIFIES that noise into per-step action jitter on hardware.
# Train the ACTOR with additive joint-state observation noise so it learns a locally smooth
# (noise-robust) mapping; the CRITIC keeps a CLEAN (privileged) view for a stable value target
# (mirrors the existing noisy-actor / *_clean-critic split). Bounded-uniform to match the forehand
# DR precedent (Unoise on joint_pos there = +/-0.01 rad).
JOINT_POS_OBS_NOISE = 0.005        # rad (~0.57 deg): raw joint-pos encoder noise (actor only)
JOINT_POS_DELTA_OBS_NOISE = 0.002 # rad: smaller -- the delta-history is the actor's velocity proxy;
# it is the channel most directly tied to jitter, so it IS corrupted, but at half the pos noise to
# keep the low-speed signal-to-noise usable (independent per-term samples, not the coupled encoder
# stream -- the standard IsaacLab approximation).
# PD gain randomization: multiply each ARM joint's stiffness/damping by U(0.9,1.1) at every reset
# (the real K/D differ from the sim nominal). Uses the built-in mdp.randomize_actuator_gains which,
# for the A1's IMPLICIT actuators, scales from default_joint_{stiffness,damping} (no cross-reset
# drift) AND writes the gains into the PhysX solver -- an in-place actuator.stiffness edit alone is a
# NO-OP on implicit-actuator dynamics (PhysX computes the PD from gains set once at init).
PD_GAIN_RAND_RANGE = (0.9, 1.1)

# Curriculum (3): baked serve source (ON by default; the curriculum (1)/(2) synthetic box is unused
# while this is True). The reset loads `HITTRACK_BAKED_PATH` (a bake_hittrack_references.py npz) and
# samples one recorded serve per env. Default = KDE-synthetic TRAIN set (933 serves densified from 50
# real anchors, held out 16). To VALIDATE generalization, point this at
# "hittrack_references_eval_real.npz" (the 15 held-out real serves) and measure hit success;
# "hittrack_references.npz" is the original 72 real serves.
HITTRACK_USE_BAKED = True
# HITTRACK_BAKED_PATH = os.path.join(os.path.dirname(__file__), "hittrack_references_train_synth.npz")
HITTRACK_BAKED_PATH = os.path.join(os.path.dirname(__file__), "hittrack_references_eval_real.npz")


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
            noise=Unoise(n_min=-JOINT_POS_OBS_NOISE, n_max=JOINT_POS_OBS_NOISE),
        )
        joint_pos_delta_history = ObsTerm(
            func=mdp.joint_pos_delta_history,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES),
                "history_length": JOINT_POS_DELTA_HISTORY_LENGTH,
            },
            noise=Unoise(n_min=-JOINT_POS_DELTA_OBS_NOISE, n_max=JOINT_POS_DELTA_OBS_NOISE),
        )
        # noisy model-derived end-effector reference command [p_ref, v_ref, n_ref, tau] (deployable)
        hit_reference_command = ObsTerm(func=mdp.hit_reference_command)
        # FK blade-center pose (encoder-exact, deployable)
        racket_pos = ObsTerm(func=mdp.racket_pos, params={"racket_body_name": RACKET_BODY_NAME})
        racket_normal = ObsTerm(func=mdp.racket_normal, params={"racket_body_name": RACKET_BODY_NAME})
        hit_ref_pos_error = ObsTerm(func=mdp.hit_ref_pos_error, params={"racket_body_name": RACKET_BODY_NAME})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Actor sees the noisy, deployable joint state (Unoise terms above) so it learns a
            # noise-robust mapping -> less on-hardware jitter. Critic overrides this to False.
            self.enable_corruption = True
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

        def __post_init__(self):
            super().__post_init__()
            # Critic is privileged: keep ALL its inputs CLEAN (the joint-state noise cfgs inherited
            # from ActorCfg are only applied when the group's corruption is enabled). A clean value
            # target is standard for asymmetric actor-critic and avoids a noisier critic.
            self.enable_corruption = False

    policy: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # --- core: model-derived hit-time reference tracking (time-gated Gaussian) ---
    hit_ref_pos = RewTerm(
        func=mdp.hit_ref_pos,
        weight=W_POS,
        params={"racket_body_name": RACKET_BODY_NAME, "sigma_t": SIGMA_T, "sigma_p": SIGMA_P, "pos_tol": POS_TOL},
    )
    hit_ref_vel = RewTerm(
        func=mdp.hit_ref_vel,
        weight=W_VEL,
        params={"racket_body_name": RACKET_BODY_NAME, "sigma_t": SIGMA_T, "sigma_v": SIGMA_V, "vel_tol": VEL_TOL},
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
            "normal_tol_deg": NORMAL_TOL_DEG,
        },
    )
    # still-face at the hit instant: reward low blade-face turn rate |dn/dt|=|omega x n|, gated at
    # tau=0 like the others AND multiplied (inside the kernel) by the velocity-matching Gaussian
    # (sigma_v=SIGMA_V, same v_ref as hit_ref_vel). hit_ref_normal sets WHERE the face points; this
    # stops it TUMBLING as it arrives (RL play tumbles ~189 deg/s vs traditional ~58), forcing speed
    # to be sourced proximally instead of via a wrist snap. The speed coupling is the key fix: without
    # it the raw still-face Gaussian is maximised by NOT MOVING, so run 2026-07-07_10-30-18 collapsed
    # to the degenerate "arrive slow, perfect still face" solution (normal 2deg, verr~1.16, success 0).
    # Coupling zeroes the still-face bonus unless the paddle is also swinging at v_ref -> a steady,
    # proximally-driven sweep becomes the reward-max way to arrive, not a snap or a stop.
    hit_ref_normal_rate = RewTerm(
        func=mdp.hit_ref_normal_rate,
        weight=W_NORMAL_RATE,
        params={
            "racket_body_name": RACKET_BODY_NAME,
            "sigma_t": SIGMA_T_NORMAL,
            "sigma_rate": SIGMA_FACE_RATE,
            "sigma_v": SIGMA_V,
        },
    )

    # --- wind-up approach guidance: wide-gate constant-v_ref velocity (2026-07-09) ---
    # Fills the gradient hole the narrow core gates leave 70-200 ms before the hit so the policy
    # cruises up to v_ref early instead of deferring speed to a late wrist snap. Gate carved to 0 at
    # tau=0 (mdp.approach_gate) -> the sharp hit_ref_vel still solely owns the hit instant. Task-space,
    # demo-free (v_ref is the model reference extended in time, not an imitation target). See the
    # SIGMA_T_APPROACH / SIGMA_V_APPROACH / W_APPROACH_VEL block above for the rationale/knobs.
    hit_ref_approach_vel = RewTerm(
        func=mdp.hit_ref_approach_vel,
        weight=W_APPROACH_VEL,
        params={
            "racket_body_name": RACKET_BODY_NAME,
            "sigma_t_wide": SIGMA_T_APPROACH,
            "sigma_t_core": SIGMA_T,
            "sigma_v": SIGMA_V_APPROACH,
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
    # PD-gain domain randomization (sim-to-real): per-joint scale of the arm's stiffness/damping by
    # U(0.9,1.1) at each reset. Built-in term -> for the implicit A1 actuators it both scales from
    # the nominal default gains (no drift) AND writes the result into PhysX (see PD_GAIN_RAND_RANGE).
    # Scoped to the 7 arm joints via joint_names (other actuators are skipped: empty intersection).
    # NOTE: for implicit actuators this does a CPU sync of the gain tensors on every reset; if it
    # dents throughput, switch mode to "startup" (fixed per-env gains, still 2048 samples across the
    # range) instead of "reset".
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES),
            "stiffness_distribution_params": PD_GAIN_RAND_RANGE,
            "damping_distribution_params": PD_GAIN_RAND_RANGE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.sac_time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.joint_state_nan)
    hit_done = DoneTerm(func=mdp.hit_window_elapsed)


@configclass
class HitTrackEnvCfg(ManagerBasedRLEnvCfg):
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
        self.scene.robot.init_state.pos = (ROBOT_BASE_X, 0.0, 0.0)
        if ROBOT_SIDE < 0:
            self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos["sj"] = READY_LIFT_POS
        self.scene.robot.init_state.joint_pos = joint_pos
        # Curriculum (3): switch the reference source to the baked real serves (lazy-loaded
        # on the first reset onto env.device).
        if HITTRACK_USE_BAKED:
            self.events.reset_reference.params["baked_path"] = HITTRACK_BAKED_PATH


class HitTrackPlayEnvCfg(HitTrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.viewer.eye = (-0.5, -1.4, 1.4)
        self.viewer.lookat = (-1.0, 0.0, 1.0)
        # Deterministic playback / deploy-prep: no observation corruption, nominal PD gains.
        self.observations.policy.enable_corruption = False
        self.events.randomize_gains = None
