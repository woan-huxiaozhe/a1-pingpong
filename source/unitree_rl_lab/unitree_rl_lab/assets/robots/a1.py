"""A1 arm robot configuration for table tennis (based on x1 chassis + A1 arms)."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

A1_USD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "data", "robots", "a1", "a1.usd"
)

A1_ARM_STIFFNESS = {
    "r1": 200.0, "r2": 200.0, "r3": 200.0,
    "r4": 90.0, "r5": 90.0, "r6": 90.0, "r7": 90.0,
}
A1_ARM_DAMPING = {
    # J1-3 5.0->3.5 (2026-07-04): the joint-delta action interface caps EE swing speed at ~K*scale/D;
    # at the hit the proximal joints sat at 8-16% velocity / 33-41% torque util (model_10000) despite a
    # saturated action, so lowering damping raises the velocity ceiling (and lowers holding torque)
    # without touching the 28 N*m effort headroom. Conservative 30% cut; the real arm's D must match.
    # J1-3 3.5->3.0 (2026-07-08): model_13000 (run A) hit-instant telemetry showed the proximal swing is
    # a SHORT torque-limited ramp (J1-2 peak 83-84% of 28 N*m) whose |ee_vx| peak lands ~47 ms AFTER the
    # hit -- the arm is still accelerating at tau=0 (reaches ~83% of v_ref, peak ~94%). Actions clamp only
    # 4-11% and joint vel is 30-44% of limit, so the interface/vel-cap is NOT the binding constraint --
    # a steeper accel transient is. Dropping D 3.5->3.0 raises K*scale/D (200*0.20/3.0=13.3 rad/s, still
    # above the 8 rad/s HW cap so it uncaps within HW) and makes the ramp reach v_ref earlier in the
    # window. ~14% step (smaller than the last 30% cut); the real arm's D must track this.
    "r1": 3.0, "r2": 3.0, "r3": 3.0,
    # J4-7 kept small (0.5); the 2026-07-04 J4 big-motor preview (K/effort/D/vel -> J1-3 class) was
    # reverted: doubling J4 torque did NOT decouple normal error from swing speed (it diverged earlier),
    # so J4 torque is not the binding cause -- the lever is the reward balance, not the wrist motor.
    "r4": 0.5, "r5": 0.5, "r6": 0.5, "r7": 0.5,
}
A1_ARM_EFFORT = {
    "r1": 28.0, "r2": 28.0, "r3": 28.0,
    "r4": 8.0, "r5": 8.0, "r6": 8.0, "r7": 8.0,
}
A1_ARM_VELOCITY = {
    "r1": 8.0, "r2": 8.0, "r3": 8.0,
    "r4": 20.0, "r5": 20.0, "r6": 20.0, "r7": 20.0,
}
A1_LIFT_EFFORT = 1000.0
A1_LIFT_STIFFNESS = 5000.0
A1_LIFT_DAMPING = 500.0

A1_TABLE_TENNIS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=A1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.7, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "sj": -0.28,           # vertical spine (was joint_lift); verify range/height in sim
            "r1": 1.769,   # whip_high3 ready pose (orig 1.56)
            "r2": -0.762,  # whip_high3 ready pose (orig -0.12)
            "r3": -1.863,  # whip_high3 ready pose (orig -1.70)
            "r4": 1.445,   # whip_high3 ready pose (orig 1.50)
            "r5": 0.206,   # whip_high3 ready pose (orig 2.03)
            "r6": -0.827,  # whip_high3 ready pose (orig 0.00)
            "r7": 1.043,   # whip_high3 ready pose (orig -0.39)
            "l1": 0.0, "l2": 0.0, "l3": 0.0, "l4": 0.0, "l5": 0.0, "l6": 0.0, "l7": 0.0,
            "t01": 0.0,
            "t02": 0.0,
            "lun_l": 0.0,
            "lun_r": 0.0,
            "wxl_1_1": 0.0, "wxl_1_2": 0.0, "wxl_2_1": 0.0, "wxl_2_2": 0.0,
            "wxl_3_1": 0.0, "wxl_3_2": 0.0, "wxl_4_1": 0.0, "wxl_4_2": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["r[1-7]"],
            effort_limit_sim=A1_ARM_EFFORT,
            velocity_limit_sim=A1_ARM_VELOCITY,
            stiffness=A1_ARM_STIFFNESS,
            damping=A1_ARM_DAMPING,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["l[1-7]"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.1,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "spine": ImplicitActuatorCfg(  # X1 prismatic lift (was joint_lift)
            joint_names_expr=["sj"],
            effort_limit_sim=A1_LIFT_EFFORT,
            velocity_limit_sim=0.0,
            stiffness=A1_LIFT_STIFFNESS,
            damping=A1_LIFT_DAMPING,
        ),
        "torso": ImplicitActuatorCfg(  # X1 torso pan/tilt (was head)
            joint_names_expr=["t0[12]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=0.1,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "base": ImplicitActuatorCfg(  # X1 drive wheels + casters, frozen (was wheels)
            joint_names_expr=["lun_.*", "wxl_.*"],
            effort_limit_sim=10.0,
            velocity_limit_sim=0.0,
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
)
