"""A1 table tennis BACKHAND environment (plan A1-TableTennis-Backhand).

sim-to-real 导向反手变体, 从头带 DR 训练. 复用 forehand 的 Scene / Observations /
Rewards / Terminations (reward 复核留下一轮, 见 spec §7), 仅改:
  A0  控制/推理 100Hz (decimation=2, 物理 200Hz 不变)
  A1  二次空气阻力 (apply_air_drag interval, k=0.125)
  A2  发球分布: serve_states.npz 表采样 (reset + relaunch)
  A3  动作延迟: 子步粒度 {5,10,15}ms
  A4  球观测延迟: 仅球派生项, 子步粒度 {5,10,15}ms (BallObsDelayAction)
  B   专家轨迹换成真机反手规范挥拍 backhand_ref.npz

DR 范围仅以上三项 (发球 + 动作延迟 + 球观测延迟); 不加 PD/力矩/观测噪声、不加
球/球桌物理随机 (plan §0/A6).
"""

from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.table_tennis.mdp as mdp

# 复用 forehand 的场景/观测/奖励/终止与常量 (保留 forehand 不动, 单一来源)
from ..forehand.env_cfg import (
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    X1TableTennisSceneCfg,
    RIGHT_ARM_JOINT_NAMES,
    RACKET_BODY_NAME,
    ROBOT_X,
    ROBOT_SIDE,
    ROBOT_BASE_X,
)

BACKHAND_DATA_DIR = os.path.dirname(__file__)
SERVE_STATES_PATH = os.path.join(BACKHAND_DATA_DIR, "serve_states.npz")
BACKHAND_REF = os.path.join(BACKHAND_DATA_DIR, "backhand_ref.npz")  # = middle (re-centered base)

# 多参考 (forehand-style per-serve matching, 2026-06-08): 单条静态参考够不到 ±0.2m 发球展开,
# 拍永远偏球 ~0.13m, 接触不在模仿流形上 -> 三次 reward reshape 全 0 回球. 改用 3 条横向变体
# (create_backhand_variants.py 按 joint_yb_2 偏置生成), match_ball_direction 按预测球 y 选桶:
# 0=middle, 1=left(+y), 2=right(-y). 顺序即 motion_id, 不可乱.
BACKHAND_REF_MIDDLE = os.path.join(BACKHAND_DATA_DIR, "backhand_ref_middle.npz")
BACKHAND_REF_LEFT = os.path.join(BACKHAND_DATA_DIR, "backhand_ref_left.npz")
BACKHAND_REF_RIGHT = os.path.join(BACKHAND_DATA_DIR, "backhand_ref_right.npz")

# provisional, 由 create_serve_states.py / create_backhand_ref.py 离线算出 (sim 内再确认见 §A5/B3)
HIT_PHASE = 0.4643            # cruise 中心在补全周期里的归一化位置
BALL_ARRIVE_TIME_EST = 0.526  # 真机 x=0.35 -> x=-1.37 用时中位数
AIR_DRAG_K = 0.125            # 二次阻力系数 (SI)

# 升降柱高度 (plan B2 配套, sim 内标定 2026-06-05): 真机反手 paddle 工作面 ~桌面上方 40cm.
# forehand 默认 joint_lift=-0.28 -> sim paddle cruise 仅 ~桌上 18cm, 够不到到达 ~桌上 32-45cm
# 的反手球. lift 与 paddle 高度 1:1: paddle_above_table = (lift+0.28)+0.18, 取 40cm -> lift=-0.06.
JOINT_LIFT = -0.18


@configclass
class BackhandStage1RewardsCfg(RewardsCfg):
    """Stage 1 (模仿热身) 权重: pose/vel 主导, 击球/回球项压低.

    行为已被 residual_scale=0.05 锁在参考挥拍附近; 这里让 reward 信号也以模仿为主.
    待 pose_tracking 饱和 (TensorBoard Episode_Reward/pose_tracking 进入平台期) 后切 Stage 2:
    把下列权重改成各注释里的 "S2 ->" 值, 并同步把 ActionsCfg.right_arm.residual_scale
    0.05 -> 0.3~0.4 (放开动作权限才能去接不同方向/速度的来球, 见 spec §7).
    """

    def __post_init__(self):
        # 模仿锚 (主导)
        self.pose_tracking.weight = 1.5           # S2 -> 0.3
        self.vel_tracking.weight = 0.5            # S2 -> 0.2
        # 引导 (适度: 让 policy 先用 phase_speed 把接触时机对齐到来球)
        self.racket_ball_proximity.weight = 0.4   # S2 -> 0.6
        self.swing_timing.weight = 0.5            # S2 -> 1.5
        self.racket_face_target.weight = 0.3      # S2 -> 1.5
        self.racket_swing_ideal.weight = 0.3      # S2 -> 1.5
        # 击球质量 (压低: residual 0.05 下基本打不动, 避免拿不到的大奖变成噪声)
        self.ball_hit.weight = 0.3                # S2 -> 0.5
        self.ball_hit_speed.weight = 0.3          # S2 -> 1.5
        self.ball_hit_direction.weight = 0.2      # S2 -> 0.5
        # 回球/落点 (压低)
        self.ball_return.weight = 0.5             # S2 -> 2.0
        self.ball_land_opponent.weight = 0.5      # S2 -> 2.0
        self.ball_land_placement.weight = 0.5     # S2 -> 2.0
        # ball_land_own_table(-0.5) 与所有正则项 (action_rate / joint_acc / joint_limit /
        # phase_speed_reg / self_collision) 保持不变.


@configclass
class BackhandStage2RewardsCfg(RewardsCfg):
    """Stage 2 (探索/回球) 权重: 放开 residual (0.05->0.3) 后, 模仿项退居锚点,
    击球/回球/落点项成为主梯度. 由 Stage 1 checkpoint 热启动 (--resume).

    切入判据 (已满足): Stage 1 pose_tracking 在 ~iter 150-300 进入平台期, 此后 mean_reward
    持平而 action_rate 反向劣化 (探索噪声空转), 任务项 (ball_hit/return) 全程为 0 ——
    residual=0.05 下球拍够不到球, 79% 回合漏球. 放开权限让任务项产生真实梯度.

    正则项 (action_rate=-0.15 / joint_acc / joint_limit / phase_speed_reg / self_collision)
    与 ball_land_own_table(-0.5) 保持不变: Stage 2 先单独观察 v/a/j, 若平滑度再次劣化
    再决定提 action_rate 权重 / 加 EMA LPF (LPF 需 train+deploy 同步, 不在此处引入).
    """

    def __post_init__(self):
        # 模仿锚 (Stage 2a 修订: 0.3 太弱 -> policy 抛弃挥拍只在球旁悬停刷 proximity, ball_hit≈0.
        # 抬回 0.8 保住挥拍 through 形状, residual 0.3 仍可调整去接偏置来球; 待接触率上来再退火)
        self.pose_tracking.weight = 0.8           # S1: 1.5 ; 初版 S2: 0.3 (太弱)
        self.vel_tracking.weight = 0.4            # S1: 0.5 ; 初版 S2: 0.2 (帮助复现挥拍时机)
        # 引导 (主梯度: 对齐接触时机/拍面/挥拍)
        # hover 修复 (R1, 2026-06-08): proximity 是失败 run 的最大正奖 (0.102, 无需接触),
        # 球拍停在球旁刷分而不挥拍 (81% 漏球, 击球链≈0). 降权 0.6->0.10 让 pose 锚点 (0.8) 重新
        # 主导, 把已验证时机正确的参考挥拍执行 through 球; 并 gate_pre_contact: 击球后 proximity
        # 清零, 不再因球飞走 dist↑ 惩罚 follow-through.
        self.racket_ball_proximity.weight = 0.10  # S1: 0.4 ; 初版 S2: 0.6 (hover 主因)
        self.racket_ball_proximity.params["gate_pre_contact"] = True
        self.racket_ball_proximity.params["command_name"] = "motion"
        self.swing_timing.weight = 1.5            # S1: 0.5
        self.racket_face_target.weight = 1.5      # S1: 0.3
        self.racket_swing_ideal.weight = 1.5      # S1: 0.3
        # 击球质量 (放开: residual 0.3 下能打动球)
        self.ball_hit.weight = 0.5                # S1: 0.3
        self.ball_hit_speed.weight = 1.5          # S1: 0.3
        self.ball_hit_direction.weight = 0.5      # S1: 0.2
        # 回球/落点 (主目标)
        self.ball_return.weight = 2.0             # S1: 0.5
        self.ball_land_opponent.weight = 2.0      # S1: 0.5
        self.ball_land_placement.weight = 2.0     # S1: 0.5

        # 接触门修复 (backhand-only, 2026-06-08): 把瞬时力门 net_forces[:,0] 换成 history 最大力,
        # 阈值 0.15->0.25, 对齐 track_ball_hit. 诊断: track_ball_hit(max力,0.25) 检出 ~25% 接住球
        # (miss 从 0.79 降到 0.57), 但 ball_hit/speed/direction(瞬时力,0.15) 全程≈0 —— 乒乓接触仅
        # 1-2 substep, 瞬时帧漏判 -> 无接触梯度 -> policy 退回纯模仿, miss 反弹回 0.75.
        # forehand 不传 use_max_force (默认 False) 行为不变.
        for _term in (self.ball_hit, self.ball_hit_speed, self.ball_hit_direction):
            _term.params["use_max_force"] = True
            _term.params["proximity_threshold"] = 0.25

        # 前瞻拦截梯度 (F2, 2026-06-08): 加 racket_at_predicted_hit.
        # R1 诊断: 降权 proximity (0.60->0.10 + gate) 成功破除 hover (TB proximity 0.102->0.017),
        # miss 一度降到 0.66, 但全程缺少"把拍移到球将到达点"的稠密位置梯度 —— policy 改去刷
        # racket_face_target (拍面朝向, 非位置, 0.055->0.079) 而放弃挥拍 (swing_timing≈0), miss
        # 反弹回 0.84, ball_return 全程 0. 本项用弹道预测球过 x=ROBOT_X 的落点 (pred_y/pred_z),
        # urgency 随到达临近 0->1 加权, 把拍拉到拦截点, 补上缺失的位置梯度. z_offset=0.0 (反手
        # 中心击球, 非 forehand 的下沉挥); robot_side=ROBOT_SIDE(-1) 使入球 (vx<0) 通过有效性判据.
        # 仅加在反手 (forehand RewardsCfg 不含此项, 行为不变).
        self.racket_at_predicted_hit = RewTerm(
            func=mdp.racket_at_predicted_hit,
            weight=1.0,
            params={
                "ball_name": "ball",
                "racket_body_name": RACKET_BODY_NAME,
                "robot_x": ROBOT_X,
                "robot_side": ROBOT_SIDE,
                "z_offset": 0.0,
                "sigma": 20.0,
                "urgency_window": 0.5,
            },
        )


@configclass
class BackhandStage3RewardsCfg(RewardsCfg):
    """Stage 3 (多参考 / per-serve matching) 权重: 配 3 条横向匹配参考 + residual 0.05.

    设计依据 (2026-06-08): 三次 reward-weight reshape (gate-fix / R1 降 hover / F2 加拦截梯度)
    全部 0 回球, 因单条参考够不到球 (~0.13m), 接触不在模仿流形上, 任何稠密奖励都能"摆姿势不接触"
    地刷满, 稀疏接触奖励无法 bootstrap. 改用 forehand 同款 3 条 per-serve 匹配参考后, 每个发球都有
    一条参考穿过来球 -> 接触自然发生在模仿流形上, residual 退回 0.05 即够. 故本配置:
      - pose/vel 主导 (骑住匹配到的参考挥拍, 接触随模仿发生);
      - 接触/回球项放到主目标权重, 让 RL 把"碰到"细化成"打回对面";
      - 保留 backhand 接触门修复 (use_max_force + 0.25 阈值), 否则乒乓 1-2 substep 接触瞬时帧漏判;
      - 不带 R1/F2 的 hack (proximity 降权+gate / racket_at_predicted_hit) —— 那是 residual 0.3
        悬停问题的补丁, residual 0.05 下不存在该问题.
    """

    def __post_init__(self):
        # 模仿锚 (主导: 骑住匹配参考)
        self.pose_tracking.weight = 1.5
        self.vel_tracking.weight = 0.5
        # 引导 (适度: 对齐接触时机/拍面/挥拍; residual 0.05 下无悬停风险, proximity 不 gate)
        self.racket_ball_proximity.weight = 0.4
        self.swing_timing.weight = 1.0
        self.racket_face_target.weight = 1.0
        self.racket_swing_ideal.weight = 1.0
        # 击球质量 (接触现在自然发生 -> 这些项拿到真实梯度)
        self.ball_hit.weight = 0.5
        self.ball_hit_speed.weight = 1.5
        self.ball_hit_direction.weight = 0.5
        # 回球/落点 (主目标)
        self.ball_return.weight = 2.0
        self.ball_land_opponent.weight = 2.0
        self.ball_land_placement.weight = 2.0
        # 接触门修复 (同 Stage 2): 瞬时力门 -> history 最大力, 阈值 0.15->0.25, 对齐 track_ball_hit.
        for _term in (self.ball_hit, self.ball_hit_speed, self.ball_hit_direction):
            _term.params["use_max_force"] = True
            _term.params["proximity_threshold"] = 0.25


@configclass
class ActionsCfg:
    """右臂残差 (子步动作延迟) + 相位速度 + 球观测子步延迟 (0 维)."""
    right_arm = mdp.ReferenceResidualJointActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        command_name="motion",
        residual_scale=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],  # 0.05->0.03: pure-ref 证实参考已对中 (middle min_gap 0.069), 残差只在 wander; 收 multi-joint FK 漂移授权 (0.05 仍漂 +0.08 dy)
        action_delay_substeps_min=8,   # {8..18} 子步 = 40-90ms (均值 65ms); 0716 实测真机 cmd->response
        action_delay_substeps_max=18,  # transport ~63ms (chirp xcorr, 频率无关 dead-time, corr0.994), 覆盖 deploy 40-110ms; 旧 1-3(5-15ms)欠延迟~4x
    )
    phase_speed = mdp.PhaseSpeedActionCfg(
        asset_name="robot",
        command_name="motion",
        speed_min=0.98,   # 0.95->0.98: 探针 trained closest-approach 仍偏早 (phase 0.459 vs ref-natural 0.479); 收窄相位授权把接触推回 0.479 (pure-ref middle min_gap 0.069)
        speed_max=1.02,   # 1.05->1.02
    )
    ball_obs_delay = mdp.BallObsDelayActionCfg(
        asset_name="robot",
        ball_name="ball",
        ball_delay_substeps_min=1,     # {1,2,3} 子步 = {5,10,15}ms
        ball_delay_substeps_max=3,
    )


@configclass
class CommandsCfg:
    motion = mdp.UpperBodyMotionCommandCfg(
        asset_name="robot",
        motion_files=[                 # 顺序 = motion_id: 0=middle, 1=left(+y), 2=right(-y)
            BACKHAND_REF_MIDDLE,
            BACKHAND_REF_LEFT,
            BACKHAND_REF_RIGHT,
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        base_y_noise_range=(0.0, 0.0),
        fixed_base=True,
        hit_phase=HIT_PHASE,
        hit_phase_noise=0.0,
        ball_arrive_time_est=BALL_ARRIVE_TIME_EST,
        ball_arrive_time_noise=0.0,
        match_ball_direction=True,     # 3 条横向参考, 按预测球 y 选桶 (需 num_motions>=3)
        ball_y_threshold=0.05,         # |pred_y|<=0.05 -> middle; >+0.05 left; <-0.05 right
        robot_x=ROBOT_X,
        robot_side=ROBOT_SIDE,
        axis_flip_indices=None,        # q_plan_i -> joint_yb_{i+1} 无翻转 (plan B2)
    )


@configclass
class EventCfg:
    # 发球: 从标定 serve_states 表采样 (替换逐维均匀)
    reset_ball = EventTerm(
        func=mdp.launch_ball,
        mode="reset",
        params={"ball_cfg": SceneEntityCfg("ball"), "serve_states_path": SERVE_STATES_PATH},
    )
    # 出界重发 + 空气阻力 + 击球追踪: 均 100Hz (每控制步触发)
    relaunch_ball = EventTerm(
        func=mdp.relaunch_ball_if_out,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={"ball_cfg": SceneEntityCfg("ball"), "serve_states_path": SERVE_STATES_PATH},
    )
    apply_air_drag = EventTerm(
        func=mdp.apply_air_drag,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={"ball_cfg": SceneEntityCfg("ball"), "k": AIR_DRAG_K},
    )
    track_hit = EventTerm(
        func=mdp.track_ball_hit,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "command_name": "motion",
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: X1TableTennisSceneCfg = X1TableTennisSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: BackhandStage3RewardsCfg = BackhandStage3RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 球观测延迟改由 BallObsDelayAction 子步处理; 整体 policy obs 不延迟
    obs_delay_min: int = 0
    obs_delay_max: int = 0

    def __post_init__(self):
        self.decimation = 2  # A0: 100Hz 控制 (物理 sim.dt=0.005 / 200Hz 不变)
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.enable_ccd = True
        self.scene.robot.init_state.pos = (ROBOT_BASE_X, 0.0, 0.0)
        if ROBOT_SIDE < 0:
            self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # 抬高升降柱: 反手 paddle 工作面对齐真机 ~桌面上方 40cm (见 JOINT_LIFT 注释).
        # copy dict 防止改到共享的 A1_TABLE_TENNIS_CFG (forehand 不受影响).
        jp = dict(self.scene.robot.init_state.joint_pos)
        jp["joint_lift"] = JOINT_LIFT
        self.scene.robot.init_state.joint_pos = jp
        # 反手 A1 臂对齐真机 MIT 锚点: r1-3 kp300/kd3.5, r4-7 kp120/kd1.0 (chirp sysid 的采集增益;
        # 2026-07-15 用户决定 sim->real 对齐, 取代 a1.py 为 HitTrack swing 调低的 kp200/90 & kd3.0/0.5).
        # 深拷贝 actuator + 新 dict, 防止污染共享的 A1_TABLE_TENNIS_CFG (forehand/hittrack 保留各自增益).
        import copy
        _acts = dict(self.scene.robot.actuators)
        _arm = copy.deepcopy(_acts["right_arm"])
        _arm.stiffness = {"r1": 300.0, "r2": 300.0, "r3": 300.0, "r4": 120.0, "r5": 120.0, "r6": 120.0, "r7": 120.0}
        _arm.damping = {"r1": 3.5, "r2": 3.5, "r3": 3.5, "r4": 1.0, "r5": 1.0, "r6": 1.0, "r7": 1.0}
        _acts["right_arm"] = _arm
        self.scene.robot.actuators = _acts
        # A6: 不加观测噪声 / PD / 力矩 / 球桌物理随机 (DR 仅发球 + 动作延迟 + 球观测延迟)
        self.observations.policy.enable_corruption = False


class RobotPlayEnvCfg(RobotEnvCfg):

    zero_ball_spin: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.viewer.eye = (-2.5, -2.0, 1.4)
        self.viewer.lookat = (-1.0, 0.0, 1.0)
        # 回放确定性: 关动作/球观测延迟
        self.commands.motion.hit_phase_noise = 0.0
        self.actions.right_arm.action_delay_substeps_max = 0
        self.actions.ball_obs_delay = None  # action_manager 跳过 None term -> 球观测无延迟

        if self.zero_ball_spin:
            self.observations.policy.ball_spin = ObsTerm(
                func=mdp.ball_spin_zero, params={"ball_name": "ball"},
            )
            self.observations.critic.ball_spin = ObsTerm(
                func=mdp.ball_spin_zero, params={"ball_name": "ball"},
            )
