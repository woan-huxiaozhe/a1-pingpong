# A1 乒乓球反手 · 环境配置 + 专家模仿 微调设计

- **日期**: 2026-06-04
- **任务**: `A1-TableTennis`(拟新建反手变体 `A1-TableTennis-Backhand`)
- **代码根**: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis`
- **目标**: 把 A1 右臂**反手**接球策略,从"确定性、手搓正手轨迹、固定中路发球"的现状,改造为 **sim-to-real 导向**:发球分布与空气阻力按真机标定、动作/球观测延迟按真机量级、专家轨迹换成真机反手示教。**从头带 DR 训练**(非 resume、非分阶段累积)。
- **本 spec 覆盖范围**: 环境配置 + 模仿/command 两块。**奖励(reward)调整不在本轮**,作为独立下一轮。

---

## 0. 决策摘要(已与用户逐项确认)

| 维度 | 决策 |
|---|---|
| 微调目标 | sim-to-real 鲁棒性,从头带 DR 训练 |
| DR 范围 | 仅:① 发球分布 ② 动作延迟 10±5ms ③ 球观测延迟 10±5ms。**不加** PD/力矩/观测噪声、不加球/球桌物理随机 |
| 发球标定 | 方案 B:**加二次空气阻力** + 用真机网速发球(经验联合采样) |
| 发球生成 | 方案 ②:对真机净速 5D 状态拟合**单个多元高斯** → 采样 → 物理滚动 → 可达过滤 → serve_states.npz |
| 专家轨迹 | 真机反手示教 `pingpong_log_1.csv` swing 1–12,用 `q_plan`,**单条规范**轨迹,FPS=100 |
| 控制/推理频率 | **100Hz**(`decimation=2`,物理 200Hz 不变),motion FPS=100 |
| 机械臂 | A1 右臂,真机 URDF 与 sim USD 关节 1:1、限位一致,无符号翻转 |

---

## 1. 背景:当前框架(简述)

- Isaac Lab `ManagerBasedRLEnv` + rsl_rl PPO。控制 50Hz(`sim.dt=0.005`、`decimation=4`、`step_dt=0.02`),物理 200Hz。
- `ROBOT_SIDE=-1`:机器人在 -X,球从 +X 飞来。`ROBOT_X=-1.5`,己方台 x∈(-1.37,0),对方台 x∈(0,1.37)。
- 动作 = 右臂 7 维残差(`目标=ref_dof+scale·action`)+ 1 维 phase_speed。当前 `residual_scale=0.05`(模仿为主)。
- 专家轨迹 = 手搓正手 whip `forehand_middle_a1_whip.npz`,相位机制对齐击球时机(`hit_phase=0.54`、`ball_arrive_time_est=0.51`)。
- 球:质量 0.0027kg,**`linear_damping=0.0`(无空气阻力)**。
- 发球:`MIDDLE_BALL` 固定中路(x=+0.35、z=1.10、vx=-3.0、vz=+0.2),`relaunch_ball_if_out` 出界重发。

---

## 2. 标定数据与发现(真机 mocap + 反手示教)

### 2.1 球轨迹(`/home/woan/kalman_filter_pingpong/data/0602`,109 条,~120Hz)
约定来自 `dataprocess/evaluate_hit_plane_prediction.py`:击球平面 `x=-1.37`,位置 m,球 +x→-x,速度用局部 deg-2 多项式拟合。

- **坐标对齐**:数据帧球桌面 z≈+0.046m → `sim_z = data_z + 0.714`。
- **三平面实测分布(已转 sim 世界系,p5–p95)**:

| 量 | x=+0.35(出生面) | x=-1.37(击球面) |
|---|---|---|
| \|vx\| | ~4.0 (3.27–4.66) | ~2.5 (1.80–3.25) |
| vy | ±0.5 | ±0.4 |
| vz | -0.09 (-0.96~+1.29) | +0.25 (-1.22~+1.56) |
| \|v\| | 4.08 (3.29–4.74) | 2.66 (2.02–3.40) |
| z(sim) | 1.23 (0.95–1.46) | 1.08 (0.94–1.21) |
| y | ±0.13 | ±0.3 |

- **关键发现**:真机球网→击球面 \|vx\| 从 4.0 衰减到 2.5(<1.7m 掉 ~1.5m/s),来自空气阻力 + 己方台弹跳摩擦。当前 sim 零阻力 → 若直接灌 vx=4.0 会到达过快。**故必须加阻力**:二次阻力 `a = -k·|v|·v`,**k≈0.125 (SI)**(由 kalman `air_drag_coeff≈1.25e-4/mm` 换算,物理估算 0.5·ρ·Cd·A/m≈0.13 吻合)。Magnus/自旋忽略。

### 2.2 反手示教(`/home/woan/robotbase_gripper/tmp/pingpong_logs/2026_06_04/pingpong_log_1.csv`)
- CSV 78 行注释头,真表头在第 79 行。列含 `swing_id, stage, q_plan_0..6, q_actual_0..6, dq_*, tau_*` 等。
- **swing 0** = 72s 待机(tracking↔recovery);**swing 1–12 = 12 次反手击球**,每条 ~0.4s @ ~100Hz,阶段 `to_hit(~16) → cruise(~11) → follow-through(~12)`,cruise=接触窗口。12 条打不同落点(击球姿态 std 达 0.3 rad on yb6/yb7)。
- **机械臂同一性(已验证)**:真机用 A1 右臂,URDF `install/armcontrol/share/armcontrol/urdf/A1/a1_r.urdf`,`joint1..7-a1_r` 限位与 sim USD **完全一致** → 同一臂、关节 1:1(`q_*_{i} ↔ joint_yb_{i+1}`)、**无符号翻转**。`q_plan` 严格守限位,`q_actual` 略过冲且 J5 有振荡 → **用 q_plan**。

---

## 3. 设计 A:环境配置

### A0. 控制/推理频率 → 100Hz
- `env_cfg.__post_init__`:`decimation = 2`(物理 `sim.dt=0.005` / 200Hz **不变**,保高速小球碰撞精度),→ `step_dt = 0.01s`、控制/推理 **100Hz**。
- `episode_length_s=10` 不变(对应步数 500→1000)。
- **PPO 配套(连带影响,细调留训练/奖励轮,此处标注)**:控制率翻倍 → 固定 `gamma=0.99` 有效时域 2s→1s、`num_steps_per_env=24` rollout 时长 0.48s→0.24s。建议 `gamma 0.99→0.995`、`num_steps_per_env 24→48` 以保持时域行为(训练轮确认)。
- **真机一致**:策略真机推理也 100Hz,与示教/servo 采样率(~100Hz)对齐。

### A1. 二次空气阻力(新增)
- 在 `mdp/events.py` 新增 `apply_air_drag(env, env_ids, ball_cfg, k=0.125)`:每控制步(interval `(0.01,0.01)`,100Hz)按当前球速给球施加外力 `F = m·a = -m·k·|v|·v`(用 `ball.set_external_force_and_torque` + `write_data_to_sim`)。
- `k` 作为可配置常量(本轮固定 0.125,不做 DR;预留日后 ±10%)。
- 在 `EventCfg` 注册该 interval 事件。

### A2. 发球分布:离线生成管线 + serve_states.npz
**离线脚本**(建议放 `robots/a1/backhand/create_serve_states.py`,可调用 kalman 仓库数据):
1. 读 0602 数据,在 x=+0.35 提取 5D 净速状态 `(y, z_sim, vx, vy, vz)`;用 x=-1.37 处可达性预筛(见 A2 可达框)得到 ~83 条有效真实状态。
2. 拟合**单个多元高斯** `N(μ,Σ)`(全协方差):
   - `μ = [y 0.00, z 1.22, vx -4.03, vy 0.02, vz -0.08]`
   - `σ = [0.09, 0.145, 0.484, 0.307, 0.746]`(协方差用样本 Σ,保留相关:vx-vz +0.43、z-vz -0.44、y-vy -0.40)
3. 采样 N(如 5000),**每维裁剪到实测 min/max**(如 vx∈[-4.91,-1.79]),防止高斯尾巴生成不可能值。
4. 用标定物理(阻力 k=0.125 + 球桌弹跳 Ch≈0.85/Cv≈0.90)前向滚动每条候选,在 **x=-1.37** 检查可达框,保留有效。
5. 写 `serve_states.npz`(字段:`states (M,6)` = `x,y,z,vx,vy,vz`,sim 世界系;x 固定 +0.35,其余采样)。

**可达框**(右臂反手,机器人右侧更大):在击球面 x=-1.37,`y ∈ [-0.20, +0.30]`、桌面上方高度 `h ∈ [0, 0.70]`(sim z∈[0.76,1.46])。真机命中率 ~83%(86/104),横向 y 是主约束,高度框是保险丝。

**env 接入**:改 `mdp/events.py` 的 `launch_ball`,支持从 serve_states 表随机抽一行(替换逐维均匀采样);`reset_ball` 与 `relaunch_ball_if_out` 都用该表。保留 env_origins 偏移与 phase 重对齐逻辑。

### A3. 动作延迟(改造为子步粒度)
- 现状:`ReferenceResidualJointAction` 延迟缓冲按控制步整数,无法表示 10±5ms。
- 改:延迟下沉到 **5ms 物理子步**粒度(`sim.dt=0.005`,**与 A0 的 decimation 改动无关**;decimation=2 时一个控制步=2 子步)。每 env 在 reset 时采样延迟 `uniform{1,2,3} 子步 = {5,10,15}ms`(均值 10ms)。`apply_actions` 每子步执行,维护一个子步级动作 ring buffer,取延迟后的目标。
- 配置:`ActionsCfg.right_arm` 增加 `action_delay_substeps_min/max`(本轮 1/3),移除/替换原 `action_delay_steps_*`。

### A4. 球观测延迟(改造:仅球派生项 + 子步粒度)
- 现状:`DelayedObsEnv` 延迟**整个 policy 观测**、20ms 粒度。
- 改:只延迟**所有球派生观测**:`ball_pos_relative, ball_vel_relative, ball_spin, ball_time_to_arrive, ball_predicted_hit, ball_bounce`;关节/相位/racket/last_action **不延迟**。
- 粒度 5ms 子步,延迟 `uniform{1,2,3} 子步 = {5,10,15}ms`(均值 10ms)。实现:env 在每子步缓存球状态到 ring buffer;构建上述球观测项时,读取延迟后子步的球状态再计算派生量(派生量随延迟球态一起算,保证视觉链路一致)。
- 替换现有 `DelayedObsEnv` 的整体延迟逻辑。

### A5. 配套依赖(必做)
- **相位重标定**:见 B3(`hit_phase` 来自 demo cruise,`ball_arrive_time_est` 在新发球+阻力下 sim 实测)。
- **球桌弹跳验证**:确认 sim 球桌 restitution/friction 使 Ch≈0.85、Cv≈0.90;加阻力+发球后跑 sim 测 x=-1.37 到达速度,确认落在 2.0–3.4,否则微调 k 或球桌材质。
- **interval 事件同步 100Hz**:`track_hit`、`relaunch_ball`(及 A1 的 `apply_air_drag`)的 `interval_range_s` 从 `(0.02,0.02)` 改为 `(0.01,0.01)`,保持每控制步触发。

### A6. 不动
- 不加 PD/力矩/观测噪声(`enable_corruption=False`)、不加球质量/球桌物理随机;`num_envs`、`episode_length_s`、`sim.dt`(物理 200Hz)、终止条件保持。
- **注意**:`decimation`(4→2)、`gamma`/`num_steps_per_env`(见 A0)**会改**,不在"不动"之列。

---

## 4. 设计 B:模仿 / command(反手)

### B1. 反手专家轨迹(单条规范)
- 源:`pingpong_log_1.csv` swing 1–12,取 `q_plan_0..6`。
- 构建脚本(建议 `robots/a1/backhand/create_backhand_ref.py`):每条 swing 按阶段时间归一化对齐 → 统一 **FPS=50** 重采样 → **相位对齐平均**成一条规范挥拍段(to_hit→cruise→follow-through)(若平均后挥拍被"抹平",回退到取 medoid swing;以 sim 内目视为准)。
- **补成完整周期**(见 B5):规范挥拍段前接 ready-hold、后接 return-to-ready hold,使总时长 ≈ 球周期(由 B3 实测 `ball_arrive_time_est` 联合确定),避免相位多绕圈。
- 输出 npz:`fps=50 / upper_body_dof (N,7) / base_y=0(fixed base) / joint_names=[joint_yb_1..7]`。
- 替换 `CommandsCfg.motion_files` 中的正手 whip。

### B2. 关节映射与验证
- 直接映射 `q_plan_{i} → joint_yb_{i+1}`,无翻转、无偏移。
- **验证(实现阶段必做)**:把规范轨迹 frame-0 姿态加载进 sim,目视/数值确认与真机该帧姿态一致(确证 USD 与 URDF 轴向约定相同);若个别关节方向不符,再按需在 loader `axis_flip_indices` 补翻转。

### B3. 相位接地
- `ball_arrive_time_est` ← 用新发球(A2)+阻力(A1)在 sim 测"球出生→到达 x=-1.37"的时间分布,取中位数回填;`relaunch_ball_if_out` 的相位重对齐用同值。
- **规范轨迹总时长**与 `ball_arrive_time_est` 联合确定(见 B5),保证挥拍段 cruise 能在球到达时刻自然对齐、相位不绕多圈。
- `hit_phase` ← `cruise` 段(接触点)在**补全后完整周期轨迹**中的归一化位置(不是裸 0.4s 挥拍段的 0.40;须在 padding 完成后重新计算),并由 phase-aligned init `initial_phase = hit_phase - arrive_time/duration` 验证落在 [0,1) 无 wrap。

### B4. 模仿权重
- 先保持模仿为主:`residual_scale≈0.05`、`pose_tracking`/`vel_tracking` 权重不变、`phase_speed∈[0.85,1.15]`。是否放开 RL 留到奖励轮。

### B5. 完整周期构建 + PREP/ready 衔接
- demo swing 只有 ~0.4s 的 `to_hit→cruise→follow-through`,**短于球飞行时间(~0.5s)**,直接当 motion 会令相位机制多绕一圈(不物理)。
- 故规范轨迹构建为**完整周期**:`[ready-hold] → [to_hit→cruise→follow-through(来自 demo)] → [return→ready-hold]`。
  - ready/PREP 位 = follow-through 末姿态或 recovery 姿态(平滑可回到的待机位)。
  - 总时长 `duration` 与 `ball_arrive_time_est` 联合选取,使挥拍段在球到达时被 phase 推进到 cruise(接触);`hit_phase` = cruise 在该完整周期里的归一化位置。
- 接回 command 的"打完(`swing_done`)冻结、等下一球"循环:打完停在末尾 ready-hold,等 relaunch 后重置相位。

### B6. 结构
- 新建 `robots/a1/backhand/`(`__init__.py` 注册 `A1-TableTennis-Backhand`、`env_cfg.py`、`create_backhand_ref.py`、`create_serve_states.py`、`backhand_ref.npz`、`serve_states.npz`),复制 forehand 结构改造,**保留 forehand 不动**。

---

## 5. 验证计划(实现后逐项过)
1. **发球管线**:serve_states.npz 中状态滚动后 100% 落在可达框;到达 x=-1.37 速度分布与实测 2.0–3.4 吻合(直方图比对)。
2. **空气阻力**:单球零控制 rollout,网→击球面 \|vx\| 衰减 ≈实测 1.5m/s。
3. **延迟**:动作/球观测延迟分布为 {5,10,15}ms;非球观测无延迟(单测断言)。
4. **专家轨迹**:sim 加载规范反手,纯参考回放姿态与真机一致;相位推进、打完冻结正常。
5. **相位接地**:`ball_arrive_time_est` 实测回填后,球到达时 racket 处于 cruise 段。
6. **端到端**:`A1-TableTennis-Backhand` 能跑起来、reset/relaunch 正常、无 NaN、若干 iter 后 reward 有上升趋势(冒烟)。

---

## 6. 文件改动清单(预计)
- `robots/a1/backhand/env_cfg.py`:`__post_init__` 设 `decimation=2`(100Hz);`track_hit`/`relaunch_ball`/`apply_air_drag` 的 `interval_range_s=(0.01,0.01)`。
- 新增 `mdp/events.py::apply_air_drag`;改 `launch_ball`/`relaunch_ball_if_out` 支持 serve_states 表采样。
- 改 `mdp/actions.py::ReferenceResidualJointAction`(子步动作延迟)+ `ReferenceResidualJointActionCfg`。
- 改 `delayed_env.py`(球派生项、子步球观测延迟)。
- 新增 `robots/a1/backhand/`:`__init__.py`、`env_cfg.py`、`create_backhand_ref.py`、`create_serve_states.py`、`backhand_ref.npz`、`serve_states.npz`。
- `agents/rsl_rl_ppo_cfg.py`:新建 backhand runner cfg,`gamma 0.99→0.995`、`num_steps_per_env 24→48`(100Hz 配套,训练轮确认)。

---

## 7. 后续轮次(本 spec 之外)
- **奖励段**:现有 16 项 reward 为正手设计;反手 + 新发球下需复核 `TARGET_X / OPTIMAL_VX / 击球阈值 / 落点框 / 各权重`。
- **多轨迹**:日后若能给 12 条 swing 标注落点,可升级 `match_ball_direction` 左/中/右多轨迹。
- **更多 DR**:PD/力矩/观测噪声、球桌物理(需先修 PhysX API)。
