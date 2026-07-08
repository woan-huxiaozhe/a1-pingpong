# HitTrack 真机部署推理节点 — 设计文档

- 日期: 2026-07-08
- 分支: `hitter`
- 状态: 待用户复核
- 相关: [2026-06-26-hittrack-design.md](2026-06-26-hittrack-design.md)、[2026-07-06-forehand-hittrack-design.md](2026-07-06-forehand-hittrack-design.md)

## 1. 背景与问题

`A1-Pingpong-HitTrack`（`source/unitree_rl_lab/unitree_rl_lab/tasks/a1_pingpong_hittrack/`）是非端到端的解析参考跟踪任务：球不在 MDP 里，策略只负责让末端跟上运行时算出的击球参考 `(p_ref, v_ref, n_ref, tau)`。训练/仿真侧的观测与动作空间已经调研清楚（见对话记录，未单独成文，本文档直接给出结论）。

现在需要为这套策略写一个**真机部署推理节点**（Python），对接两套已存在的下位机 ROS2 接口：

1. **机械臂底层控制**：`/home/woan/robotbase_gripper/robotbase/src/armcontrol/src/inference_arm_control_node.cpp`（C++，`inference_arm_control_node`）。
2. **球体追踪/预测**：`pingpong_kalman` 包（Kalman 滤波 + 击球面预测），话题见第 4 节。

本设计只覆盖**新增的 Python 推理节点**；不改动上述两个既有 C++/ROS 包的实现（`armcontrol` 侧一个待新增的"回准备姿态"信号除外，接口本文档已固定为 `/model_control/reset`，具体订阅与执行逻辑由用户自行在 `armcontrol` 里实现）。

## 2. 目标 / 非目标

### 目标
- 新建一个独立 Python 脚本（`PPO-pingpong/scripts/deploy/` 下），加载训练好的 HitTrack PPO 策略（`policy.pt`），通过 ROS2 与 `armcontrol` + `pingpong_kalman` 通信，闭环完成"锁定来球 → 跟踪挥拍 → 回准备姿态"的单次发球周期，并可循环执行。
- 观测/动作的组装、坐标系变换、后处理逻辑与训练侧数值对齐（能复用的直接 import 训练代码，不能复用的新写但要有交叉验证手段）。
- 给出一个可在无真实球体/无 KF 话题时先跑通链路的降级路径（用固定/手动球状态代替）。

### 非目标（本轮不做）
- 不修改 `inference_arm_control_node.cpp`。
- 不修改 `pingpong_kalman` 包。
- 不做训练侧的域随机化/新增奖励项等改动。
- 不实现"回准备姿态"信号在 `armcontrol` 侧的具体逻辑（那是用户自己在 C++ 侧加的新功能，本文档只消费其接口）。
- 不做多球连续发球的调度/节奏控制（假设是"一次一球，人工或发球机控制节奏"）。

## 3. 决策（来自 brainstorming 问答）

| # | 决策 | 选择 |
|---|---|---|
| 1 | 部署节点位置 | 独立脚本，`PPO-pingpong/scripts/deploy/`，不新建 ROS2 ament 包 |
| 2 | 推理后端 | 训练机本地跑推理（同机装好 PyTorch/Isaac），走 ROS2 网络对接机载板卡上的 `armcontrol`；用 `torch.jit`，不需要 onnxruntime |
| 3 | 主循环驱动方式 | 方案 C：挂在 `/right_joint_states` 订阅回调上（反应式，跟 armcontrol 的真实控制节奏对齐），不自建独立定时器 |
| 4 | 发球周期触发 | 完全自动，由 `pingpong_kalman` 的 `/resetKalman` 信号驱动进入 TRACKING |
| 5 | 回准备姿态触发 | 推理节点直接发布 `/model_control/reset`（`std_msgs/msg/Bool`，新增专用信号，`True`=触发一次归位）；`armcontrol` 侧订阅与 `movej` 归位执行由用户后续实现，接口（topic 名/类型）现在就固定下来，不再用占位封装；`enable` 语义不变，仍然是"停止执行 = 原地悬停" |
| 6 | 末端位姿坐标系 | 统一用训练侧的"世界系"（= 桌面/mocap 标定系）；FK 天然给出 base 系结果，变换（base 平移 `ROBOT_BASE_X` + 旋转，当前配置下旋转=单位阵）在 Python 推理节点侧做，因为 `armcontrol` 从不发布任何笛卡尔位姿话题 |
| 7 | 坐标系变换正确性验证方法 | 同一关节状态 q 分别喂给"部署侧 FK+变换"和"仿真侧 FK"，对比末端位姿是否一致（验证的是模型/系数一致性，不验证 base 物理标定，后者假定已经在传统控制器阶段做好） |
| 8 | `/resetKalman` 触发时的 OOD 防护 | `/resetKalman` 只代表"检测到新球"，不保证轨迹起点落在训练分布内；触发后先做一次 x 门控（用 `/kalman/pingpong_pos` 的球体当前 x，与烘焙真实发球数据的 x 范围核对，初始建议 `[0.7, 1.4]` m，可配置）——通过才进入 TRACKING，不通过则忽略本次 reset，继续停留 READY |

## 4. 外部接口（本节点只消费，不定义）

### 4.1 `armcontrol`（`inference_arm_control_node.cpp`，已存在，不改动）

| 方向 | 话题 | 类型 | 说明 |
|---|---|---|---|
| 订阅 | `/right_joint_states` | `sensor_msgs/msg/JointState` | 反馈 `position/velocity/effort`（7 关节，`RIGHT_ARM_JOINT_NAMES` 对应的 `joint1-a1_r..joint7-a1_r`）；同时是本节点方案 C 的 tick 源 |
| 发布 | `/model_action` | `std_msgs/msg/Float64MultiArray` | `controlled_arms=right` 时 POSITION 格式 = 7 个 double（关节目标位置，弧度）。启动参数需配 `action_format=position`（不用 `auto`，避免尺寸歧义）、`expected_action_rate_hz=100`、`interpolation_duration_mode=fixed`、`fixed_interpolation_duration_s=0.01`（详见第 8 节） |
| 发布 | `/model_control/enable` | `std_msgs/msg/Bool` | `True`=按 `/model_action` 伺服跟踪；`False`=停止执行、原地悬停（语义不变，急停安全） |
| 发布 | `/model_control/reset` | `std_msgs/msg/Bool` | 本节点发布 `True` 触发一次"回准备姿态"；`armcontrol` 侧收到后用配置文件里给定的准备姿态走 `movej` 规划回位（订阅与执行逻辑由用户后续在 `armcontrol` 里实现，不影响本节点开发——接口本身已固定，无需占位封装）。本节点不等待任何完成通知，发布后等待一个可配置的固定超时即视为"归位完成" |

不使用 `movej_right_angle` / `/arm_move_state`（这两个是给别的调用方用的点位规划通道，不适合被推理节点在每个发球周期里高频调用，详见对话中的讨论）。

### 4.2 `pingpong_kalman`（已存在，不改动）

**输出话题（本节点订阅）：**

| 话题 | 类型 | 说明 |
|---|---|---|
| `/resetKalman` | `std_msgs/msg/Bool` | 检测到新来球并重置滤波器时发布 `true`——本节点把**任何一条消息**（约定按 `data=true` 触发，防御性处理其他取值）当作"新发球开始"事件，进入 TRACKING |
| `/kalman/pingpong_pred` | `pingpong_kalman/msg/PredictedHit` | 击球面预测结果，见 4.3；核心输入，每次收到（无论 `valid`）都要显式处理 |
| `/kalman/pingpong_pos` | `geometry_msgs/msg/PoseStamped` | KF 滤波后球位置。**新增用途**：`/resetKalman` 触发时用其 `pose.position.x` 做 OOD 门控（见第 5 节）；门控之外仍可选用于日志/诊断 |
| `/kalman/pingpong_vel` | `geometry_msgs/msg/TwistStamped` | KF 估计速度，同上，仅诊断 |

### 4.3 `PredictedHit` 消息字段

```
header      # 沿用输入动捕消息的时间戳和坐标系（约定与训练侧"世界系"一致，即桌面/mocap 标定系）
pred_y      # 击球面处球的 y 坐标 [m]
pred_z      # 击球面处球的 z 坐标 [m]
pred_vx     # 击球面处 x 方向速度 [m/s]
pred_vy     # 击球面处 y 方向速度 [m/s]
pred_vz     # 击球面处 z 方向速度 [m/s]
pred_t      # 从当前状态到达击球面的预计时间 [s]（= tau，非负，代表"还有多久到")
valid       # 是否在 predict_max_time 内成功穿过目标 x 平面
```

`valid=false` 时该消息仍会发布，本节点必须显式处理（见 5.3 的容错/丢失逻辑），不能默认当作合法数据消费。

**需要用户确认/配置对齐的一项**：`pingpong_kalman` 预测所用的目标 x 平面（其内部参数，消息里没有携带具体数值）必须等于训练侧 `HIT_PLANE_X`（当前 `-1.44`）。这是一个配置层面要核对的常数，不在本节点代码里体现，但要在联调时显式检查一次。

## 5. 状态机

```
INIT ──[发归位信号]──> (等归位完成，固定超时) ──> READY(idle)
READY(idle) ──[/resetKalman 触发]──> GATE(x 门控)
GATE(x 门控) ──[球体 x∈[0.7,1.4] 通过]──> TRACKING
GATE(x 门控) ──[x 不在范围内 / 尚无最新球位置]──> READY(idle)（忽略本次 reset）
TRACKING ──[tau<=-post_margin_s(0.12) 或 预测持续无效/丢失]──> [enable=False + 发归位信号]
[enable=False + 发归位信号] ──(等归位完成，固定超时)──> READY(idle)
```

- **INIT**：`torch.jit.load` 策略；等第一帧 `/right_joint_states`；发一次归位信号（`/model_control/reset`=True），固定超时后视为归位完成。
- **READY(idle)**：`/model_control/enable=False`（默认状态，不必每 tick 重复发），不发 `/model_action`；订阅 `/resetKalman` 等触发。
  - **GATE(x 门控)**：收到 `/resetKalman` 后，取最近一帧 `/kalman/pingpong_pos.pose.position.x`（本地缓存最新值）。若 `x∈[0.7,1.4]`（可配置常量，需与烘焙数据分布核对，见第 8 决策）→ 进入 TRACKING；若不在范围内，或尚未收到过任何 `/kalman/pingpong_pos` 消息 → 忽略本次 reset 事件，留在 READY(idle)，继续等下一次 `/resetKalman`（可选打一条日志记录本次门控拒绝，不算故障，不触发第 9 节的安全中断路径）。
- **TRACKING**：
  - 进入时：重置 `prev_target=当前真实 q`、`joint_pos_delta_history`（5 帧填当前 q）、`last_action=0`、`tau_anchor=None`；发 `/model_control/enable=True`。
  - 在收到第一条有效 `PredictedHit`（`valid=true`）之前，**不做推理/不发 `/model_action`**（没有合法参考，宁可先原地悬停，也不能拿默认零值当参考去驱动动作）。
  - 每个 `/right_joint_states` tick（有合法 `tau_anchor` 之后）：
    1. `tau_live = tau_anchor.pred_t - (now - tau_anchor.recv_time)`（本地实时外推，见 5.2）
    2. 用 `tau_anchor` 里锚定的 `(pred_y,pred_z,pred_vx,pred_vy,pred_vz)` 调训练侧 `plan_hit_reference` 得到 `p_ref,v_ref,n_ref`
    3. 用当前真实 q 做 FK + base 变换得到 `racket_pos,racket_normal`（世界系）
    4. 拼 68 维 obs → policy 前向 → 训练侧 `compute_joint_delta_target` 同款后处理 → `q_target`
    5. 发布 `/model_action`（POSITION 格式）
  - 结束条件：`tau_live <= -post_margin_s` **或** 预测流丢失超过容忍时间（见 5.3）。
- **[enable=False + 发归位信号]**：停发 `/model_action`；发 `enable=False`；发归位信号（`/model_control/reset`=True）；固定超时后视为归位完成，回 READY，重新武装 `/resetKalman` 监听（含 x 门控）。

## 6. `tau` / 参考量的实时锚定与外推（5.2 详细展开）

训练侧的 `tau_true_stream` 是"reset 时锚定一个初始估计，之后按固定 `step_dt` 线性倒数"（见 `reference_commands.reset_reference_command` / `tau_streams`），并不是每步重新从球的当前状态反算。`PredictedHit.pred_t` 语义与此一致（"距离到达击球面还有多久"），所以部署侧采用同构做法，而不是直接把每条消息的 `pred_t` 原样当成 tau：

- 维护 `tau_anchor = (pred_y, pred_z, pred_vx, pred_vy, pred_vz, pred_t, recv_time)`，仅在收到 `valid=true` 的新消息时**更新**（覆盖）。
- 每个控制 tick 用 `tau_anchor.pred_t - (now - tau_anchor.recv_time)` 做实时外推得到 `tau_live`，而不是等下一条 KF 消息才更新——这样即使 KF 消息率低于 100Hz 控制环，obs 里的 `tau` 依然是连续倒数的，行为上对齐训练。
- `valid=false` 的消息**不更新** `tau_anchor`（继续用上一个合法锚定外推），但要计入"丢失容忍"计时（5.3）。
- 这个设计同时回答了"球已经过击球面之后 `pred_t` 语义是否继续有效"的疑问：过了击球面之后我们不再依赖 KF 继续给出负值或任何东西，纯本地时钟外推，天然能算出负的 `tau_live`，直到触发 `post_margin` 结束条件——这与训练侧 `hit_window_elapsed` 的判定方式（基于固定时间线，不是每步重新预测）是同构的。

## 7. 坐标系与 FK

- **统一坐标系** = 训练侧定义的"世界系"（对单机械臂部署场景，env-local 退化为世界系本身，即 `env_origins=(0,0,0)`，建议用 `play_hittrack.py --num_envs 1` 打印一次 `env.scene.env_origins` 做最终确认，不假设）。`hit_reference_command`、`racket_pos`、`racket_normal`、`hit_ref_pos_error` 四组观测全部在这个系里，`PredictedHit` 的坐标（沿用动捕系）按现有标定惯例也应与此一致。
- **FK 模块（新写）**：优先用 RBDL + 同一份 `a1_r.urdf`（跟 `armcontrol.cpp` 内部用的模型一致，避免"两套 FK 各自实现、悄悄对不齐"），需要确认训练机上是否有 RBDL 的 python binding；没有则退化到 `pytorch_kinematics` 解析 URDF。
- **base 变换**：FK 原生输出是 base 系；平移 `T=(ROBOT_BASE_X,0,0)=(-1.82,0,0)`，旋转 `R`=单位阵（当前 `env_cfg.py` 里 `ROBOT_SIDE<0` 分支的配置）。`racket_pos_world = R@racket_pos_fk(q) + T`；`racket_normal_world = R@racket_normal_fk(q)`（方向量只旋转不平移）。
- **正确性验证（不是物理标定验证）**：取若干组关节状态 q（至少含 `READY_JOINT_POS`），分别用"部署侧 FK+变换"和"跑一个单 env 仿真直接读 `_racket_body_state`"算末端位姿，两者应一致。这只验证部署代码的运动学模型/坐标系数是否跟训练一致；机械臂物理 base 是否真的摆在 mocap 世界系 `(-1.82,0,0)`，属于台面/机械臂/mocap 三方标定问题，假定已经在传统控制器调试阶段解决，不在本节点验证范围内。

## 8. 观测/动作对齐细节

**68 维 Actor 观测**，按训练侧 `ObservationsCfg.ActorCfg` 声明顺序拼接：

| 顺序 | 名称 | 维度 | 部署侧来源 |
|---|---|---|---|
| 1 | joint_pos_rel | 7 | 真实 q − `default_joint_pos`（需从训练侧 a1 资产配置扒出精确数值，核对与 `READY_JOINT_POS` 的关系） |
| 2 | joint_pos_delta_history | 35 | 本地维护 6 帧滚动缓冲（含当前帧），`history[0]-history[1:]`，clip ±1.0，TRACKING 进入时重置 |
| 3 | hit_reference_command | 10 | `[p_ref,v_ref,n_ref,tau_live]`，见第 6 节 |
| 4 | racket_pos | 3 | FK+base 变换，见第 7 节 |
| 5 | racket_normal | 3 | FK+base 变换 |
| 6 | hit_ref_pos_error | 3 | `racket_pos - p_ref` |
| 7 | last_action | 7 | **裁剪前**的上一步策略原始输出（不是送进 `compute_joint_delta_target` 之后的值——这是训练侧的定义，容易踩错） |

**可直接复用、零重新实现风险**（都是纯 torch/无 isaaclab 依赖，直接 import）：
- `unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp.reference_planner.plan_hit_reference`
- `unitree_rl_lab.tasks.table_tennis_sac.control.compute_joint_delta_target`

**策略导出缺口（需要新增）**：现有 `scripts/rsl_rl/play_hittrack.py` 能正确加载 HitTrack checkpoint（处理了 `handle_deprecated_rsl_rl_cfg` 迁移）但不导出 jit/onnx；通用 `scripts/rsl_rl/play.py` 有导出逻辑但加载不了 HitTrack 的 legacy cfg（`KeyError: 'class_name'`）。需要新增一个小脚本/在 `play_hittrack.py` 里补一段，把两者结合：正确加载 + `export_policy_as_jit(policy_nn, normalizer=actor_obs_normalizer)`，这样 `empirical_normalization` 会随导出打包，部署侧拿到的 `policy.pt` 直接吃原始 obs，不用自己维护归一化统计量。

## 9. 安全与容错

- **预测丢失/持续无效**：`valid=false` 或 `/resetKalman` 之后一直没收到任何 `PredictedHit` 超过一个容忍窗口（可调参数，初始建议几十毫秒量级/几个控制周期，具体数值留给实现/联调阶段定） → 立即中断 TRACKING（不等 `tau_live` 自然归零），走 `enable=False + 归位信号`。
- **推理异常/NaN 动作**：捕获异常或检测到非法数值 → 同上，直接中断到安全态。
- **心跳看门狗**：方案 C 主循环挂在 `/right_joint_states` 回调上，额外加一个低频（如 10Hz）独立看门狗定时器，检测该反馈是否长时间未到达（网络断/节点挂，阈值同样是可调参数），触发同样的中断路径。
- **纵深防御**：`armcontrol` 自身的 `applySafetyLimits`（限速/限加速度/关节限位/`action_timeout_s`）是硬件侧最后一道保险，本节点的裁剪逻辑不是唯一防线，但也不能因此放松自己这一侧的限位。

## 10. 测试计划

1. **纯 Python 单测**（无 ROS/isaaclab）：obs 拼装维度与切片顺序；`tau_anchor` 外推逻辑；状态机转换（用假的消息序列驱动）。
2. **FK 交叉验证**：离线脚本对比新 FK 模块 vs 仿真真值（第 7 节方法），产出一份固定样本的 fixture 做回归测试。
3. **ROS 集成干跑（无硬件）**：假的 `/right_joint_states` + 假的 `/resetKalman`/`PredictedHit` 序列喂节点，断言 `/model_action` 输出形状/幅值/连续性正常，且在预测无效/丢失时正确中断。
4. **真机 bench 测试**：先不接真实 KF，手动/脚本模拟一次 `PredictedHit` 序列，机械臂降速/防护到位，跑通"锁定→追踪→归位"整圈。
5. **接入真实 `pingpong_kalman` 闭环验证**。

## 11. 开放依赖 / 待确认项

| # | 项 | 状态 |
|---|---|---|
| A | 球体预测话题（`/resetKalman`、`/kalman/pingpong_pred`/`PredictedHit`） | **已确认**（本文档第 4.2/4.3 节） |
| B | `armcontrol` 侧对 `/model_control/reset` 的订阅与 `movej` 归位执行 | 接口已固定（topic/类型见 4.1）；具体实现由用户后续在 `armcontrol` 里完成，不阻塞本节点开发 |
| C | 仿真 `A1_ARM_STIFFNESS/DAMPING` 与 `armcontrol` 启动参数 `kps`/`kds` 是否物理匹配 | 待核实（不阻塞本节点开发） |
| D | 训练机上 RBDL python binding 是否可用（决定 FK 实现路径） | 待核实 |
| E | `default_joint_pos`（obs 用）与真实 soft joint limits 的精确数值 | 待从训练侧资产配置提取 |
| F | `pingpong_kalman` 预测目标 x 平面参数是否等于 `HIT_PLANE_X=-1.44` | 待联调时核对 |
| G | reset x 门控范围 `[0.7,1.4]`（决策 8）是否与烘焙真实发球数据的 x 分布吻合 | 待核对 `bake_hittrack_references.py` 产出的烘焙数据确认 |

## 12. 后续步骤

设计通过后，转 `writing-plans`，把 §4-§10 拆成可执行的实现任务（模块划分预期：ROS 接口层 / 状态机 / obs 组装 / FK / 策略导出脚本，各自独立可测）。
