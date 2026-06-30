# A1 乒乓球 · 模型化击球参考跟踪任务 (HitTrack) 设计

- **日期**: 2026-06-26
- **分支**: `hitter`
- **新任务**: `A1-Pingpong-HitTrack`(**不**退役现有 `A1-TableTennis-SAC-Catch`)
- **代码根**: `source/unitree_rl_lab/unitree_rl_lab/tasks/a1_pingpong_hittrack`
- **重构注记 (2026-06-30)**: HitTrack 已从 `table_tennis_sac` 抽成上述独立 task 包,仅 import 复用其共享基建(场景/机器人/通用 MDP 项/`hitting.py`);本 spec 余下正文沿用原 `table_tennis_sac` 增量布局与旧任务 id 描述,当前布局以 `docs/hittrack_方案简介.md` 为准。
- **来源**: 本 spec 是 `docs/sac_table_tennis_architecture.md` §15「Proposed Model-Based Hit-Reference Refactor」的逐条细化与落地版,与该节配套阅读。
- **目标**: 把当前端到端的接球/回球 SAC 任务,改造为**非端到端、模型化的"击球参考跟踪"任务**——轨迹预测器给出击球面球态 → 参考规划器换算成末端击球参考 `(p_ref, v_ref, n_ref)` → 机械臂策略在击球时刻跟踪该参考。**球离开 RL 的 MDP**;奖励从"球结果"改为"击球时刻参考跟踪误差"。预测问题(KF)与控制问题(策略)彻底解耦。

---

## 0. 决策摘要(已与用户逐项确认)

| 维度 | 决策 |
|---|---|
| 任务性质 | 非端到端、模型化击球参考**跟踪**;球不在 MDP 内 |
| 控制/推理频率 | **100Hz**(`decimation=2`,`sim.dt=0.005` 物理 200Hz 不变);匹配真机控制环 100Hz,部署免插值。Catch 保持 50Hz 不动 |
| 数据源 | **真机部署 KF 录制的真实轨迹**(120Hz),含 mocap 真值 + KF 实时预测 |
| 跟踪目标 (D1) | **跟 actor 收到的 noisy 指令**(纯跟踪 MPC);clean 真值只给 critic / success 指标 |
| 时间门 (D2) | **连续高斯门**,门用 `tau_true`(privileged) |
| 速度项 (D3) | **全向量** `‖v_racket − v_ref‖` |
| 动作契约 | 保留现有 `JointDeltaTargetAction`(选项 A);**否决** IK 残差(选项 B) |
| 法向奖励 | v1 **关闭**(`w_normal=0`),但 `n_ref` 仍在 10 维参考观测里 |
| planner 位置 | **运行时模块**,与部署共用;烘焙脚本**只产原始球态**,不烘焙末端参考 |
| 训练器 | **暂不敲定**;设计算法无关,SAC vs `rsl_rl` PPO 设计定稿后再选 |

---

## 1. 背景:当前框架(简述)

- Isaac Lab `ManagerBasedRLEnv`。当前 SAC Catch:控制 50Hz(`sim.dt=0.005`、`decimation=4`、`step_dt=0.02`),物理 200Hz。
- `ROBOT_SIDE=-1`:机器人在 −X,球从 +X 飞来。击球面 `HIT_PLANE_X = SAC_ROBOT_X = -1.37`,己方台 x∈(−1.37,0),对方台 x∈(0,1.37),`TABLE_Z=0.76`。
- 动作 = 右臂 7 维关节增量(`JointDeltaTargetAction`,`action_scale=0.12`、`smoothing=0.5`、`max_joint_velocity` 取 `A1_ARM_VELOCITY`、`limit_margin=1e-3`)。
- 现有奖励 = 三段式球结果阶梯(hit/return/valid_return + 多项 post-hit 球塑形)+ 硬件平滑正则。依赖 `update_sac_episode_state`(接触传感器 + 球位置)。
- 物理核心 `mdp/hitting.py`(纯 torch,无 Isaac 依赖)已实现:
  - `ideal_racket_velocity(origin, v_in, target, theta, restitution, drag_k, lin_damp, ...) -> (v_paddle_ideal, v_out, face_normal)`:无旋刚性接触求逆 `v_out = v_in − (1+e)((v_in−v_p)·n)n` + 带阻力发射边值解。
  - `contact_inverse`、`predict_landing_xy`、`predict_z_at_x`。
  - `NEUTRAL_THETA=radians(28°)`、`PADDLE_RESTITUTION=0.75`。

---

## 2. 核心概念:noisy vs clean(前瞻 vs 事后)

整个非对称设计的根基。录制数据里**同时**含两样东西:

- **noisy 参考** = `planner(KF 的击球面预测)`。KF 在球未到时**实时外推**出"球将在 `x=−1.37` 处的状态",带误差(非 Magnus KF 对旋转误判 → per-serve 一致性偏置 + 小幅逐步抖动)。**这是真机部署时机器人唯一拥有的信号(前瞻)。**
- **clean 真值** = `planner(球实际穿越 x=−1.37 的状态)`。只有**事后**才知道——回看完整录制的 mocap 轨迹,解出球真实穿越击球面的状态。**部署实时拿不到(事后真值)。**

即便 KF 的输入传感器也是 mocap,KF 的**预测**(外推)和球**实际**走的(回看)之间那道缝就是 noisy↔clean。

**用途分工:**
- actor **全程只消费 noisy**(KF 预测)。
- clean 用于:① **success/验证指标**(对真实结果衡量,这条省不掉);② **critic 输入**(privileged 方差抑制,可开关 ablation)。
- **部署即拆**:丢掉 critic 与 clean,actor 吃的还是训练时一模一样的 KF 预测输入,无缝迁移。

> 诚实说明:在 D1=a(奖励对 noisy)下,环境对 actor **近乎全可观**(奖励是"拍 FK 状态 vs actor 看得到的 noisy 指令"的函数)。唯一真正 privileged 的是「门用 `tau_true`、actor 只看 `tau_noisy`」那点时序差。故 critic-clean 是**轻量稳定器,非命根子**,设为可开关。

---

## 3. 数据录制与烘焙

### 3.1 原始录制 schema(真机部署侧,120Hz)
每行:
```
t (统一时间戳), serve_id,
mocap   : x, y, z, vx, vy, vz      # 动捕真值球态
KF      : x, y, z, vx, vy, vz      # KF 当前滤波估计
KF_pred : y, z, vx, vy, vz, tau    # KF 对击球面(x=−1.37 固定,不记)的预测 + 预测到达时距
```
约束:
- **保留每条 serve 的完整 mocap 轨迹**(离线据此解真值穿越)。
- **只录飞穿球(拍子不接触)**,或加一个 `contact` 标志位剔除被击打后的轨迹——否则 mocap"穿越点"不是真实自由飞行的穿越点。
- 真实捕捉帧 → sim 世界帧的轴翻转/偏移、`ROBOT_SIDE` 换算在烘焙时统一处理(沿用 `create_serve_states.py` 既有约定)。

### 3.2 烘焙脚本 `scripts/sac_table_tennis/bake_hittrack_references.py`(只产原始球态,**不跑 planner**)
读 120Hz 日志 → 产出每条 serve 的:
1. **noisy 球态流**:`KF_pred (y,z,vx,vy,vz)(t)`,重采样到 **100Hz**(线性插值;KF 原始更新若慢于 100Hz 则保持/插值)。
2. **clean 真值球态**:从 mocap 连续轨迹解出 `x=−1.37` 穿越点的球态(单个,固定)。
3. **`tau_true(t)`** = (真值穿越时刻 − 该 100Hz 网格点时间)。
4. **reachability 标志**:**宽松的粗粒度运动学可行性兜底**——只剔掉"真值击球点在手臂工作空间外(含拍面半幅 + 充裕 margin)、任何策略都够不到"的离群 serve。对真实录制 serve 基本是空操作的安全网。
- 输出 `hittrack_references.npz`。**不存任何末端参考**(planner 运行时跑)。
- 100Hz Nyquist 50Hz,击球面预测量缓变(动态 ≪ 10Hz),无混叠。

---

## 4. 参考规划器(运行时模块)

`source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis_sac/mdp/reference_planner.py`

- **运行时模块,与部署共用同一份代码**(非离线烘焙)。`击球面球态 → (p_ref, v_ref, n_ref)`:
  - 选 target = 对方台中心 `OPP_TABLE_CENTER_X`(高度 `TABLE_Z`),`theta=NEUTRAL_THETA`。
  - 调 `hitting.ideal_racket_velocity` → `(v_paddle_ideal, v_out, face_normal)`。
  - `p_ref = 穿越点位置`;`v_ref = v_paddle_ideal`(全向量);`n_ref = face_normal`。
  - 纯 torch,可脱离 Isaac 单测。
- **运行时如何跑**:env 在 **reset 时对加载的整条流批量跑一次**填满本回合参考 buffer。planner 逐时刻无记忆(`ref(t)=planner(球态(t))`,不耦合历史),故"reset 批量"与"每步流式"**数值完全相同**;批量更省(异步 reset 下每步仅 ~1/70 env 在跑求解)。
- noisy 球态流 → **actor 的 noisy 末端参考(随 KF 收敛逐步演化)**;clean 真值球态 → **clean 末端参考(整回合恒定)**。到击球时刻 noisy≈clean。
- **平价收益**:① 仿真运行时数据流 `KF预测 → planner → 末端参考 → 策略` 与真机逐字一致,planner 代码路径在训练回路内被验证;② planner 改动(`theta`/target/restitution)**不必重烘焙**,改模块重训即可(烘焙产物是 planner 无关的"球态")。
- **否决备选**:把 planner 烘焙进 npz——会引入"烘焙版/部署版分叉"风险且改 planner 要重烘焙。

---

## 5. 观测契约

50Hz→100Hz 后 `step_dt=0.01`。`JOINT_POS_DELTA_HISTORY_LENGTH = 5`(由 3 加深,Kd=5)。

### 5.1 Actor (`policy`,可部署,`enable_corruption=False`)
| 项 | 来源 | 维度 |
|---|---|---|
| `joint_pos` | `joint_pos_rel`(右臂 7 关节) | 7 |
| `joint_pos_delta_history` | `joint_pos_delta_history(Kd=5)` | 7×5=35 |
| `reference_command`(**noisy**) | 运行时 planner(noisy 球态) | `p_ref`(3)+`v_ref`(3)+`n_ref`(3)+`tau_noisy`(1)=**10** |
| `racket_pos` | FK 拍中心(`RACKET_OFFSET_Z=0.045`,encoder 可精确算) | 3 |
| `racket_normal` | FK 拍面法向 | 3 |
| `ref_pos_error`(可选便利特征) | `racket_pos − p_ref_noisy`(均 actor 可得) | 3 |
| `last_action` | | 7 |

> actor **不含** `joint_vel` / `racket_vel`(均为 sim 真值或噪声 encoder 差分 → privileged)。速度信息靠 `joint_pos_delta_history` 提供。

### 5.2 Critic(扩展 Actor + privileged)
| 项 | 来源 |
|---|---|
| `joint_vel` | `joint_vel_rel`(sim 真值) |
| `racket_vel` / `racket_ang_vel` / `racket_axes` | sim 真值 FK 速度 |
| `reference_command`(**clean**) | 运行时 planner(clean 真值球态)+ `tau_true` |
| `ref_vel_error`(**critic-only**,D3 决策) | `racket_vel − v_ref` |

- **硬删**旧球观测函数:`ball_pos_history`、`ball_pos_relative_to_racket`、`ball_vel_relative_to_racket`、`ball_vel_w`、`hit_command_at_robot_x`、`estimated_hit_command_at_robot_x`、`predicted_hit_point_at_robot_x`、`time_to_predicted_intercept`。
- **复用** `_racket_body_state` / `racket_pos` / `racket_vel` / `racket_normal` / `racket_axes` / `joint_pos_delta_history`。

---

## 6. 动作契约

- **保留现有 `JointDeltaTargetAction` 原样(选项 A)**:`policy action [-1,1]^7 → 关节目标增量 → smoothing/限速 → q_target`。7 关节全开,`limit_violation` 追踪保留(供 `joint_limit` 正则)。
- 100Hz 调整:`max_joint_velocity` 为物理 rad/s,限速 `max_delta = vel×step_dt` 自动减半,挥拍峰速不变;`action_scale` 0.12 偏大,**降到 ~0.06 或靠限速兜底**(留调参)。
- **风险点(进验证指标盯)**:新奖励要求末端在击球面达 `v_ref`(拍速 ~1.2 m/s 量级),`max_joint_velocity` 才是峰速真正瓶颈。若 `vel_score` 学不上去,**第一优先级放宽限速 / 调大 `action_scale`**,而非动奖励。
- **否决备选(选项 B,IK 残差控制器)**:用户意图就是用 RL 策略**替代** IK——策略=约束下做末端控制的学习型 MPC,隐式解 IK+时序+约束。显式 IK 与设计目标冲突,不实现。

---

## 7. 奖励契约

### 7.1 删除(全部球结果项 + 依赖)
`hit_bonus, return_cross_net, bad_hit, table_proximity, return_bonus, landing_placement, flat_return, post_hit_outgoing, post_hit_net_clearance` 及所有 pre/post-hit 球塑形 hook。连带 `update_sac_episode_state`(接触+球)废弃。

### 7.2 保留(硬件/平滑正则,100Hz 重训时复核权重)
`action_rate(-0.005)`、`joint_acc(-5e-7)`、`joint_jerk(-2e-10)`、`joint_limit(margin 0.05, -1.0)`、`joint_effort_margin(margin_frac 0.85, -0.25)`。

### 7.3 新增:击球面参考跟踪(核心)
```
time_gate = exp(-0.5 (tau_true / sigma_t)^2)          # 门用 tau_true (privileged)
pos_score = exp(-‖p_racket - p_ref‖^2 / (2 sigma_p^2))
vel_score = exp(-‖v_racket - v_ref‖^2 / (2 sigma_v^2)) # 全向量 (D3)
reward = w_pos·gate·pos_score + w_vel·gate·vel_score + 正则
```
- `p_racket / v_racket` = 拍**几何中心**位姿/速度(`_racket_body_state`,sim 真值,privileged 给奖励用)。
- **D1=a**:`p_ref / v_ref` = actor 收到的 **noisy 指令**(纯跟踪;天花板被 KF 残差封顶,与真机一致)。clean → critic / success。
- **D2**:连续高斯门(每步评估)。
- pos 与 vel **加性**(不做乘性门控)。
- `normal_score` **关闭**(`w_normal=0`),`n_ref` 仍观测不进奖励。

### 7.4 起始数值(可调)
| 参数 | 起始 | 说明 |
|---|---:|---|
| `sigma_t` | 0.03 s | ~3 步 @100Hz |
| `sigma_p` | 0.03 m | ≈ KF 击球点误差;拍面半幅 ~0.075×0.125 m |
| `sigma_v` | 0.3 m/s | 拍速 ~1.2 m/s 的松端;稳定后收到 ~0.15–0.2 |
| `sigma_n` | 12° | 仅 normal 开启时 |
| `w_pos` | 20 | 100Hz 起始值;先等权 |
| `w_vel` | 20 | 同上 |
| `w_normal` | 0 | v1 关 |

> **数值标定说明**:本任务生来即 100Hz,连续高斯门按步求和,门内约 2×σ_t/step_dt≈6 个采样点。`w_pos=w_vel=20` 选取使"完美击球时刻的门内累计奖励"显著压过正则项(正则量级 1e-1~1e-7)。均为可调起点,训练后按 `reward_terms/*` 占比微调。

### 7.5 放松旋钮(若太稀疏)
**仅**加宽 `sigma_t` 或加 time-to-go 轨迹参考。**绝不重新引入球结果奖励**(红线)。

---

## 8. 回合与 Reset 逻辑

### 8.1 Reset(无球)
- **删** `launch_ball`、`apply_air_drag`、`update_sac_episode_state`。
- **加** `reset_reference_command`:从 npz 采一条 serve,载入 noisy/clean 球态流 + `tau_true`,reset 时批量跑 planner 填满末端参考 buffer,置参考时间游标到回合起点。
- **保留** `reset_robot_to_ready_pose`(`SAC_READY_JOINT_POS` + `SAC_READY_LIFT_POS` 不变;**E4: ready pose 固定,v1 不随机**)。
- **加** `update_hit_track_state`(替代旧 tracker):每步推进游标、吐当前 noisy/clean 参考 + `tau_true`、在击球窗口记 success。

### 8.2 时序参数
- **E1: `tau_initial = min(该 serve 自然时距, max_prep)`**,变长、加上限。`max_prep ≈ 0.6 s`(挥拍约需 0.5s,参见背手 demo arrive_time≈0.526s)。快球自然短、慢球砍到 0.6s 起(跳过早期干等)。策略观测含 `tau`,据此自适应。
- **E2: `episode_length = max_prep + post_margin`**,`post_margin ≈ 0.10–0.15 s`(σ_t 的 ~3σ)→ **≈0.70 s = 70 步 @100Hz**。
- **击球后不训回位**:无旋模型下出球在接触瞬间已定,follow-through 对已飞走的球零影响;`post_margin` 段保持正则项开启把瞬时动作拽平滑即可。回原位是**部署层**职责(用速度感知减速轨迹,**非裸 movej**,因交接时手臂高速运动)。

### 8.3 可达性 / 软约束(E3)
- 击球面**位置与速度都是软目标**(高斯奖励 + 拍面有尺寸,差几 cm 照样打到)。
- 离线"过滤"**仅**粗粒度运动学可行性兜底(§3.2),非精度门;速度可达留软(够不到由奖励天花板体现)。

### 8.4 Success flag(E5)
在 `tau_true` 过零那一步:
- **success(对 clean 真值判)**:`‖p_racket − p_ref_clean‖ < 0.05 m` **且** `‖v_racket − v_ref_clean‖ < 0.2 m/s`(可调)。天花板 = KF 预测残差 = 真机"能否真打回去"的上限。
- 同时记 **vs-noisy** 误差(衡量纯控制质量)——区分失败是预测问题还是控制问题。

### 8.5 终止(`TerminationsCfg`)
- 保 `time_out`(到 `episode_length`)、`nan_state`。
- `sac_episode_done`(球结果)→ 换 `hit_window_elapsed`(`tau_true < −post_margin`)。
- 保留硬关节违例终止(若现有)。

---

## 9. Replay 与训练

- **训练器算法无关**;actor/critic 非对称观测 SAC 与 `rsl_rl` PPO 均支持。SAC vs PPO 设计定稿后再选(用户倾向先评估 `rsl_rl` PPO 的成熟度)。
- **保留 actor/critic 拆分,但设为可开关 ablation**(默认开)。理由见 §2:D1=a 下环境近乎全可观,critic-clean + `tau_true` 是轻量稳定器。
- 可选 `hit_window` 表存 `|tau_true|<0.10s` 的 transition(仅 off-policy 用得上)。
- 可选 clean 参考/clean 速度**辅助重构头**(只当 aux loss,**不进 reward**),v1 **不上**。

---

## 10. 实现切片(文件 & 任务 id)

任务 id:**`A1-TableTennis-SAC-HitTrack`**。

| 件 | 文件 | 职责 |
|---|---|---|
| 烘焙脚本(新) | `scripts/sac_table_tennis/bake_hittrack_references.py` | 读 120Hz mocap+KF 日志 → 重采样 100Hz → noisy 球态流 + clean 真值穿越 + `tau_true` + reachability;写 `hittrack_references.npz`。**无 planner** |
| reference planner(新,**运行时**) | `mdp/reference_planner.py` | `球态 → (p_ref,v_ref,n_ref)`,调 `hitting.ideal_racket_velocity`;与部署共用;纯 torch 单测 |
| HitReferenceCommand(新,运行时) | `mdp/reference_commands.py` | reset 采 serve、批量跑 planner、置游标;每步吐 noisy(actor)/clean(critic)参考 + `tau_true` |
| observations | `mdp/observations.py` | 加 10 维参考指令 + FK 参考误差;**硬删** ball_* 观测;复用 racket FK |
| rewards | `mdp/rewards.py` | `hit_ref_pos` / `hit_ref_vel`(高斯门)+ 保留正则;删球结果项 |
| terminations | `mdp/terminations.py` | `hit_window_elapsed` + nan + 硬限位 |
| env cfg(新 task) | 新建 `hittrack_env_cfg.py`(**不碰** Catch 的 `env_cfg.py`)+ gym 任务注册 | `decimation=2`(100Hz)、`episode_length≈0.7s` |
| tests | 新测试文件 | planner 形状/范围 + 逐调用流式 NaN/边界、游标单调、完美跟踪奖励取极大、无球结果项激活、烘焙输出 schema |

`hitting.py` **不动**。

---

## 11. 验证指标

TensorBoard(从球结果转向参考跟踪):
- `ref_pos_error_at_hit`、`ref_vel_error_at_hit`:**同时报 vs-clean(真实质量)与 vs-noisy(纯控制)**。
- **`success_rate`**(headline,§8.4,对 clean)。
- `hit_time_abs_error`:**最佳跟踪时刻 vs 真值击球时刻**的偏移(挥拍时序)。
- `reachable_reference_rate`、`joint_limit_violation_rate`、`effort_margin_mean`、各 `reward_terms/*`。

部署 dry-run(离线、RL 外):planner 输出 vs 真机 FK 在击球时戳实现值、预测更新→下发指令延迟、被拒参考及原因。

**离线回球质量验证(§13 第7步,RL 奖励之外)**:把策略实现的拍 `p/v/n` 喂进 `hitting.predict_landing_xy` / `predict_z_at_x`,估计落点/过网,评回球质量。**保持在奖励之外**(红线 + 非端到端安全网)。

---

## 12. 迁移顺序

1. Catch 冻结(已存在,不碰)。
2. 加 HitTrack,无球结果奖励;env trainer 无关。
3. planner 模块 standalone + 单测;烘焙脚本(重采样 + 真值穿越提取)。
4. 复用 FK + joint-history 观测。
5. noisy 参考直接来自**录制的真实 KF 流**(速度误差天然含在内,doc 当时发愁的"velocity noise 缺失"自动解决);合成噪声 `SAC_HIT_COMMAND_NOISE` **降级为可选增强**,仅给下面随机 curriculum 用。
6. **课程**:① 固定单参考(取背手 demo 击球态,在线 plan,验证跟踪奖励+控制能学会打一个点)→ ② 小随机参考(中心附近小 box 在线 plan,验证泛化)→ ③ 全真实 serve 分布(加载烘焙 npz)。
7. 跟踪误差小后,跑 §11 离线回球质量验证。

---

## 13. 否决的备选(文档卫生)

| 备选 | 否决理由 |
|---|---|
| 选项 B:IK 残差控制器 | 用户意图用 RL **替代** IK;显式 IK 与"策略=学习型 MPC"目标冲突 |
| 奖励对 clean 真值(D1=b) | 逼策略在噪声输入下还原真值 → 把预测又塞回策略,违背解耦初衷 |
| 单发奖励(D2=b,仅 tau≈0 一步) | 太稀疏,初期学不出挥拍;σ_t→0 时与连续门等价,先用连续门 |
| 速度只比法向分量(D3=b) | 物理上更正确(无旋切向自由)但需 n_ref 投影;v1 先全向量,逼僵硬再换 |
| 把 planner 烘焙进 npz | 引入烘焙/部署分叉风险 + 改 planner 要重烘焙;改为运行时模块 |
| 仿真跑 120Hz(对齐数据率) | 与真机 100Hz 控制环失配,重新制造插值;数据 120Hz 只作源、降采样到 100Hz |

---

## 14. 待定 / 进入 plan 阶段细化

- 100Hz 下 `action_scale` 与各正则权重的具体重调值(训练后实测)。
- 真机录制 schema 的 `contact` 标志位 vs "只录飞穿"的最终采集约定。
- 手臂工作空间可达框的具体 (y,z) 边界(用于 §3.2 离群剔除)。
- critic-clean ablation 的开关默认与对比实验。
- 训练器最终选型(SAC vs `rsl_rl` PPO)。
