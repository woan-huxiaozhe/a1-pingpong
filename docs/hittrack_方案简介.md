# HitTrack 方案简介

> 独立速览文档。任务 ID：`A1-Pingpong-HitTrack`。
> 完整设计与实现细节见 `docs/superpowers/specs/2026-06-26-hittrack-design.md` 与
> `docs/superpowers/plans/2026-06-26-hittrack-implementation.md`，本文不依赖它们即可读懂。

## 1. 一句话概括

让机械臂**只学"在预测击球时刻把球拍送到正确的位姿与速度"**，球不进入 MDP。
击球参考量 `(p_ref, v_ref, n_ref)` 由解析模型从球的状态推出，奖励是对该参考量的
**时间门控高斯跟踪误差**。这是一个**非端到端**的子任务，把"接球→回球"中最难学的
"拍速/拍向规划"从强化学习里剥离出来，交给模型，RL 只负责"按时跟到"。

## 2. 为什么这么做

端到端的 Catch 任务里，策略要同时学：感知球的轨迹、预测落点、规划理想拍速、再控制
机械臂跟上。这条链路又长又稀疏，容易陷入 lob、欠摆、开环定式等局部最优（见 catch
方向的多份诊断 memo）。

HitTrack 把链路**切成两段**：

- **模型段（解析、可信、可部署）**：给定球在击球面的状态 + 目标落点，用理想拍速模型
  解析地算出该时刻球拍应有的 `(位置 p_ref, 速度 v_ref, 法向 n_ref)`。
- **学习段（RL）**：策略只需在正确时刻把 FK 拍心跟到 `(p_ref, v_ref)`。奖励稠密、
  几何直观、不依赖触球这一稀疏事件 —— 跟踪问题远比"打中并回到指定落点"好学。

球完全不在仿真 MDP 里：reset 时采样一个"虚拟来球"算出参考流，整段 episode 只跟踪
参考量。这让环境轻量（无球碰撞）、可大规模并行、且训练信号干净。

## 3. MDP 设计

| 维度 | 设定 |
|---|---|
| 控制频率 | **100 Hz**（`decimation=2`, `sim.dt=0.005`, `step_dt=0.01`），对齐真机控制环；Catch 仍 50 Hz |
| Episode 长度 | `0.72 s`（≈ `MAX_PREP_S 0.6` + `POST_MARGIN_S 0.12`，约 72 步） |
| 动作 | 右臂关节增量目标，`action_scale=0.06`（100 Hz 下从 Catch 的 0.12 减半），带平滑与速度限幅 |
| 击球面 | `HIT_PLANE_X = -1.37`（= `SAC_ROBOT_X`） |
| 目标落点 | 对方台面中心 `(OPP_TABLE_CENTER_X≈0.685, 0, TABLE_Z=0.76)` |

**观测（actor / 可部署集合）**：右臂关节角、关节增量历史、噪声参考指令
`[p_ref, v_ref, n_ref, tau]`(10维)、FK 拍心位置/法向、拍心位置误差、上一步动作。
**critic 额外特权观测**：关节速度、拍的速度/角速度/坐标轴、**clean 参考指令 + tau_true**、拍心速度误差。

**奖励（核心三项 + 平滑正则）**：

- `hit_ref_pos`（weight 20）、`hit_ref_vel`（weight 40）：时间门控高斯跟踪
  `gate = exp(-0.5 (tau_true/σ_t)²)`，`score = exp(-||e||² / 2σ²)`，
  其中 `σ_t=0.03`、`σ_p=0.05`、`σ_v=0.4`。只有逼近击球时刻、误差才计入，
  避免策略提前/事后"蹭分"。
- `hit_ref_normal`（weight 12）：同样时间门控的拍面法向角度高斯，
  `score = exp(-angle(n_racket,n_ref)^2 / 2σ_normal^2)`，当前 `σ_normal=20 deg`。
- 平滑正则（从 Catch env_cfg 原样照搬）：`action_rate`、`joint_acc`、`joint_jerk`、
  `joint_limit_margin`、`joint_effort_margin` —— 服务 sim-to-real。

**终止**：超时、关节 NaN、`hit_window_elapsed`（越过击球步 + post-margin 即结束）。

**成功判据**：击球步上 clean 参考的位置误差 `<0.05 m` 且速度误差 `<0.2 m/s`。

## 4. 双轨参考量：clean vs noisy

每个 episode 维护两套参考流，对应"可部署"与"评测/批评"两种视角：

- **noisy（actor 可见、可部署）**：模拟真机 KF 对来球的预测 —— 越早预测越不准，
  随 `tau` 衰减的偏置 + 抖动注入（`phase_scaled_ball_noise`）。策略只能看到它，
  因此学到的是"在不确定预测下鲁棒跟踪"。
- **clean（critic / 成功判定专用）**：来球的真实穿越状态，无噪声。用于特权 critic 观测
  与成功判据，给训练提供无偏的价值信号。

二者都喂给**同一个 memoryless 的解析规划器** `plan_hit_reference` 生成参考位姿/速度。
全部坐标在 **env-local 系**（world − `env.scene.env_origins`），规避历史上的 frame-flip bug。

## 5. 三段式课程（curriculum）

参考流的来源可切换，支持由易到难：

1. **合成盒采样**：在击球面附近的可达盒 `HITTRACK_BOX` 内随机采来球状态，
   `tau_initial = MAX_PREP_S`，无噪声 —— 学基本跟踪。
2. **合成 + 噪声**：同上但叠加相位缩放的 KF 噪声 —— 学鲁棒性。
3. **真机烘焙**（默认关闭，`HITTRACK_USE_BAKED=False`）：从真实 120 Hz 录制日志离线
   烘焙出每个发球的参考流，reset 时随机抽一条真实发球 —— 缩小 sim-to-real gap。

三段共用同一套 reset/update 与规划代码路径，只是参考量来源不同。

## 6. 代码结构

HitTrack **自成独立 task 包** `a1_pingpong_hittrack`：所有 HitTrack 专属代码集中于此，
对 Catch(`table_tennis_sac`)只做 **import 复用**（场景/机器人常量/通用 MDP 项/击球物理
`hitting.py`），不改、不删 Catch 任何代码。

```
tasks/a1_pingpong_hittrack/           # 独立 task 包（HitTrack 专属）
├── env_cfg.py                    # HitTrackEnvCfg / HitTrackPlayEnvCfg（100 Hz）
├── bake_hittrack_references.py   # 真机日志 → 100 Hz 参考流（纯 numpy，离线）
├── hittrack_references.npz       # 烘焙输出的参考流数据
├── __init__.py                   # gym.register("A1-Pingpong-HitTrack")（含 PPO entry point）
├── agents/
│   └── rsl_rl_ppo_cfg.py         # HitTrackPPORunnerCfg（RSL-RL PPO 训练）
└── mdp/
    ├── tracking.py            # time_gate / gaussian_score / hit_track_terms（纯 torch）
    ├── reference_planner.py   # plan_hit_reference（包裹 table_tennis_sac.mdp.hitting）
    ├── reference_source.py    # 采样盒 / tau 流 / 相位噪声 / 横向增强 / 可达性（纯 torch）
    ├── reference_commands.py  # env 端 reset_reference_command / update_hit_track_state
    ├── observations.py        # hit_reference_command(_clean) / hit_ref_pos|vel_error
    ├── rewards.py             # hit_ref_pos / hit_ref_vel
    ├── terminations.py        # hit_window_elapsed
    └── __init__.py            # 聚合：复用 table_tennis_sac.mdp + 导出上述 HitTrack 模块
```

`reset_reference_command` 在 reset 时一次性把整段 noisy 流 + clean 状态喂给 memoryless
规划器算出全程参考（reset 批量 == 逐步，因规划器无状态），并播种 cursor-0；
`update_hit_track_state`（`mode="interval"`）每控制步推进 cursor、刷新当前参考，
并在击球步**惰性**取 FK 拍态来锁存成功/误差。

## 7. 与 Catch 任务的关系

- HitTrack 自成独立包 `a1_pingpong_hittrack`；仅 **import 复用**（不修改）Catch 的场景、就绪位姿、机器人摆位、物理参数、通用 MDP 项与击球物理 `hitting.py`。
- 平滑正则项原样复用 Catch 配置，保证两任务 sim-to-real 行为一致。

## 8. 现状与待办

**已完成（TDD，全绿）**：10 个任务全部实现并提交（分支 `hitter`，3 个分组 commit）。
测试 **32 passed / 4 skipped**。纯模块 + 奖励核 + gym 注册均有真实覆盖；4 个 skip 是
env 端测试 —— `isaaclab` 依赖 USD(`pxr`)，仅存在于 Isaac Sim 运行时。

**未做 / 受阻**：

- **Isaac GPU 冒烟测试**：按要求跳过；env 配置目前只经 `py_compile` + 注册校验，
  完整构建/rollout 待在 Isaac 内实跑验证。
- **真机数据烘焙**：受阻于尚未录制的新格式 120 Hz 日志（mocap + 部署 KF + KF 预测）。
  烘焙脚本与其单测不依赖该数据；合成课程 (1)/(2) 无需它即可训练。
- **训练器选择**（SAC vs `rsl_rl` PPO）按 spec 暂缓；环境与训练器无关。
- 环境中无 `ruff`，isaaclab 相关文件以 `py_compile` 替代静态检查。

## 9. 关键超参一览

| 参数 | 值 | 含义 |
|---|---|---|
| `MAX_PREP_S` / `POST_MARGIN_S` | 0.6 / 0.12 s | 准备时长 / 击球后余量 |
| `STEP_DT` | 0.01 s | 100 Hz 控制步 |
| `σ_t` / `σ_p` / `σ_v` | 0.03 / 0.05 / 0.4 | 时间门控 / 位置 / 速度高斯宽度 |
| `σ_normal` | 20 deg | 拍面法向角度高斯宽度 |
| `W_POS` / `W_VEL` / `W_NORMAL` | 20 / 40 / 12 | 位置 / 速度 / 拍面法向奖励权重 |
| `SUCCESS_POS` / `SUCCESS_VEL` | 0.05 m / 0.2 m/s | 成功判据阈值 |
| `REACH_Y` / `REACH_Z` | (−0.2,0.2) / (0.7,1.5) | 可达工作空间门 |
| `action_scale` | 0.10 | 关节增量动作尺度（100 Hz） |
