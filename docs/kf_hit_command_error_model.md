# KF 击球点预测误差建模（sim `estimated_hit_command`）

> **目的**：在仿真里把"策略真机部署时看到的 KF 击球点预测误差"复现出来，注入到观测项
> `estimated_hit_command`。本文件自包含——按 §6 的流程可从原始动捕数据一步步复现出 §4 的
> 全部参数，并接进 §7 的代码。
>
> 相关文件：
> - 误差来源滤波器：`kalman_filter_pingpong/src/kalman.py`（部署 KF）
> - 发球门控：`kalman_filter_pingpong/docs/serve_onset_reset_gating.md`
> - sim 实现：`source/unitree_rl_lab/.../table_tennis_sac/mdp/observations.py`、`env_cfg.py`

---

## 0. 直接复现：函数 + 参数

> 把下面的函数和参数原样拿走即可复现当前误差建模。原理/标定/复现细节见 §1 之后。
> 依赖：`hit_command_at_robot_x`（干净预测，下附）→ `ball_predicted_hit_point`（纯解析抛体预测，
> 位于 `tasks/table_tennis/mdp/observations.py`）。误差只加在 **actor/policy 观测组**，critic 用干净的
> `groundtruth_hit_command`。击球平面 `robot_x = -1.47 m`。

**参数**（`(y, z, tau)`，单位 `m / m / s`）：

| 参数 | y | z | tau |
|------|------|------|------|
| `bias_std_far`（per-episode 偏置 std @ far） | 0.024 | 0.022 | 0.016 |
| `jitter_std`（per-step 抖动 std） | 0.003 | 0.008 | 0.003 |
| `fixed_offset_far`（系统偏置 @ far） | 0.0 | −0.010 | −0.005 |
| `far_tau`（该轴 phase 饱和 horizon, s） | 0.65 | 0.40 | 0.55 |

```python
# env_cfg.py
SAC_HIT_COMMAND_NOISE = {
    "bias_std_far": (0.024, 0.022, 0.016),
    "jitter_std": (0.003, 0.008, 0.003),
    "fixed_offset_far": (0.0, -0.010, -0.005),
    "far_tau": (0.65, 0.40, 0.55),
}
# ObsTerm（ActorCfg）
estimated_hit_command = ObsTerm(
    func=mdp.estimated_hit_command_at_robot_x,
    params={"ball_name": "ball", "robot_x": SAC_ROBOT_X, "robot_side": ROBOT_SIDE,
            **SAC_HIT_COMMAND_NOISE},
)
```

```python
# mdp/observations.py
def hit_command_at_robot_x(env, ball_name: str, robot_x: float, robot_side: int) -> torch.Tensor:
    """Clean simulator hit command: [p_hit_x, p_hit_y, p_hit_z, tau]."""
    hit = ball_predicted_hit_point(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)
    x = torch.full((hit.shape[0], 1), robot_x, device=hit.device, dtype=hit.dtype)
    return torch.cat([x, hit[:, 0:2], hit[:, 2:3]], dim=-1)


def estimated_hit_command_at_robot_x(
    env,
    ball_name: str,
    robot_x: float,
    robot_side: int,
    bias_std_far: tuple[float, float, float] = (0.024, 0.022, 0.016),
    jitter_std: tuple[float, float, float] = (0.003, 0.008, 0.003),
    fixed_offset_far: tuple[float, float, float] = (0.0, -0.010, -0.005),
    far_tau: tuple[float, float, float] = (0.65, 0.40, 0.55),
    y_abs_limit: float = 2.0,
    z_min: float = 0.45,
    z_max: float = 2.0,
) -> torch.Tensor:
    """Deployment-style hit command with KF-like prediction error.

    Calibrated from the deployment Kalman (kalman_filter_pingpong) open-loop hit-point
    residuals on the 0617 mocap serves. The real KF error is dominated by a *per-serve
    consistent bias* (random direction each serve, set by that ball's spin/launch that the
    non-Magnus KF mispredicts), NOT per-step white noise -- so a per-step Gaussian would be
    averaged out by the policy. Three components, each scaled by the per-axis horizon phase
    tau/far_tau so the command converges to truth as the ball arrives:
      - bias  : per-episode unit normal (y,z,tau) drawn once per ball reset, * bias_std_far.
      - jitter: per-step white noise, jitter_std (small).
      - offset: fixed systematic offset fixed_offset_far shared by all serves (mainly z ~ -1cm).
    far_tau is per-axis: z saturates early (~0.40s), y ~linear to ~0.65s, tau plateaus (~0.55s).
    The strike-plane x stays fixed at robot_x; error is applied to predicted y/z/tau.
    Cached per policy step so the actor and critic groups see the same command in one step.
    """
    clean = hit_command_at_robot_x(env, ball_name=ball_name, robot_x=robot_x, robot_side=robot_side)
    num = clean.shape[0]
    device = clean.device
    step = env.episode_length_buf.to(torch.long)

    # --- per-episode bias unit vector (y, z, tau), redrawn at episode reset ---
    bias_attr = "_sac_hit_bias_unit"
    bias_step_attr = "_sac_hit_bias_step"
    if not hasattr(env, bias_attr):
        setattr(env, bias_attr, torch.randn(num, 3, device=device))
        setattr(env, bias_step_attr, torch.full((num,), -1, dtype=torch.long, device=device))
    bias_unit = getattr(env, bias_attr)
    bias_last = getattr(env, bias_step_attr)
    reset = (bias_last < 0) | (step == 0) | (step < bias_last)
    if torch.any(reset):
        bias_unit[reset] = torch.randn(int(reset.sum()), 3, device=device)
        bias_last[reset] = step[reset]

    # --- per-step cached estimate (actor/critic consistency within a step) ---
    attr = "_sac_estimated_hit_command"
    step_attr = "_sac_estimated_hit_command_step"
    if not hasattr(env, attr):
        setattr(env, attr, clean.clone())
        setattr(env, step_attr, torch.full((num,), -1, dtype=torch.long, device=device))

    estimate = getattr(env, attr)
    last_step = getattr(env, step_attr)
    refresh = step != last_step
    if torch.any(refresh):
        next_estimate = clean.clone()
        tau = clean[:, 3:4].clamp(min=0.0)  # [N, 1]

        far = torch.tensor(far_tau, device=device, dtype=clean.dtype).clamp(min=1.0e-6)  # [3]
        phase = (tau / far.unsqueeze(0)).clamp(min=0.0, max=1.0)  # [N, 3], per-axis horizon scaling

        bias_std = torch.tensor(bias_std_far, device=device, dtype=clean.dtype).unsqueeze(0)  # [1, 3]
        jit_std = torch.tensor(jitter_std, device=device, dtype=clean.dtype).unsqueeze(0)
        offset = torch.tensor(fixed_offset_far, device=device, dtype=clean.dtype).unsqueeze(0)

        # err on (y, z, tau): bias keeps its per-episode direction, magnitude scales with phase
        err = bias_unit * (bias_std * phase) + torch.randn(num, 3, device=device) * jit_std + offset * phase

        next_estimate[:, 1:3] += err[:, 0:2]  # y, z
        next_estimate[:, 3:4] += err[:, 2:3]  # tau
        next_estimate[:, 1:2] = next_estimate[:, 1:2].clamp(-y_abs_limit, y_abs_limit)
        next_estimate[:, 2:3] = next_estimate[:, 2:3].clamp(z_min, z_max)
        next_estimate[:, 3:4] = next_estimate[:, 3:4].clamp(0.0, 3.0)

        estimate[refresh] = next_estimate[refresh]
        last_step[refresh] = step[refresh]

    return estimate
```

clamp：`y∈[-2,2]`、`z∈[0.45,2.0]`、`tau∈[0,3]`。`estimated = clean + err`，`err` 只作用于 `(y,z,tau)`，
`x` 固定为击球平面不加误差。`tau→0` 时 `phase→0`，偏置与系统 offset 收敛到 0（指令收敛真值）。

---

## 1. 为什么要建这个误差

- **sim 的球是干净的**：`launch_ball` 发射时 `ang_vel=0`，SAC 未注册空气阻力，球只受重力 + PhysX
  弹跳；`ball_predicted_hit_point` 是纯解析抛体。所以 **sim 的"干净预测"≈ 真实未来落点**。
- **真机不是**：策略部署时看到的是 **KF 的预测**，而 KF 误差来自对真实球（阻力、旋转/Magnus、
  弹跳旋转耦合）的**模型失配**。
- **路线 B（本方案）**：不在 sim 里跑 KF（把真机 KF 套到干净球上会近乎完美、复现不出误差——
  误差的瓶颈是球物理而非滤波器），而是**保留 sim 干净预测 + 注入按真机 KF 残差标定的误差**。
- 只注入 actor 组；critic 仍用 `groundtruth_hit_command`（干净），保持值函数训练稳定。

---

## 2. 误差的真实结构（关键发现，决定模型形态）

用部署 KF（叠加发球门控，见 §6.1）在 0617 动捕发球上做**开环击球点预测残差**统计，得到：

1. **误差由 per-episode 一致偏置主导，不是每步白噪声。** 决策段（horizon 0.2–0.5s）误差方差里
   **73%–98% 是"每球一致的偏置"**（同一个球连续帧的预测误差方向/大小高度相关），每步白抖动只占很小一部分。
   - 物理原因：KF 关闭了飞行 Magnus（`magnus_coeff=0`），**每个球的旋转把预测往一个固定方向带偏**，
     整段飞行一致。
   - **建模含义**：必须每球抽一次偏置（球内固定方向），不能用每步重抽的白高斯——否则策略会把噪声
     低通滤掉，真机上却面对一个滤不掉的持续偏置而崩。

2. **偏置不是单一固定方向。** 决策段每球平均偏置：y 总体均值≈0、正负球数对半（→ 随机方向）；
   z 有 −9mm 的固定系统分量 + 随机分量；tau 类似。所以模型 = **每球随机方向偏置 `N(0,σ)` + 小固定 offset（主要 z）**。

3. **误差随 horizon 增长、tau→0 收敛**（越接近击球预测越准），且三轴 horizon profile 不同
   （z 早饱和、y 近线性、tau 中段平后段跳）→ 用**逐轴** `far_tau`。

---

## 3. 模型定义

记干净预测 `clean = [x_robot, y*, z*, tau*]`。每个 env：

- **per-episode 偏置**：每次球 reset（`episode_length_buf==0`）抽 `b = (b_y,b_z,b_tau) ~ N(0,I)`，整段固定。
- 每个 policy step（带 per-step 缓存，保证同一 step 内 actor/critic 取到同一值）：
  ```
  phase = clip(tau* / far_tau, 0, 1)                      # 逐轴, shape (3,)
  err_y   = b_y   * bias_std_far_y   * phase_y + N(0, jitter_y) + offset_y * phase_y
  err_z   = b_z   * bias_std_far_z   * phase_z + N(0, jitter_z) + offset_z * phase_z
  err_tau = b_tau * bias_std_far_tau * phase_tau + N(0, jitter_tau) + offset_tau * phase_tau
  estimated = clean + [0, err_y, err_z, err_tau]，再做 clamp
  ```
- `tau→0` 时 `phase→0`，偏置与系统 offset 都收敛到 0（指令收敛到真值），抖动保留——与真 KF 行为一致。

---

## 4. 标定参数（即 §0 的表，附依据）

| 轴 | `bias_std_far` | `jitter_std` | `fixed_offset_far` | `far_tau` | profile 依据 |
|----|------|------|------|------|------|
| y | 0.024 | 0.003 | 0.0 | 0.65 | 近线性增长到 ~0.65s，无系统方向 |
| z | 0.022 | 0.008 | −0.010 | 0.40 | ~0.4s 即饱和到 ~22mm；KF 系统性把击球高度预测偏低 ~1cm |
| tau | 0.016 | 0.003 | −0.005 | 0.55 | 中段平台 ~13–16ms；小负系统偏置（略预测早到） |

---

## 5. 标定数据（0617 残差实测，复现目标）

数据：`kalman_filter_pingpong/data/0617_traj_data`（53 个 120Hz VRPN 文件，52 个有效发球，5100 个
`(horizon, 误差)` 样本）。门控 KF。击球平面 `x=-1.47m`。

**总体 mean/std（按 horizon 分箱；y/z 单位 mm，tau 单位 ms）：**

| horizon | n | err_y mean/std | err_z mean/std | err_tau mean/std |
|---------|---|------|------|------|
| 0.1–0.2s | 615 | −0.8 / 3.9 | +1.6 / 8.6 | +0.3 / 4.0 |
| 0.2–0.3s | 616 | +0.5 / 11.2 | −5.2 / 15.4 | −1.1 / 11.0 |
| 0.3–0.4s | 620 | +0.6 / 13.5 | −7.8 / 22.3 | −4.4 / 13.5 |
| 0.4–0.5s | 617 | −1.1 / 14.8 | −10.0 / 23.6 | −7.6 / 13.1 |
| 0.5–0.6s | 620 | −3.2 / 18.3 | −10.1 / 21.9 | −6.2 / 16.9 |
| 0.6–0.7s | 612 | −4.6 / 24.9 | −9.6 / 25.2 | +3.1 / 29.0 |

**bias/jitter 分解（`bias_frac` = per-episode 偏置方差占比）：**

| horizon | y | z | tau |
|---------|------|------|------|
| 0.2–0.3s | 0.92 | 0.73 | 0.90 |
| 0.3–0.4s | 0.97 | 0.73 | 0.96 |
| 0.4–0.5s | 0.98 | 0.89 | 0.98 |
| 0.5–0.6s | 0.98 | 0.84 | 0.81 |

**每球偏置方向（horizon 0.30–0.55s 段）：**

| 轴 | 总体均值 | 每球偏置 std | 正/负球数 |
|----|------|------|------|
| y | −0.7mm | 14.1mm | 24 / 28（对半→随机方向） |
| z | −9.2mm | 19.7mm | 20 / 32（偏负→含固定分量） |
| tau | −6.4ms | 12.6ms | 19 / 33 |

---

## 6. 复现流程（从原始数据到参数）

### 6.1 发球门控（前置）

部署 KF 持球期 `P_vv` 坍缩会导致发球后速度估计滞后 ~58ms、预测全错，必须先门控修复。
见 `kalman_filter_pingpong/docs/serve_onset_reset_gating.md`，实现 `src/serve_gated_kalman.py`
（`ServeGatedKalman`：单帧差分 `|vx|>2.0m/s` 连续 2 帧 → 整体 re-init）。残差提取**必须用门控 KF**，
否则发球后大 horizon 段会被未门控的垃圾预测污染。

### 6.2 残差提取

方法：复用部署侧开环误差方法论——在线跑门控 KF、每帧存快照，从每个 TRACKING 帧开环
predict-only 前滚到 `x=-1.47` 平面得到预测击球点，与 mocap 真实穿越点比，得 `err_y/err_z/err_tau`。
触发覆盖每个 TRACKING 帧 → 稠密 `(horizon, 误差)`。

```python
# extract_residuals.py（要点）
import glob, os, csv, numpy as np
from pathlib import Path
import sys; sys.path.insert(0, "kalman_filter_pingpong"); sys.path.insert(0, "kalman_filter_pingpong/src")
from src.kalman import KalmanConfig
from dataprocess.interactive_kalman_visualizer import (
    load_vrpn_file, select_best_continuous_segment, clone_kf, find_time_for_real_x)
from serve_gated_kalman import ServeGatedKalman

ROBOT_X = -1.47; DT_ROLL = 0.005; T_MAX = 1.5
def roll_to_plane(snap, cfg, t0):
    kf = clone_kf(snap, cfg); prev = kf.get_position()/1000; pt = t0; t = t0
    while t < t0 + T_MAX:
        t += DT_ROLL; kf.predict(t); cur = kf.get_position()/1000
        if prev[0] >= ROBOT_X and cur[0] < ROBOT_X and cur[0] != prev[0]:
            a = (ROBOT_X-prev[0])/(cur[0]-prev[0])
            return prev[1]+a*(cur[1]-prev[1]), prev[2]+a*(cur[2]-prev[2]), pt+a*(t-pt)
        prev, pt = cur, t
    return None

cfg = KalmanConfig(); rows = []
for f in sorted(glob.glob("kalman_filter_pingpong/data/0617_traj_data/vrpn_pose_ball_*.txt")):
    sid = int(f.split("_")[-1].split(".")[0])
    t_abs, real_m = load_vrpn_file(Path(f))
    t_abs, real_m, _ = select_best_continuous_segment(t_abs, real_m, preferred_x_m=ROBOT_X)
    if len(t_abs) < 10: continue
    t_rel = t_abs - float(t_abs[0]); real_mm = real_m*1000
    t_target = find_time_for_real_x(t_rel, real_m, ROBOT_X)
    if t_target is None: continue
    # 真实穿越点 (y*,z*)
    xi = real_m[:,0]; cross = None
    for i in range(1, len(xi)):
        if (xi[i-1]-ROBOT_X)*(xi[i]-ROBOT_X) <= 0 and xi[i] != xi[i-1]:
            a = (ROBOT_X-xi[i-1])/(xi[i]-xi[i-1])
            cross = (real_m[i-1,1]+a*(real_m[i,1]-real_m[i-1,1]),
                     real_m[i-1,2]+a*(real_m[i,2]-real_m[i-1,2])); break
    if cross is None: continue
    ry, rz = cross
    trk = ServeGatedKalman(cfg); snaps, gates = [], []
    for i in range(len(t_rel)):
        trk.step(real_mm[i], float(t_rel[i]))
        snaps.append(clone_kf(trk.kf, cfg) if trk.kf.initialized else None); gates.append(trk.state)
    for j in range(len(t_rel)):
        if gates[j] != "TRACKING" or snaps[j] is None: continue
        h = t_target - float(t_rel[j])
        if h <= 0: continue
        pr = roll_to_plane(snaps[j], cfg, float(t_rel[j]))
        if pr is None: continue
        yp, zp, tc = pr
        rows.append((sid, h, yp-ry, zp-rz, (tc-float(t_rel[j]))-h))
# 落盘 CSV: serve_id, horizon_s, err_y_m, err_z_m, err_tau_s
```

### 6.3 分解 + 拟合

对每个 horizon 分箱，按 `serve_id` 把误差分成"每球均值的方差（between-serve = 偏置）"与
"球内方差（within-serve = 抖动）"：
```
total_std^2 ≈ bias_std^2 + jitter_std^2
bias_std   = std_over_serves( mean_within_serve )
jitter_std = sqrt( mean_over_serves( var_within_serve ) )
bias_frac  = bias_std^2 / (bias_std^2 + jitter_std^2)
```
固定 offset = 该箱总体均值（`fixed_offset`）。`σ_bias(h)` 随 horizon 拟合，按各轴 profile 选
`far_tau`（z 早饱和→0.40；y 近线性→0.65；tau 中段平→0.55），令 `bias_std_far = σ_bias(far_tau)`。
得到 §4 参数。

### 6.4 验证

numpy 复刻 §3 模型，模拟多回合（每回合 horizon 从 ~0.72s 递减），跑与 §6.3 相同的分解，
对比 §5 实测。结果（model / real，std）：

| horizon | y | z | tau |
|---------|------|------|------|
| 0.35s | 13.8 / 13.5 | 21.4 / 22.3 | 10.9 / 13.5 |
| 0.45s | 17.4 / 14.8 | 23.5 / 23.6 | 13.8 / 13.1 |
| 0.55s | 21.1 / 18.3 | 23.6 / 21.9 | 16.1 / 16.9 |

`bias_frac` 复刻 0.75–0.99、z 系统偏置复刻 ~−10mm，决策段全面对齐。

---

## 7. sim 代码集成

函数与配置见 **§0**（可直接抄）。落地位置：

- **`mdp/observations.py`** — `estimated_hit_command_at_robot_x(...)`（§0 即其完整实现）。
  env 上的缓冲：`_sac_hit_bias_unit` / `_sac_hit_bias_step`（每球 reset 重抽偏置）、
  `_sac_estimated_hit_command` / `_sac_estimated_hit_command_step`（每 step 缓存，保证同 step 内
  actor 与 critic 取同一值）。
- **`env_cfg.py`** — `SAC_HIT_COMMAND_NOISE` 字典 + `ActorCfg.estimated_hit_command` 这个 ObsTerm。
- critic 组（`CriticCfg`）额外带干净的 `groundtruth_hit_command`（`hit_command_at_robot_x`），不受误差影响。

---

## 8. 局限与维护

- **误差全程开启**，训练 / play / replay buffer 一致——它是观测的固有定义，真机永远如此，**不设开关**。
- **tau 在 0.65s+（发球后瞬态尾部）被低估**（模型 ~16ms vs 实测 ~29ms）：该区是刚发球的最大 horizon、
  策略本不该出手，低风险。
- **z 用线性-饱和近似**：`far_tau_z=0.40` 后 phase 钳到 1，0.4s 内近似线性，足够贴合决策段。
- **何时需要重新标定**：① 部署 KF 重新调参/改结构（如开启 Magnus）；② 来球速度/旋转分布变化；
  ③ 击球平面 `robot_x` 改变。重跑 §6 即可。
- 当前未建模 #3 `ball_pos_history` 的小 robustness 噪声（独立、可选，~2–3mm）。
