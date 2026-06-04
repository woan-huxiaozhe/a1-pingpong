# A1 乒乓球反手 · 环境配置 + 专家模仿 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 A1 右臂乒乓球任务改造成 sim-to-real 导向的**反手**变体 `A1-TableTennis-Backhand`:100Hz 控制、真机标定的发球分布 + 空气阻力、子步级动作/球观测延迟、真机示教反手专家轨迹。

**Architecture:** 新建独立配置目录 `robots/a1/backhand/`(不动 forehand)。两条**离线 numpy 管线**生成产物(`backhand_ref.npz` 专家轨迹、`serve_states.npz` 可达发球状态表),一个**共享物理模块** `mdp/ball_physics.py`(二次阻力 + 弹跳,离线与 sim 共用,保证一致),以及对 `events.py / actions.py / delayed_env.py` 的子步级机制改造。

**Tech Stack:** Isaac Lab `ManagerBasedRLEnv` + rsl_rl PPO;Python/numpy/scipy(离线,base python 可跑)、torch(sim,需 conda 环境)。规范来自 `docs/superpowers/specs/2026-06-04-tabletennis-backhand-env-imitation-design.md`。

---

## 测试与运行约定(全程适用)

- **离线 numpy 脚本/纯函数** → 真 TDD,用 base python:`python3 -m pytest tests/table_tennis/<file>.py -v`(base python 有 numpy/scipy)。
- **torch/sim 集成代码**(events/actions/delayed_env/env_cfg)→ base python 无 torch,**在项目 conda 环境**跑:`python scripts/rsl_rl/...`(见各任务 Verify)。这些用"跑 sim 观察输出"验证,不强求 pytest。
- 数据源(只读,勿改):
  - 球轨迹 mocap:`/home/woan/kalman_filter_pingpong/data/0602/vrpn_pose_ball_*.txt`
  - 反手示教:`/home/woan/robotbase_gripper/tmp/pingpong_logs/2026_06_04/pingpong_log_1.csv`
- 提交粒度:每个 Task 末尾 commit。**commit message 先贴草稿给用户确认再执行**(项目 commit 协议)。计划中的 `git commit` 步骤=生成草稿并请确认。

---

## Phase A — 反手骨架 + 100Hz 控制

### Task 1: 新建 backhand 配置目录 + 注册任务 + 100Hz

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/backhand/__init__.py`
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/backhand/env_cfg.py`(本任务先整体复制 forehand,只改 decimation/interval;后续任务再逐步替换发球/延迟/motion)
- Reference: `robots/a1/forehand/__init__.py`、`robots/a1/forehand/env_cfg.py`、`agents/rsl_rl_ppo_cfg.py`

- [ ] **Step 1: 复制 forehand 配置为 backhand 起点**

```bash
cd source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1
cp forehand/env_cfg.py backhand/env_cfg.py
cp forehand/__init__.py backhand/__init__.py
```

- [ ] **Step 2: 改 `backhand/__init__.py` 注册新 task id**

把 forehand 的注册改成(只改 id 字段,entry 指向本目录的 `env_cfg`):

```python
import gymnasium as gym

gym.register(
    id="A1-TableTennis-Backhand",
    entry_point="unitree_rl_lab.tasks.table_tennis.delayed_env:DelayedObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.table_tennis.agents.rsl_rl_ppo_cfg:A1TableTennisPPORunnerCfg",
    },
)
```

- [ ] **Step 3: 确认 backhand 目录被父级 import**

`robots/a1/__init__.py` 当前只 import forehand。追加 backhand 子模块导入,使 gym.register 生效。查看现有内容:

Run: `cat source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/__init__.py`

按其现有写法(可能是 `from . import forehand`)追加一行 `from . import backhand`(若用了别的自动发现机制则遵循原样,不要新造模式)。

- [ ] **Step 4: 在 `backhand/env_cfg.py` 的 `RobotEnvCfg.__post_init__` 设 100Hz**

定位 `__post_init__`(forehand 里是 `self.decimation = 4`),改为:

```python
        self.decimation = 2          # 100Hz 控制 (was 4 -> 50Hz); 物理保持 sim.dt=0.005 (200Hz)
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
```

(其余 `__post_init__` 内容暂保持不变,后续任务再改。)

- [ ] **Step 5: 把三个 interval 事件同步到 100Hz(每控制步=0.01s)**

在 `backhand/env_cfg.py` 的 `EventCfg` 里,把 `relaunch_ball` 和 `track_hit` 的 `interval_range_s=(0.02, 0.02)` 改为 `(0.01, 0.01)`。(`apply_air_drag` 在 Task 5 新增时直接用 0.01。)

- [ ] **Step 6: 验证任务注册 + 环境可构建 + step_dt 正确**

Run（conda 环境）: `python scripts/rsl_rl/ref_generate/list_envs.py 2>&1 | grep -i backhand`
Expected: 列出 `A1-TableTennis-Backhand`。

Run: `python scripts/rsl_rl/train.py --task A1-TableTennis-Backhand --num_envs 8 --max_iterations 1 --headless 2>&1 | tail -30`
Expected: 环境成功构建并完成 1 次迭代(此刻仍用 forehand 的发球/motion,只验证骨架+100Hz 不报错)。日志中 `decimation=2`、physics dt 0.005。

- [ ] **Step 7: Commit**

```bash
git add source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/backhand/ source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/__init__.py
git commit  # 先贴草稿确认: "新增 A1 反手任务骨架并切到 100Hz 控制"
```

---

## Phase B — 共享球物理模块(纯函数,可测)

### Task 2: `ball_physics.py` — 二次阻力 + 弹跳前向滚动

**Files:**
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/mdp/ball_physics.py`
- Test: `tests/table_tennis/test_ball_physics.py`

设计:离线(numpy)与 sim(torch)共用同一物理常量与阻力公式,保证 serve_states 生成时的滚动与 sim 内动力学一致(spec A2/A5 要求)。阻力公式对 np.ndarray / torch.Tensor 都适用(只用 `*`、`sum`、`sqrt`);弹跳滚动仅离线用 numpy。

- [ ] **Step 1: 写失败测试**

```python
# tests/table_tennis/test_ball_physics.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
    "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/mdp"))
import ball_physics as bp

def test_drag_accel_opposes_velocity_and_quadratic():
    v = np.array([[4.0, 0.0, 0.0]])
    a = bp.drag_accel(v, k=0.125)
    # a = -k|v|v ; |v|=4 -> a_x = -0.125*4*4 = -2.0
    assert np.allclose(a, [[-2.0, 0.0, 0.0]], atol=1e-6)
    # 反向: 速度翻倍 -> 阻力加速度 4x (二次)
    a2 = bp.drag_accel(2 * v, k=0.125)
    assert np.allclose(a2[0, 0] / a[0, 0], 4.0, atol=1e-6)

def test_rollout_decelerates_and_bounces():
    # 从出生面 x=+0.35, z=1.23(sim系), vx=-4.0 滚到击球面 x=-1.37
    state = np.array([0.35, 0.0, 1.23, -4.0, 0.0, -0.1])  # x,y,z,vx,vy,vz
    hit = bp.rollout_to_plane(state, target_x=-1.37, k=0.125,
                              table_z=0.76, ch=0.85, cv=0.90, gravity=9.81, dt=0.001)
    assert hit is not None
    # 到达击球面时纵深速度应衰减(|vx| < 4.0, 且与实测 2.0~3.4 量级相符)
    assert 1.5 < abs(hit["vx"]) < 4.0
    assert hit["x"] == -1.37 or abs(hit["pos"][0] - (-1.37)) < 1e-2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/table_tennis/test_ball_physics.py -v`
Expected: FAIL（`No module named 'ball_physics'`）。

- [ ] **Step 3: 实现 `ball_physics.py`**

```python
"""共享乒乓球物理: 二次空气阻力 + 桌面弹跳. 离线(numpy)与 sim(torch)共用阻力公式.

a_drag = -k * |v| * v   (k≈0.125 SI, 由真机标定; 见 spec §2.1)
弹跳: 触桌(z<=table_z 且 vz<0)时 vz -> -cv*vz, 水平 (vx,vy) -> ch*(vx,vy).
"""
from __future__ import annotations

import numpy as np

DRAG_K = 0.125          # SI, 标定值
TABLE_Z = 0.76          # sim 球桌面高度
CH = 0.85               # 弹跳水平保持系数 (kalman 标定)
CV = 0.90               # 弹跳垂直恢复系数
GRAVITY = 9.81


def drag_accel(vel, k: float = DRAG_K):
    """二次空气阻力加速度 a=-k|v|v. vel: (...,3) np.ndarray 或 torch.Tensor."""
    if isinstance(vel, np.ndarray):
        speed = np.sqrt((vel * vel).sum(axis=-1, keepdims=True))
    else:  # torch.Tensor
        speed = (vel * vel).sum(dim=-1, keepdim=True).sqrt()
    return -k * speed * vel


def rollout_to_plane(state, target_x: float, k: float = DRAG_K,
                     table_z: float = TABLE_Z, ch: float = CH, cv: float = CV,
                     gravity: float = GRAVITY, dt: float = 0.001,
                     max_t: float = 3.0):
    """numpy 前向积分单球, 返回首次穿越 x=target_x 时的状态 dict, 否则 None.

    state: [x,y,z,vx,vy,vz] (sim 世界系, m / m/s). 球从 +x 向 -x 运动 (target_x<出生x).
    """
    p = np.array(state[:3], dtype=float)
    v = np.array(state[3:6], dtype=float)
    g = np.array([0.0, 0.0, -gravity])
    n = int(max_t / dt)
    for _ in range(n):
        a = drag_accel(v[None, :], k)[0] + g
        v_new = v + a * dt
        p_new = p + v_new * dt
        # 弹跳: 穿过桌面且下行
        if p_new[2] <= table_z and v_new[2] < 0.0:
            v_new[2] = -cv * v_new[2]
            v_new[0] *= ch
            v_new[1] *= ch
            p_new[2] = table_z
        # 穿越目标平面 (从 x>target 到 x<=target)
        if p[0] > target_x >= p_new[0]:
            alpha = (p[0] - target_x) / (p[0] - p_new[0] + 1e-12)
            pc = p + alpha * (p_new - p)
            vc = v + alpha * (v_new - v)
            return {"pos": pc, "vel": vc, "x": target_x,
                    "y": float(pc[1]), "z": float(pc[2]),
                    "vx": float(vc[0]), "vy": float(vc[1]), "vz": float(vc[2])}
        p, v = p_new, v_new
    return None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/table_tennis/test_ball_physics.py -v`
Expected: PASS（2 passed）。

- [ ] **Step 5: Commit**

```bash
git add source/.../table_tennis/mdp/ball_physics.py tests/table_tennis/test_ball_physics.py
git commit  # 草稿: "新增共享球物理模块(二次阻力+弹跳滚动)"
```

---

## Phase C — 离线产物生成器(纯 numpy,可测)

### Task 3: `create_backhand_ref.py` — 真机示教 → 单条规范反手 npz

**Files:**
- Create: `source/unitree_rl_lab/.../robots/a1/backhand/create_backhand_ref.py`
- Create (产物): `robots/a1/backhand/backhand_ref.npz`
- Test: `tests/table_tennis/test_backhand_ref.py`

要点(spec B1/B5):取 `pingpong_log_1.csv` swing 1–12 的 `q_plan_0..6`;按阶段(to_hit→cruise→follow-through)时间归一化对齐 → 重采样 → 相位对齐平均成挥拍段;前后补 ready-hold 成完整周期;FPS=100;输出 `fps/upper_body_dof(N,7)/base_y/joint_names`;记录 cruise 归一化区间(写进 npz 便于 Task 10 设 hit_phase)。

- [ ] **Step 1: 写失败测试**

```python
# tests/table_tennis/test_backhand_ref.py
import numpy as np, os, subprocess, sys

REF = os.path.join(os.path.dirname(__file__), "..", "..",
    "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/backhand/backhand_ref.npz")
GEN = os.path.join(os.path.dirname(REF), "create_backhand_ref.py")

A1_LIMITS = np.array([[-1.04,3.14],[-3.14,0.26],[-2.758,2.758],[-1.92,1.92],
                      [-2.758,2.758],[-1.57,1.57],[-2.758,2.758]])

def test_generate_and_validate():
    subprocess.run([sys.executable, GEN], check=True)
    d = np.load(REF, allow_pickle=True)
    dof = d["upper_body_dof"]
    assert dof.ndim == 2 and dof.shape[1] == 7
    assert float(d["fps"]) == 100.0
    assert dof.shape[0] >= 40                      # 完整周期 (含 ready-hold)
    # 守 USD 限位
    for j in range(7):
        assert dof[:, j].min() >= A1_LIMITS[j,0] - 1e-3
        assert dof[:, j].max() <= A1_LIMITS[j,1] + 1e-3
    # 关节名
    names = [str(n) for n in d["joint_names"]]
    assert names == [f"joint_yb_{i}" for i in range(1, 8)]
    # cruise 区间已记录且在 (0,1)
    cl, cr = float(d["cruise_phase_lo"]), float(d["cruise_phase_hi"])
    assert 0.0 < cl < cr < 1.0
    # 首尾为 ready-hold(近似静止)
    assert np.allclose(dof[0], dof[1], atol=1e-3)
    assert np.allclose(dof[-1], dof[-2], atol=1e-3)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/table_tennis/test_backhand_ref.py -v`
Expected: FAIL（找不到 create_backhand_ref.py / npz）。

- [ ] **Step 3: 实现 `create_backhand_ref.py`**

```python
"""真机反手示教 -> 单条规范反手参考轨迹 (spec B1/B5).

源: /home/woan/robotbase_gripper/tmp/pingpong_logs/2026_06_04/pingpong_log_1.csv
取 swing 1-12 的 q_plan_0..6, 阶段对齐 (to_hit->cruise->follow-through), 重采样到统一
长度后相位对齐平均成挥拍段, 前后补 ready-hold 成完整周期, FPS=100 写 npz.
关节 1:1 映射 q_plan_{i} -> joint_yb_{i+1} (同一 A1 右臂, 无翻转; 见 spec §2.2).
"""
import os
import numpy as np

CSV = "/home/woan/robotbase_gripper/tmp/pingpong_logs/2026_06_04/pingpong_log_1.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backhand_ref.npz")
FPS = 100
JOINT_NAMES = np.array([f"joint_yb_{i}" for i in range(1, 8)])
A1_LIMITS = np.array([[-1.04,3.14],[-3.14,0.26],[-2.758,2.758],[-1.92,1.92],
                      [-2.758,2.758],[-1.57,1.57],[-2.758,2.758]])
SWING_LEN = 40          # 重采样后每条挥拍段统一帧数 (~0.4s @100Hz)
READY_HOLD = 20         # 首尾各补的 ready-hold 帧数 (Task 10 据 arrive_time 再调)


def load_csv(path):
    header = None
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        if header is None:
            header = line.strip().split(",")
            continue
        rows.append(line.strip().split(","))
    col = {n: i for i, n in enumerate(header)}
    return rows, col


def extract_swings(rows, col):
    """返回 {swing_id: {"q":(F,7), "stage":[...]}} for swing_id 1..12."""
    out = {}
    for r in rows:
        sid = int(r[col["swing_id"]])
        if sid < 1:                # swing 0 = 待机, 跳过
            continue
        q = [float(r[col[f"q_plan_{j}"]]) for j in range(7)]
        out.setdefault(sid, {"q": [], "stage": []})
        out[sid]["q"].append(q)
        out[sid]["stage"].append(r[col["stage"]])
    for sid in out:
        out[sid]["q"] = np.array(out[sid]["q"], dtype=np.float32)
    return out


def resample(arr, n):
    """线性重采样 (F,7) -> (n,7)."""
    F = arr.shape[0]
    src = np.linspace(0, 1, F)
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(7)], axis=1).astype(np.float32)


def main():
    rows, col = load_csv(CSV)
    swings = extract_swings(rows, col)

    # 每条挥拍段重采样到 SWING_LEN, 记录 cruise 在 [0,1] 的归一化区间
    resampled = []
    cruise_los, cruise_his = [], []
    for sid, sw in sorted(swings.items()):
        q = sw["q"]
        F = q.shape[0]
        stage = np.array(sw["stage"])
        cruise_idx = np.where(stage == "cruise")[0]
        if len(cruise_idx) == 0:
            continue
        cruise_los.append(cruise_idx[0] / (F - 1))
        cruise_his.append(cruise_idx[-1] / (F - 1))
        resampled.append(resample(q, SWING_LEN))
    R = np.stack(resampled, axis=0)            # (S, SWING_LEN, 7)
    swing_avg = R.mean(axis=0)                 # (SWING_LEN, 7) 相位对齐平均

    # cruise 归一化区间(挥拍段内)
    cruise_lo_swing = float(np.mean(cruise_los))
    cruise_hi_swing = float(np.mean(cruise_his))

    # 前后补 ready-hold: ready 位 = 挥拍段首帧 (to_hit 起始姿态, 平滑可待机)
    ready = swing_avg[0:1]
    hold_pre = np.repeat(ready, READY_HOLD, axis=0)
    hold_post = np.repeat(swing_avg[-1:], READY_HOLD, axis=0)
    dof = np.concatenate([hold_pre, swing_avg, hold_post], axis=0).astype(np.float32)

    # cruise 在完整周期里的归一化区间
    total = dof.shape[0]
    cl = (READY_HOLD + cruise_lo_swing * (SWING_LEN - 1)) / (total - 1)
    ch = (READY_HOLD + cruise_hi_swing * (SWING_LEN - 1)) / (total - 1)

    # 守限位(平均可能微越界 -> clip)
    for j in range(7):
        dof[:, j] = np.clip(dof[:, j], A1_LIMITS[j, 0], A1_LIMITS[j, 1])

    base_y = np.zeros(total, dtype=np.float32)
    np.savez(OUT, fps=np.float64(FPS), upper_body_dof=dof, base_y=base_y,
             joint_names=JOINT_NAMES,
             cruise_phase_lo=np.float64(cl), cruise_phase_hi=np.float64(ch))
    print(f"[backhand_ref] {OUT}  frames={total}  cruise_phase=[{cl:.3f},{ch:.3f}]  "
          f"from {len(resampled)} swings")
    for j in range(7):
        print(f"  yb{j+1}: [{dof[:,j].min():+.3f}, {dof[:,j].max():+.3f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/table_tennis/test_backhand_ref.py -v`
Expected: PASS。同时 `python3 source/.../backhand/create_backhand_ref.py` 打印每关节范围与 cruise_phase。

- [ ] **Step 5: Commit**

```bash
git add source/.../backhand/create_backhand_ref.py source/.../backhand/backhand_ref.npz tests/table_tennis/test_backhand_ref.py
git commit  # 草稿: "新增反手参考生成脚本+产物(真机示教单条规范轨迹)"
```

---

### Task 4: `create_serve_states.py` — mocap → 可达发球状态表 npz

**Files:**
- Create: `source/.../robots/a1/backhand/create_serve_states.py`
- Create (产物): `robots/a1/backhand/serve_states.npz`
- Test: `tests/table_tennis/test_serve_states.py`

要点(spec A2):在 x=+0.35 提取真机净速 5D 状态 → 用 x=-1.37 可达框预筛 → 拟合多元高斯 → 采样 N → 每维裁剪到实测 min/max → 用 `ball_physics.rollout_to_plane` 滚动到 x=-1.37 → 可达框过滤 → 写有效 launch 状态 `(M,6)=x,y,z,vx,vy,vz`。可达框:`y∈[-0.20,0.30]`、`h=z-0.76∈[0,0.70]`。

- [ ] **Step 1: 写失败测试**

```python
# tests/table_tennis/test_serve_states.py
import numpy as np, os, subprocess, sys

BASE = os.path.join(os.path.dirname(__file__), "..", "..",
    "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/backhand")
GEN = os.path.join(BASE, "create_serve_states.py")
OUT = os.path.join(BASE, "serve_states.npz")

def test_generate_reachable_serves():
    subprocess.run([sys.executable, GEN, "--n", "2000"], check=True)
    d = np.load(OUT)
    s = d["states"]
    assert s.ndim == 2 and s.shape[1] == 6          # x,y,z,vx,vy,vz
    assert s.shape[0] >= 500                         # 足够多有效发球
    # 出生面 x ≈ +0.35
    assert np.allclose(s[:, 0], 0.35, atol=1e-6)
    # vx 朝 -x (负) 且在实测裁剪范围内
    assert (s[:, 3] < 0).all()
    assert s[:, 3].min() >= -4.92 and s[:, 3].max() <= -1.78
    # z 在实测出生高度范围
    assert s[:, 2].min() >= 0.84 and s[:, 2].max() <= 1.55
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/table_tennis/test_serve_states.py -v`
Expected: FAIL（脚本/产物不存在）。

- [ ] **Step 3: 实现 `create_serve_states.py`**

```python
"""真机 mocap -> 可达发球状态表 (spec A2, 方案②).

流程: 0602 轨迹在 x=+0.35 提取净速 5D 状态(y,z_sim,vx,vy,vz) -> x=-1.37 可达框预筛
 -> 拟合单个多元高斯 N(mu,Sigma) -> 采样 -> 每维裁剪到实测 min/max -> 用标定物理
(ball_physics)滚动到 x=-1.37 -> 可达框过滤 -> 写有效 launch 状态 (M,6).
"""
import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "mdp"))
import ball_physics as bp

DATA_DIR = "/home/woan/kalman_filter_pingpong/data/0602"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_states.npz")
SPAWN_X = 0.35
HIT_X = -1.37
TABLE_Z_DATA = 0.046
Z_OFFSET = bp.TABLE_Z - TABLE_Z_DATA          # data_z -> sim_z
Y_MIN, Y_MAX = -0.20, 0.30                    # 可达框 (右臂反手)
H_MIN, H_MAX = 0.0, 0.70


def load_vrpn(path):
    rows = []
    for raw in open(path, errors="ignore"):
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        p = t.split()
        if len(p) < 4:
            continue
        try:
            row = [float(p[i]) for i in range(4)]
        except ValueError:
            continue
        if np.isfinite(row).all():
            rows.append(row)
    a = np.array(rows, float)
    return a[:, 0], a[:, 1:4]


def select_segment(t, p, tx, max_gap=0.20):
    if len(t) <= 1:
        return t, p
    cuts = [0]
    for i in range(1, len(t)):
        if (t[i] - t[i-1]) <= 0 or (t[i] - t[i-1]) > max_gap:
            cuts.append(i)
    cuts.append(len(t))
    segs = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1]-cuts[i] >= 2]
    if not segs:
        return t, p
    cr = [(s, e) for (s, e) in segs if np.nanmin(p[s:e, 0]) <= tx <= np.nanmax(p[s:e, 0])]
    s, e = max(cr or segs, key=lambda it: it[1]-it[0])
    return t[s:e], p[s:e]


def crossing(t, p, tx, win=0.06):
    x = p[:, 0]
    for i in range(1, len(x)):
        x0, x1 = float(x[i-1]), float(x[i])
        if x0 >= tx >= x1 and not np.isclose(x0, x1):
            a = float(np.clip((tx-x0)/(x1-x0), 0, 1))
            tc = (1-a)*t[i-1] + a*t[i]
            pc = (1-a)*p[i-1] + a*p[i]
            m = (t >= tc-win) & (t <= tc+1e-9)
            tt, pp = t[m], p[m]
            if len(tt) < 3:
                m = (t >= tc-2*win) & (t <= tc+1e-9); tt, pp = t[m], p[m]
                if len(tt) < 3:
                    return None
            deg = min(2, len(tt)-1); tau = tt - tc; v = np.empty(3)
            for ax in range(3):
                c = np.polyfit(tau, pp[:, ax], deg); v[ax] = np.polyval(np.polyder(c), 0.0)
            return pc, v
    return None


def reachable(y, z_sim):
    return (Y_MIN <= y <= Y_MAX) and (H_MIN <= (z_sim - bp.TABLE_Z) <= H_MAX)


def collect_net_states():
    states = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "vrpn_pose_ball_*.txt"))):
        t, p = load_vrpn(path)
        ht, hp = select_segment(t, p, HIT_X)
        hit = crossing(ht, hp, HIT_X)
        if hit is None:
            continue
        hy, hz = hit[0][1], hit[0][2] + Z_OFFSET
        if not reachable(hy, hz):
            continue
        nt, npp = select_segment(t, p, SPAWN_X)
        net = crossing(nt, npp, SPAWN_X)
        if net is None:
            continue
        pc, v = net
        states.append([pc[1], pc[2] + Z_OFFSET, v[0], v[1], v[2]])  # y,z_sim,vx,vy,vz
    return np.array(states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="高斯采样候选数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    X = collect_net_states()              # (n0,5) y,z,vx,vy,vz
    print(f"valid-reachable real net states: {len(X)}")
    mu = X.mean(0)
    cov = np.cov(X.T)
    lo, hi = X.min(0), X.max(0)

    cand = rng.multivariate_normal(mu, cov, size=args.n)
    cand = np.clip(cand, lo, hi)           # 每维裁剪到实测范围

    out = []
    for y, z, vx, vy, vz in cand:
        if vx >= 0:                        # 必须朝 -x 飞
            continue
        hit = bp.rollout_to_plane([SPAWN_X, y, z, vx, vy, vz], target_x=HIT_X)
        if hit is None:
            continue
        if reachable(hit["y"], hit["z"]):
            out.append([SPAWN_X, y, z, vx, vy, vz])
    S = np.array(out, dtype=np.float32)
    np.savez(OUT, states=S, mu=mu, cov=cov, lo=lo, hi=hi)
    print(f"[serve_states] {OUT}  kept {len(S)}/{args.n}  "
          f"vx[{S[:,3].min():.2f},{S[:,3].max():.2f}]  z[{S[:,2].min():.2f},{S[:,2].max():.2f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/table_tennis/test_serve_states.py -v`
Expected: PASS。脚本打印有效真机净速条数(~83)与保留发球数。

- [ ] **Step 5: 生成正式产物(默认 N)并 Commit**

```bash
python3 source/.../backhand/create_serve_states.py --n 5000
git add source/.../backhand/create_serve_states.py source/.../backhand/serve_states.npz tests/table_tennis/test_serve_states.py
git commit  # 草稿: "新增可达发球状态表生成脚本+产物(高斯采样+物理滚动+可达过滤)"
```

---

## Phase D — sim 机制改造(torch / 需 conda 环境验证)

### Task 5: 空气阻力事件 `apply_air_drag`

**Files:**
- Modify: `source/.../table_tennis/mdp/events.py`(新增函数)
- Modify: `source/.../table_tennis/mdp/__init__.py`(导出,若该文件用显式导出)
- Modify: `robots/a1/backhand/env_cfg.py`(`EventCfg` 注册)

- [ ] **Step 1: 在 `events.py` 新增 `apply_air_drag`**

```python
from unitree_rl_lab.tasks.table_tennis.mdp import ball_physics as bp


def apply_air_drag(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    k: float = 0.125,
):
    """每控制步给球施加二次空气阻力外力 F = m * (-k|v|v). 见 spec A1 / ball_physics."""
    ball: RigidObject = env.scene[ball_cfg.name]
    vel = ball.data.root_lin_vel_w                      # (N,3)
    accel = bp.drag_accel(vel, k)                       # torch, (N,3)
    mass = ball.data.default_mass.to(env.device).sum(dim=-1, keepdim=True)  # (N,1)
    force = (mass * accel).unsqueeze(1)                 # (N,1,3) 单 body
    torque = torch.zeros_like(force)
    ball.set_external_force_and_torque(force, torque)
    ball.write_data_to_sim()
```

> 注:`default_mass` 形状/API 以本仓库 Isaac Lab 版本为准——若 `default_mass` 不可用,用常量球质量 `0.0027`。Step 3 验证时确认数值正确。

- [ ] **Step 2: 在 backhand `EventCfg` 注册阻力事件**

```python
    apply_air_drag = EventTerm(
        func=mdp.apply_air_drag,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={"ball_cfg": SceneEntityCfg("ball"), "k": 0.125},
    )
```

- [ ] **Step 3: 验证阻力使球减速(单球零控制 rollout)**

Run（conda）: `python scripts/rsl_rl/play_pure_ref.py --task A1-TableTennis-Backhand --npz <abs>/backhand_ref.npz --num_envs 1 --video --video_length 200 2>&1 | tail -20`

Expected: 不报错;在日志/视频里球从网附近到击球面纵深速度明显衰减(对照 spec:|vx| 网 ~4.0 → 击球面 ~2.5,衰减约 1.5 m/s)。若衰减明显偏离,核对 `k`、质量与 `set_external_force_and_torque` 形状。

- [ ] **Step 4: Commit**

```bash
git add source/.../mdp/events.py source/.../mdp/__init__.py source/.../backhand/env_cfg.py
git commit  # 草稿: "给 sim 球加二次空气阻力事件(100Hz)"
```

---

### Task 6: 发球从 `serve_states.npz` 采样

**Files:**
- Modify: `source/.../table_tennis/mdp/events.py`(`launch_ball` 支持状态表;`relaunch_ball_if_out` 同步)
- Modify: `robots/a1/backhand/env_cfg.py`(`reset_ball`/`relaunch_ball` 传入表路径)

- [ ] **Step 1: `launch_ball` 增加"状态表采样"分支**

在 `launch_ball` 顶部加参数 `serve_states_path: str | None = None` 与一个模块级缓存,优先用表采样(否则保留原逐维均匀采样,向后兼容 forehand):

```python
_SERVE_TABLE_CACHE: dict[str, torch.Tensor] = {}

def _load_serve_table(path: str, device) -> torch.Tensor:
    if path not in _SERVE_TABLE_CACHE:
        import numpy as np
        s = np.load(path)["states"]
        _SERVE_TABLE_CACHE[path] = torch.tensor(s, dtype=torch.float32, device=device)
    return _SERVE_TABLE_CACHE[path]
```

在 `launch_ball` 内,采样位置/速度处改为:

```python
    if serve_states_path is not None:
        table = _load_serve_table(serve_states_path, env.device)   # (M,6) x,y,z,vx,vy,vz
        idx = torch.randint(0, table.shape[0], (num,), device=env.device)
        sample = table[idx]
        pos = sample[:, :3].clone()
        pos += env.scene.env_origins[env_ids]
        vel = sample[:, 3:6].clone()
    else:
        # 原逐维均匀采样 (保持不变)
        ...
```

(`quat`、`ang_vel`、`write_root_state_to_sim` 保持原样。)

- [ ] **Step 2: `relaunch_ball_if_out` 传递同一表**

`relaunch_ball_if_out` 调用 `launch_ball(...)` 处把 `serve_states_path` 透传(给它加同名参数,默认 None)。其后的相位重对齐逻辑保持不变(Task 10 会回填 `ball_arrive_time_est`)。

- [ ] **Step 3: backhand `EventCfg` 传入表路径**

```python
    SERVE_STATES = os.path.join(os.path.dirname(__file__), "serve_states.npz")
    reset_ball = EventTerm(func=mdp.launch_ball, mode="reset",
        params={"ball_cfg": SceneEntityCfg("ball"), "serve_states_path": SERVE_STATES})
    relaunch_ball = EventTerm(func=mdp.relaunch_ball_if_out, mode="interval",
        interval_range_s=(0.01, 0.01),
        params={"ball_cfg": SceneEntityCfg("ball"), "serve_states_path": SERVE_STATES})
```

(移除原 `**MIDDLE_BALL` 展开;`MIDDLE_*` 常量可留作注释参考。)

- [ ] **Step 4: 验证发球分布 + 到达速度**

Run（conda）: `python scripts/rsl_rl/play_pure_ref.py --task A1-TableTennis-Backhand --npz <abs>/backhand_ref.npz --num_envs 1 --video --video_length 400 2>&1 | tail -20`

Expected: 球在 x≈+0.35 以 |vx|≈3.3–4.7 出生,经阻力+弹跳后到达击球区 |v|≈2.0–3.4,落点都在可达框内(目视:球都飞到机器人可够到的范围)。

- [ ] **Step 5: Commit**

```bash
git add source/.../mdp/events.py source/.../backhand/env_cfg.py
git commit  # 草稿: "发球改为从可达发球状态表采样(替换逐维均匀)"
```

---

### Task 7: 子步级动作延迟

**Files:**
- Modify: `source/.../table_tennis/mdp/actions.py`(`ReferenceResidualJointAction` + `Cfg`)
- Test: `tests/table_tennis/test_delay_index.py`(纯索引逻辑,可 base python 测)
- Modify: `robots/a1/backhand/env_cfg.py`(`ActionsCfg.right_arm` 参数)

要点(spec A3):`apply_actions()` 每物理子步(5ms)被调用;维护子步级 ring buffer,延迟 `uniform{1,2,3}子步={5,10,15}ms`。把延迟从 `process_actions`(控制步)移到 `apply_actions`(子步)。

- [ ] **Step 1: 写失败测试(纯延迟索引逻辑)**

把"环形缓冲取延迟项"的索引计算抽成可单测的纯函数 `delayed_index(step_counter, delay, buf_len)`。

```python
# tests/table_tennis/test_delay_index.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
    "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/mdp"))
from delay_utils import delayed_index

def test_delay_zero_returns_latest():
    # buffer 写入位置 = step % buf_len; delay=0 -> 取最新
    assert delayed_index(write_pos=3, delay=0, buf_len=4) == 3

def test_delay_wraps():
    assert delayed_index(write_pos=0, delay=1, buf_len=4) == 3
    assert delayed_index(write_pos=1, delay=3, buf_len=4) == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest tests/table_tennis/test_delay_index.py -v`
Expected: FAIL（`No module named 'delay_utils'`）。

- [ ] **Step 3: 新建 `mdp/delay_utils.py`**

```python
"""子步延迟环形缓冲的纯索引逻辑 (可单测, 不依赖 torch)."""

def delayed_index(write_pos: int, delay: int, buf_len: int) -> int:
    """最近写入位置 write_pos, 取 delay 子步前的项的环形下标."""
    return (write_pos - delay) % buf_len
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest tests/table_tennis/test_delay_index.py -v`
Expected: PASS。

- [ ] **Step 5: 改 `ReferenceResidualJointAction` 用子步延迟**

在 `__init__` 用 `cfg.action_delay_substeps_max` 建子步缓冲与 per-env 延迟(reset 时采样 `randint(min,max+1)`);`process_actions` 只算 `target = ref + raw*scale` 存为"最新目标";新增/改 `apply_actions` 每子步写入环形缓冲并按 `delayed_index` 取延迟目标下发:

```python
    def apply_actions(self):
        if self._max_delay_sub > 0:
            self._buf[:, self._write_pos] = self._latest_target
            from .delay_utils import delayed_index
            # per-env 不同 delay: 用 gather 取各自延迟项
            read = (self._write_pos - self._delay_sub) % self._buf.shape[1]   # (N,)
            idx = read.view(-1, 1, 1).expand(-1, 1, self._num_joints)
            out = self._buf.gather(1, idx).squeeze(1)
            self._write_pos = (self._write_pos + 1) % self._buf.shape[1]
        else:
            out = self._latest_target
        self._robot.set_joint_position_target(out, joint_ids=self._joint_ids)
```

(具体张量初始化:`_buf` 形状 `(N, max_sub+1, num_joints)`;`_delay_sub` `(N,)` long;`_latest_target` 在 `process_actions` 末尾赋值;`reset` 清零并重采样 `_delay_sub`。保留无延迟分支兼容 forehand。)`Cfg` 增加 `action_delay_substeps_min/max`。

- [ ] **Step 6: backhand `ActionsCfg` 设延迟 1~3 子步**

```python
    right_arm = mdp.ReferenceResidualJointActionCfg(
        asset_name="robot", joint_names=RIGHT_ARM_JOINT_NAMES, command_name="motion",
        residual_scale=[0.05]*7,
        action_delay_substeps_min=1, action_delay_substeps_max=3,   # {5,10,15}ms @5ms子步
    )
```

- [ ] **Step 7: 验证(sim 跑通 + 延迟生效)**

Run（conda）: `python scripts/rsl_rl/train.py --task A1-TableTennis-Backhand --num_envs 32 --max_iterations 2 --headless 2>&1 | tail -20`
Expected: 正常构建/迭代,无形状错误。

- [ ] **Step 8: Commit**

```bash
git add source/.../mdp/actions.py source/.../mdp/delay_utils.py tests/table_tennis/test_delay_index.py source/.../backhand/env_cfg.py
git commit  # 草稿: "动作延迟改为子步级 ring buffer(10±5ms)"
```

---

### Task 8: 子步级"球派生观测"延迟

**Files:**
- Modify: `source/.../table_tennis/delayed_env.py`
- Modify: `source/.../table_tennis/mdp/observations.py`(球观测项读延迟球态)
- Modify: `robots/a1/backhand/env_cfg.py`(若需开关)

要点(spec A4):只延迟球派生观测(`ball_pos_relative/ball_vel_relative/ball_spin/ball_time_to_arrive/ball_predicted_hit/ball_bounce`),子步 5ms 粒度,延迟 `{5,10,15}ms`;关节/相位/racket/last_action 不延迟。实现:env 每子步把球 root pos/vel 存进环形缓冲,obs 计算前把"延迟后球态"暴露为 `env._delayed_ball_pos_w / _delayed_ball_vel_w`;球观测函数优先读它。

- [ ] **Step 1: `DelayedObsEnv` 维护子步球态缓冲**

由于 `apply_action()` 每子步被调用,而 env 无现成子步钩子,这里在 `DelayedObsEnv` 重写 `step` 的子步循环外不可行;改用**最简健壮方案**:复用 Task 7 的子步节拍——在球阻力事件已是每控制步,不足子步。**采用动作项搭载捕获**:在 `ReferenceResidualJointAction.apply_actions`(每子步)末尾追加一行,把球 root state 推进 env 持有的缓冲。

在 `DelayedObsEnv.__init__` 建缓冲:

```python
        self._ball_delay_max = getattr(cfg, "ball_obs_delay_substeps_max", 0)
        if self._ball_delay_max > 0:
            self._ball_buf_pos = torch.zeros(self.num_envs, self._ball_delay_max + 1, 3, device=self.device)
            self._ball_buf_vel = torch.zeros(self.num_envs, self._ball_delay_max + 1, 3, device=self.device)
            self._ball_write_pos = 0
            self._ball_delay = torch.randint(getattr(cfg, "ball_obs_delay_substeps_min", 1),
                self._ball_delay_max + 1, (self.num_envs,), device=self.device)
            self._delayed_ball_pos_w = None
            self._delayed_ball_vel_w = None
```

新增 `push_ball_substep()`(动作项每子步调用)与 `refresh_delayed_ball()`(每控制步 obs 前调用):

```python
    def push_ball_substep(self):
        if self._ball_delay_max <= 0: return
        ball = self.scene["ball"]
        self._ball_buf_pos[:, self._ball_write_pos] = ball.data.root_pos_w
        self._ball_buf_vel[:, self._ball_write_pos] = ball.data.root_lin_vel_w
        self._ball_write_pos = (self._ball_write_pos + 1) % self._ball_buf_pos.shape[1]

    def refresh_delayed_ball(self):
        if self._ball_delay_max <= 0: return
        read = (self._ball_write_pos - 1 - self._ball_delay) % self._ball_buf_pos.shape[1]
        idx = read.view(-1, 1, 1)
        self._delayed_ball_pos_w = self._ball_buf_pos.gather(1, idx.expand(-1,1,3)).squeeze(1)
        self._delayed_ball_vel_w = self._ball_buf_vel.gather(1, idx.expand(-1,1,3)).squeeze(1)
```

在 `step()` 内 `super().step()` 之后、返回 obs 之前调用 `refresh_delayed_ball()`(注意 obs 已在 super().step 内算过——因此改为在 super().step 调用前不行;见 Step 2 的顺序处理)。

- [ ] **Step 2: 处理"obs 在 super().step 内计算"的顺序问题**

ManagerBasedRLEnv 在 `step()` 内部算 obs。为让 obs 读到延迟球态,需在 obs 计算前刷新。最稳妥:重写 `DelayedObsEnv` 不依赖时序——让**球观测函数自己**按需取延迟项,即 `refresh_delayed_ball()` 由观测函数首次访问时惰性触发一次/控制步。实现:给观测函数加一个 helper:

在 `observations.py` 顶部:

```python
def _ball_pos_w(env, ball):
    dp = getattr(env, "_delayed_ball_pos_w", None)
    return dp if dp is not None else ball.data.root_pos_w

def _ball_vel_w(env, ball):
    dv = getattr(env, "_delayed_ball_vel_w", None)
    return dv if dv is not None else ball.data.root_lin_vel_w
```

并在 `DelayedObsEnv.step()` 中,在 `super().step()` **之前**调用 `refresh_delayed_ball()`(用上一控制步末尾子步缓冲的快照——即 obs 用的是"截至上一步末的延迟球态",符合观测滞后语义)。

- [ ] **Step 3: 球派生观测函数改读延迟球态**

把 `observations.py` 中这些函数内 `ball.data.root_pos_w` / `root_lin_vel_w` 替换为 `_ball_pos_w(env, ball)` / `_ball_vel_w(env, ball)`:`ball_pos_relative`、`ball_vel_relative`、`ball_time_to_arrive`、`ball_predicted_hit_point`、`ball_bounce_state`(`ball_spin_zero` 返回零无需改)。**不改**任何关节/相位/racket 函数。

- [ ] **Step 4: 动作项每子步推进球缓冲**

在 `ReferenceResidualJointAction.apply_actions` 末尾加:

```python
        push = getattr(self._env, "push_ball_substep", None)
        if push is not None:
            push()
```

- [ ] **Step 5: backhand env_cfg 开启球观测延迟**

在 `RobotEnvCfg` 加字段并在 `__post_init__` 设:`ball_obs_delay_substeps_min=1`、`ball_obs_delay_substeps_max=3`。`RobotPlayEnvCfg` 里设 0(回放无延迟)。

- [ ] **Step 6: 验证(sim 跑通 + 仅球观测延迟)**

Run（conda）: `python scripts/rsl_rl/train.py --task A1-TableTennis-Backhand --num_envs 32 --max_iterations 2 --headless 2>&1 | tail -20`
Expected: 正常迭代无错。临时打印:延迟球态与实时球态在球运动时存在 1~3 子步差(可在 `refresh_delayed_ball` 末加一次性 debug print 比较 norm,验证后删)。

- [ ] **Step 7: Commit**

```bash
git add source/.../delayed_env.py source/.../mdp/observations.py source/.../mdp/actions.py source/.../backhand/env_cfg.py
git commit  # 草稿: "球派生观测加子步级延迟(10±5ms),非球观测不延迟"
```

---

## Phase E — 接线 + 相位接地 + 端到端

### Task 9: 接线 backhand env_cfg(motion + PPO cfg)

**Files:**
- Modify: `robots/a1/backhand/env_cfg.py`(`CommandsCfg.motion`)
- Modify: `agents/rsl_rl_ppo_cfg.py`(新增 backhand runner cfg)
- Modify: `robots/a1/backhand/__init__.py`(指向新 runner cfg)

- [ ] **Step 1: `CommandsCfg.motion` 指向反手参考**

```python
    motion = mdp.UpperBodyMotionCommandCfg(
        asset_name="robot",
        motion_files=[os.path.join(os.path.dirname(__file__), "backhand_ref.npz")],
        resampling_time_range=(1.0e9, 1.0e9),
        base_y_noise_range=(0.0, 0.0), fixed_base=True,
        hit_phase=0.45,                 # 占位; Task 10 用 cruise_phase 回填
        hit_phase_noise=0.0,
        ball_arrive_time_est=0.51,      # 占位; Task 10 sim 实测回填
        ball_arrive_time_noise=0.0,
        match_ball_direction=False,     # 单条规范, 不按方向选
        robot_x=ROBOT_X, robot_side=ROBOT_SIDE, axis_flip_indices=None,
    )
```

- [ ] **Step 2: 新增 `A1BackhandTableTennisPPORunnerCfg`(gamma/num_steps)**

在 `agents/rsl_rl_ppo_cfg.py` 追加:

```python
@configclass
class A1BackhandTableTennisPPORunnerCfg(A1TableTennisPPORunnerCfg):
    """100Hz 反手: gamma/num_steps_per_env 按控制率翻倍调整 (spec A0)."""
    experiment_name = "a1_tabletennis_backhand"
    num_steps_per_env = 48          # 24->48 保持 rollout 时长 ~0.48s @100Hz
    def __post_init__(self):
        self.algorithm.gamma = 0.995  # 0.99->0.995 保持时域折扣 ~2s @100Hz
```

> 若 `RslRlPpoAlgorithmCfg` 不支持 `__post_init__` 改字段,改为直接复制 algorithm 块并设 `gamma=0.995`。

- [ ] **Step 3: `backhand/__init__.py` 指向新 runner cfg**

把 `rsl_rl_cfg_entry_point` 改为 `...agents.rsl_rl_ppo_cfg:A1BackhandTableTennisPPORunnerCfg`。

- [ ] **Step 4: 验证构建**

Run（conda）: `python scripts/rsl_rl/train.py --task A1-TableTennis-Backhand --num_envs 16 --max_iterations 1 --headless 2>&1 | tail -20`
Expected: 用反手 motion 构建成功,日志 `gamma=0.995`、`num_steps_per_env=48`。

- [ ] **Step 5: Commit**

```bash
git add source/.../backhand/env_cfg.py source/.../backhand/__init__.py source/.../agents/rsl_rl_ppo_cfg.py
git commit  # 草稿: "接线反手 motion 与 100Hz PPO 配置"
```

---

### Task 10: 相位接地(`hit_phase` + `ball_arrive_time_est` 实测回填)

**Files:**
- Create: `scripts/rsl_rl/ref_generate/measure_backhand_arrive.py`(测量脚本,仿 `ref_generate` 现有诊断脚本风格)
- Modify: `robots/a1/backhand/env_cfg.py`(回填两个数)

- [ ] **Step 1: `hit_phase` 用 cruise_phase 中点**

读 `backhand_ref.npz` 的 `cruise_phase_lo/hi`(Task 3 已写入),取接触点(中点或 hi)设 `hit_phase`:

```bash
python3 -c "import numpy as np; d=np.load('source/.../backhand/backhand_ref.npz'); print('cruise', float(d['cruise_phase_lo']), float(d['cruise_phase_hi']))"
```

把 `CommandsCfg.motion.hit_phase` 设为 `round((lo+hi)/2, 3)`(或 hi,按接触语义)。

- [ ] **Step 2: 写测量脚本测 `ball_arrive_time_est`**

新建 `measure_backhand_arrive.py`,**复制 `scripts/rsl_rl/play_pure_ref.py` 的 env 构建/AppLauncher 样板**(同样的 `--task` 加载、`gym.make`、`env.reset()`),改成测量逻辑:
- `num_envs=256`,加载 `A1-TableTennis-Backhand`(play cfg)。
- `env.reset()` 后循环若干步,**动作恒为 0**(`torch.zeros(num_envs, action_dim)`),纯惯性让发球飞向机器人。
- 每步读 `ball_x_local = ball.data.root_pos_w[:,0] - env.scene.env_origins[:,0]`;对每个尚未记录的 env,当 `ball_x_local` 首次穿越 `ROBOT_X=-1.5`(由 + 向 - 穿过)时,记录当前 `step * step_dt` 为到达时间。
- 跑满后 `print` 到达时间的中位数与 p25/p75。
参考同目录 `play_reference_motion.py` / `v85_track_hit_debug.py` 的 env 步进写法,勿新造 launcher 模式。

- [ ] **Step 3: 跑测量并回填**

Run（conda）: `python scripts/rsl_rl/ref_generate/measure_backhand_arrive.py 2>&1 | tail -5`
Expected: 打印到达时间中位数(秒)。把 `CommandsCfg.motion.ball_arrive_time_est` 设为该中位数。

- [ ] **Step 4: 验证相位对齐(纯参考回放)**

Run（conda）: `python scripts/rsl_rl/play_pure_ref.py --task A1-TableTennis-Backhand --npz <abs>/backhand_ref.npz --num_envs 1 --video --video_length 400 --output_dir logs/pure_ref/backhand 2>&1 | tail -20`
Expected: 视频里球到达击球区时,机械臂正处于 cruise(挥拍接触)段;`initial_phase = hit_phase - arrive_time/duration` 落在 [0,1) 无 wrap(可在 commands 临时打印验证)。若挥拍过早/过晚,微调 hit_phase 或 READY_HOLD(回到 Task 3 调 padding)。

- [ ] **Step 5: Commit**

```bash
git add scripts/rsl_rl/ref_generate/measure_backhand_arrive.py source/.../backhand/env_cfg.py
git commit  # 草稿: "相位接地: cruise->hit_phase, sim实测回填 ball_arrive_time_est"
```

---

### Task 11: 端到端冒烟 + 球桌弹跳系数验证

**Files:**
- Modify(按需): `robots/a1/backhand/env_cfg.py` 或球桌 USD 材质(仅当到达速度偏离时)

- [ ] **Step 1: 球桌弹跳系数验证**

用 Task 10 的测量脚本或临时打印,统计球到达 x=-1.37 的 `|v|` 分布,确认落在 **2.0–3.4**(spec 实测)。若系统性偏快/偏慢:
- 偏快 → sim 球桌水平摩擦不足(Ch>0.85) 或阻力 k 偏小;偏慢反之。
- 微调:优先调 `apply_air_drag` 的 `k`(±20% 内),其次核对球桌 USD `restitution`(目标 Cv≈0.90)。记录最终值到 spec/记忆。

- [ ] **Step 2: 短训冒烟**

Run（conda, 后台）: `python scripts/rsl_rl/train.py --task A1-TableTennis-Backhand --num_envs 1024 --max_iterations 100 --headless 2>&1 | tee logs/backhand_smoke.log`
Expected: 无 NaN/崩溃;`reset`/`relaunch` 正常;100 iter 后 `Mean reward` 有上升趋势(冒烟,不要求收敛)。

- [ ] **Step 3: 核对验证清单(spec §5)**

逐项确认:① serve_states 全可达;② 阻力衰减 ~1.5m/s;③ 动作/球观测延迟 {5,10,15}ms 且非球观测无延迟;④ 反手回放姿态合理、相位推进/打完冻结正常;⑤ arrive_time 回填后接触对齐;⑥ 端到端冒烟通过。

- [ ] **Step 4: Commit + 更新记忆**

```bash
git add -A
git commit  # 草稿: "反手100Hz环境冒烟通过+球桌系数标定收尾"
```

更新 `~/.claude/projects/-data-PPO-pingpong/memory/` 的 scope 记忆:记录最终 `k`、`hit_phase`、`ball_arrive_time_est`、到达速度分布、冒烟结论。

---

## 实现顺序与依赖
- 强制顺序:Task 1 → 2 → 3/4(可并行,都依赖 2) → 5 → 6(依赖 4) → 7 → 8(依赖 7 的子步钩子) → 9 → 10(依赖 3 的 cruise_phase + 6 的发球) → 11。
- Task 2/3/4 为纯 numpy,可在 base python TDD;5–11 需 conda(torch/isaaclab)。

## 范围外(后续轮次,见 spec §7)
- 奖励(reward)针对反手+新发球的复核与权重调整。
- 多轨迹 `match_ball_direction`(需给 12 swing 标注落点)。
- 更多 DR(PD/力矩/观测噪声、球桌物理)。
