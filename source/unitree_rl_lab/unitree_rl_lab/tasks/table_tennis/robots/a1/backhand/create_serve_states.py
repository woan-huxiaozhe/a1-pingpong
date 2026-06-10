"""Offline serve-state generator (plan A2).

把 0602 真机 mocap 球轨迹 (vrpn) 标定成 sim 发球状态表 serve_states.npz:

  1. 每条轨迹在出生面 x=+0.35 提取 5D 净速状态 (y, z_sim, vx, vy, vz);
     在击球面 x=-1.37 提取实测落点, 落在可达框内 → 该条有效 (预筛, 用真实数据).
  2. 对有效条目的出生状态拟合单个全协方差多元高斯 N(mu, Sigma).
  3. 采样 N 条, 每维裁剪到实测 [min,max] (防高斯尾巴), 用标定物理
     (二次阻力 k=0.125 + 己方台弹跳 Ch=0.85/Cv=0.90) 前向滚动到 x=-1.37,
     落在可达框内 → 保留.
  4. 写 serve_states.npz: states (M,6) = x,y,z,vx,vy,vz (sim 世界系, x 固定 +0.35).

坐标对齐 (data->sim): x,y 同向; z_sim = z_data + 0.714 (数据帧桌面 ~0.046m -> sim 0.76).
所有速度三轴不变 (z 仅常数平移).

provisional ball_arrive_time_est = 有效真实轨迹 x=0.35 -> x=-1.37 用时中位数,
直接喂给 env_cfg / commands (sim 内再确认见 plan A5/B3).

用法:  python3 create_serve_states.py [--data_dir DIR] [--n 5000] [--plot]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# ------------------------- 标定常量 (plan A2/§2.1) -------------------------
BIRTH_X = 0.35          # 出生面 (sim x = data x)
HIT_X = -1.37           # 击球面
Z_OFFSET = 0.714        # data z -> sim z
TABLE_TOP = 0.76        # sim 桌面顶 z
GRAVITY = 9.81
DRAG_K = 0.125          # 二次空气阻力系数 (SI), a = -k*|v|*v
BOUNCE_CH = 0.85        # 弹跳水平速度保留 (己方台摩擦)
BOUNCE_CV = 0.90        # 弹跳垂直恢复系数

# 球网 (x=0): 网顶 z 与 observations.py/rewards.py 的 net_z_top 常量一致.
# 发球从 x=+0.35 飞向 -x, 经 x=0 时球心须清网, 否则 sim 内会真撞网 (用户观察到的现象).
NET_X = 0.0
NET_Z_TOP = 0.9125      # 桌面 0.76 + 标准网高 0.1525
BALL_RADIUS = 0.02      # 乒乓球半径 ~20mm
NET_MIN_Z = NET_Z_TOP + BALL_RADIUS + 0.02   # 球心过网最低 z (半径 + 2cm 余量) = 0.9525

# 可达框 (右臂反手, plan A2): 击球面 x=-1.37 处. 用户标定窗口:
#   高度 0~60cm (桌面上方), 横向 y in [-0.20, 0.30].
REACH_Y = (-0.20, 0.30)
REACH_H = (0.0, 0.60)   # 桌面上方高度 -> z_sim in [0.76, 1.36]
REACH_Z = (TABLE_TOP + REACH_H[0], TABLE_TOP + REACH_H[1])

DEFAULT_DATA_DIR = "/home/woan/kalman_filter_pingpong/data/0602"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_states.npz")

VEL_WINDOW_S = 0.035    # 局部 deg-2 速度拟合窗口 (对齐 kalman evaluate)


def load_vrpn(path: str) -> tuple[np.ndarray, np.ndarray]:
    """读取一条 vrpn 轨迹: 返回 (t[s], pos[N,3] m). 跳过注释/非法行."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                row = [float(parts[i]) for i in range(4)]
            except ValueError:
                continue
            if np.isfinite(row).all():
                rows.append(row)
    if not rows:
        raise ValueError("no valid rows")
    arr = np.array(rows, dtype=float)
    return arr[:, 0] - arr[0, 0], arr[:, 1:4]


def state_at_plane(t: np.ndarray, pos: np.ndarray, target_x: float):
    """在 x=target_x (球朝 -x 飞) 处提取 (t_cross, y, z_data, vx, vy, vz).

    线性插值定位过面位置与时间; 局部 deg-2 多项式拟合取该处速度.
    取第一个 "向 -x 运动" 的过面点. 无则返回 None.
    """
    x = pos[:, 0]
    for i in range(len(x) - 1):
        if (x[i] - target_x) >= 0 >= (x[i + 1] - target_x):  # 自 +x 向 -x 穿过
            dx = x[i] - x[i + 1]
            a = (x[i] - target_x) / dx if abs(dx) > 1e-9 else 0.0
            t_cross = t[i] + a * (t[i + 1] - t[i])
            p_cross = pos[i] + a * (pos[i + 1] - pos[i])
            mask = np.abs(t - t_cross) <= VEL_WINDOW_S
            if mask.sum() < 5:
                lo = max(0, i - 4)
                mask = slice(lo, min(len(t), i + 5))
            tau = t[mask] - t_cross
            vel = np.empty(3)
            ok = True
            for ax in range(3):
                try:
                    c = np.polyfit(tau, pos[mask, ax], 2)
                except Exception:
                    ok = False
                    break
                vel[ax] = c[1]  # d/dtau at tau=0
            if not ok or vel[0] >= 0:  # 必须朝 -x
                continue
            return t_cross, p_cross[1], p_cross[2], vel[0], vel[1], vel[2]
    return None


def in_reach(y: float, z_sim: float) -> bool:
    return REACH_Y[0] <= y <= REACH_Y[1] and REACH_Z[0] <= z_sim <= REACH_Z[1]


def roll_to_hit(state6: np.ndarray, dt: float = 0.002, t_max: float = 2.0):
    """用标定物理 (阻力+己方台弹跳) 前向滚动到 x=HIT_X.

    返回 (reason, payload):
      ("net",  None)                       过网 x=0 时球心低于 NET_MIN_Z -> 判为撞网, 丢弃
      ("ok",   (y,z,vx,vy,vz,t))           成功穿过击球面 x=HIT_X
      ("miss", None)                       t_max 内未到击球面

    state6 = [x,y,z,vx,vy,vz] (sim 世界系). 半隐式欧拉, 桌面以上一次弹跳.
    """
    p = state6[:3].astype(float).copy()
    v = state6[3:].astype(float).copy()
    t = 0.0
    g = np.array([0.0, 0.0, -GRAVITY])
    while t < t_max:
        speed = np.linalg.norm(v)
        a = g - DRAG_K * speed * v
        v = v + a * dt
        p_new = p + v * dt
        # 过网检查: 自 +x 向 -x 穿过 x=0 时球心须清网, 否则 sim 内会真撞网 -> 丢弃该发球
        if p[0] > NET_X >= p_new[0]:
            an = (p[0] - NET_X) / (p[0] - p_new[0]) if (p[0] - p_new[0]) > 1e-9 else 0.0
            z_net = p[2] + an * (p_new[2] - p[2])
            if z_net < NET_MIN_Z:
                return "net", None
        # 己方台弹跳: 在己方台范围内 (x in (-1.37,0)) 穿过桌面且下行
        if p_new[2] <= TABLE_TOP and v[2] < 0 and (HIT_X < p_new[0] < 0.0):
            v[2] = -BOUNCE_CV * v[2]
            v[0] *= BOUNCE_CH
            v[1] *= BOUNCE_CH
            p_new[2] = TABLE_TOP
        # 穿过击球面 x=HIT_X
        if p_new[0] <= HIT_X:
            a_lerp = (p[0] - HIT_X) / (p[0] - p_new[0]) if (p[0] - p_new[0]) > 1e-9 else 0.0
            pc = p + a_lerp * (p_new - p)
            return "ok", (pc[1], pc[2], v[0], v[1], v[2], t + a_lerp * dt)
        p = p_new
        t += dt
    return "miss", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--n", type=int, default=5000, help="高斯采样数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
    print(f"[load] {len(files)} files from {args.data_dir}")

    birth_states = []      # 有效条目的出生 5D 状态 (y, z_sim, vx, vy, vz)
    arrive_times = []      # 实测 x=0.35 -> x=-1.37 用时
    hit_yz = []            # 实测击球面 (y, z_sim)
    n_birth = n_hit = 0
    for p in files:
        try:
            t, pos = load_vrpn(p)
        except Exception:
            continue
        b = state_at_plane(t, pos, BIRTH_X)
        h = state_at_plane(t, pos, HIT_X)
        if b is None:
            continue
        n_birth += 1
        tb, by, bz, bvx, bvy, bvz = b
        bz_sim = bz + Z_OFFSET
        if h is None:
            continue
        n_hit += 1
        th, hy, hz, _, _, _ = h
        hz_sim = hz + Z_OFFSET
        hit_yz.append((hy, hz_sim))
        if in_reach(hy, hz_sim) and th > tb:
            birth_states.append((by, bz_sim, bvx, bvy, bvz))
            arrive_times.append(th - tb)

    birth = np.array(birth_states)              # (K,5)
    print(f"[extract] birth-plane hits={n_birth}  hit-plane hits={n_hit}  "
          f"reachable(valid)={len(birth)}")
    if len(birth) < 5:
        raise SystemExit("too few valid trajectories; check data_dir / planes")

    mu = birth.mean(0)
    sigma = birth.std(0)
    cov = np.cov(birth, rowvar=False)
    lo = birth.min(0)
    hi = birth.max(0)
    arrive_med = float(np.median(arrive_times))
    labels = ["y", "z_sim", "vx", "vy", "vz"]
    print("[fit] single multivariate Gaussian on valid birth states:")
    for i, lb in enumerate(labels):
        print(f"   {lb:5s} mu={mu[i]:+.3f} sigma={sigma[i]:.3f} "
              f"[min {lo[i]:+.3f}, max {hi[i]:+.3f}]")
    print(f"[fit] ball_arrive_time_est (median real x=0.35->-1.37) = {arrive_med:.3f} s")

    # --- 采样 + 裁剪 + 物理滚动 + 可达过滤 ---
    rng = np.random.default_rng(args.seed)
    samp = rng.multivariate_normal(mu, cov, size=args.n)
    samp = np.clip(samp, lo, hi)              # 每维裁剪到实测范围

    kept = []
    hit_vx = []
    hit_z = []
    roll_times = []
    n_net = n_miss = n_oob = 0
    for s in samp:
        y, z_sim, vx, vy, vz = s
        state6 = np.array([BIRTH_X, y, z_sim, vx, vy, vz])
        reason, r = roll_to_hit(state6)
        if reason == "net":
            n_net += 1
            continue
        if reason != "ok":
            n_miss += 1
            continue
        hy, hz, hvx, hvy, hvz, tt = r
        if in_reach(hy, hz):
            kept.append([BIRTH_X, y, z_sim, vx, vy, vz])
            hit_vx.append(abs(hvx))
            hit_z.append(hz)
            roll_times.append(tt)
        else:
            n_oob += 1

    states = np.array(kept, dtype=np.float32)
    keep_rate = len(states) / args.n
    print(f"[roll] sampled={args.n} kept={len(states)} ({keep_rate*100:.1f}%)  "
          f"rejected: net={n_net} out-of-box@-1.37={n_oob} no-arrival={n_miss}")
    if len(hit_vx) > 0:
        hv = np.array(hit_vx)
        hzc = np.array(hit_z) - TABLE_TOP   # 到达 -1.37 时桌面上方高度 (m)
        print(f"[roll] arrival |vx| at x=-1.37: mean={hv.mean():.2f} "
              f"p5={np.percentile(hv,5):.2f} p95={np.percentile(hv,95):.2f} "
              f"(target 2.0-3.4)")
        print(f"[roll] arrival height above table: mean={hzc.mean()*100:.1f}cm "
              f"[{hzc.min()*100:.1f}, {hzc.max()*100:.1f}]cm (window 0-60cm)")
        print(f"[roll] rollout arrive_time: median={np.median(roll_times):.3f}s")

    np.savez(
        args.out,
        states=states,
        ball_arrive_time_est=np.float64(arrive_med),
        mu=mu, sigma=sigma, cov=cov,
        clip_lo=lo, clip_hi=hi,
    )
    print(f"[save] {args.out}  states shape={states.shape}")
    print(f"[NEXT] env_cfg: ball_arrive_time_est ~= {arrive_med:.3f} (sim 内再确认)")


if __name__ == "__main__":
    main()
