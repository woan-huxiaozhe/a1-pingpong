"""HitTrack 部署用击球参考规划的 numpy 移植——与训练侧
``a1_pingpong_hittrack.mdp.reference_planner.plan_hit_reference`` 的等价实现。

为什么需要它：训练侧 plan_hit_reference 是为 GPU 批量(数千 env)写的纯 torch 函数，其核心
``solve_launch_velocity`` 用二分法(18 次)×轨迹前向积分(~70 步)数值求解带二次空气阻力的抛射
边值问题——有 drag 时无闭形式解。部署是单 env(N=1)、CPU、每 /right_joint_states tick 调一次，
torch 每个算子的 Python dispatch 开销(≈µs)全砸在 1 个 env 上 → 单次 ~16ms，把控制回路从训练的
100Hz 拖到 ~55Hz(见 obs 录制诊断)。

本移植逐字复刻同一套离散动力学与常数，但把「批-1 的张量运算」换成**纯 Python float 标量运算**
（deploy 恒为 N=1；plan_hit_reference 对 N 行逐行标量求解），彻底消除 torch/numpy 的 per-op
dispatch 开销。签名、默认参数、返回类型(torch float32 张量)与原 plan_hit_reference 完全一致，
故 obs.assemble_obs 无需改动即可直接注入本函数。标量以 Python float(=C double)计算，末端转回
float32 与原实现对齐；两者差异远低于域随机化噪声（见 tests/test_reference_planner_np.py）。
"""
from __future__ import annotations

import math

import numpy as np
import torch

# 与 table_tennis_sac.mdp.hitting 的常数保持单一来源一致（此处复刻，避免部署链引入 isaaclab）。
NEUTRAL_THETA = math.radians(28.0)
PADDLE_RESTITUTION = 0.75


def _height_at_range_scalar(z0, speed, cos_t, sin_t, horiz_dist, *, drag_k, damp_total,
                            control_dt, gravity, n_steps):
    """标量前向积分：从高度 z0、速度 speed、仰角(cos_t,sin_t)发射，返回累计水平行程首次达到
    horiz_dist 时刻的高度；n_steps 内未到达(欠射)返回大负哨兵。半隐式：位置用步前速度积分。
    damp_total = (1/(1+lin_damp*sub_dt))**substeps 已由调用方预算。"""
    vr = speed * cos_t
    vz = speed * sin_t
    r = 0.0
    z = z0
    for _ in range(n_steps):
        r_next = r + vr * control_dt
        z_next = z + vz * control_dt
        if r < horiz_dist <= r_next:                    # 本步跨越目标射程
            denom = r_next - r
            if denom < 1.0e-9:
                denom = 1.0e-9
            frac = (horiz_dist - r) / denom
            frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
            return z + frac * (z_next - z)
        r, z = r_next, z_next
        # 一控制步动力学：重力 + 线性阻尼(解析) + 一次二次空气阻力 patch（同 _step_dynamics）
        vz = vz - gravity * control_dt
        vr *= damp_total
        vz *= damp_total
        speed_mag = math.sqrt(vr * vr + vz * vz)
        if speed_mag < 1.0e-9:
            speed_mag = 1.0e-9
        factor = 1.0 - drag_k * speed_mag * control_dt
        if factor < 0.0:
            factor = 0.0
        vr *= factor
        vz *= factor
    return -1.0e6


def _solve_launch_velocity_scalar(ox, oy, oz, tx, ty, tz, *, theta, drag_k, lin_damp,
                                  control_dt, substeps, gravity, speed_lo, speed_hi,
                                  bisection_iters, max_flight_s):
    """单个球：落到 (tx,ty,tz) 所需的出射球速 (vox,voy,voz)。航向由 origin->target 定、仰角由
    theta 定，唯一自由标量=打出速率，用单调二分在「射程处高度 − 目标高度」上求解。"""
    dx, dy = tx - ox, ty - oy
    horiz = math.sqrt(dx * dx + dy * dy)
    if horiz < 1.0e-4:
        horiz = 1.0e-4
    hx, hy = dx / horiz, dy / horiz

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    n_steps = int(round(max_flight_s / control_dt))
    sub_dt = control_dt / substeps
    damp_total = (1.0 / (1.0 + lin_damp * sub_dt)) ** substeps

    lo, hi = speed_lo, speed_hi
    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        z_at = _height_at_range_scalar(
            oz, mid, cos_t, sin_t, horiz,
            drag_k=drag_k, damp_total=damp_total,
            control_dt=control_dt, gravity=gravity, n_steps=n_steps)
        if (z_at - tz) > 0.0:                           # 射程处偏高 -> 降速
            hi = mid
        else:
            lo = mid
    speed = 0.5 * (lo + hi)
    return speed * cos_t * hx, speed * cos_t * hy, speed * sin_t


def _contact_inverse_scalar(vix, viy, viz, vox, voy, voz, restitution):
    """闭形式刚体无摩擦碰撞反解：产生 v_out 的最小拍速与单位法向。返回 (v_paddle(3,), n(3,))。"""
    wx, wy, wz = vox - vix, voy - viy, voz - viz
    w_norm = math.sqrt(wx * wx + wy * wy + wz * wz)
    if w_norm < 1.0e-9:
        w_norm = 1.0e-9
    nx, ny, nz = wx / w_norm, wy / w_norm, wz / w_norm
    v_in_n = vix * nx + viy * ny + viz * nz
    v_p_n = v_in_n + w_norm / (1.0 + restitution)
    return (v_p_n * nx, v_p_n * ny, v_p_n * nz), (nx, ny, nz)


def plan_hit_reference(p_ball_hit, v_ball_hit, target, *, theta=NEUTRAL_THETA,
                       restitution=PADDLE_RESTITUTION, drag_k=0.08, lin_damp=0.05,
                       control_dt=0.02, substeps=4, gravity=9.81, speed_lo=0.5,
                       speed_hi=12.0, bisection_iters=18, max_flight_s=1.4):
    """返回 (p_ref, v_ref, n_ref)，签名/语义同训练侧 plan_hit_reference。
    p_ref=接触点(原样透传输入张量)；v_ref=理想拍速；n_ref=拍面法向。对 N 行逐行标量求解。"""
    p_np = p_ball_hit.detach().cpu().numpy().astype(np.float64, copy=False)
    v_np = v_ball_hit.detach().cpu().numpy().astype(np.float64, copy=False)
    t_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
    n_env = p_np.shape[0]
    # target 可以是 [1,3]（广播到所有 env）或 [N,3]
    t_bcast = t_np if t_np.shape[0] == n_env else np.broadcast_to(t_np, (n_env, 3))

    v_ref = np.empty((n_env, 3), dtype=np.float32)
    n_ref = np.empty((n_env, 3), dtype=np.float32)
    for i in range(n_env):
        vox, voy, voz = _solve_launch_velocity_scalar(
            p_np[i, 0], p_np[i, 1], p_np[i, 2],
            t_bcast[i, 0], t_bcast[i, 1], t_bcast[i, 2],
            theta=theta, drag_k=drag_k, lin_damp=lin_damp, control_dt=control_dt,
            substeps=substeps, gravity=gravity, speed_lo=speed_lo, speed_hi=speed_hi,
            bisection_iters=bisection_iters, max_flight_s=max_flight_s)
        vp, nn = _contact_inverse_scalar(
            v_np[i, 0], v_np[i, 1], v_np[i, 2], vox, voy, voz, restitution)
        v_ref[i] = vp
        n_ref[i] = nn

    return p_ball_hit, torch.from_numpy(v_ref), torch.from_numpy(n_ref)

