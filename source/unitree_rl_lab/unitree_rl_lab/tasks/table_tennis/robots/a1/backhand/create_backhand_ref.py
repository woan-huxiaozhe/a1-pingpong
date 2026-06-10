"""Offline backhand reference builder (plan B1/B5).

把真机反手示教 pingpong_log_1.csv 的 swing 1-12 (列 q_plan_0..6) 做成
单条规范挥拍, 再补成完整周期 motion, 写 backhand_ref.npz.

流程:
  1. 解析 swing 1-12, 每条按 stage (to_hit -> cruise -> follow-through) 切段.
  2. 每条把各 stage 重采样到规范帧数 (各 stage 取 12 条中位帧数), 拼成等长归一化挥拍.
  3. 相位对齐平均成一条规范挥拍 (canonical). 若平均把挥拍幅度抹平 (yb4 主关节
     range 掉太多) 则回退到 medoid swing.
  4. 补成完整周期: [ready-hold] -> [canonical 挥拍] -> [return->ready] -> [ready-hold],
     总时长按 ball_arrive_time_est 选取, 使 cruise 中心落在 hit_phase 且 phase 不绕圈.
  5. 写 npz: fps=100 / upper_body_dof (N,7) / base_y=0 / joint_names=joint_yb_1..7.
     打印 hit_phase, duration, initial_phase 供 env_cfg 回填 (sim 内再确认见 plan B3).

映射: q_plan_{i} -> joint_yb_{i+1}, 无翻转无偏移 (plan B2, 真机=sim 同一右臂).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

FPS = 100
STAGES = ["to_hit", "cruise", "follow-through"]
JOINT_NAMES = np.array([f"joint_yb_{i}" for i in range(1, 8)])
A1_LIMITS = np.array([
    [-1.04, 3.14], [-3.14, 0.26], [-2.758, 2.758], [-1.92, 1.92],
    [-2.758, 2.758], [-1.57, 1.57], [-2.758, 2.758],
])

DEFAULT_CSV = "/home/woan/robotbase_gripper/tmp/pingpong_logs/2026_06_04/pingpong_log_1.csv"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backhand_ref.npz")

# provisional: 来自 create_serve_states.py (median real x=0.35 -> -1.37)
BALL_ARRIVE_TIME_EST = 0.526


def load_swings(csv_path: str):
    """返回 {swing_id: {"q": (n,7), "stage": [n]}} for swing 1-12."""
    with open(csv_path) as f:
        lines = f.readlines()
    hi = next(i for i, l in enumerate(lines) if l.startswith("sample_id,"))
    hdr = lines[hi].strip().split(",")
    si, sti = hdr.index("swing_id"), hdr.index("stage")
    qcols = [hdr.index(f"q_plan_{i}") for i in range(7)]

    swings: dict[int, dict] = {}
    for l in lines[hi + 1:]:
        p = l.strip().split(",")
        if len(p) < len(hdr):
            continue
        sw = int(float(p[si]))
        if sw < 1:  # swing 0 = 待机
            continue
        d = swings.setdefault(sw, {"q": [], "stage": []})
        d["q"].append([float(p[c]) for c in qcols])
        d["stage"].append(p[sti])
    for d in swings.values():
        d["q"] = np.array(d["q"], dtype=np.float64)
    return swings


def resample(seg: np.ndarray, n_out: int) -> np.ndarray:
    """线性重采样 (m,7) -> (n_out,7) 在归一化时间 [0,1]."""
    if len(seg) == n_out:
        return seg.copy()
    src = np.linspace(0.0, 1.0, len(seg))
    dst = np.linspace(0.0, 1.0, n_out)
    return np.stack([np.interp(dst, src, seg[:, j]) for j in range(7)], axis=1)


def stage_aligned(swings: dict):
    """每条 swing 各 stage 重采样到规范帧数, 拼成等长. 返回 (K,T,7) + 各 stage 帧数."""
    # 各 stage 规范帧数 = 12 条中位
    stage_lens = {st: [] for st in STAGES}
    for d in swings.values():
        for st in STAGES:
            stage_lens[st].append(int(np.sum(np.array(d["stage"]) == st)))
    canon = {st: int(round(np.median(stage_lens[st]))) for st in STAGES}

    aligned = []
    for d in swings.values():
        stages = np.array(d["stage"])
        parts = []
        ok = True
        for st in STAGES:
            seg = d["q"][stages == st]
            if len(seg) < 2:
                ok = False
                break
            parts.append(resample(seg, canon[st]))
        if ok:
            aligned.append(np.concatenate(parts, axis=0))
    return np.stack(aligned, axis=0), canon


def smootherstep(n: int) -> np.ndarray:
    """0->1 minjerk ramp (端点零速度零加速度)."""
    t = np.linspace(0.0, 1.0, n)
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def build(csv_path: str, out_path: str, arrive_time: float):
    swings = load_swings(csv_path)
    print(f"[load] swings {sorted(swings)} (n={len(swings)})")

    aligned, canon = stage_aligned(swings)        # (K,S,7)
    K, S, _ = aligned.shape
    to_hit, cruise, follow = canon["to_hit"], canon["cruise"], canon["follow-through"]
    print(f"[align] {K} swings, canonical frames: "
          f"to_hit={to_hit} cruise={cruise} follow={follow} (S={S})")

    mean_sw = aligned.mean(0)                      # (S,7)
    # medoid = 离均值最近的真实 swing
    dist = np.linalg.norm((aligned - mean_sw).reshape(K, -1), axis=1)
    medoid = aligned[int(dist.argmin())]
    # 抹平检测: yb4 (肘, 主挥拍关节) range
    j = 3
    if (mean_sw[:, j].ptp()) < 0.75 * np.median([sw[:, j].ptp() for sw in aligned]):
        canonical = medoid
        print(f"[canon] mean flattened yb4 range -> use MEDOID swing "
              f"(idx {int(dist.argmin())})")
    else:
        canonical = mean_sw
        print(f"[canon] use MEAN of {K} swings (yb4 range {mean_sw[:, j].ptp():.3f})")

    ready = canonical[0].copy()                    # to_hit 起始 = 待机/launch 位

    # --- 补完整周期: pre-hold + 挥拍 + return + post-hold ---
    # pre-hold 帧数: 保证 initial_phase = hit_phase - arrive_time/duration >= 0
    # cruise 中心(挥拍内) = to_hit + cruise/2; 需 pre + to_hit + cruise/2 >= arrive_time*FPS
    return_n = int(round(0.30 * FPS))              # follow-end -> ready 平滑回位
    post_n = int(round(0.20 * FPS))                # 待下一球的 ready-hold
    need = arrive_time * FPS - (to_hit + cruise / 2.0)
    pre_n = max(int(np.ceil(need)) + 5, int(round(0.10 * FPS)))

    pre = np.repeat(ready[None], pre_n, axis=0)
    follow_end = canonical[-1]
    ramp = smootherstep(return_n)[:, None]
    ret = follow_end[None] * (1 - ramp) + ready[None] * ramp
    post = np.repeat(ready[None], post_n, axis=0)
    dof = np.concatenate([pre, canonical, ret, post], axis=0).astype(np.float32)

    T = dof.shape[0]
    duration = T / FPS
    cruise_center = pre_n + to_hit + cruise / 2.0
    hit_phase = cruise_center / T
    initial_phase = hit_phase - arrive_time / duration

    print(f"[cycle] frames: pre={pre_n} swing={S} return={return_n} post={post_n} "
          f"total={T}  duration={duration:.3f}s")
    print(f"[phase] hit_phase={hit_phase:.4f}  arrive_time={arrive_time:.3f}  "
          f"initial_phase={initial_phase:.4f} (need in [0,1))")
    if not (0.0 <= initial_phase < 1.0):
        print("   [WARN] initial_phase 越界, 调整 pre/post 帧数或 arrive_time")

    # 限位检查
    ok = True
    for i in range(7):
        lo, hi_ = A1_LIMITS[i]
        vmin, vmax = dof[:, i].min(), dof[:, i].max()
        if vmin < lo - 1e-4 or vmax > hi_ + 1e-4:
            print(f"   [WARN] joint_yb_{i+1} [{vmin:.3f},{vmax:.3f}] vs [{lo:.3f},{hi_:.3f}]")
            ok = False
    if ok:
        print("[limits] all joints within A1 limits OK")

    base_y = np.zeros(T, dtype=np.float32)
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof, base_y=base_y,
             joint_names=JOINT_NAMES,
             hit_phase=np.float64(hit_phase),
             ball_arrive_time_est=np.float64(arrive_time))
    print(f"[save] {out_path}  ({T} frames @ {FPS}fps)")
    print(f"[NEXT] env_cfg commands.motion: hit_phase={hit_phase:.4f}  "
          f"ball_arrive_time_est={arrive_time:.3f} (sim 内再确认)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--arrive_time", type=float, default=BALL_ARRIVE_TIME_EST)
    args = ap.parse_args()
    build(args.csv, args.out, args.arrive_time)
