#!/usr/bin/env python3
"""HitTrack obs 录制(recorder.ObsRecorder 输出的 CSV)可视化 dashboard。

一个 matplotlib 窗口，左侧 RadioButtons 切换标签页、Prev/Next 按钮切换 episode：
  · 观测·关节      —— 关节位置 q / 速度 dq/dt
  · 观测·末端&球   —— 6行2列：列1 末端位置/速度 xyz(实线真实 vs 虚线 planner)；
                       列2 法向 xyz+夹角误差+位置误差+tau
  · 策略输出 vs 关节 —— 每关节：实际 q / 指令 target / 策略原始输出 act
  · 关节全状态     —— 位置 / 速度 / 转矩(转矩需 recorder 采集 effort，当前 CSV 无 -> 占位提示)
  · 时序/推理      —— tick 间隔(实达周期) / infer_ms / tick_ms / 直方图

obs 列布局(见 scripts/deploy/hittrack/obs.py)：
  0:7   q-DEFAULT(关节相对位置)   7:42  关节 delta 历史[5x7]   42:45 p_ref(球预测击球点)
  45:48 v_ref(理想拍速)          48:51 n_ref(目标拍面法向)   51 tau_live
  52:55 racket_pos(末端位置)     55:58 racket_normal(末端法向) 58:61 racket_pos-p_ref(位置误差)
  61:68 last_action
注意：obs 里没有“球速度”原始量(只有 planner 求得的理想拍速 v_ref)；转矩也未记录。

用法：
  python scripts/viz_obs_record.py <csv>              # 交互窗口
  python scripts/viz_obs_record.py <csv> --save out/  # 无显示环境，导出每 episode 各标签 PNG
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "deploy"))
try:
    from hittrack import config as _C
    DEFAULT_Q = np.array(_C.DEFAULT_JOINT_POS, dtype=float)
    JOINT_NAMES = [n.replace("-a1_r", "") for n in _C.RIGHT_ARM_JOINT_NAMES]
    HIT_PLANE_X = _C.HIT_PLANE_X
    STEP_DT = _C.STEP_DT
except Exception:  # 独立运行兜底（与 config 保持一致的字面量）
    DEFAULT_Q = np.array([1.769, -0.762, -1.863, 1.445, 0.206, -0.827, 1.043])
    JOINT_NAMES = [f"J{i}" for i in range(1, 8)]
    HIT_PLANE_X = -1.44
    STEP_DT = 0.01

TABS = ["观测·关节", "观测·末端&球", "策略输出vs关节", "关节全状态", "时序/推理"]
AXIS_COL = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
J_COLORS = None  # 延迟到 matplotlib 导入后


def _derive(df: pd.DataFrame) -> dict:
    """从一个 episode 的 DataFrame 抽出物理量数组。"""
    def O(a, b=None):
        if b is None:
            return df[f"obs_{a}"].to_numpy(dtype=float)
        return df[[f"obs_{i}" for i in range(a, b)]].to_numpy(dtype=float)

    t = df["t"].to_numpy(dtype=float)
    t = t - t[0]
    # 保证严格递增供 np.gradient 用（ROS 时钟通常已递增，加极小抖动防止 dt=0）
    t_mono = t + np.arange(len(t)) * 1e-9

    q_abs = O(0, 7) + DEFAULT_Q
    racket_pos = O(52, 55)
    D = dict(
        step=df["step"].to_numpy(dtype=float),
        t=t,
        tick_dt_ms=df["tick_dt_ms"].to_numpy(dtype=float),
        infer_ms=df["infer_ms"].to_numpy(dtype=float),
        tick_ms=df["tick_ms"].to_numpy(dtype=float),
        tau=O(51),
        q_abs=q_abs,
        p_ref=O(42, 45),
        v_ref=O(45, 48),
        n_ref=O(48, 51),
        racket_pos=racket_pos,
        racket_normal=O(55, 58),
        pos_err=O(58, 61),
        act=df[[f"act_{i}" for i in range(7)]].to_numpy(dtype=float),
        tgt=df[[f"tgt_{i}" for i in range(7)]].to_numpy(dtype=float),
    )
    n = len(t)
    D["q_vel"] = np.gradient(q_abs, t_mono, axis=0) if n >= 2 else np.zeros_like(q_abs)
    D["racket_vel"] = np.gradient(racket_pos, t_mono, axis=0) if n >= 2 else np.zeros_like(racket_pos)
    D["hit"] = _hit_step(D["tau"])
    # 电机反馈转矩：新版 recorder 才有 mtau_* 列；全 NaN（真机未发 effort）当作没有
    mcols = [f"mtau_{i}" for i in range(7)]
    if all(c in df.columns for c in mcols):
        mtau = df[mcols].to_numpy(dtype=float)
        D["mtau"] = mtau if np.isfinite(mtau).any() else None
    else:
        D["mtau"] = None

    # 原始球观测 / 预测：新版 recorder 才有；全 NaN（旧文件或未收到）当作没有
    def _opt(cols):
        if all(c in df.columns for c in cols):
            arr = df[cols].to_numpy(dtype=float)
            return arr if np.isfinite(arr).any() else None
        return None

    D["ball"] = _opt(["ball_x", "ball_y", "ball_z"])
    D["pred"] = _opt(["pred_y", "pred_z", "pred_vx", "pred_vy", "pred_vz", "pred_t"])
    return D


def _hit_step(tau: np.ndarray):
    """tau_live 由正穿零的(分数)步号 = 击球时刻。找不到返回 None。"""
    for i in range(1, len(tau)):
        if tau[i - 1] > 0.0 >= tau[i]:
            frac = tau[i - 1] / (tau[i - 1] - tau[i] + 1e-12)
            return (i - 1) + float(np.clip(frac, 0.0, 1.0))
    return None


def _mark_hit(ax, hit):
    if hit is not None:
        ax.axvline(hit, color="k", ls=":", lw=1.2, alpha=0.7, zorder=0)


def _setup_cjk_font(mpl):
    """选一个系统可用的中文字体，避免中文标题渲染成豆腐块。"""
    from matplotlib import font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    prefer = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
              "WenQuanYi Zen Hei", "Source Han Sans CN", "Microsoft YaHei",
              "SimHei", "Droid Sans Fallback", "AR PL UMing CN"]
    found = [n for n in prefer if n in avail]
    mpl.rcParams["font.sans-serif"] = found + ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return found[0] if found else None


def build_and_run(csv_path: str, save_dir: str | None):
    import matplotlib
    if save_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons

    _setup_cjk_font(matplotlib)
    global J_COLORS
    J_COLORS = plt.get_cmap("tab10").colors

    df = pd.read_csv(csv_path)
    ep_ids = sorted(int(e) for e in df["episode"].unique())
    per_ep = {e: _derive(df[df["episode"] == e].reset_index(drop=True)) for e in ep_ids}

    fig = plt.figure(figsize=(15.5, 9.0))
    fig.canvas.manager.set_window_title(f"HitTrack obs viz — {os.path.basename(csv_path)}")
    state = {"tab": TABS[0], "ei": 0}
    content_axes: list = []

    # ---- 各标签绘制 ----
    def grid(nrows, ncols, hspace=0.42, wspace=0.30, top=0.90):
        gs = fig.add_gridspec(nrows, ncols, left=0.28, right=0.985, top=top, bottom=0.07,
                              hspace=hspace, wspace=wspace)
        axes = []
        for r in range(nrows):
            for c in range(ncols):
                ax = fig.add_subplot(gs[r, c])
                content_axes.append(ax)
                axes.append(ax)
        return axes

    def tab_joints(D):
        # 7 关节各一行；左列位置、右列速度，趋势看得更清楚
        axes = grid(7, 2, hspace=0.62, wspace=0.20, top=0.93)
        for j in range(7):
            ap, av = axes[2 * j], axes[2 * j + 1]
            ap.plot(D["q_abs"][:, j], color=J_COLORS[j], lw=1.3)
            av.plot(D["q_vel"][:, j], color=J_COLORS[j], lw=1.3)
            ap.set_ylabel(JOINT_NAMES[j], fontsize=8)
            for a in (ap, av):
                _mark_hit(a, D["hit"]); a.grid(alpha=0.25); a.tick_params(labelsize=7)
            if j == 0:
                ap.set_title("位置 q (rad)", fontsize=9)
                av.set_title("速度 dq/dt (rad/s)", fontsize=9)
            if j == 6:
                ap.set_xlabel("step", fontsize=8); av.set_xlabel("step", fontsize=8)

    def tab_ee_ball(D):
        # 6 行 × 2 列：列1=末端跟踪(位置xyz+速度xyz)，列2=法向/误差/时间
        axes = grid(6, 2, hspace=0.5, wspace=0.28, top=0.94)

        def cell(r, c):
            return axes[r * 2 + c]

        def track(ax, real, ref, key, ylab):
            ax.plot(real, color=AXIS_COL.get(key, "#1f77b4"), lw=1.5, label="真实")
            ax.plot(ref, color="k", lw=1.2, ls="--", alpha=0.85, label="planner")
            ax.set_ylabel(ylab, fontsize=8)
            _mark_hit(ax, D["hit"]); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

        # ---- 列1：末端位置 xyz / 速度 xyz（实线=真实观测, 虚线=planner 的 p_ref/v_ref）----
        for r, key in enumerate("xyz"):
            track(cell(r, 0), D["racket_pos"][:, r], D["p_ref"][:, r], key, f"pos.{key} (m)")
            track(cell(r + 3, 0), D["racket_vel"][:, r], D["v_ref"][:, r], key, f"vel.{key} (m/s)")
        cell(0, 0).set_title("末端跟踪  实线=真实, 虚线=planner", fontsize=9)
        cell(0, 0).legend(fontsize=6, loc="best", framealpha=0.5)

        # ---- 列2：法向 xyz / 夹角误差 / 位置误差 / tau ----
        for r, key in enumerate("xyz"):
            track(cell(r, 1), D["racket_normal"][:, r], D["n_ref"][:, r], key, f"n.{key}")
        cell(0, 1).set_title("法向跟踪 & 误差/时间", fontsize=9)
        dot = np.clip(np.sum(D["racket_normal"] * D["n_ref"], axis=1), -1.0, 1.0)
        a3 = cell(3, 1); a3.plot(np.degrees(np.arccos(dot)), color="#8c564b", lw=1.5)
        a3.set_ylabel("法向夹角误差(deg)", fontsize=8)
        a4 = cell(4, 1); a4.plot(np.linalg.norm(D["pos_err"], axis=1), color="#9467bd", lw=1.6)
        a4.set_ylabel("|末端-p_ref|(m)", fontsize=8)
        a5 = cell(5, 1); a5.plot(D["tau"], color="#ff7f0e", lw=1.5)
        a5.axhline(0, color="#ff7f0e", ls=":", lw=1, alpha=0.6); a5.set_ylabel("tau_live(s)", fontsize=8)
        for a in (a3, a4, a5):
            _mark_hit(a, D["hit"]); a.grid(alpha=0.25); a.tick_params(labelsize=7)

        cell(5, 0).set_xlabel("step", fontsize=8)
        cell(5, 1).set_xlabel("step", fontsize=8)

    def tab_policy_vs_joint(D):
        axes = grid(3, 3, hspace=0.5, wspace=0.38)
        for j in range(7):
            ax = axes[j]
            ax.plot(D["q_abs"][:, j], color="#1f77b4", lw=1.5, label="实际 q")
            ax.plot(D["tgt"][:, j], color="#2ca02c", lw=1.3, ls="--", label="指令 target")
            ax.set_ylabel("rad", fontsize=8); ax.grid(alpha=0.25)
            tw = ax.twinx(); content_axes.append(tw)
            tw.plot(D["act"][:, j], color="#d62728", lw=1.0, ls=":", alpha=0.8, label="策略输出 act")
            tw.set_ylabel("act", color="#d62728", fontsize=8)
            ax.set_title(JOINT_NAMES[j], fontsize=9)
            _mark_hit(ax, D["hit"]); ax.set_xlabel("step", fontsize=8)
        # 图例放到第 8 格
        leg = axes[7]; leg.axis("off")
        leg.plot([], [], color="#1f77b4", lw=1.5, label="实际 q (rad, 左轴)")
        leg.plot([], [], color="#2ca02c", lw=1.3, ls="--", label="指令 target (rad, 左轴)")
        leg.plot([], [], color="#d62728", lw=1.0, ls=":", label="策略原始输出 act (右轴)")
        leg.plot([], [], color="k", ls=":", label="击球时刻 tau=0")
        leg.legend(loc="center", fontsize=9, framealpha=0.6)
        axes[8].axis("off")

    def tab_full_state(D):
        # 7 关节各一行：位置 | 速度 | 转矩(有 mtau 则画，无则列中部一次性说明)
        axes = grid(7, 3, hspace=0.62, wspace=0.26, top=0.93)
        mtau = D.get("mtau")
        for j in range(7):
            ap, av, at = axes[3 * j], axes[3 * j + 1], axes[3 * j + 2]
            ap.plot(D["q_abs"][:, j], color=J_COLORS[j], lw=1.3)
            av.plot(D["q_vel"][:, j], color=J_COLORS[j], lw=1.3)
            ap.set_ylabel(JOINT_NAMES[j], fontsize=8)
            cells = [ap, av]
            if mtau is not None:
                at.plot(mtau[:, j], color=J_COLORS[j], lw=1.3); cells.append(at)
            else:
                at.axis("off")
            for a in cells:
                _mark_hit(a, D["hit"]); a.grid(alpha=0.25); a.tick_params(labelsize=7)
            if j == 0:
                ap.set_title("位置 q (rad)", fontsize=9)
                av.set_title("速度 dq/dt (rad/s)", fontsize=9)
                at.set_title("转矩 (Nm)" if mtau is not None else "转矩 (未记录)", fontsize=9)
            if j == 6:
                ap.set_xlabel("step", fontsize=8); av.set_xlabel("step", fontsize=8)
                if mtau is not None:
                    at.set_xlabel("step", fontsize=8)
        if mtau is None:
            axes[3 * 3 + 2].text(  # 转矩列中部(第 4 行)一次性说明
                0.5, 0.5, "转矩未记录\n\nobs 录制未采集\n/right_joint_states\n的 effort 字段。\n"
                "如需：扩展 recorder\n在 on_joint_state 里\n一并记录 q/qd/effort。",
                ha="center", va="center", fontsize=8.5, color="#b03030",
                bbox=dict(boxstyle="round", fc="#fbeaea", ec="#b03030"))

    def tab_timing(D):
        a = grid(2, 2)
        a[0].plot(D["tick_dt_ms"], color="#1f77b4", marker=".", lw=1.2)
        a[0].axhline(STEP_DT * 1e3, color="r", ls="--", lw=1.2, label=f"训练周期 {STEP_DT*1e3:.0f}ms")
        a[0].set_title("tick 间隔(实达周期) [ms]"); a[0].set_ylabel("ms"); a[0].legend(fontsize=8); a[0].grid(alpha=0.25)
        a[1].plot(D["infer_ms"], color="#2ca02c", label="infer_ms(policy前向)")
        a[1].plot(D["tick_ms"], color="#ff7f0e", label="tick_ms(组obs+policy+cjdt)")
        a[1].set_title("计算耗时 [ms]"); a[1].set_ylabel("ms"); a[1].legend(fontsize=8); a[1].grid(alpha=0.25)
        dt = D["tick_dt_ms"][np.isfinite(D["tick_dt_ms"])]
        if len(dt):
            a[2].hist(dt, bins=min(20, max(5, len(dt) // 2)), color="#1f77b4", alpha=0.8)
            a[2].axvline(STEP_DT * 1e3, color="r", ls="--", lw=1.2)
            a[2].axvline(dt.mean(), color="k", ls="-", lw=1.2, label=f"均值 {dt.mean():.1f}ms")
            a[2].set_title("tick 间隔分布"); a[2].set_xlabel("ms"); a[2].legend(fontsize=8)
        rate = 1000.0 / dt.mean() if len(dt) and dt.mean() > 0 else float("nan")
        a[3].axis("off")
        a[3].text(0.5, 0.5,
                  f"平均 tick 间隔 = {dt.mean():.1f} ms\n平均频率 ≈ {rate:.1f} Hz\n"
                  f"目标 = {STEP_DT*1e3:.0f} ms ({1/STEP_DT:.0f} Hz)\n\n"
                  f"infer 均值 {D['infer_ms'].mean():.2f} ms · tick 均值 {D['tick_ms'].mean():.2f} ms",
                  ha="center", va="center", fontsize=11,
                  bbox=dict(boxstyle="round", fc="#eef3fb", ec="#3a6ea5"))
        for ax in a[:2]:
            ax.set_xlabel("step")

    DRAW = {
        TABS[0]: tab_joints, TABS[1]: tab_ee_ball, TABS[2]: tab_policy_vs_joint,
        TABS[3]: tab_full_state, TABS[4]: tab_timing,
    }

    def redraw():
        for ax in content_axes:
            try:
                ax.remove()
            except Exception:
                pass
        content_axes.clear()
        ep = ep_ids[state["ei"]]
        D = per_ep[ep]
        DRAW[state["tab"]](D)
        hit_txt = f" · 击球@step≈{D['hit']:.1f}" if D["hit"] is not None else " · 未见击球(tau未穿零)"
        fig.suptitle(f"{state['tab']}    |    episode {ep} (共{len(ep_ids)})  steps={len(D['t'])}{hit_txt}",
                     fontsize=12, fontweight="bold")
        fig.canvas.draw_idle()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ei, ep in enumerate(ep_ids):
            state["ei"] = ei
            for tab in TABS:
                state["tab"] = tab
                redraw()
                safe = tab.replace("/", "_").replace("·", "_").replace("vs", "vs")
                out = os.path.join(save_dir, f"ep{ep}_{safe}.png")
                fig.savefig(out, dpi=110)
                print("saved", out)
        print(f"\n完成：{len(ep_ids)} episode × {len(TABS)} 标签 -> {save_dir}")
        return

    # ---- 交互控件 ----
    ax_radio = fig.add_axes([0.015, 0.30, 0.20, 0.42]); ax_radio.set_title("标签页", fontsize=10)
    radio = RadioButtons(ax_radio, TABS)
    for lbl in radio.labels:
        lbl.set_fontsize(10)

    ax_prev = fig.add_axes([0.02, 0.20, 0.085, 0.05])
    ax_next = fig.add_axes([0.12, 0.20, 0.085, 0.05])
    b_prev = Button(ax_prev, "◀ Prev ep"); b_next = Button(ax_next, "Next ep ▶")
    ax_hint = fig.add_axes([0.015, 0.08, 0.20, 0.09]); ax_hint.axis("off")
    ax_hint.text(0, 1, "左选标签页\nPrev/Next 切 episode\n虚线=击球时刻 tau=0",
                 va="top", fontsize=9, color="#555")

    def on_tab(label):
        state["tab"] = label; redraw()

    def on_prev(_):
        state["ei"] = (state["ei"] - 1) % len(ep_ids); redraw()

    def on_next(_):
        state["ei"] = (state["ei"] + 1) % len(ep_ids); redraw()

    radio.on_clicked(on_tab)
    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)

    redraw()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="HitTrack obs 录制可视化")
    ap.add_argument("csv", help="obs_record_*.csv 路径")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="无显示环境：把每 episode 各标签导出 PNG 到该目录（不弹窗）")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        ap.error(f"文件不存在: {args.csv}")
    build_and_run(args.csv, args.save)


if __name__ == "__main__":
    main()
