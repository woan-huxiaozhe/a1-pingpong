#!/usr/bin/env python3
"""一次性诊断：逐帧记录 /right_joint_states 的【到达时刻】与【header.stamp】。

用途：与开控制的 deploy **同时跑**，定位「deploy 出现 ~700ms 控制黑洞」的根因层次——
到达间隔 vs 源时间戳(header.stamp) 交叉判定：
  · 到达空、stamp 连续 -> 传输/DDS 投递冻结（从机在发，网络层卡）
  · 到达空、stamp 也空 -> 从机发布端自己停了（上游）
  · 到达顺、deploy 却空 -> deploy 单线程自阻塞（对比 deploy 的 obs_record 得知）

⚠️ 干净 ROS shell 跑（先 conda deactivate 再 source ROS），且
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp（与 deploy 一致）。

用法（与开控制 deploy 并行；secs=0 直到 Ctrl-C）：
  python scripts/deploy/record_joint_arrival.py --out /tmp/js_arrival.csv --secs 120
复看（并与 deploy obs_record 的空洞时刻对齐）：
  python scripts/deploy/record_joint_arrival.py --analyze /tmp/js_arrival.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

HEADER = ["t_arrival", "stamp", "n_names"]


def _analyze(path: str) -> None:
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path)
    if len(df) < 2:
        print(f"行数不足({len(df)})"); return
    ta = df["t_arrival"].to_numpy(float)
    ga = np.diff(ta)
    i = int(np.argmax(ga))
    print(f"总帧 {len(df)}  时长 {(ta[-1]-ta[0]):.1f}s  平均 {len(df)/(ta[-1]-ta[0]):.1f}Hz")
    print(f"到达间隔: 中位 {np.median(ga)*1e3:.1f}ms  最大 {ga.max()*1e3:.0f}ms "
          f"@ t_arrival={ta[i]:.6f}(与前一帧 {ta[i]:.6f})  >100ms 空洞数 {(ga>0.1).sum()}")
    st = df["stamp"].to_numpy(float)
    if np.any(st > 0):
        gs = np.diff(st)
        j = int(np.argmax(gs))
        print(f"源 header.stamp 间隔: 中位 {np.median(gs)*1e3:.1f}ms  最大 {gs.max()*1e3:.0f}ms "
              f"@ stamp={st[j]:.6f}  >100ms 空洞数 {(gs>0.1).sum()}")
        print("  -> 到达空但 stamp 连续 = 传输冻结；stamp 也空 = 从机发布端停。")
    else:
        print("源 header.stamp 全 0/缺失，无法判发布端 vs 传输。")


def _record(out: str, secs: float) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("js_arrival_recorder")
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(HEADER)
    state = {"n": 0}

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def on_js(m: JointState):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        w.writerow([f"{now():.6f}", f"{stamp:.6f}", len(m.name)])
        state["n"] += 1

    # 与 deploy 一致的 depth=10，暴露同样的 reader 缓冲行为
    node.create_subscription(JointState, "/right_joint_states", on_js, 10)
    node.get_logger().info(f"逐帧记录 /right_joint_states -> {out}。Ctrl-C 停止。")

    t0 = now()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if secs > 0 and now() - t0 >= secs:
                break
    except KeyboardInterrupt:
        pass
    finally:
        f.flush(); f.close()
        node.get_logger().info(f"停止。已写 {state['n']} 帧 -> {out}")
        node.destroy_node(); rclpy.shutdown()
    _analyze(out)


def main():
    ap = argparse.ArgumentParser(description="逐帧记录 /right_joint_states 到达时刻与源时间戳")
    ap.add_argument("--out", default="/tmp/js_arrival.csv")
    ap.add_argument("--secs", type=float, default=120.0, help="录制时长秒(0=直到 Ctrl-C)")
    ap.add_argument("--analyze", metavar="CSV", default=None, help="只分析已录 CSV")
    args = ap.parse_args()
    if args.analyze:
        _analyze(args.analyze)
    else:
        _record(args.out, args.secs)


if __name__ == "__main__":
    main()
