#!/usr/bin/env python3
"""被动录制 Kalman 预测/球位管线（只订阅、绝不 publish，机器人不受影响）。

用途：与开控制的 deploy **同时跑**，抓从球一出现起的原始 KF 流，判定
「开控制后跟踪窗口变短」的根因——是 KF 晚发(外部耦合)还是 deploy 丢/压预测(内部)：
  · valid 何时翻 true（早=门口 / 晚=贴面）
  · /pos 有无检测空洞（有=手臂遮挡视觉；无但 valid 晚=CPU 争用/误 reset）

只订阅 /kalman/pingpong_pred、/kalman/pingpong_pos、/resetKalman；每条消息落一行 CSV，
按 /resetKalman 递增 serve 计数；退出时按发球汇总首个 valid 的 pred_t / valid 计数 / 检测空洞。

⚠️ 运行前（见项目 memory）：
  1) 干净 ROS shell —— 先 `conda deactivate` 再 source ROS，否则 isaac 环境 numpy 冲突崩。
  2) `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`（与真机/ deploy 一致），否则收不到话题。
  3) source pingpong_kalman 工作区（提供 PredictedHit）。

用法（与开控制 deploy 并行）：
  # 终端A: 正常开控制跑 deploy（enable 在底层手动 arm；deploy 只发 model_action+reset）
  # 终端B（干净 shell）:
  python scripts/deploy/record_pred.py --out /tmp/pred_ctrl.csv --secs 120
  # 复看:
  python scripts/deploy/record_pred.py --analyze /tmp/pred_ctrl.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HEADER = ["t_wall", "serve", "kind", "valid",
          "pred_y", "pred_z", "pred_vx", "pred_vy", "pred_vz", "pred_t",
          "ball_x", "ball_y", "ball_z"]


def _analyze(path: str) -> None:
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path)
    print(f"总行数 {len(df)}  发球次数 {int(df['serve'].max()) if len(df) else 0}")
    for s, g in df.groupby("serve"):
        pred = g[g["kind"] == "pred"]
        pos = g[g["kind"] == "pos"]
        val = pred[pred["valid"] == 1]
        line = f"  发球#{int(s)}: pred{len(pred)}条(valid {len(val)}) pos{len(pos)}条"
        if len(val):
            t0 = pred["t_wall"].to_numpy(float)[0]
            fv = val.iloc[0]
            dt_valid = float(fv["t_wall"]) - t0
            line += (f" | 首valid: 距首pred {dt_valid*1e3:.0f}ms, pred_t={fv['pred_t']:.3f}s"
                     f" (小=晚锁/贴面)")
        else:
            line += " | 无 valid！"
        if len(pos) >= 2:
            tp = pos["t_wall"].to_numpy(float)
            gaps = np.diff(tp)
            line += f" | pos间隔 中位{np.median(gaps)*1e3:.0f}ms 最大{gaps.max()*1e3:.0f}ms(大=检测空洞/遮挡)"
        print(line)


def _record(out: str, secs: float) -> None:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
    from hittrack import config as C
    try:
        from pingpong_kalman.msg import PredictedHit
    except ImportError:
        print("[fatal] 无法 import pingpong_kalman.msg.PredictedHit —— 先 source 该工作区。", file=sys.stderr)
        sys.exit(2)

    rclpy.init()
    node = Node("hittrack_pred_recorder")
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(HEADER)
    state = {"serve": 0, "n": 0}

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def on_pred(m):
        w.writerow([now(), state["serve"], "pred", int(bool(m.valid)),
                    m.pred_y, m.pred_z, m.pred_vx, m.pred_vy, m.pred_vz, m.pred_t, "", "", ""])
        state["n"] += 1

    def on_pos(m):
        p = m.pose.position
        w.writerow([now(), state["serve"], "pos", "", "", "", "", "", "", "", p.x, p.y, p.z])
        state["n"] += 1

    def on_reset(_):
        state["serve"] += 1
        node.get_logger().info(f"resetKalman -> 发球#{state['serve']}")

    node.create_subscription(PredictedHit, C.TOPIC_KALMAN_PRED, on_pred, 50)
    node.create_subscription(PoseStamped, C.TOPIC_KALMAN_POS, on_pos, 50)
    node.create_subscription(Bool, C.TOPIC_KALMAN_RESET, on_reset, 10)
    node.get_logger().info(f"被动录制中 -> {out}（不发布任何话题）。Ctrl-C 停止。")

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
        node.get_logger().info(f"停止。已写 {state['n']} 行、{state['serve']} 次发球 -> {out}")
        node.destroy_node(); rclpy.shutdown()
    _analyze(out)


def main():
    ap = argparse.ArgumentParser(description="被动录制 Kalman 预测管线（不接控制）")
    ap.add_argument("--out", default="/tmp/pred_rec.csv", help="录制 CSV 输出路径")
    ap.add_argument("--secs", type=float, default=120.0, help="录制时长秒(0=直到 Ctrl-C)")
    ap.add_argument("--analyze", metavar="CSV", default=None, help="只分析已录 CSV，不订阅")
    args = ap.parse_args()
    if args.analyze:
        _analyze(args.analyze)
    else:
        _record(args.out, args.secs)


if __name__ == "__main__":
    main()
