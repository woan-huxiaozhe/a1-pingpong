"""HitTrack 真机部署入口。系统 python3.10 + rclpy + torch(CPU)。

用法：
  source /opt/ros/humble/setup.bash
  source <pingpong_kalman_ws>/install/setup.bash   # 提供 PredictedHit（联调时）
  /usr/bin/python3 scripts/deploy/hittrack/deploy.py --policy <.../exported/policy.pt>

策略启动 armcontrol 时须显式传匹配 sim 的 kps/kds（见设计文档 §8、计划"联调阶段"）。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import rclpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hittrack.ros_node import HitTrackDeployNode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="导出的 torch.jit policy.pt 路径")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = torch.jit.load(args.policy, map_location=args.device)
    model.eval()

    @torch.no_grad()
    def policy(o):  # o: [1,68] -> [1,7]
        return model(o)

    rclpy.init()
    node = HitTrackDeployNode(policy=policy)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
