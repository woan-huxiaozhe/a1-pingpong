"""HitTrack 真机部署入口。系统 python3.10 + rclpy + torch(CPU)。

用法：
  source /opt/ros/humble/setup.bash
  source /data/pingpong_rl_traj/install/setup.bash   # 提供 PredictedHit（联调时）
  /usr/bin/python3 scripts/deploy/hittrack/deploy.py --policy <.../exported/policy.pt>

策略启动 armcontrol 时须显式传匹配 sim 的 kps/kds（见设计文档 §8、计划"联调阶段"）。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch
import rclpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hittrack.ros_node import HitTrackDeployNode
from hittrack.recorder import ObsRecorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="导出的 torch.jit policy.pt 路径")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-record", action="store_true",
                    help="关闭观测录制（默认开启，CSV 落在 policy 同目录）")
    args = ap.parse_args()

    model = torch.jit.load(args.policy, map_location=args.device)
    model.eval()

    @torch.no_grad()
    def policy(o):  # o: [1,68] cpu(state_machine 全程用 cpu tensor 拼 obs) -> [1,7] cpu
        return model(o.to(args.device)).to("cpu")

    with torch.no_grad():
        policy(torch.zeros(1, 68))  # 预热：把首次前向的 kernel 编译/内存分配开销挪到 spin 之前，不占用首次追踪 tick

    recorder = None
    if not args.no_record:
        policy_dir = os.path.dirname(os.path.abspath(args.policy))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(policy_dir, f"obs_record_{stamp}.csv")
        recorder = ObsRecorder(csv_path)
        print(f"[hittrack] 观测录制已开启 -> {csv_path}", flush=True)

    rclpy.init()
    node = HitTrackDeployNode(policy=policy, recorder=recorder)
    print("[hittrack] enable(arm/disarm) 交底层手动；本节点只发 model_action + reset(归位)，"
          "永不发 /model_control/enable", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if recorder is not None:
            stats = recorder.close()
            print(f"[hittrack] 录制结束：写入 {stats['written']} 行，丢弃 {stats['dropped']} 行 "
                  f"-> {stats['path']}", flush=True)


if __name__ == "__main__":
    main()
