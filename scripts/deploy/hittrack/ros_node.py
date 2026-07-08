"""HitTrack 部署 ROS2 节点（设计文档 §4、§5）。

方案 C：主循环挂在 /right_joint_states 回调上（与 armcontrol 控制节奏对齐）。订阅球体预测/位置/
reset，把事件转交纯逻辑状态机 HitTrackStateMachine；状态机的副作用经回调发布到 /model_action、
/model_control/enable、/model_control/reset。额外一个 10Hz 看门狗检测反馈心跳丢失。

本文件是薄接线层：不含控制/坐标/观测数学（那些在 fk/obs/tau_anchor/state_machine 里）。
"""
from __future__ import annotations

import os
import sys

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hittrack import config as C
from hittrack.pure_import import load_compute_joint_delta_target, load_plan_hit_reference
from hittrack.state_machine import HitTrackStateMachine

try:
    from pingpong_kalman.msg import PredictedHit
    _HAVE_PRED_MSG = True
except ImportError:  # pingpong_kalman 工作区未 source：允许纯链路冒烟，仅缺预测订阅
    PredictedHit = None
    _HAVE_PRED_MSG = False


class _Callbacks:
    """把状态机副作用桥接到 ROS 发布器/日志。"""

    def __init__(self, node: "HitTrackDeployNode"):
        self._n = node

    def publish_action(self, target_list):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in target_list]
        self._n._pub_action.publish(msg)

    def publish_enable(self, flag):
        self._n._pub_enable.publish(Bool(data=bool(flag)))

    def publish_reset(self):
        self._n._pub_reset.publish(Bool(data=True))

    def log(self, message):
        self._n.get_logger().info(message)


class HitTrackDeployNode(Node):
    def __init__(self, *, policy):
        super().__init__("hittrack_deploy")
        self._pub_action = self.create_publisher(Float64MultiArray, C.TOPIC_MODEL_ACTION, 10)
        self._pub_enable = self.create_publisher(Bool, C.TOPIC_ENABLE, 10)
        self._pub_reset = self.create_publisher(Bool, C.TOPIC_RESET, 10)

        self._sm = HitTrackStateMachine(
            policy=policy,
            plan_fn=load_plan_hit_reference(),
            cjdt_fn=load_compute_joint_delta_target(),
            callbacks=_Callbacks(self),
            now_fn=self._now,
        )
        self._last_joint_time = None

        self.create_subscription(JointState, C.TOPIC_JOINT_STATES, self._on_joint_states, 10)
        self.create_subscription(PoseStamped, C.TOPIC_KALMAN_POS, self._on_ball_pos, 10)
        self.create_subscription(Bool, C.TOPIC_KALMAN_RESET, self._on_kalman_reset, 10)
        if _HAVE_PRED_MSG:
            self.create_subscription(PredictedHit, C.TOPIC_KALMAN_PRED, self._on_pred, 10)
        else:
            self.get_logger().warn(
                "pingpong_kalman.msg.PredictedHit 不可用——已跳过预测订阅（纯链路冒烟降级）。"
                "真机联调前请先 source pingpong_kalman 工作区。")

        self.create_timer(C.WATCHDOG_PERIOD_S, self._on_watchdog)  # 时间-housekeeping + 心跳看门狗
        self._sm.on_startup()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_joint_states(self, msg: JointState):
        # 按关节名匹配（不假设数组顺序）：armcontrol 的 jointN-a1_r 与 RIGHT_ARM_JOINT_NAMES 一一对应
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            q = [msg.position[index[name]] for name in C.RIGHT_ARM_JOINT_NAMES]
        except KeyError:
            # 反馈里缺右臂关节名（可能是别的关节组的消息）——忽略本帧
            return
        now = self._now()
        self._last_joint_time = now
        self._sm.on_joint_state(q, now)

    def _on_pred(self, msg):
        fields = (msg.pred_y, msg.pred_z, msg.pred_vx, msg.pred_vy, msg.pred_vz,
                  msg.pred_t, bool(msg.valid))
        self._sm.on_pred(fields, self._now())

    def _on_ball_pos(self, msg: PoseStamped):
        self._sm.on_ball_pos(msg.pose.position.x)

    def _on_kalman_reset(self, msg: Bool):  # 任何消息都当"新发球"触发（data 值防御性处理）
        self._sm.on_kalman_reset()

    def _on_watchdog(self):
        now = self._now()
        self._sm.on_tick(now)  # RETURNING 归位超时推进，与关节反馈是否到达无关（防卡死）
        if self._last_joint_time is not None and now - self._last_joint_time > C.JOINT_STATE_WATCHDOG_S:
            self._sm.on_watchdog(now)
