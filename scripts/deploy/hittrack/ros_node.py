"""HitTrack 部署 ROS2 节点（设计文档 §4、§5）。

方案 C：主循环挂在 /right_joint_states 回调上（与 armcontrol 控制节奏对齐）。订阅球体预测/位置/
reset，把事件转交纯逻辑状态机 HitTrackStateMachine；状态机的副作用经回调发布到 /model_action、
/model_control/reset。额外一个 10Hz 看门狗检测反馈心跳丢失。
enable(arm/disarm) 交底层手动掌管，本节点永不发布 /model_control/enable。

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
from hittrack.pure_import import load_compute_joint_delta_target
from hittrack.reference_planner_np import plan_hit_reference
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

    def publish_reset(self):
        self._n._pub_reset.publish(Bool(data=True))

    def log(self, message):
        self._n.get_logger().info(message)


class HitTrackDeployNode(Node):
    def __init__(self, *, policy, recorder=None):
        super().__init__("hittrack_deploy")
        self._pub_action = self.create_publisher(Float64MultiArray, C.TOPIC_MODEL_ACTION, 10)
        self._pub_reset = self.create_publisher(Bool, C.TOPIC_RESET, 10)

        self._sm = HitTrackStateMachine(
            policy=policy,
            plan_fn=plan_hit_reference,
            cjdt_fn=load_compute_joint_delta_target(),
            callbacks=_Callbacks(self),
            now_fn=self._now,
            recorder=recorder,
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

        # 固定 100Hz 控制 tick（周期 STEP_DT=0.01）：推理/发布不再挂在 /right_joint_states 回调上，
        # 而由本定时器驱动，消除 tick_dt 抖动、对齐训练的 10ms 步进。joint_states 回调降级为纯 q 缓存。
        self._control_timer = self.create_timer(C.STEP_DT, self._on_control_tick)
        self._sm.on_startup()
        self._joint_name_warned = False

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_joint_states(self, msg: JointState):
        # 按关节名匹配（不假设数组顺序）：armcontrol 的 jointN-a1_r 与 RIGHT_ARM_JOINT_NAMES 一一对应
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            q = [msg.position[index[name]] for name in C.RIGHT_ARM_JOINT_NAMES]
        except KeyError:
            # 反馈里缺右臂关节名——名字不匹配则整帧被丢，_q 恒为 None，发球门控会卡死。
            # 一次性告警把这类静默失效暴露出来（否则心跳看门狗也不会触发，无任何提示）。
            if not self._joint_name_warned:
                self._joint_name_warned = True
                self.get_logger().warn(
                    f"/right_joint_states 关节名不匹配，本帧被丢弃：收到 {list(msg.name)}，"
                    f"期望包含 {C.RIGHT_ARM_JOINT_NAMES}（config.RIGHT_ARM_JOINT_NAMES）")
            return
        # 电机反馈转矩（effort）：与 q 同帧同名对齐，仅供录制/诊断，不进 obs。
        # 真机 JointState 可能不带 effort（空数组或长度不一致）→ 传 None，录制填 NaN。
        tau_motor = None
        if msg.effort is not None and len(msg.effort) == len(msg.name):
            try:
                tau_motor = [float(msg.effort[index[name]]) for name in C.RIGHT_ARM_JOINT_NAMES]
            except (KeyError, IndexError):
                tau_motor = None
        now = self._now()
        self._last_joint_time = now
        self._sm.on_joint_state(q, now, tau_motor=tau_motor)

    def _on_pred(self, msg):
        # 桌面系 -> 地面系：真机 KF 的 pred_z 以桌面为原点，训练 npz 已 += z_offset。
        # 在此把同一偏置加回 pred_z（仅 z；y/速度/时间不受影响），下游全部地面系。
        fields = (msg.pred_y, msg.pred_z + C.KF_Z_OFFSET, msg.pred_vx, msg.pred_vy,
                  msg.pred_vz, msg.pred_t, bool(msg.valid))
        self._sm.on_pred(fields, self._now())

    def _on_ball_pos(self, msg: PoseStamped):
        p = msg.pose.position
        # x 用于门控（不偏移）；z 桌面系 -> 地面系与 pred 对齐（录制/诊断一致）。
        self._sm.on_ball_pos(p.x, p.y, p.z + C.KF_Z_OFFSET)

    def _on_kalman_reset(self, msg: Bool):  # 任何消息都当"新发球"触发（data 值防御性处理）
        self._sm.on_kalman_reset()

    def _on_control_tick(self):
        """固定 100Hz 控制 tick：驱动状态机 on_control_tick（TRACKING 推理/发布、RETURNING 归位超时）。
        额外做反馈心跳看门狗——tick 不再依赖 /right_joint_states 到达，若反馈长时间未到必须主动中断，
        否则会拿过期 q 持续推理。"""
        now = self._now()
        if self._last_joint_time is not None and now - self._last_joint_time > C.JOINT_STATE_WATCHDOG_S:
            self._sm.on_watchdog(now)     # TRACKING 中则中断到安全态并发 reset 归位
        self._sm.on_control_tick(now)
