"""HitTrack 部署状态机（设计文档 §5、§9）。

状态：INIT → READY(idle) → [GATE x 门控（内联判定）] → TRACKING → [enable=False+归位] → RETURNING → READY。
纯逻辑、无 ROS 依赖：事件由 ROS 层（Task 7）调用，副作用经 callbacks 回调发出，时钟经 now_fn 注入
（便于用假时钟单测）。TRACKING 每 tick：tau 实时外推 → 组装 68 维 obs → policy → compute_joint_delta_target
后处理 → 发布关节目标。安全中断：tau 越过 post_margin、预测丢失/超时、非有限动作。
"""
from __future__ import annotations

import torch

from . import config as C
from .obs import JointDeltaHistory, assemble_obs
from .tau_anchor import TauAnchor


class HitTrackStateMachine:
    def __init__(self, *, policy, plan_fn, cjdt_fn, callbacks, now_fn):
        self._policy = policy
        self._plan_fn = plan_fn
        self._cjdt_fn = cjdt_fn
        self._cb = callbacks
        self._now_fn = now_fn

        self.state = "INIT"
        self._ball_x = None            # 最近一帧 /kalman/pingpong_pos.x（门控用）
        self._q = None                 # 最近一帧真实关节角 [7] float32
        self._tau = TauAnchor()
        self._hist = JointDeltaHistory()
        self._prev_target = None       # compute_joint_delta_target 的滚动目标
        self._last_action = torch.zeros(7, dtype=torch.float32)
        self._last_activity = 0.0      # 进入 TRACKING 或最近一次 valid 预测的时刻（丢失计时基准）
        self._return_start = 0.0

        # 预计算后处理常量（float32 张量，[7]）
        margin = C.LIMIT_MARGIN
        self._lower = torch.tensor(C.SOFT_LIMIT_LOWER, dtype=torch.float32) + margin
        self._upper = torch.tensor(C.SOFT_LIMIT_UPPER, dtype=torch.float32) - margin
        self._action_scale = torch.tensor(C.ACTION_SCALE, dtype=torch.float32)
        self._max_delta = torch.tensor(C.MAX_JOINT_VELOCITY, dtype=torch.float32) * C.STEP_DT

    # ---- 事件（由 ROS 层调用）----

    def on_startup(self):
        """INIT：发一次归位信号，进入 RETURNING，固定超时后转 READY。"""
        self._begin_return(self._now_fn())

    def on_ball_pos(self, x):
        self._ball_x = float(x)

    def on_kalman_reset(self):
        """READY 时的 x 门控判定：通过→TRACKING，否则记日志留 READY。"""
        if self.state != "READY":
            return
        lo, hi = C.RESET_X_GATE
        if self._ball_x is None or not (lo <= self._ball_x <= hi):
            self._cb.log(f"gate reject: ball x={self._ball_x} not in {C.RESET_X_GATE}")
            return
        self._enter_tracking()

    def on_pred(self, fields, recv_time):
        """喂 PredictedHit：仅 valid=true 更新锚点，并刷新丢失计时。"""
        if self.state != "TRACKING":
            return
        py, pz, vx, vy, vz, pred_t, valid = fields
        if valid:
            self._tau.update(py, pz, vx, vy, vz, pred_t, recv_time)
            self._last_activity = float(recv_time)

    def on_joint_state(self, q, now):
        """方案 C 的 tick 源。"""
        self._q = torch.as_tensor(q, dtype=torch.float32)
        if self.state == "RETURNING":
            if now - self._return_start >= C.READY_RETURN_TIMEOUT_S:
                self.state = "READY"
            return
        if self.state == "TRACKING":
            self._track_tick(now)

    # ---- 内部 ----

    def _enter_tracking(self):
        self._tau.reset()
        self._hist.reset(self._q)
        self._prev_target = self._q.clone()
        self._last_action = torch.zeros(7, dtype=torch.float32)
        self._last_activity = self._now_fn()
        self.state = "TRACKING"
        self._cb.publish_enable(True)

    def _track_tick(self, now):
        anchored = self._tau.has_anchor()
        if anchored:
            tau_live = self._tau.tau_live(now)
            if tau_live <= -C.POST_MARGIN_S:
                self._interrupt(now, "hit window elapsed")
                return
        # 丢失/超时（覆盖首个 valid 预测之前的等待与追踪中途丢失）
        if now - self._last_activity > C.PRED_LOSS_TOLERANCE_S:
            self._interrupt(now, "prediction lost/timeout")
            return
        if not anchored:
            return  # 收到首个 valid 预测前原地悬停，不推理不发动作
        obs = assemble_obs(self._q, self._tau, tau_live, self._last_action, self._plan_fn, self._hist)
        raw = self._policy(obs.reshape(1, -1))[0]
        if not bool(torch.isfinite(raw).all()):
            self._interrupt(now, "non-finite action")
            return
        target = self._cjdt_fn(
            self._q, raw, self._lower, self._upper,
            action_scale=self._action_scale,
            previous_target=self._prev_target,
            smoothing=C.SMOOTHING,
            max_delta_per_step=self._max_delta,
        )
        self._prev_target = target
        self._cb.publish_action(target.tolist())
        self._last_action = raw
        self._hist.push(self._q)

    def _begin_return(self, now):
        self._cb.publish_enable(False)
        self._cb.publish_reset()
        self._return_start = now
        self.state = "RETURNING"

    def _interrupt(self, now, reason):
        self._cb.log(f"interrupt: {reason}")
        self._begin_return(now)
