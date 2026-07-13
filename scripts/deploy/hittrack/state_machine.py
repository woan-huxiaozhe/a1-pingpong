"""HitTrack 部署状态机（设计文档 §5、§9）。

状态：INIT → READY(idle) → [GATE x 门控（内联判定）] → TRACKING → [reset 归位] → RETURNING → READY。
enable(arm/disarm) 由底层手动掌管，推理层永不下发（发 enable=True 会触发从机 ~700ms 阻塞式
模式切换、吃掉逼近段反馈）；推理层只下发 model_action，并在 episode 结束时发 reset 触发归位。
纯逻辑、无 ROS 依赖：事件由 ROS 层（Task 7）调用，副作用经 callbacks 回调发出，时钟经 now_fn 注入
（便于用假时钟单测）。TRACKING 每 tick：tau 实时外推 → 组装 68 维 obs → policy → compute_joint_delta_target
后处理 → 发布关节目标。安全中断：tau 越过 post_margin、预测丢失/超时、非有限动作。
"""
from __future__ import annotations

import time

import torch

from . import config as C
from .obs import JointDeltaHistory, assemble_obs
from .tau_anchor import TauAnchor


class HitTrackStateMachine:
    def __init__(self, *, policy, plan_fn, cjdt_fn, callbacks, now_fn, recorder=None):
        self._policy = policy
        self._plan_fn = plan_fn
        self._cjdt_fn = cjdt_fn
        self._cb = callbacks
        self._now_fn = now_fn
        self._recorder = recorder

        self.state = "INIT"
        self._ball_x = None            # 最近一帧 /kalman/pingpong_pos.x（门控用）
        self._ball_pos = None          # 最近一帧原始球位置 (x,y,z)（录制/诊断用；仅 x 到手则 None）
        self._last_pred = None         # 最近一次 valid 预测 (py,pz,vx,vy,vz,pred_t)（录制/诊断用）
        self._q = None                 # 最近一帧真实关节角 [7] float32
        self._tau_motor = None         # 最近一帧电机反馈转矩 [7]（effort，录制用；真机无则 None）
        self._tau = TauAnchor()
        self._hist = JointDeltaHistory()
        self._prev_target = None       # compute_joint_delta_target 的滚动目标
        self._last_action = torch.zeros(7, dtype=torch.float32)
        self._act_filt = None          # 原始 act 一阶低通(方案B)的滚动状态；None=未初始化/追踪起始
        self._lpf_enabled = bool(C.ACT_LPF_ENABLED)  # 从 config 读默认(单一来源)；可在构造后按实例覆盖
        self._lpf_alpha = float(C.ACT_LPF_ALPHA)
        self._last_activity = 0.0      # 进入 TRACKING 或最近一次 valid 预测的时刻（丢失计时基准）
        self._return_start = 0.0
        self._first_infer = True       # 首个推理 tick = 训练 reset step（delta 历史置零）
        self._episode = 0              # 每进入一次 TRACKING +1（录制用，区分每次追踪）
        self._step = 0                 # 当前 episode 内的推理步计数（录制用）
        self._prev_tick_t = None       # 上一推理 tick 的 now（算 tick 实际间隔/达成频率）

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

    def on_ball_pos(self, x, y=None, z=None):
        self._ball_x = float(x)
        if y is not None and z is not None:      # 全 xyz 到手才存原始球位置（录制用）
            self._ball_pos = (float(x), float(y), float(z))

    def on_kalman_reset(self):
        """READY 时的 x 门控判定：通过→TRACKING，否则记日志留 READY。"""
        if self.state != "READY":
            return
        if self._q is None:
            # 还没收到任何 /right_joint_states：无当前关节角无法初始化历史/prev_target/obs，
            # 拒绝进入 TRACKING（READY 可纯靠归位超时到达，不保证已有关节反馈）。
            self._cb.log("gate reject: no joint feedback yet (q is None)")
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
            self._last_pred = (float(py), float(pz), float(vx), float(vy), float(vz), float(pred_t))

    def on_joint_state(self, q, now, tau_motor=None):
        """缓存最近一帧真实关节角/转矩。tick 已改为固定 100Hz 定时器(on_control_tick)驱动，本回调
        不再直接触发推理——消除随 /right_joint_states 到达抖动的 tick_dt(实测 1.6–24.7ms)，让 obs 的
        joint_pos_delta_history 与 tau 外推都按训练的固定 10ms 步进。tau_motor=同帧电机反馈转矩[7]。
        仍在 RETURNING 顺带推进归位超时(与定时器互为冗余，反馈在时也能归位)。"""
        self._q = torch.as_tensor(q, dtype=torch.float32)
        self._tau_motor = tau_motor
        if self.state == "RETURNING":
            self._maybe_finish_return(now)

    def on_control_tick(self, now):
        """固定 100Hz 控制 tick（ROS 定时器驱动，周期 STEP_DT）：与 /right_joint_states 到达节奏解耦。
        TRACKING：用最近缓存的 self._q 组 68 维 obs→policy→后处理→发布关节目标；
        RETURNING：推进归位超时；其余状态空转。"""
        if self.state == "RETURNING":
            self._maybe_finish_return(now)
        elif self.state == "TRACKING":
            self._track_tick(now)

    def on_watchdog(self, now):
        """反馈心跳丢失（/right_joint_states 长时间未到）→ 若在 TRACKING 立即中断到安全态。"""
        if self.state == "TRACKING":
            self._interrupt(now, "joint-state feedback watchdog")

    # ---- 内部 ----

    def _maybe_finish_return(self, now):
        if self.state == "RETURNING" and now - self._return_start >= C.READY_RETURN_TIMEOUT_S:
            self.state = "READY"

    def _enter_tracking(self):
        self._tau.reset()
        self._hist.reset(self._q)
        self._prev_target = self._q.clone()
        self._last_action = torch.zeros(7, dtype=torch.float32)
        self._act_filt = None          # 每次进入追踪清零低通状态，首个推理 tick 直接采用 raw(无起跳)
        self._last_activity = self._now_fn()
        self._first_infer = True
        self._episode += 1
        self._step = 0
        self._prev_tick_t = None
        self.state = "TRACKING"
        # enable(arm/disarm) 交底层手动，推理层不下发（enable=True 会触发从机 ~700ms 冻结、
        # 吃掉逼近段反馈）；进入追踪只管组 obs / 发 model_action。
        self._cb.log("进入 TRACKING（enable 交底层，推理只下发 model_action）")

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
        # 关节 delta 历史与训练同构（observations.joint_pos_delta_history 是 push-then-read）：
        # 先把当前 q 压入历史再读 deltas，使最新一格 = q_t - q_{t-1}（瞬时关节速度，不滞后一拍）。
        # 首个推理 tick 等价训练 reset step（step==0）：整段填当前 q，deltas 为 0。
        if self._first_infer:
            self._hist.reset(self._q)
            self._first_infer = False
        else:
            self._hist.push(self._q)
        tick_t0 = time.perf_counter()
        obs = assemble_obs(self._q, self._tau, tau_live, self._last_action, self._plan_fn, self._hist)
        infer_t0 = time.perf_counter()
        raw = self._policy(obs.reshape(1, -1))[0]
        infer_ms = (time.perf_counter() - infer_t0) * 1e3
        if not bool(torch.isfinite(raw).all()):
            self._interrupt(now, "non-finite action")
            return
        # 方案B：对【原始 act】做一阶指数低通再送动作接口，压高频抖动。ENABLED=False 或 α>=1
        # 时 act_cmd 即 raw（逐字节等价旧行为）。首个 tick self._act_filt 直接采用 raw，避免从 0 起跳。
        if self._lpf_enabled and self._lpf_alpha < 1.0:
            if self._act_filt is None:
                self._act_filt = raw.clone()
            else:
                self._act_filt = self._lpf_alpha * raw + (1.0 - self._lpf_alpha) * self._act_filt
            act_cmd = self._act_filt
        else:
            act_cmd = raw
        target = self._cjdt_fn(
            self._q, act_cmd, self._lower, self._upper,
            action_scale=self._action_scale,
            previous_target=self._prev_target,
            smoothing=C.SMOOTHING,
            max_delta_per_step=self._max_delta,
        )
        self._prev_target = target
        target_list = target.tolist()
        self._cb.publish_action(target_list)
        # last_action 回灌【未滤波 raw】：训练侧 obs.last_action 记录的是策略原始输出，保持一致。
        self._last_action = raw
        if self._recorder is not None:
            tick_ms = (time.perf_counter() - tick_t0) * 1e3
            tick_dt_ms = (now - self._prev_tick_t) * 1e3 if self._prev_tick_t is not None else float("nan")
            self._recorder.record(
                t=now, episode=self._episode, step=self._step,
                tick_dt_ms=tick_dt_ms, infer_ms=infer_ms, tick_ms=tick_ms,
                tau_live=float(tau_live),
                obs=obs.tolist(), action=raw.tolist(), target=target_list,
                motor_tau=self._tau_motor, ball_obs=self._ball_pos, pred=self._last_pred,
            )
            self._step += 1
        self._prev_tick_t = now

    def _begin_return(self, now):
        # 推理层不碰 enable（arm/disarm 交底层手动）；仅发 reset 触发底层归位到 ready。
        # reset 只在 episode 结束/中断时发，此时冻结落在击球后的归位段，无害。
        self._cb.publish_reset()
        self._return_start = now
        self.state = "RETURNING"

    def _interrupt(self, now, reason):
        self._cb.log(f"interrupt: {reason}")
        self._begin_return(now)
