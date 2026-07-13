import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "deploy"))
from hittrack.state_machine import HitTrackStateMachine
from hittrack.pure_import import load_plan_hit_reference, load_compute_joint_delta_target
from hittrack import config as C


class Rec:
    def __init__(self):
        self.actions = []
        self.enables = []
        self.resets = 0
        self.logs = []

    def publish_action(self, a):
        self.actions.append(list(a))

    def publish_enable(self, b):
        self.enables.append(b)

    def publish_reset(self):
        self.resets += 1

    def log(self, m):
        self.logs.append(m)


def _sm(now_box, rec):
    return HitTrackStateMachine(
        policy=lambda o: torch.zeros(1, 7),
        plan_fn=load_plan_hit_reference(),
        cjdt_fn=load_compute_joint_delta_target(),
        callbacks=rec, now_fn=lambda: now_box[0])


def test_gate_rejects_out_of_range():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0])   # 推进出 RETURNING -> READY
    sm.on_ball_pos(0.2)                            # x 在门控范围外
    sm.on_kalman_reset()
    assert sm.state == "READY" and any("gate" in m.lower() for m in rec.logs)


def test_gate_accepts_and_enters_tracking():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0])
    sm.on_ball_pos(1.0); sm.on_kalman_reset()
    assert sm.state == "TRACKING" and rec.enables == []   # 进入追踪不下发 enable（交底层）


def test_no_action_before_first_valid_pred():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    now[0] += 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_control_tick(now[0])   # 还没喂 valid pred
    assert rec.actions == []                                       # 不发动作


def test_tracks_then_publishes_action():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])    # valid pred
    now[0] += 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_control_tick(now[0])
    assert len(rec.actions) == 1 and len(rec.actions[0]) == 7


def test_pred_loss_interrupts():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])
    now[0] += C.PRED_LOSS_TOLERANCE_S + 0.05; sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_control_tick(now[0])  # 长时间无新 pred
    assert rec.enables == [] and rec.resets >= 1   # 不碰 enable；发 reset 归位
    assert sm.state in ("RETURNING", "READY")


def test_normal_end_on_tau_past_margin():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.02, True), now[0])
    now[0] += 0.02 + C.POST_MARGIN_S + 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_control_tick(now[0])
    assert rec.enables == [] and rec.resets >= 1   # 不碰 enable；发 reset 归位


def test_delta_history_not_lagged_vs_training():
    # 关节 delta 历史必须 push-then-read（最新一格 = q_t - q_{t-1}），不滞后一拍。
    now = [0.0]; rec = Rec(); captured = []

    def policy(o):
        captured.append(o.clone())
        return torch.zeros(1, 7)

    sm = HitTrackStateMachine(
        policy=policy, plan_fn=load_plan_hit_reference(),
        cjdt_fn=load_compute_joint_delta_target(), callbacks=rec, now_fn=lambda: now[0])
    sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    q0 = list(C.READY_JOINT_POS)
    sm.on_joint_state(q0, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])
    now[0] += 0.01; sm.on_joint_state(q0, now[0]); sm.on_control_tick(now[0])   # 首个推理 tick = reset step -> deltas 全 0
    assert torch.allclose(captured[-1][0, 7:42], torch.zeros(35), atol=1e-6)
    q1 = list(q0); q1[0] += 0.05
    now[0] += 0.01; sm.on_joint_state(q1, now[0]); sm.on_control_tick(now[0])   # 第二 tick: 最新一格 = q1 - q0
    newest = captured[-1][0, 7:14]                          # history[0]-history[1] 的 7 个关节
    assert abs(float(newest[0]) - 0.05) < 1e-5
    assert torch.allclose(newest[1:], torch.zeros(6), atol=1e-6)


def test_recorder_receives_obs_during_tracking():
    # 录制器在 TRACKING 推理 tick 上被喂：obs=68、action/target=7，episode/step 正确。
    now = [0.0]; rec = Rec()

    class FakeRecorder:
        def __init__(self):
            self.rows = []

        def record(self, **kw):
            self.rows.append(kw)

    fr = FakeRecorder()
    sm = HitTrackStateMachine(
        policy=lambda o: torch.zeros(1, 7), plan_fn=load_plan_hit_reference(),
        cjdt_fn=load_compute_joint_delta_target(), callbacks=rec,
        now_fn=lambda: now[0], recorder=fr)
    sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0, 0.2, 0.9); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])
    now[0] += 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0], tau_motor=[1.5] * 7); sm.on_control_tick(now[0])
    assert len(fr.rows) == 1
    r0 = fr.rows[0]
    assert len(r0["obs"]) == 68 and len(r0["action"]) == 7 and len(r0["target"]) == 7
    assert r0["episode"] == 1 and r0["step"] == 0
    assert r0["infer_ms"] >= 0.0
    assert r0["motor_tau"] == [1.5] * 7                          # 电机转矩同帧串到录制行
    assert r0["ball_obs"] == (1.0, 0.2, 0.9)                     # 原始球位置(全xyz)
    assert r0["pred"] == (0.05, 1.0, -4.0, 0.0, -0.5, 0.4)       # 原始预测


def test_never_publishes_enable_low_level_owns_arm():
    # 新约定：推理层永不下发 enable（arm/disarm 交底层手动）；只发 model_action + reset(归位)。
    now = [0.0]; rec = Rec()
    sm = HitTrackStateMachine(
        policy=lambda o: torch.zeros(1, 7), plan_fn=load_plan_hit_reference(),
        cjdt_fn=load_compute_joint_delta_target(), callbacks=rec, now_fn=lambda: now[0])
    sm.on_startup()                                      # 起步发一次 reset 归位（不碰 enable）
    now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    assert sm.state == "TRACKING"
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])
    now[0] += 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_control_tick(now[0])
    assert rec.enables == []                             # 全程从不碰 enable 开关
    assert len(rec.actions) == 1                         # 追踪 tick 只发 model_action
    assert rec.resets >= 1                               # 起步/归位仍用 reset 触发底层回 ready


def test_returning_advances_without_joint_states():
    # 反馈停了也不能卡在 RETURNING：固定 100Hz 控制 tick(on_control_tick)应推进归位超时。
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup()
    assert sm.state == "RETURNING"
    now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_control_tick(now[0])                             # 无任何 on_joint_state
    assert sm.state == "READY"


def test_kalman_reset_before_joint_feedback_stays_ready():
    # READY 可纯靠时间到达（无 on_joint_state -> self._q 仍为 None）；此时来发球触发
    # 不得崩溃：无当前关节角无法初始化历史/obs，应拒绝进入 TRACKING 并留在 READY。
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup()
    now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_control_tick(now[0])                            # RETURNING -> READY，全程无关节反馈
    assert sm.state == "READY"
    sm.on_ball_pos(1.0)                                   # 球在门控内
    sm.on_kalman_reset()                                  # self._q is None
    assert sm.state == "READY"
    assert True not in rec.enables                        # 从未 publish_enable(True)
    assert any("joint" in m.lower() for m in rec.logs)


def test_act_lpf_smooths_and_keeps_last_action_raw():
    # 方案B 原始 act 一阶低通：关闭=raw 直通；开启 alpha=0.5=指数移动平均；
    # 且 last_action 回灌始终是【未滤波 raw】（与训练侧 obs.last_action 记录原始输出一致）。
    def build(enabled, alpha):
        now = [0.0]; rec = Rec(); seen = []; call = [0]

        def policy(o):
            seen.append(o.clone())
            call[0] += 1
            return torch.full((1, 7), float(call[0]), dtype=torch.float32)  # raw=1,2,3(变化才能区分滤波)

        cmds = []

        def fake_cjdt(q, action, lo, hi, *, action_scale, previous_target, smoothing, max_delta_per_step):
            cmds.append(action.clone())      # 截获真正送进动作接口的命令 act
            return q.clone()                 # 返回合法 [7] target

        sm = HitTrackStateMachine(
            policy=policy, plan_fn=load_plan_hit_reference(), cjdt_fn=fake_cjdt,
            callbacks=rec, now_fn=lambda: now[0])
        sm._lpf_enabled = enabled; sm._lpf_alpha = alpha
        sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
        sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
        sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])
        for _ in range(3):
            now[0] += 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_control_tick(now[0])
        return cmds, seen

    # 关闭：命令即 raw = 1,2,3（逐字节等价旧行为）
    cmds_off, _ = build(False, 0.5)
    assert [float(c[0]) for c in cmds_off] == [1.0, 2.0, 3.0]

    # 开启 alpha=0.5：首步采 raw，之后 EMA -> 1, 1.5, 2.25
    cmds_on, seen = build(True, 0.5)
    assert [round(float(c[0]), 4) for c in cmds_on] == [1.0, 1.5, 2.25]
    # 第3个推理 tick 看到的 obs.last_action(61) == 上一步【未滤波 raw】(=2)，而非滤波值(1.5)
    assert abs(float(seen[2][0, 61]) - 2.0) < 1e-6


