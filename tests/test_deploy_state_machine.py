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
    assert sm.state == "TRACKING" and rec.enables[-1] is True


def test_no_action_before_first_valid_pred():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    now[0] += 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0])   # 还没喂 valid pred
    assert rec.actions == []                                       # 不发动作


def test_tracks_then_publishes_action():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])    # valid pred
    now[0] += 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0])
    assert len(rec.actions) == 1 and len(rec.actions[0]) == 7


def test_pred_loss_interrupts():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.4, True), now[0])
    now[0] += C.PRED_LOSS_TOLERANCE_S + 0.05; sm.on_joint_state(C.READY_JOINT_POS, now[0])  # 长时间无新 pred
    assert rec.enables[-1] is False and rec.resets >= 1
    assert sm.state in ("RETURNING", "READY")


def test_normal_end_on_tau_past_margin():
    now = [0.0]; rec = Rec(); sm = _sm(now, rec); sm.on_startup(); now[0] = C.READY_RETURN_TIMEOUT_S + 0.1
    sm.on_joint_state(C.READY_JOINT_POS, now[0]); sm.on_ball_pos(1.0); sm.on_kalman_reset()
    sm.on_pred((0.05, 1.0, -4.0, 0.0, -0.5, 0.02, True), now[0])
    now[0] += 0.02 + C.POST_MARGIN_S + 0.01; sm.on_joint_state(C.READY_JOINT_POS, now[0])
    assert rec.enables[-1] is False and rec.resets >= 1
