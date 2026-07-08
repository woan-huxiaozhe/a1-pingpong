"""68 维 Actor 观测组装（设计文档 §8）。顺序：joint_pos_rel(7)+history(35)+hit_reference_command(10)
+racket_pos(3)+racket_normal(3)+hit_ref_pos_error(3)+last_action(7)。"""
from __future__ import annotations
import torch
from . import config as C
from .fk import racket_pose_world

_DEFAULT = torch.tensor(C.DEFAULT_JOINT_POS, dtype=torch.float32)
_TARGET = torch.tensor([C.HITTRACK_TARGET_XYZ], dtype=torch.float32)

class JointDeltaHistory:
    """镜像训练 observations.joint_pos_delta_history：存 HISTORY_LENGTH+1 帧，deltas=history[0]-history[1:]。"""
    def __init__(self):
        self._h = None  # [L+1, 7]
    def reset(self, q):
        self._h = q.reshape(1, 7).repeat(C.HISTORY_LENGTH + 1, 1).clone()
    def push(self, q):
        self._h = torch.roll(self._h, shifts=1, dims=0)
        self._h[0] = q
    def deltas(self):
        d = self._h[0:1] - self._h[1:]                 # [L,7]
        return d.reshape(-1).clamp(-1.0, 1.0)          # [35]

def assemble_obs(q, tau_anchor, tau_live, last_action, plan_fn, hist):
    py, pz, vx, vy, vz = tau_anchor.ball_state()
    p_ball = torch.tensor([[C.HIT_PLANE_X, py, pz]], dtype=torch.float32)
    v_ball = torch.tensor([[vx, vy, vz]], dtype=torch.float32)
    p_ref, v_ref, n_ref = plan_fn(p_ball, v_ball, _TARGET, restitution=C.HIT_RESTITUTION)
    p_ref, v_ref, n_ref = p_ref[0], v_ref[0], n_ref[0]
    racket_pos, racket_normal = racket_pose_world(q.float())
    racket_pos = racket_pos.float(); racket_normal = racket_normal.float()
    hit_ref = torch.cat([p_ref, v_ref, n_ref, torch.tensor([tau_live], dtype=torch.float32)])
    return torch.cat([
        q - _DEFAULT,                       # 0:7
        hist.deltas(),                      # 7:42
        hit_ref,                            # 42:52
        racket_pos,                         # 52:55
        racket_normal,                      # 55:58
        racket_pos - p_ref,                 # 58:61
        last_action,                        # 61:68
    ])
