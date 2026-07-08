"""tau 实时锚定与外推（设计文档 §6）。仅 valid=true 时 update；tick 用本地时钟外推。"""
from __future__ import annotations


class TauAnchor:
    def __init__(self):
        self._s = None  # (py, pz, vx, vy, vz, pred_t, recv_time)

    def update(self, pred_y, pred_z, pred_vx, pred_vy, pred_vz, pred_t, recv_time):
        self._s = (float(pred_y), float(pred_z), float(pred_vx), float(pred_vy),
                   float(pred_vz), float(pred_t), float(recv_time))

    def has_anchor(self):
        return self._s is not None

    def ball_state(self):
        py, pz, vx, vy, vz, _, _ = self._s
        return (py, pz, vx, vy, vz)

    def tau_live(self, now):
        _, _, _, _, _, pred_t, recv_time = self._s
        return pred_t - (float(now) - recv_time)

    def reset(self):
        self._s = None
