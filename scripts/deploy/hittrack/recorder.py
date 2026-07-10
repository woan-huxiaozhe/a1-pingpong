"""HitTrack 部署观测录制器：后台线程 + 队列的异步 CSV 写入。

设计要点（不让读写影响推理）：
- 推理线程只做 `queue.put_nowait`（队列满则丢弃并计数，绝不阻塞 /right_joint_states tick）。
- 格式化与磁盘 I/O 全部在守护线程里做，不占用追踪 tick 的推理预算。
- 热路径唯一成本是调用方对 obs/action/target 的 `.tolist()`（~89 个 float，微秒级）。
CSV 落在策略 policy.pt 同目录，一次部署进程一个文件（用 episode 列区分每次追踪）。
列 mtau_0..6 = 该 tick 电机反馈的关节转矩(effort)；真机若不发 effort 则记 NaN。
列 ball_x/y/z = /kalman/pingpong_pos 原始球位置；pred_* = PredictedHit 原始预测
（击球点 y,z 与球速 vx,vy,vz、pred_t）——obs 里没有这些原始球量，专为诊断另存；未收到则 NaN。
"""
from __future__ import annotations

import csv
import queue
import threading

_SENTINEL = object()


class ObsRecorder:
    def __init__(self, csv_path, *, obs_dim=68, act_dim=7, maxsize=50000, flush_every=200):
        self._path = csv_path
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._written = 0
        self._flush_every = flush_every
        self._act_dim = act_dim

        header = (
            ["t", "episode", "step", "tick_dt_ms", "infer_ms", "tick_ms", "tau_live"]
            + [f"obs_{i}" for i in range(obs_dim)]
            + [f"act_{i}" for i in range(act_dim)]
            + [f"tgt_{i}" for i in range(act_dim)]
            + [f"mtau_{i}" for i in range(act_dim)]
            + ["ball_x", "ball_y", "ball_z",
               "pred_y", "pred_z", "pred_vx", "pred_vy", "pred_vz", "pred_t"]
        )
        self._f = open(csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(header)
        self._f.flush()

        self._thread = threading.Thread(target=self._drain, name="obs-recorder", daemon=True)
        self._thread.start()

    def record(self, *, t, episode, step, tick_dt_ms, infer_ms, tick_ms, tau_live,
               obs, action, target, motor_tau=None, ball_obs=None, pred=None):
        """热路径调用：obs/action/target 须已是 python float 列表。永不阻塞。
        motor_tau=电机反馈的 7 关节转矩列表；None（真机未发 effort）时写 NaN。
        ball_obs=(x,y,z) 原始球位置；pred=(pred_y,pred_z,pred_vx,pred_vy,pred_vz,pred_t)
        原始预测；两者 None（尚未收到）时对应列写 NaN。"""
        row = [t, episode, step, tick_dt_ms, infer_ms, tick_ms, tau_live]
        row.extend(obs)
        row.extend(action)
        row.extend(target)
        row.extend(motor_tau if motor_tau is not None else [float("nan")] * self._act_dim)
        row.extend(ball_obs if ball_obs is not None else [float("nan")] * 3)
        row.extend(pred if pred is not None else [float("nan")] * 6)
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self._dropped += 1

    def _drain(self):
        pending = 0
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                break
            self._w.writerow(item)
            self._written += 1
            pending += 1
            if pending >= self._flush_every or self._q.empty():
                self._f.flush()
                pending = 0
        self._f.flush()

    def close(self, timeout=5.0):
        """发哨兵、等后台线程排干队列并 flush，返回统计。"""
        self._q.put(_SENTINEL)
        self._thread.join(timeout=timeout)
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
        return {"written": self._written, "dropped": self._dropped, "path": self._path}
