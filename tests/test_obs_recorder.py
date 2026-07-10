import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "deploy"))
from hittrack.recorder import ObsRecorder


def test_records_header_rows_and_flushes(tmp_path):
    p = str(tmp_path / "rec.csv")
    r = ObsRecorder(p, obs_dim=68, act_dim=7)
    for step in range(5):
        r.record(t=float(step), episode=1, step=step, tick_dt_ms=10.0, infer_ms=1.2,
                 tick_ms=2.3, tau_live=0.4, obs=[0.1] * 68, action=[0.2] * 7, target=[0.3] * 7,
                 motor_tau=[0.4] * 7, ball_obs=(1.0, 2.0, 3.0),
                 pred=(0.1, 0.2, -4.0, 0.3, -0.5, 0.42))
    stats = r.close()
    assert stats["written"] == 5 and stats["dropped"] == 0
    with open(p) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]; idx = {n: i for i, n in enumerate(hdr)}
    assert hdr[:7] == ["t", "episode", "step", "tick_dt_ms", "infer_ms", "tick_ms", "tau_live"]
    assert len(rows) == 6                                     # header + 5 行
    assert len(rows[1]) == 7 + 68 + 7 + 7 + 7 + 3 + 6         # 105 列：+mtau7 +ball3 +pred6
    row1 = rows[1]
    assert [float(row1[idx[f"mtau_{i}"]]) for i in range(7)] == [0.4] * 7
    assert [float(row1[idx[c]]) for c in ("ball_x", "ball_y", "ball_z")] == [1.0, 2.0, 3.0]
    assert ([float(row1[idx[c]]) for c in
             ("pred_y", "pred_z", "pred_vx", "pred_vy", "pred_vz", "pred_t")]
            == [0.1, 0.2, -4.0, 0.3, -0.5, 0.42])


def test_missing_optional_fields_write_nan(tmp_path):
    # 真机未发 effort/球观测/预测（缺省）时，mtau_*/ball_*/pred_* 应写 NaN 而非报错或错位。
    p = str(tmp_path / "rec.csv")
    r = ObsRecorder(p, obs_dim=68, act_dim=7)
    r.record(t=0.0, episode=1, step=0, tick_dt_ms=10.0, infer_ms=1.0, tick_ms=1.0,
             tau_live=0.0, obs=[0.0] * 68, action=[0.0] * 7, target=[0.0] * 7)
    r.close()
    with open(p) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]; idx = {n: i for i, n in enumerate(hdr)}
    assert len(rows[1]) == 105
    opt = ([f"mtau_{i}" for i in range(7)]
           + ["ball_x", "ball_y", "ball_z",
              "pred_y", "pred_z", "pred_vx", "pred_vy", "pred_vz", "pred_t"])
    assert all(math.isnan(float(rows[1][idx[c]])) for c in opt)


def test_put_never_blocks_when_full(tmp_path):
    # 队列满时 record 必须立刻返回并计数，绝不阻塞推理线程。
    p = str(tmp_path / "rec.csv")
    r = ObsRecorder(p, maxsize=1)
    for i in range(2000):
        r.record(t=float(i), episode=1, step=i, tick_dt_ms=10.0, infer_ms=1.0,
                 tick_ms=1.0, tau_live=0.0, obs=[0.0] * 68, action=[0.0] * 7, target=[0.0] * 7)
    stats = r.close()
    assert stats["written"] + stats["dropped"] == 2000   # 每行要么写入要么被丢弃，无丢失/无阻塞
