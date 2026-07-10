"""numpy/标量移植的击球参考规划与训练侧 torch 版逐球态对拍。

保证部署链换用 reference_planner_np.plan_hit_reference 后，obs 的 hit_reference / hit_ref_pos_error
与训练侧数值一致（差异远低于域随机化噪声），且不再依赖 batch-1 torch 的 dispatch 开销。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "deploy"))
from hittrack import config as C
from hittrack.pure_import import load_plan_hit_reference
from hittrack.reference_planner_np import plan_hit_reference as plan_np

_TARGET = torch.tensor([[*C.HITTRACK_TARGET_XYZ]], dtype=torch.float32)


def test_matches_torch_reference_over_ball_states():
    plan_torch = load_plan_hit_reference()
    rng = np.random.default_rng(0)
    max_v = max_n = 0.0
    for _ in range(200):
        py = rng.uniform(-0.35, 0.35); pz = rng.uniform(0.70, 1.30)
        vx = rng.uniform(-6.0, -2.0); vy = rng.uniform(-1.5, 1.5); vz = rng.uniform(-2.5, 1.0)
        p = torch.tensor([[C.HIT_PLANE_X, py, pz]], dtype=torch.float32)
        v = torch.tensor([[vx, vy, vz]], dtype=torch.float32)
        _, vt, nt = plan_torch(p, v, _TARGET, restitution=C.HIT_RESTITUTION)
        _, vn, nn = plan_np(p, v, _TARGET, restitution=C.HIT_RESTITUTION)
        max_v = max(max_v, float((vt - vn).abs().max()))
        max_n = max(max_n, float((nt - nn).abs().max()))
    # 二分求解速率精度 ~(12-0.5)/2^18≈4e-5，float32/64 差异更小；1e-3 已远超物理可辨阈值
    assert max_v < 1.0e-3, f"v_ref 偏差过大: {max_v}"
    assert max_n < 1.0e-3, f"n_ref 偏差过大: {max_n}"


def test_shapes_and_passthrough_pref():
    p = torch.tensor([[C.HIT_PLANE_X, 0.05, 1.0]], dtype=torch.float32)
    v = torch.tensor([[-4.0, 0.0, -0.5]], dtype=torch.float32)
    p_ref, v_ref, n_ref = plan_np(p, v, _TARGET, restitution=C.HIT_RESTITUTION)
    assert p_ref is p                                    # 接触点原样透传
    assert v_ref.shape == (1, 3) and n_ref.shape == (1, 3)
    assert v_ref.dtype == torch.float32 and n_ref.dtype == torch.float32
    assert torch.isfinite(v_ref).all() and torch.isfinite(n_ref).all()
    assert abs(float(torch.linalg.norm(n_ref)) - 1.0) < 1e-5   # 法向为单位向量


def test_batch_broadcast_target():
    P = torch.tensor([[C.HIT_PLANE_X, 0.1, 1.0], [C.HIT_PLANE_X, -0.2, 0.9]], dtype=torch.float32)
    V = torch.tensor([[-4.0, 0.3, -0.5], [-3.0, -0.4, 0.2]], dtype=torch.float32)
    _, v_ref, n_ref = plan_np(P, V, _TARGET, restitution=C.HIT_RESTITUTION)   # target [1,3] 广播
    assert v_ref.shape == (2, 3) and n_ref.shape == (2, 3)
