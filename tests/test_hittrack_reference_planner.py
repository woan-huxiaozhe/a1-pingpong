from __future__ import annotations

import torch

from _hittrack_loader import load_pure

load_pure("hitting")  # resolve reference_planner's intra-package dependency first
plan_hit_reference = load_pure("reference_planner").plan_hit_reference


def _case():
    # ball crosses hit plane x=-1.37; ROBOT_SIDE=-1, ball travels -x toward robot
    p = torch.tensor([[-1.37, 0.0, 1.0]])
    v_in = torch.tensor([[-4.0, 0.0, -0.5]])  # incoming, toward -x
    target = torch.tensor([[0.685, 0.0, 0.76]])  # opponent table center
    return p, v_in, target


def test_shapes_and_p_ref_identity():
    p, v_in, target = _case()
    p_ref, v_ref, n_ref = plan_hit_reference(p, v_in, target)
    assert p_ref.shape == (1, 3) and v_ref.shape == (1, 3) and n_ref.shape == (1, 3)
    assert torch.allclose(p_ref, p)


def test_v_ref_parallel_to_normal():
    # contact_inverse returns a normal-only paddle velocity -> v_ref ∥ n_ref
    p, v_in, target = _case()
    _, v_ref, n_ref = plan_hit_reference(p, v_in, target)
    cross = torch.cross(v_ref, n_ref, dim=-1)
    assert torch.norm(cross) < 1e-4


def test_returns_ball_toward_opponent():
    # the planned outgoing direction must push the ball back toward +x (opponent)
    p, v_in, target = _case()
    _, v_ref, _ = plan_hit_reference(p, v_in, target)
    assert v_ref[0, 0] > 0.0  # paddle pushes in +x
