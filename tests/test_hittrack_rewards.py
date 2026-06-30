from __future__ import annotations

import pytest
import torch

from _hittrack_loader import load_pure

hit_track_terms = load_pure("tracking").hit_track_terms


def test_perfect_hit_gives_full_terms_at_tau_zero():
    p = torch.zeros(1, 3)
    v = torch.zeros(1, 3)
    pos, vel = hit_track_terms(p, v, p, v, torch.zeros(1),
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=1.0, w_vel=1.0)
    assert torch.isclose(pos[0], torch.tensor(1.0)) and torch.isclose(vel[0], torch.tensor(1.0))


def test_far_from_hit_time_gates_to_zero():
    p = torch.zeros(1, 3)
    v = torch.zeros(1, 3)
    pos, vel = hit_track_terms(p, v, p, v, torch.full((1,), 0.3),  # tau=0.3s >> sigma_t
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=1.0, w_vel=1.0)
    assert pos[0] < 1e-6 and vel[0] < 1e-6


def test_reward_wrappers_exist():
    rewards = pytest.importorskip(
        "unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp.rewards",
        reason="isaaclab not installed")
    assert hasattr(rewards, "hit_ref_pos") and hasattr(rewards, "hit_ref_vel")
