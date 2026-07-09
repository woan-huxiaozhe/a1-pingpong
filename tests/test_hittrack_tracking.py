from __future__ import annotations

import math

import torch

from _hittrack_loader import load_pure

_tracking = load_pure("tracking")
time_gate = _tracking.time_gate
gaussian_score = _tracking.gaussian_score
hit_track_terms = _tracking.hit_track_terms
normal_align_term = _tracking.normal_align_term
approach_gate = _tracking.approach_gate
approach_vel_term = _tracking.approach_vel_term


def test_time_gate_peaks_at_zero():
    tau = torch.tensor([0.0, 0.03, -0.03])
    g = time_gate(tau, sigma_t=0.03)
    assert torch.isclose(g[0], torch.tensor(1.0))
    assert torch.isclose(g[1], torch.tensor(math.exp(-0.5)), atol=1e-6)
    assert torch.isclose(g[1], g[2])  # symmetric


def test_gaussian_score_perfect_and_decay():
    assert torch.isclose(gaussian_score(torch.tensor(0.0), 0.03), torch.tensor(1.0))
    assert gaussian_score(torch.tensor(0.06), 0.03) < gaussian_score(torch.tensor(0.03), 0.03)


def test_hit_track_terms_perfect_hit():
    p = torch.zeros(1, 3)
    v = torch.zeros(1, 3)
    pos_term, vel_term = hit_track_terms(
        p, v, p_ref=p.clone(), v_ref=v.clone(), tau=torch.zeros(1),
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=20.0, w_vel=20.0)
    assert torch.isclose(pos_term[0], torch.tensor(20.0))
    assert torch.isclose(vel_term[0], torch.tensor(20.0))


def test_hit_track_terms_velocity_is_full_vector():
    # tangential velocity error must be penalized (full-vector, D3=a)
    p = torch.zeros(1, 3)
    v = torch.tensor([[0.0, 1.0, 0.0]])
    v_ref = torch.zeros(1, 3)
    _, vel_term = hit_track_terms(p, v, p, v_ref, torch.zeros(1),
        sigma_t=0.03, sigma_p=0.03, sigma_v=0.3, w_pos=20.0, w_vel=20.0)
    assert vel_term[0] < 20.0  # 1 m/s tangential error reduces the score


def test_normal_align_term_uses_angular_gaussian():
    n_ref = torch.tensor([[1.0, 0.0, 0.0]])
    n_same = torch.tensor([[1.0, 0.0, 0.0]])
    n_20deg = torch.tensor([[math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0]])
    tau = torch.zeros(1)

    perfect = normal_align_term(n_same, n_ref, tau, sigma_t=0.03, sigma_normal_deg=20.0, w_normal=1.0)
    off = normal_align_term(n_20deg, n_ref, tau, sigma_t=0.03, sigma_normal_deg=20.0, w_normal=1.0)

    assert torch.isclose(perfect[0], torch.tensor(1.0))
    assert torch.isclose(off[0], torch.tensor(math.exp(-0.5)), atol=1e-6)


def test_approach_gate_zero_at_hit_alive_in_windup():
    # carved to ZERO at tau=0 (the sharp core gate owns the hit instant), alive in the wind-up
    tau = torch.tensor([0.0, 0.05, 0.10])
    g = approach_gate(tau, sigma_t_wide=0.09, sigma_t_core=0.03)
    assert g[0] < 1e-6
    assert g[1] > 0.3 and g[2] > 0.1


def test_approach_vel_rewards_cruise_in_windup_not_at_hit():
    v_ref = torch.tensor([[1.2, 0.0, 0.0]])
    kw = dict(sigma_t_wide=0.09, sigma_t_core=0.03, sigma_v=0.7, w=10.0)
    # at the hit instant the gate is ~0, so even a perfect velocity match pays ~nothing here
    at_hit = approach_vel_term(v_ref, v_ref, torch.zeros(1), **kw)
    assert at_hit[0] < 1e-3
    # in the wind-up, cruising at v_ref pays more than being at rest, and the broad sigma_v keeps a
    # non-zero (bootstrap) gradient even from rest
    tau = torch.full((1,), 0.10)
    matched = approach_vel_term(v_ref, v_ref.clone(), tau, **kw)
    at_rest = approach_vel_term(torch.zeros(1, 3), v_ref, tau, **kw)
    assert matched[0] > at_rest[0] > 0.0
