from __future__ import annotations

import torch

from _hittrack_loader import load_pure

_rs = load_pure("reference_source")
sample_hit_ball_states = _rs.sample_hit_ball_states
tau_streams = _rs.tau_streams
phase_scaled_ball_noise = _rs.phase_scaled_ball_noise
is_reachable = _rs.is_reachable

BOX = {"y": (-0.2, 0.3), "z": (0.9, 1.25), "vx": (-4.5, -3.0), "vy": (-0.3, 0.3), "vz": (-1.0, 0.5)}


def test_sample_fixes_x_and_respects_box():
    s = sample_hit_ball_states(64, BOX, hit_plane_x=-1.37, device="cpu", gen=torch.Generator().manual_seed(0))
    assert s.shape == (64, 6)
    assert torch.allclose(s[:, 0], torch.full((64,), -1.37))
    assert (s[:, 1] >= -0.2).all() and (s[:, 1] <= 0.3).all()
    assert (s[:, 3] <= -3.0).all() and (s[:, 3] >= -4.5).all()


def test_tau_streams_counts_down():
    tau0 = torch.tensor([0.5, 0.3])
    ts = tau_streams(tau0, n_steps=5, step_dt=0.01)
    assert ts.shape == (2, 5)
    assert torch.isclose(ts[0, 0], torch.tensor(0.5))
    assert torch.isclose(ts[0, 1], torch.tensor(0.49))


def test_noise_off_by_default_is_zero():
    tau = torch.full((3, 4), 0.4)
    bias = torch.randn(3, 5)
    noise = phase_scaled_ball_noise(tau, bias, bias_std=(0, 0, 0, 0, 0), jitter_std=(0, 0, 0, 0, 0),
                                    fixed_offset=(0, 0, 0, 0, 0), far_tau=(0.6, 0.4, 0.5, 0.6, 0.6))
    assert torch.allclose(noise, torch.zeros_like(noise))


def test_reachable_box():
    p = torch.tensor([[-1.37, 0.0, 1.0], [-1.37, 2.0, 1.0]])  # 2nd is far out in y
    ok = is_reachable(p, y_range=(-0.6, 0.6), z_range=(0.7, 1.5))
    assert bool(ok[0]) and not bool(ok[1])
