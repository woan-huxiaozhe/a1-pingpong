"""Pure-torch synthetic reference-source kernels (Isaac-free): sample hit-plane ball states,
build tau countdown streams, optional KF-style phase-scaled noise, loose reachability gate."""

from __future__ import annotations

import torch


def sample_hit_ball_states(n: int, box: dict, hit_plane_x: float, *, device, gen=None) -> torch.Tensor:
    """``[n,6]`` ``[x,y,z,vx,vy,vz]`` with ``x=hit_plane_x``; other dims uniform in ``box``."""

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)

    y, z = u(*box["y"]), u(*box["z"])
    vx, vy, vz = u(*box["vx"]), u(*box["vy"]), u(*box["vz"])
    x = torch.full((n,), hit_plane_x, device=device)
    return torch.stack([x, y, z, vx, vy, vz], dim=-1)


def tau_streams(tau_initial: torch.Tensor, n_steps: int, step_dt: float) -> torch.Tensor:
    """``tau_true[k, s] = tau_initial[k] - s*step_dt`` -> ``[n, n_steps]``."""
    s = torch.arange(n_steps, device=tau_initial.device, dtype=tau_initial.dtype)
    return tau_initial.unsqueeze(1) - s.unsqueeze(0) * step_dt


def phase_scaled_ball_noise(tau, bias_unit, *, bias_std, jitter_std, fixed_offset, far_tau) -> torch.Tensor:
    """Per-episode-biased noise on ``(y,z,vx,vy,vz)``, magnitude scaled by ``clamp(tau/far_tau,0,1)``.
    Mirrors the calibrated KF model; defaults to all-zero stds (noise off). Returns ``[n,T,5]``."""
    device, dtype = tau.device, tau.dtype
    far = torch.tensor(far_tau, device=device, dtype=dtype).clamp(min=1e-6)
    phase = (tau.unsqueeze(-1) / far).clamp(0.0, 1.0)  # [n,T,5]
    bs = torch.tensor(bias_std, device=device, dtype=dtype)
    js = torch.tensor(jitter_std, device=device, dtype=dtype)
    off = torch.tensor(fixed_offset, device=device, dtype=dtype)
    bias = bias_unit.unsqueeze(1) * (bs * phase)  # [n,T,5]
    jitter = torch.randn(tau.shape[0], tau.shape[1], 5, device=device) * js
    return bias + jitter + off * phase


def is_reachable(p_hit: torch.Tensor, y_range, z_range) -> torch.Tensor:
    """Loose workspace box gate on the hit-plane ``(y,z)``. Returns ``[n]`` bool."""
    y, z = p_hit[:, 1], p_hit[:, 2]
    return (y >= y_range[0]) & (y <= y_range[1]) & (z >= z_range[0]) & (z <= z_range[1])
