"""Pure-torch hit-time reference-tracking reward kernels (no isaaclab import, unit-testable)."""

from __future__ import annotations

import torch


def time_gate(tau: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """Gaussian gate centered at the hit instant tau=0 (tau, sigma_t in seconds)."""
    return torch.exp(-0.5 * (tau / sigma_t) ** 2)


def gaussian_score(error_norm: torch.Tensor, sigma: float) -> torch.Tensor:
    """exp(-||e||^2 / (2 sigma^2)) given the L2 norm ||e||."""
    return torch.exp(-(error_norm**2) / (2.0 * sigma * sigma))


def hit_track_terms(
    p_racket: torch.Tensor,
    v_racket: torch.Tensor,
    p_ref: torch.Tensor,
    v_ref: torch.Tensor,
    tau: torch.Tensor,
    *,
    sigma_t: float,
    sigma_p: float,
    sigma_v: float,
    w_pos: float,
    w_vel: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive, time-gated position + (full-vector) velocity tracking terms. Each [N]."""
    gate = time_gate(tau, sigma_t)
    pos_err = torch.norm(p_racket - p_ref, dim=-1)
    vel_err = torch.norm(v_racket - v_ref, dim=-1)
    pos_term = w_pos * gate * gaussian_score(pos_err, sigma_p)
    vel_term = w_vel * gate * gaussian_score(vel_err, sigma_v)
    return pos_term, vel_term


def normal_align_term(
    n_racket: torch.Tensor,
    n_ref: torch.Tensor,
    tau: torch.Tensor,
    *,
    sigma_t: float,
    sigma_normal_deg: float,
    w_normal: float,
) -> torch.Tensor:
    """Time-gated blade-normal alignment term [N] using a Gaussian over angular error."""
    gate = time_gate(tau, sigma_t)
    cos = (n_racket * n_ref).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos)
    sigma = torch.as_tensor(sigma_normal_deg, dtype=angle.dtype, device=angle.device) * torch.pi / 180.0
    return w_normal * gate * gaussian_score(angle, sigma)


def face_still_term(
    n_racket: torch.Tensor,
    ang_vel: torch.Tensor,
    tau: torch.Tensor,
    *,
    sigma_t: float,
    sigma_rate: float,
    w: float,
) -> torch.Tensor:
    """Time-gated reward [N] for a *still* blade face at the hit instant.

    ``normal_align_term`` rewards WHERE the face points; this rewards that it is not TUMBLING when
    it gets there. The face-normal turn rate is ``|dn/dt| = |omega x n|`` (rad/s): only omega
    perpendicular to ``n`` tilts the face, so paddle spin *about* its own normal is correctly ignored.
    A Gaussian over that rate makes a steady face (low ``|omega x n|``) the reward-maximising way to
    arrive -- which forces the swing speed to be sourced from the proximal joints (steady-face,
    like the traditional cruise at ~1 rad/s) instead of a distal wrist snap (which tumbles the face
    at ~3 rad/s and is why normal_err spikes exactly at peak speed). Same tau-gate as the other
    hit terms (does NOT move the gate)."""
    gate = time_gate(tau, sigma_t)
    face_rate = torch.norm(torch.cross(ang_vel, n_racket, dim=-1), dim=-1)
    return w * gate * gaussian_score(face_rate, sigma_rate)
