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
    w_normal: float,
) -> torch.Tensor:
    """Time-gated blade-normal alignment term [N].

    ``n_racket`` and ``n_ref`` are (approximately) unit direction vectors, so alignment is their
    dot product. Uses a cosine kernel ``0.5*(1+cos)`` rather than a narrow Gaussian on the angle:
    at the current ~100 deg error a Gaussian gives ~0 gradient, whereas ``0.5*(1+cos)`` has a
    near-maximal restoring gradient around 90-100 deg and degrades to a quadratic near alignment
    (so it also fine-tunes). Gated identically to the pos/vel terms so orientation is only enforced
    around the hit instant.
    """
    gate = time_gate(tau, sigma_t)
    cos = (n_racket * n_ref).sum(dim=-1).clamp(-1.0, 1.0)
    return w_normal * gate * 0.5 * (1.0 + cos)
