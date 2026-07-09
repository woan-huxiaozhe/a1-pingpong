"""Pure-torch hit-time reference-tracking reward kernels (no isaaclab import, unit-testable)."""

from __future__ import annotations

import torch


def time_gate(tau: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """Gaussian gate centered at the hit instant tau=0 (tau, sigma_t in seconds)."""
    return torch.exp(-0.5 * (tau / sigma_t) ** 2)


def gaussian_score(error_norm: torch.Tensor, sigma: float, tol: float = 0.0) -> torch.Tensor:
    """exp(-max(0, ||e|| - tol)^2 / (2 sigma^2)): a flat top (=1) inside the tolerance box
    ``||e|| <= tol`` with Gaussian decay beyond it. ``tol=0.0`` recovers the plain Gaussian.

    Why the flat top: the pos/vel/normal terms are otherwise unbounded Gaussians that keep rewarding
    error->0, so the optimizer perpetually trades the cheapest term against the others (closing verr
    physically raises normal via the wrist tumble/torque saturation, so vel and normal fight forever
    along a fixed Pareto front). Zeroing the marginal gradient once a term is INSIDE its tolerance
    stops that tug-of-war -- a "good enough" term no longer pays to over-optimize at another's expense
    -- and concentrates gradient on the errors still OUTSIDE tolerance (the hard tail serves)."""
    e = torch.clamp(error_norm - tol, min=0.0)
    return torch.exp(-(e**2) / (2.0 * sigma * sigma))


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
    pos_tol: float = 0.0,
    vel_tol: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Additive, time-gated position + (full-vector) velocity tracking terms. Each [N].

    **Position tracks a MOVING reference**, not the static crossing point. ``p_ref`` is where the
    racket should be at the hit instant (tau=0) and ``v_ref`` is its velocity there, so the correct
    constant-velocity approach line is ``p_ref(tau) = p_ref - v_ref*tau``: at ``tau`` seconds BEFORE
    the hit (tau>0) the paddle should sit ``v_ref*tau`` behind ``p_ref`` along the swing, and AFTER
    it (tau<0) it continues ``v_ref*|tau|`` past into the follow-through. This makes the position and
    velocity targets mutually consistent over the WHOLE time-gate window instead of only at tau=0.

    Why it matters: with a static ``p_ref`` the broad time gate (~0.06 s at sigma_t=0.03) rewards
    being near the fixed point anywhere in the window, so a paddle can (a) park early at ``p_ref`` and
    still collect position reward while barely moving, and (b) let the true crossing "slide"
    ~``|v_ref|*window`` (≈7 cm at 1.2 m/s) along the swing -- both fight the velocity term, which
    wants a paddle streaming through at ``v_ref``. The moving reference removes the degeneracy
    (parking / sliding now lands off the line and loses position reward) and turns the broad gate into
    a dense approach-*trajectory* signal -- the traditional-controller cruise pattern -- rather than a
    single smeared point. The velocity target is already constant (= the cruise velocity), so only
    position needed the fix; ``sigma_t`` stays broad on purpose (dense signal, matches the ~±30 ms
    real cruise plateau)."""
    gate = time_gate(tau, sigma_t)
    p_ref_t = p_ref - v_ref * tau.unsqueeze(-1)  # constant-velocity approach line through p_ref @ tau=0
    pos_err = torch.norm(p_racket - p_ref_t, dim=-1)
    vel_err = torch.norm(v_racket - v_ref, dim=-1)
    pos_term = w_pos * gate * gaussian_score(pos_err, sigma_p, tol=pos_tol)
    vel_term = w_vel * gate * gaussian_score(vel_err, sigma_v, tol=vel_tol)
    return pos_term, vel_term


def normal_align_term(
    n_racket: torch.Tensor,
    n_ref: torch.Tensor,
    tau: torch.Tensor,
    *,
    sigma_t: float,
    sigma_normal_deg: float,
    w_normal: float,
    normal_tol_deg: float = 0.0,
) -> torch.Tensor:
    """Time-gated blade-normal alignment term [N] using a Gaussian over angular error."""
    gate = time_gate(tau, sigma_t)
    cos = (n_racket * n_ref).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos)
    deg2rad = torch.pi / 180.0
    sigma = torch.as_tensor(sigma_normal_deg, dtype=angle.dtype, device=angle.device) * deg2rad
    tol = float(normal_tol_deg) * deg2rad
    return w_normal * gate * gaussian_score(angle, sigma, tol=tol)


def face_still_term(
    n_racket: torch.Tensor,
    ang_vel: torch.Tensor,
    v_racket: torch.Tensor,
    v_ref: torch.Tensor,
    tau: torch.Tensor,
    *,
    sigma_t: float,
    sigma_rate: float,
    sigma_v: float,
    w: float,
) -> torch.Tensor:
    """Time-gated reward [N] for a *still* blade face while actually swinging through the hit.

    ``normal_align_term`` rewards WHERE the face points; this rewards that it is not TUMBLING when
    it gets there. The face-normal turn rate is ``|dn/dt| = |omega x n|`` (rad/s): only omega
    perpendicular to ``n`` tilts the face, so paddle spin *about* its own normal is correctly ignored.

    **Speed coupling (the fix that matters):** the raw face-still Gaussian is trivially maximised by
    NOT MOVING -- ``|omega x n| -> 0`` when the whole arm decelerates -- so on its own it hands the
    degenerate "arrive slow with a perfect still face" solution a free bonus (observed: run
    2026-07-07 collapsed to verr~1.16 with normal 2deg and near-full still-face reward). We multiply
    the face-still score by the SAME velocity-matching Gaussian ``exp(-verr^2/2 sigma_v^2)`` that the
    velocity term uses, so a still face pays ONLY to the extent the paddle is also swinging at v_ref.
    "Still because stopped" -> speed_gate ~= 0 -> zero reward; "still because proximal-sourced while
    swinging fast" (the traditional cruise) -> speed_gate ~= 1 -> full reward. This makes a steady,
    proximally-driven sweep the reward-maximising way to arrive instead of a distal wrist snap (which
    tumbles the face at ~3 rad/s and is why normal_err spikes exactly at peak speed). Same tau-gate
    as the other hit terms (does NOT move the gate)."""
    gate = time_gate(tau, sigma_t)
    face_rate = torch.norm(torch.cross(ang_vel, n_racket, dim=-1), dim=-1)
    vel_err = torch.norm(v_racket - v_ref, dim=-1)
    speed_gate = gaussian_score(vel_err, sigma_v)  # no still-face credit unless swinging at v_ref
    return w * gate * gaussian_score(face_rate, sigma_rate) * speed_gate
