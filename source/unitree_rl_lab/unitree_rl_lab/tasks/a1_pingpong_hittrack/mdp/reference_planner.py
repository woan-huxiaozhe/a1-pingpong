"""Runtime hit-reference planner (shared with deployment). Pure torch, no isaaclab import.

Converts a hit-plane ball state into the end-effector reference the arm policy tracks:
target landing -> drag-aware launch solve + closed-form contact inversion (mdp.hitting).
"""

from __future__ import annotations

import torch

from unitree_rl_lab.tasks.table_tennis_sac.mdp.hitting import (
    NEUTRAL_THETA,
    PADDLE_RESTITUTION,
    ideal_racket_velocity,
)


def plan_hit_reference(
    p_ball_hit: torch.Tensor,
    v_ball_hit: torch.Tensor,
    target: torch.Tensor,
    *,
    theta: float = NEUTRAL_THETA,
    restitution: float = PADDLE_RESTITUTION,
    drag_k: float = 0.08,
    lin_damp: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (p_ref, v_ref, n_ref). p_ref is the contact point; v_ref the ideal paddle
    velocity; n_ref the blade normal. All world/env-frame; the origin offset cancels."""
    v_paddle, _v_out, n = ideal_racket_velocity(
        p_ball_hit,
        v_ball_hit,
        target,
        theta=theta,
        restitution=restitution,
        drag_k=drag_k,
        lin_damp=lin_damp,
    )
    return p_ball_hit, v_paddle, n
