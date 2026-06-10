from __future__ import annotations

import torch


def compute_joint_delta_target(
    joint_pos: torch.Tensor,
    action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    *,
    action_scale: float | torch.Tensor = 0.12,
    previous_target: torch.Tensor | None = None,
    smoothing: float = 1.0,
    max_delta_per_step: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Map normalized actions to bounded joint position targets.

    The policy action is interpreted as a delta from the current joint position,
    then optionally smoothed and rate-limited in target space before joint-limit
    clamping. This helper is intentionally Isaac-free so the clamp/rate logic can
    be regression-tested without launching the simulator.
    """

    raw_target = joint_pos + action.clamp(-1.0, 1.0) * action_scale
    target = raw_target if previous_target is None else previous_target + smoothing * (raw_target - previous_target)

    if previous_target is not None and max_delta_per_step is not None:
        delta = (target - previous_target).clamp(-max_delta_per_step, max_delta_per_step)
        target = previous_target + delta

    return torch.max(torch.min(target, upper_limits), lower_limits)
