"""Pure-function (no Isaac) checks for the sparse/event-heavy SAC reward ladder.

The reward functions in ``mdp/rewards.py`` import ``isaaclab`` transitively, which is not
available in CI. These tests therefore re-implement the *pure* numerical kernel of each new
terminal reward against a lightweight fake ``env`` (a namespace holding the ``_sac_*`` cache
tensors and ``_sac_step_event_mask``), mirroring the source one-to-one. They guard the three
properties the refactor depends on:

  (a) each terminal reward fires only under its own event bit;
  (b) the active outcome ladder keeps bad-hit terminal outcomes below cross-net/valid return;
  (c) ``racket_spin_penalty`` rises with contact angular velocity;
  (d) the ungated ``landing_placement`` scores an edge landing and a center landing equally
      (i.e. removing the center gate makes placement depend only on landing position).

Keep this file Isaac-free: do NOT import from ``mdp.rewards``/``mdp.events`` directly.
"""

from __future__ import annotations

import math
import types

import torch

# Event bit layout, mirrored from table_tennis_sac/event_tags.py (kept in sync by hand; it is a
# tiny stable enum). Re-importing event_tags is Isaac-free, but inlining keeps the test fully
# self-contained and documents the contract the rewards rely on.
EVENT_TAGS = ("near_miss", "hit", "bad_hit", "return", "valid_return", "miss")
EVENT_TO_BIT = {name: 1 << idx for idx, name in enumerate(EVENT_TAGS)}


def _fired(env, event):
    return (env._sac_step_event_mask & EVENT_TO_BIT[event]) != 0


# --- pure kernels mirroring mdp/rewards.py (the four new Ace terminal rewards) ---


def miss_approach(env, sigma=8.0):
    fired = _fired(env, "miss")
    min_dist = torch.nan_to_num(env._sac_min_dist, nan=0.0, posinf=1.0e3, neginf=1.0e3)
    score = torch.exp(-sigma * min_dist * min_dist)
    return torch.where(fired, score, torch.zeros_like(score))


def table_proximity(env, table_x_min=0.0, table_x_max=1.37, table_y_half=0.7625, scale=1.0, floor=-1.0):
    fired = _fired(env, "bad_hit")
    cross_x = env._sac_table_cross_x
    cross_y = env._sac_table_cross_y
    x_progress = (cross_x - table_x_min) / max(scale, 1.0e-6)
    y_miss = (cross_y.abs() - table_y_half).clamp(min=0.0) / max(scale, 1.0e-6)
    score = (x_progress - y_miss).clamp(min=floor, max=1.0)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(fired, score, torch.zeros_like(score))


def racket_spin_penalty(env, scale=12.0):
    fired = _fired(env, "hit")
    ang_vel = torch.nan_to_num(env._sac_hit_racket_ang_vel, nan=0.0, posinf=0.0, neginf=0.0)
    score = (ang_vel / max(scale, 1.0e-6)).clamp(min=0.0, max=1.0)
    return torch.where(fired, score, torch.zeros_like(score))


def flat_return(env, ref_height=1.4, band=0.4):
    fired = _fired(env, "valid_return")
    max_height = torch.nan_to_num(env._sac_post_hit_max_height, nan=ref_height, posinf=ref_height, neginf=ref_height)
    score = ((ref_height - max_height) / max(band, 1.0e-6)).clamp(min=0.0, max=1.0)
    return torch.where(fired, score, torch.zeros_like(score))


def net_clearance_score(clearance, ramp_low=-0.3, ramp_high=0.1):
    positive = (clearance / max(ramp_high, 1.0e-6)).clamp(min=0.0, max=1.0)
    negative = -(clearance / min(ramp_low, -1.0e-6)).clamp(min=0.0, max=1.0)
    return torch.where(clearance >= 0.0, positive, negative)


def landing_placement(env, target_x, target_y=0.0, sigma_x=0.25, sigma_y=0.3):
    fired = _fired(env, "valid_return")
    dx = env._sac_landing_x - target_x
    dy = env._sac_landing_y - target_y
    score = torch.exp(-(dx * dx / (2.0 * sigma_x**2) + dy * dy / (2.0 * sigma_y**2)))
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(fired, score, torch.zeros_like(score))


def _make_env(n=4):
    """Fake env: one row per outcome (miss / hit-only / bad_hit / valid_return)."""
    env = types.SimpleNamespace()
    nan = float("nan")
    env._sac_step_event_mask = torch.zeros(n, dtype=torch.long)
    env._sac_min_dist = torch.full((n,), float("inf"))
    env._sac_table_cross_x = torch.full((n,), nan)
    env._sac_table_cross_y = torch.full((n,), nan)
    env._sac_hit_racket_ang_vel = torch.full((n,), nan)
    env._sac_post_hit_max_height = torch.full((n,), float("-inf"))
    env._sac_landing_x = torch.full((n,), nan)
    env._sac_landing_y = torch.full((n,), nan)
    return env


def test_each_reward_fires_only_under_its_event_bit():
    env = _make_env(4)
    # row0 miss, row1 hit, row2 bad_hit, row3 valid_return
    env._sac_step_event_mask[0] = EVENT_TO_BIT["miss"]
    env._sac_step_event_mask[1] = EVENT_TO_BIT["hit"]
    env._sac_step_event_mask[2] = EVENT_TO_BIT["bad_hit"]
    env._sac_step_event_mask[3] = EVENT_TO_BIT["valid_return"]
    # populate caches so every term *could* be non-zero if it (wrongly) fired
    env._sac_min_dist[:] = 0.1
    env._sac_table_cross_x[:] = 0.6
    env._sac_table_cross_y[:] = 0.0
    env._sac_hit_racket_ang_vel[:] = 6.0
    env._sac_post_hit_max_height[:] = 1.0
    env._sac_landing_x[:] = 0.6
    env._sac_landing_y[:] = 0.0

    ma = miss_approach(env)
    assert ma[0] > 0.0
    assert torch.all(ma[[1, 2, 3]] == 0.0)

    spin = racket_spin_penalty(env)
    assert spin[1] > 0.0
    assert torch.all(spin[[0, 2, 3]] == 0.0)

    tp = table_proximity(env)
    assert tp[2] > 0.0
    assert torch.all(tp[[0, 1, 3]] == 0.0)

    fr = flat_return(env)
    assert fr[3] > 0.0
    assert torch.all(fr[[0, 1, 2]] == 0.0)


def test_ladder_keeps_bad_hit_below_cross_net_and_return_tier():
    weights = {
        "hit_bonus": 20.0,
        "return_cross_net": 40.0,
        "table_proximity": 10.0,
        "bad_hit": -50.0,
        "racket_spin_penalty": -2.0,
        "return_bonus": 100.0,
        "landing_placement": 25.0,
        "flat_return": 10.0,
    }
    # Miss has no active positive reward in the sparse/event-heavy pass.
    miss_total = 0.0

    # Best-case terminal bad-hit still must not out-earn crossing the net.
    env = _make_env(1)
    env._sac_step_event_mask[0] = EVENT_TO_BIT["hit"] | EVENT_TO_BIT["bad_hit"]
    env._sac_table_cross_x[0] = 1.0
    env._sac_table_cross_y[0] = 0.0
    env._sac_hit_racket_ang_vel[0] = 0.0  # cleanest contact -> no penalty
    hit_total = (
        weights["hit_bonus"] * 1.0
        + weights["table_proximity"] * float(table_proximity(env)[0])
        + weights["bad_hit"] * 1.0
        + weights["racket_spin_penalty"] * float(racket_spin_penalty(env)[0])
    )

    # A crossed-net return should dominate a bad-hit bridge.
    cross_total = weights["hit_bonus"] + weights["return_cross_net"]

    # Best-case return tier (a valid_return also implies the hit event fired earlier).
    env = _make_env(1)
    env._sac_step_event_mask[0] = EVENT_TO_BIT["hit"] | EVENT_TO_BIT["return"] | EVENT_TO_BIT["valid_return"]
    env._sac_landing_x[0] = 0.6  # at target center -> placement 1
    env._sac_landing_y[0] = 0.0
    env._sac_post_hit_max_height[0] = 0.9  # flat drive -> flat_return 1
    env._sac_hit_racket_ang_vel[0] = 0.0
    return_total = (
        weights["hit_bonus"] * 1.0
        + weights["return_cross_net"] * 1.0
        + weights["return_bonus"] * 1.0
        + weights["landing_placement"] * float(landing_placement(env, target_x=0.6)[0])
        + weights["flat_return"] * float(flat_return(env)[0])
        + weights["racket_spin_penalty"] * float(racket_spin_penalty(env)[0])
    )

    assert hit_total <= miss_total, (hit_total, miss_total)
    assert hit_total < cross_total, (hit_total, cross_total)
    assert cross_total < return_total, (cross_total, return_total)
    # Effective-value (post step_dt) ladder caps from the plan also hold.
    assert math.isclose(weights["hit_bonus"] * 0.02, 0.40)
    assert math.isclose(weights["return_cross_net"] * 0.02, 0.80)
    assert math.isclose(weights["bad_hit"] * 0.02, -1.00)
    assert math.isclose(weights["return_bonus"] * 0.02, 2.00)


def test_net_clearance_score_is_negative_below_net_and_positive_above():
    clearance = torch.tensor([-0.6, -0.15, 0.0, 0.05, 0.2])
    out = net_clearance_score(clearance, ramp_low=-0.3, ramp_high=0.1)
    expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert torch.allclose(out, expected)


def test_racket_spin_penalty_rises_with_angular_velocity():
    env = _make_env(3)
    env._sac_step_event_mask[:] = EVENT_TO_BIT["hit"]
    env._sac_hit_racket_ang_vel[0] = 1.0
    env._sac_hit_racket_ang_vel[1] = 6.0
    env._sac_hit_racket_ang_vel[2] = 30.0  # beyond scale -> clamps to 1
    out = racket_spin_penalty(env, scale=12.0)
    assert out[0] < out[1] < out[2]
    assert math.isclose(float(out[2]), 1.0)


def test_ungated_landing_placement_scores_edge_and_center_equally_by_position():
    """Removing the center gate => placement depends only on landing position, not on where
    the ball struck the blade. Two valid returns landing at the same spot must score the same
    regardless of any (now-unused) contact-offset state."""
    env = _make_env(2)
    env._sac_step_event_mask[:] = EVENT_TO_BIT["valid_return"]
    # Both land at the exact target center; there is no center-offset input to the ungated form.
    env._sac_landing_x[:] = 0.6
    env._sac_landing_y[:] = 0.0
    out = landing_placement(env, target_x=0.6)
    assert math.isclose(float(out[0]), float(out[1]))
    assert math.isclose(float(out[0]), 1.0, rel_tol=1e-6)

    # A landing off-center scores strictly less than the centered landing (gradient preserved).
    env_edge = _make_env(1)
    env_edge._sac_step_event_mask[0] = EVENT_TO_BIT["valid_return"]
    env_edge._sac_landing_x[0] = 0.6 + 0.4
    env_edge._sac_landing_y[0] = 0.3
    edge = landing_placement(env_edge, target_x=0.6)
    assert float(edge[0]) < float(out[0])
