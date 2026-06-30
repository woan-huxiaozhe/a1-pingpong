"""HitTrack termination terms (ball/tracker-free). Used only by the A1-Pingpong-HitTrack task."""

from __future__ import annotations

import torch


def hit_window_elapsed(env) -> torch.Tensor:
    """End the episode once the hit-time tracking window has fully closed.

    True once ``episode_length_buf > hit_step + round(post_margin/step_dt)`` (i.e.
    ``tau_true < -post_margin``). Reads the ``_ht_*`` buffers maintained by
    ``mdp.reference_commands`` and is ball/tracker-free.
    """
    margin = getattr(env, "_ht_post_margin_steps", 12)
    done = env.episode_length_buf.to(torch.long) > (env._ht_hit_step + margin)
    return done
