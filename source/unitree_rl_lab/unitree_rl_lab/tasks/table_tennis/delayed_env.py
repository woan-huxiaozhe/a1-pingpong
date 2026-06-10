"""ManagerBasedRLEnv with observation delay randomization."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.buffers import DelayBuffer


class DelayedObsEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that applies random observation delay to the policy group.

    Only the policy observation is delayed; the critic observation remains undelayed
    for more stable value function training.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._obs_delay_min = getattr(cfg, "obs_delay_min", 0)
        self._obs_delay_max = getattr(cfg, "obs_delay_max", 0)

        if self._obs_delay_max > 0:
            self._obs_delay_buffer = DelayBuffer(
                history_length=self._obs_delay_max,
                batch_size=self.num_envs,
                device=self.device,
            )
            delays = torch.randint(
                self._obs_delay_min, self._obs_delay_max + 1,
                (self.num_envs,), dtype=torch.int, device=self.device,
            )
            self._obs_delay_buffer.set_time_lag(delays)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._obs_delay_max > 0:
            self._obs_delay_buffer.reset(env_ids)
            delays = torch.randint(
                self._obs_delay_min, self._obs_delay_max + 1,
                (len(env_ids),), dtype=torch.int, device=self.device,
            )
            self._obs_delay_buffer.set_time_lag(delays, env_ids)

    def step(self, action):
        obs_buf, rew, term, trunc, extras = super().step(action)
        if self._obs_delay_max > 0:
            obs_buf["policy"] = self._obs_delay_buffer.compute(obs_buf["policy"])
        return obs_buf, rew, term, trunc, extras


class BackhandDelayedEnv(ManagerBasedRLEnv):
    """Backhand env (plan A4): ball-observation delay only, at physics sub-step granularity.

    Unlike DelayedObsEnv (which delays the whole policy observation), the backhand delays
    *only* the ball-derived observations. That delay is produced by the zero-dim
    BallObsDelayAction term (the per-sub-step hook), which writes a delayed ball snapshot
    onto this env. This class just guarantees the snapshot attributes exist with a safe
    default so the ball-obs functions fall back to the live state if the term is absent.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._delayed_ball_pos = None
        self._delayed_ball_vel = None
        super().__init__(cfg, render_mode, **kwargs)
