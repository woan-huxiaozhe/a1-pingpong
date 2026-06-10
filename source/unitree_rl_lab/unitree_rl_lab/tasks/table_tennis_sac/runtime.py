from __future__ import annotations

import torch

from unitree_rl_lab.tasks.table_tennis_sac.replay import EpisodeInfo


def unwrap_env(env):
    return getattr(env, "unwrapped", env)


def split_actor_critic_obs(obs) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(obs, dict):
        actor = obs.get("policy", obs.get("actor"))
        critic = obs.get("critic", actor)
        if actor is None:
            raise KeyError(f"Observation dict has no policy/actor key: {list(obs.keys())}")
        return actor, critic
    return obs, obs


def extract_final_episode_infos(env, done: torch.Tensor) -> dict[int, EpisodeInfo]:
    raw = unwrap_env(env)
    done_cpu = done.detach().cpu().bool().reshape(-1)
    infos: dict[int, EpisodeInfo] = {}
    if not torch.any(done_cpu):
        return infos

    def _get(name: str, env_id: int, default):
        value = getattr(raw, name, None)
        if value is None:
            return default
        item = value.detach().cpu()[env_id]
        if isinstance(default, float):
            return float(item)
        return int(item)

    for env_id in done_cpu.nonzero(as_tuple=True)[0].tolist():
        infos[env_id] = EpisodeInfo(
            event_mask=_get("_sac_final_event_mask", env_id, 0),
            min_dist=_get("_sac_final_min_dist", env_id, float("nan")),
            closest_step=_get("_sac_final_closest_step", env_id, -1),
            hit_step=_get("_sac_final_hit_step", env_id, -1),
            return_step=_get("_sac_final_return_step", env_id, -1),
            valid_return_step=_get("_sac_final_valid_return_step", env_id, -1),
            bad_hit_step=_get("_sac_final_bad_hit_step", env_id, -1),
            miss_step=_get("_sac_final_miss_step", env_id, -1),
            landing_y=_get("_sac_final_landing_y", env_id, float("nan")),
            hit_outgoing_speed=_get("_sac_final_hit_outgoing_speed", env_id, float("nan")),
            hit_up_speed=_get("_sac_final_hit_up_speed", env_id, float("nan")),
            post_hit_max_outgoing_speed=_get("_sac_final_post_hit_max_outgoing_speed", env_id, float("nan")),
            post_hit_max_height=_get("_sac_final_post_hit_max_height", env_id, float("nan")),
        )
    return infos
