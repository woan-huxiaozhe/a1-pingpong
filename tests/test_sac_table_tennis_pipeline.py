from __future__ import annotations

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "unitree_rl_lab"))

from unitree_rl_lab.tasks.table_tennis_sac.control import compute_joint_delta_target
from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TO_BIT, encode_events
from unitree_rl_lab.tasks.table_tennis_sac.replay import (
    EpisodeInfo,
    EpisodeTraceBuffer,
    StratifiedReplaySampler,
    UniformReplayBuffer,
    extract_event_windows,
    make_event_tables,
    recompute_landing_y_reward,
)
from unitree_rl_lab.tasks.table_tennis_sac.sac import SACAgent, SACConfig


def _add_fake_batch(replay: UniformReplayBuffer, n: int, obs_dim: int = 5, critic_dim: int = 7, action_dim: int = 3):
    return replay.add_batch(
        obs_actor=torch.randn(n, obs_dim),
        obs_critic=torch.randn(n, critic_dim),
        action=torch.randn(n, action_dim).clamp(-1, 1),
        reward=torch.randn(n),
        next_obs_actor=torch.randn(n, obs_dim),
        next_obs_critic=torch.randn(n, critic_dim),
        done=torch.zeros(n, dtype=torch.bool),
        episode_id=torch.arange(n),
        step_index=torch.arange(n),
    )


def test_joint_delta_target_clamps_and_rate_limits():
    q = torch.zeros(1, 3)
    action = torch.tensor([[2.0, -2.0, 0.5]])
    lo = torch.tensor([[-0.05, -0.20, -1.0]])
    hi = torch.tensor([[0.20, 0.05, 1.0]])
    prev = torch.zeros_like(q)
    target = compute_joint_delta_target(
        q,
        action,
        lo,
        hi,
        action_scale=0.12,
        previous_target=prev,
        smoothing=1.0,
        max_delta_per_step=0.06,
    )
    assert torch.allclose(target, torch.tensor([[0.06, -0.06, 0.06]]))


def test_joint_delta_target_accepts_per_joint_rate_limits():
    q = torch.zeros(1, 3)
    action = torch.ones(1, 3)
    lo = torch.full((1, 3), -1.0)
    hi = torch.full((1, 3), 1.0)
    prev = torch.zeros_like(q)
    target = compute_joint_delta_target(
        q,
        action,
        lo,
        hi,
        action_scale=0.2,
        previous_target=prev,
        smoothing=1.0,
        max_delta_per_step=torch.tensor([0.04, 0.08, 0.12]),
    )
    assert torch.allclose(target, torch.tensor([[0.04, 0.08, 0.12]]))


def test_replay_insert_overwrite_sampling_shapes_and_versions():
    replay = UniformReplayBuffer(4)
    old_indices, old_versions = _add_fake_batch(replay, 3)
    _add_fake_batch(replay, 3)
    assert len(replay) == 4
    assert not replay.is_valid(old_indices[:1], old_versions[:1]).item()
    batch = replay.sample(2)
    assert batch["obs_actor"].shape == (2, 5)
    assert batch["obs_critic"].shape == (2, 7)
    assert batch["action"].shape == (2, 3)
    assert batch["event_mask"].shape == (2,)


def test_event_windows_and_trace_finalization():
    info = EpisodeInfo(
        event_mask=encode_events(["near_miss", "hit", "return", "valid_return"]),
        closest_step=12,
        hit_step=20,
        return_step=27,
        valid_return_step=35,
    )
    windows = extract_event_windows(50, info)
    assert windows["near_miss"][0] == 0
    assert windows["near_miss"][-1] == 12
    assert windows["hit"][0] == 0
    assert windows["hit"][-1] == 30
    assert windows["valid_return"][-1] == 35

    trace = EpisodeTraceBuffer(num_envs=1)
    trace.append(torch.zeros(40, dtype=torch.long), torch.arange(40), torch.arange(100, 140))
    finalized = trace.finalize(0, info)
    assert "hit" in finalized
    hit_indices, hit_versions = finalized["hit"]
    assert hit_indices[0].item() == 0
    assert hit_versions[-1].item() == 130


def test_stratified_sampler_falls_back_to_uniform_and_uses_valid_table_entries():
    replay = UniformReplayBuffer(32)
    indices, versions = _add_fake_batch(replay, 20)
    tables = make_event_tables(capacity=64)
    sampler = StratifiedReplaySampler(replay, tables)
    batch, composition = sampler.sample(16)
    assert batch["action"].shape[0] == 16
    assert composition["uniform"] == 16

    tables["near_miss"].add(indices[:8], versions[:8])
    replay.or_event_masks(indices[:8], EVENT_TO_BIT["near_miss"])
    batch, composition = sampler.sample(16)
    assert batch["action"].shape[0] == 16
    assert composition["near_miss"] > 0


def test_her_landing_reward_recompute_skips_invalid_episodes():
    achieved = torch.tensor([0.0, 0.2, float("nan")])
    target = torch.tensor([0.0, 0.0, 0.0])
    reward = recompute_landing_y_reward(achieved, target, sigma=0.2)
    assert reward[0].item() == 1.0
    assert 0.0 < reward[1].item() < 1.0
    assert reward[2].item() == 0.0


def test_sac_update_on_synthetic_batch():
    agent = SACAgent(
        actor_obs_dim=5,
        critic_obs_dim=7,
        action_dim=3,
        config=SACConfig(actor_hidden_dims=(16, 16), critic_hidden_dims=(16, 16)),
        device="cpu",
    )
    batch = {
        "obs_actor": torch.randn(32, 5),
        "obs_critic": torch.randn(32, 7),
        "action": torch.randn(32, 3).clamp(-1, 1),
        "reward": torch.randn(32, 1),
        "next_obs_actor": torch.randn(32, 5),
        "next_obs_critic": torch.randn(32, 7),
        "done": torch.zeros(32, 1),
    }
    losses = agent.update(batch)
    assert all(math.isfinite(value) for value in losses.values())


def test_sac_alpha_respects_minimum():
    agent = SACAgent(
        actor_obs_dim=5,
        critic_obs_dim=7,
        action_dim=3,
        config=SACConfig(
            actor_hidden_dims=(16, 16),
            critic_hidden_dims=(16, 16),
            initial_alpha=0.001,
            min_alpha=0.005,
        ),
        device="cpu",
    )
    batch = {
        "obs_actor": torch.randn(32, 5),
        "obs_critic": torch.randn(32, 7),
        "action": torch.randn(32, 3).clamp(-1, 1),
        "reward": torch.randn(32, 1),
        "next_obs_actor": torch.randn(32, 5),
        "next_obs_critic": torch.randn(32, 7),
        "done": torch.zeros(32, 1),
    }
    assert agent.alpha.item() >= 0.005 - 1.0e-7
    agent.update(batch)
    assert agent.alpha.item() >= 0.005 - 1.0e-7
