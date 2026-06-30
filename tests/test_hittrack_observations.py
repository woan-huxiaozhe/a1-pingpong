from __future__ import annotations

import types

import pytest
import torch

obs = pytest.importorskip(
    "unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp.observations",
    reason="isaaclab not installed")


def _fake_env(n=2):
    env = types.SimpleNamespace()
    env.num_envs = n
    env.device = "cpu"
    env._ht_p_ref_noisy = torch.zeros(n, 3)
    env._ht_v_ref_noisy = torch.ones(n, 3)
    env._ht_n_ref_noisy = torch.zeros(n, 3)
    env._ht_n_ref_noisy[:, 1] = 1.0
    env._ht_tau_noisy = torch.full((n,), 0.4)
    return env


def test_reference_command_is_10_dim_and_ordered():
    env = _fake_env()
    cmd = obs.hit_reference_command(env)
    assert cmd.shape == (2, 10)
    assert torch.allclose(cmd[:, 6:9], env._ht_n_ref_noisy)  # normal block
    assert torch.allclose(cmd[:, 9], env._ht_tau_noisy)  # tau last
