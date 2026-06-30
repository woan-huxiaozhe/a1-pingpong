from __future__ import annotations

import types

import pytest
import torch

term = pytest.importorskip("unitree_rl_lab.tasks.a1_pingpong_hittrack.mdp.terminations",
                           reason="isaaclab not installed")


def test_done_after_post_margin():
    env = types.SimpleNamespace(num_envs=2, device="cpu")
    env._ht_hit_step = torch.tensor([50, 50])
    env._ht_post_margin_steps = 12
    env.episode_length_buf = torch.tensor([55, 63])  # 55<=62 not done; 63>62 done
    done = term.hit_window_elapsed(env)
    assert not bool(done[0]) and bool(done[1])
