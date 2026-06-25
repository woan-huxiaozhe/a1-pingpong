from __future__ import annotations

import torch

from unitree_rl_lab.tasks.table_tennis_sac.sac import SACAgent, TanhGaussianActor, mlp


def test_aux_reconstruction_head_uses_critic_private_block():
    agent = SACAgent(actor_obs_dim=60, critic_obs_dim=92, action_dim=7, device="cpu")
    assert agent.actor.aux_head is not None
    assert agent.actor.aux_head.out_features == 32

    batch = {
        "obs_actor": torch.randn(8, 60),
        "obs_critic": torch.randn(8, 92),
        "action": torch.tanh(torch.randn(8, 7)),
        "reward": torch.randn(8, 1),
        "next_obs_actor": torch.randn(8, 60),
        "next_obs_critic": torch.randn(8, 92),
        "done": torch.zeros(8, 1),
    }
    losses = agent.update(batch)
    assert "policy_loss" in losses
    assert "aux_reconstruction_loss" in losses
    assert torch.isfinite(torch.tensor(losses["aux_reconstruction_loss"]))


def test_actor_loads_legacy_pre_aux_layout():
    legacy = mlp(60, (512, 256, 128), 14, "elu")
    legacy_state = {"net." + key: value for key, value in legacy.state_dict().items()}
    actor = TanhGaussianActor(60, 7, (512, 256, 128), "elu", aux_target_dim=32)

    exact = actor.load_compatible_state_dict(legacy_state)
    assert exact is False
    assert actor.aux_head is not None
