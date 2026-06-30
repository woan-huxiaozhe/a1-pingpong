from __future__ import annotations

import pytest


def test_cfg_constructs_at_100hz_with_tracking_rewards():
    cfgmod = pytest.importorskip(
        "unitree_rl_lab.tasks.a1_pingpong_hittrack.env_cfg", reason="isaaclab not installed")
    cfg = cfgmod.HitTrackEnvCfg()
    assert cfg.decimation == 2 and abs(cfg.sim.dt - 0.005) < 1e-9
    rew = cfg.rewards
    assert hasattr(rew, "hit_ref_pos") and hasattr(rew, "hit_ref_vel")
    # no ball-outcome reward terms wired
    for banned in ("hit_bonus", "return_cross_net", "bad_hit", "landing_placement"):
        assert not hasattr(rew, banned)


def test_task_registered():
    import sys

    import gymnasium as gym

    # The Isaac-free module loader (tests/_hittrack_loader.py) registers stub `unitree_rl_lab.*`
    # packages in sys.modules to bypass the isaaclab-importing package __init__. Those stubs would
    # shadow the real package here, so drop them and import the real one -- importing
    # `a1_pingpong_hittrack` only runs gym.register with lazy string entry points (no isaaclab), so
    # this stays Isaac-free.
    for name in [m for m in sys.modules if m == "unitree_rl_lab" or m.startswith("unitree_rl_lab.")]:
        del sys.modules[name]

    import unitree_rl_lab.tasks.a1_pingpong_hittrack  # noqa: F401  triggers registration
    assert "A1-Pingpong-HitTrack" in gym.registry


def test_task_registered_with_ppo_entry_point():
    """HitTrack is trained with RSL-RL PPO (not SAC): the registration must expose an
    `rsl_rl_cfg_entry_point` so `scripts/rsl_rl/train.py` / `play.py` can resolve the runner cfg.

    `gym.register` only stores the entry-point *string* (it never imports `isaaclab_rl.rsl_rl`,
    which needs USD/`pxr`), so this assertion stays Isaac-free -- it just inspects the stored kwargs.
    """
    import sys

    import gymnasium as gym

    for name in [m for m in sys.modules if m == "unitree_rl_lab" or m.startswith("unitree_rl_lab.")]:
        del sys.modules[name]

    import unitree_rl_lab.tasks.a1_pingpong_hittrack  # noqa: F401  triggers registration

    spec = gym.registry["A1-Pingpong-HitTrack"]
    assert spec.kwargs.get("rsl_rl_cfg_entry_point") == (
        "unitree_rl_lab.tasks.a1_pingpong_hittrack.agents.rsl_rl_ppo_cfg:HitTrackPPORunnerCfg"
    )
