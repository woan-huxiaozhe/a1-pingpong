import gymnasium as gym

# A1-Pingpong-HitTrack: model-derived end-effector hit-reference TRACKING task (the ball is out of
# the MDP), trained with RSL-RL PPO (dense tracking + massively parallel sim) via
# scripts/rsl_rl/{train,play}.py. This is an independent task package: it OWNS all HitTrack logic
# (env cfg, reference kernels/manager, tracking obs/reward/termination, baking script). The env cfg
# sources its scene / robot placement from the forehand base task and reuses the generic MDP terms
# from the ``table_tennis_sac`` package by import (see ``mdp/__init__.py``); it no longer depends on
# the Catch ``table_tennis_sac.env_cfg`` for the scene or any tuning constants.
gym.register(
    id="A1-Pingpong-HitTrack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HitTrackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:HitTrackPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HitTrackPPORunnerCfg",
    },
)
