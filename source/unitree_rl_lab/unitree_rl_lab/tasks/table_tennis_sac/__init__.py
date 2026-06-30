import gymnasium as gym

gym.register(
    id="A1-TableTennis-SAC-Catch",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:RobotPlayEnvCfg",
    },
)

# HitTrack trains with RSL-RL PPO (dense tracking + parallel sim), not the custom SAC used by
# Catch -- so it exposes rsl_rl_cfg_entry_point and runs via scripts/rsl_rl/{train,play}.py.
# (Env lives in the shared table_tennis_sac package; the gym id is package/trainer-agnostic.)
gym.register(
    id="A1-Pingpong-HitTrack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hittrack_env_cfg:HitTrackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.hittrack_env_cfg:HitTrackPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HitTrackPPORunnerCfg",
    },
)
