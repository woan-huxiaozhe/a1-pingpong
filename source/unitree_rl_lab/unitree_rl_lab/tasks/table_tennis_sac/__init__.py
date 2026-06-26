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

gym.register(
    id="A1-TableTennis-SAC-HitTrack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hittrack_env_cfg:HitTrackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.hittrack_env_cfg:HitTrackPlayEnvCfg",
    },
)
