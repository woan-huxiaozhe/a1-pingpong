import gymnasium as gym

# A1-Pingpong-HitTrack-Forehand: forehand-stance sibling of A1-Pingpong-HitTrack. Same analytic
# hit-reference TRACKING task (ball out of the MDP), same shared mdp / baked references; ONLY the
# ready pose + base stance differ (see env_cfg.py). Auto-discovered by tasks/__init__.py's
# import_packages, so no global file needs editing.
gym.register(
    id="A1-Pingpong-HitTrack-Forehand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForehandHitTrackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:ForehandHitTrackPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ForehandHitTrackPPORunnerCfg",
    },
)
