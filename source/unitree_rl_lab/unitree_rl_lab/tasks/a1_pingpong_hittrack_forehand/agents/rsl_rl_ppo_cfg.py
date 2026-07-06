"""RSL-RL PPO runner cfg for the FOREHAND HitTrack task (``A1-Pingpong-HitTrack-Forehand``).

Identical hyperparameters to the backhand HitTrack runner -- same robot, same 100 Hz control rate,
same dense tracking profile -- so we subclass it and only redirect the experiment (log/checkpoint)
directory. Keeping the subclass means any future backhand-runner hyperparameter change propagates
here automatically; override individual fields below if the forehand run needs to diverge.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.a1_pingpong_hittrack.agents.rsl_rl_ppo_cfg import HitTrackPPORunnerCfg


@configclass
class ForehandHitTrackPPORunnerCfg(HitTrackPPORunnerCfg):
    experiment_name = "a1_tabletennis_hittrack_forehand"
