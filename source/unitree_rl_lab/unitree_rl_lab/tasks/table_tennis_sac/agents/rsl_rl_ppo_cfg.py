"""RSL-RL PPO runner config for the HitTrack task (``A1-Pingpong-HitTrack``).

HitTrack is a dense, time-gated end-effector tracking task on the A1 right arm at 100 Hz
(``decimation=2``, ``sim.dt=0.005``). The ball is out of the MDP, the reward is a Gaussian
position+velocity tracking score, and episodes are short (~72 steps @100 Hz). That profile --
dense reward + massively parallel Isaac sim -- is exactly where on-policy PPO is the better fit
than the off-policy SAC used by the Catch task (see ``docs/sac_vs_ppo_对比.md``), and it lets
HitTrack share the same trainer / deployment-export path as the forehand/backhand tasks.

Hyperparameters mirror the A1 backhand 100 Hz PPO cfg
(``table_tennis.agents.rsl_rl_ppo_cfg:A1TableTennisBackhandPPORunnerCfg``) -- same robot, same
control rate -- which is the closest validated precedent:

  * ``gamma=0.995`` -- at 100 Hz this keeps the effective horizon ~2 s (``1/(1-0.995)=200``
    steps), matching ``0.99`` @ 50 Hz, so the early "prep" actions still get credit for the
    terminal hit-time tracking reward (``0.995**72 ≈ 0.70`` discount across a full ~72-step
    episode, vs ``0.99**72 ≈ 0.48``).
  * ``num_steps_per_env=48`` -- ~0.48 s rollout @100 Hz; with ``init_at_random_ep_len=True`` in
    ``runner.learn`` the per-env phases are staggered, so each rollout batch spans hit events
    across the whole prep window.
  * legacy ``policy=RslRlPpoActorCriticCfg`` form -- ``scripts/rsl_rl/train.py`` calls
    ``handle_deprecated_rsl_rl_cfg()`` at runtime to migrate it to the rsl-rl 5.x
    actor/critic + ``distribution_cfg`` schema (identical to forehand/backhand).

The HitTrack env already exposes ``policy`` (actor) and ``critic`` (privileged) observation
groups, so ``RslRlVecEnvWrapper``'s asymmetric actor-critic works without any env change.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HitTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner cfg for the 100 Hz HitTrack tracking task."""

    num_steps_per_env = 48  # ~0.48 s rollout @100 Hz (matches A1 backhand 100 Hz precedent)
    max_iterations = 20000  # dense tracking converges faster than the sparse Catch task; override with --max_iterations
    save_interval = 500
    experiment_name = "a1_tabletennis_hittrack"
    empirical_normalization = True  # obs mixes joint angles / positions / velocities of differing scale
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        noise_std_type="log",  # constrain std>0 (default "scalar" can be pushed negative mid-training)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # small: HitTrack needs precise tracking, not broad exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # desired_kl drives lr down once the policy sharpens
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
