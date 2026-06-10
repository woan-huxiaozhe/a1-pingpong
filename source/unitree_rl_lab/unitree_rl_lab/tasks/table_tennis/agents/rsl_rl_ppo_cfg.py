from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TableTennisPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = ""
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        noise_std_type="log",  # 防止 std 学到负值 (默认 "scalar" 无正值约束, 训中可推过零 → normal expects std>=0)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class A1TableTennisPPORunnerCfg(TableTennisPPORunnerCfg):
    """A1 乒乓球 (模仿为主) PPO 配置: lr=1e-3, value_loss_coef=1.0, epochs=5, desired_kl=0.01."""

    experiment_name = "a1_tabletennis"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class A1TableTennisBackhandPPORunnerCfg(A1TableTennisPPORunnerCfg):
    """A1 反手 100Hz 配套 (plan A0/§6): 控制率翻倍 50->100Hz, 为保持时域行为:
    gamma 0.99 -> 0.995 (有效时域 ~1s 维持), num_steps_per_env 24 -> 48 (rollout ~0.48s 维持).

    沿用 forehand 的 policy=RslRlPpoActorCriticCfg 旧写法; rsl-rl 5.x 下由 train.py 的
    handle_deprecated_rsl_rl_cfg() 在运行时把 policy 转成 actor/critic + distribution_cfg.
    """

    experiment_name = "a1_tabletennis_backhand"
    num_steps_per_env = 48
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
