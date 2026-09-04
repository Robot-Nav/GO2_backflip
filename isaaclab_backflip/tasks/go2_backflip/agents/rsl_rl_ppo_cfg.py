"""RSL-RL PPO configuration matching the original backflip training setup."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2BackflipPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 1
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "go2_backflip"
    run_name = ""
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    clip_actions = 100.0
    # Match the original Gym runner.  The initial standard deviation is 1.0;
    # this lower bound only prevents late training from becoming deterministic.
    min_action_std = 0.35
    max_action_std = 1.50
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

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
