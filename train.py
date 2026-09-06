"""Train the fine-tuned Gym-equivalent Go2 backflip task in Isaac Lab."""

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=1, help="Environment and PPO random seed.")
parser.add_argument(
    "--max_iterations", type=int, default=None,
    help="Number of PPO updates; defaults to 5000, matching the fine-tuned Gym config.",
)
parser.add_argument("--run_name", type=str, default="", help="Optional suffix for the run directory.")
parser.add_argument(
    "--trainer", choices=("gym_ppo", "rsl_rl"), default="rsl_rl",
    help="PPO implementation: Isaac Lab RSL-RL or the original Gym AsymmetricPPO.",
)
parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume.")
parser.add_argument(
    "--reset_optimizer", action="store_true",
    help="Load actor/critic weights but initialize a fresh optimizer.",
)
parser.add_argument(
    "--reset_curriculum", action="store_true",
    help="On resume, restart the safety curriculum instead of restoring checkpoint progress.",
)
parser.add_argument("--min_action_std", type=float, default=None, help="Override PPO noise lower bound.")
parser.add_argument("--max_action_std", type=float, default=None, help="Override PPO noise upper bound.")
parser.add_argument(
    "--safety_warmup_iterations", type=int, default=None,
    help="Override safety-curriculum warmup in PPO updates.",
)
parser.add_argument(
    "--safety_ramp_iterations", type=int, default=None,
    help="Override safety-curriculum ramp in PPO updates.",
)
parser.add_argument(
    "--validate_only", action="store_true",
    help="Validate task/agent configuration and imports without creating a PhysX scene.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if "isaacsim.asset.importer.urdf" not in args.kit_args:
    args.kit_args = f"{args.kit_args} --enable isaacsim.asset.importer.urdf".strip()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_backflip  # noqa: F401, E402
from isaaclab_backflip.gym_compat import GymBackflipVecEnvAdapter
from isaaclab_backflip.rsl_rl_runner import BackflipOnPolicyRunner
from isaaclab_backflip.tasks.go2_backflip.agents.rsl_rl_ppo_cfg import Go2BackflipPPORunnerCfg
from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env_cfg import Go2BackflipEnvCfg
from rl.Backflip import BackflipRunner


TASK_ID = "Isaac-Go2-Backflip-Direct-v0"


def gym_ppo_cfg(agent_cfg) -> dict:
    """Return the dictionary consumed by the original BackflipRunner."""
    policy = agent_cfg.policy
    algorithm = agent_cfg.algorithm
    return {
        "runner": {
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "save_interval": agent_cfg.save_interval,
        },
        "policy": {
            "init_noise_std": policy.init_noise_std,
            "actor_hidden_dims": list(policy.actor_hidden_dims),
            "critic_hidden_dims": list(policy.critic_hidden_dims),
            "activation": policy.activation,
        },
        "algorithm": {
            "clip_param": algorithm.clip_param,
            "desired_kl": algorithm.desired_kl,
            "entropy_coef": algorithm.entropy_coef,
            "min_action_std": agent_cfg.min_action_std,
            "max_action_std": agent_cfg.max_action_std,
            "gamma": algorithm.gamma,
            "lam": algorithm.lam,
            "learning_rate": algorithm.learning_rate,
            "max_grad_norm": algorithm.max_grad_norm,
            "num_learning_epochs": algorithm.num_learning_epochs,
            "num_mini_batches": algorithm.num_mini_batches,
            "schedule": algorithm.schedule,
            "use_clipped_value_loss": algorithm.use_clipped_value_loss,
            "value_loss_coef": algorithm.value_loss_coef,
        },
    }


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    env_cfg = Go2BackflipEnvCfg()
    agent_cfg = Go2BackflipPPORunnerCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.run_name = args.run_name
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.min_action_std is not None:
        agent_cfg.min_action_std = args.min_action_std
    if args.max_action_std is not None:
        agent_cfg.max_action_std = args.max_action_std
    if args.safety_warmup_iterations is not None:
        env_cfg.rewards.safety_curriculum_warmup_steps = (
            args.safety_warmup_iterations * agent_cfg.num_steps_per_env
        )
    if args.safety_ramp_iterations is not None:
        env_cfg.rewards.safety_curriculum_ramp_steps = (
            args.safety_ramp_iterations * agent_cfg.num_steps_per_env
        )

    if agent_cfg.min_action_std <= 0.0 or agent_cfg.min_action_std > agent_cfg.max_action_std:
        raise ValueError("action std bounds must satisfy 0 < min_action_std <= max_action_std")
    if len(env_cfg.control.joint_velocity_limits) != env_cfg.action_space:
        raise ValueError("joint_velocity_limits must contain one value per action")
    if len(env_cfg.control.target_velocity_limits) != env_cfg.action_space:
        raise ValueError("target_velocity_limits must contain one value per action")
    if not (
        0.0 <= env_cfg.control.position_termination_margin
        <= env_cfg.control.position_termination_start_margin
    ):
        raise ValueError("position termination margin must tighten from start to final")
    if not 0.0 <= env_cfg.rewards.unsafe_contact_termination_curriculum <= 1.0:
        raise ValueError("unsafe contact termination curriculum must be in [0, 1]")

    print(
        "[INFO] Fine-tuned Gym-equivalent schedule: "
        f"velocity_ratio=[{env_cfg.control.velocity_termination_start_ratio:.2f}, "
        f"{env_cfg.control.velocity_termination_ratio:.2f}], "
        f"position_margin=[{env_cfg.control.position_termination_start_margin:.3f}, "
        f"{env_cfg.control.position_termination_margin:.3f}] rad, "
        f"safety_warmup={env_cfg.rewards.safety_curriculum_warmup_steps // agent_cfg.num_steps_per_env} iters, "
        f"safety_ramp={env_cfg.rewards.safety_curriculum_ramp_steps // agent_cfg.num_steps_per_env} iters, "
        f"action_std=[{agent_cfg.min_action_std:.2f}, {agent_cfg.max_action_std:.2f}], "
        f"trainer={args.trainer}"
    )

    if args.validate_only:
        from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env import Go2BackflipEnv

        env_cfg.validate()
        agent_cfg.validate()
        print(
            "[INFO] Configuration valid: "
            f"env={Go2BackflipEnv.__name__}, actor_obs={env_cfg.observation_space}, "
            f"critic_obs={env_cfg.state_space}, actions={env_cfg.action_space}, "
            f"num_envs={env_cfg.scene.num_envs}"
        )
        return

    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_dir_name += f"_{agent_cfg.run_name}"
    log_dir = (log_root / run_dir_name).resolve()
    os.makedirs(log_dir / "params", exist_ok=True)
    print(f"[INFO] Logging experiment to: {log_dir}")

    if args.resume and not args.checkpoint:
        raise ValueError("--resume requires --checkpoint=/path/to/model.pt")

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    if args.trainer == "gym_ppo":
        env = GymBackflipVecEnvAdapter(raw_env)
        runner = BackflipRunner(env, gym_ppo_cfg(agent_cfg), str(log_dir), agent_cfg.device)
    else:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = BackflipOnPolicyRunner(env, agent_cfg.to_dict(), str(log_dir), agent_cfg.device)

    if args.resume:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        runner.load(str(checkpoint), load_optimizer=not args.reset_optimizer)
        if not args.reset_curriculum:
            env.unwrapped.common_step_counter = (
                runner.current_learning_iteration * agent_cfg.num_steps_per_env
            )

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    if exit_code:
        sys.exit(exit_code)
