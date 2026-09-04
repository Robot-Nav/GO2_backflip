"""Run and optionally export a trained Isaac Lab Go2 backflip policy."""

import argparse
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL model checkpoint (required to play).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--seed", type=int, default=1, help="Environment seed.")
parser.add_argument("--steps", type=int, default=-1, help="Exit after this many policy steps; -1 runs continuously.")
parser.add_argument("--real_time", action="store_true", help="Sleep to match the 50 Hz policy rate.")
parser.add_argument("--randomized", action="store_true", help="Use training randomization and sensor noise.")
parser.add_argument("--export", action="store_true", help="Export policy.pt and policy.onnx beside the checkpoint.")
parser.add_argument(
    "--profile",
    choices=(
        "gym_discovery",
        "lab_discovery",
        "safety_targets",
        "safety_landing",
        "safety_initial_repro",
        "hardware_targets",
        "hardware_velocity",
    ),
    default="safety_initial_repro",
    help="Controller/recovery settings used when replaying the checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if "isaacsim.asset.importer.urdf" not in args.kit_args:
    args.kit_args = f"{args.kit_args} --enable isaacsim.asset.importer.urdf".strip()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_backflip  # noqa: F401, E402
from isaaclab_backflip.rsl_rl_runner import BackflipOnPolicyRunner
from isaaclab_backflip.tasks.go2_backflip.agents.rsl_rl_ppo_cfg import Go2BackflipPPORunnerCfg
from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env_cfg import (
    Go2BackflipEnvCfg,
    Go2BackflipPlayEnvCfg,
)


TASK_ID = "Isaac-Go2-Backflip-Direct-v0"


def configure_play_profile(env_cfg):
    """Apply the deterministic deployment settings for a training stage."""
    # Both discovery profiles use the original Gym raw PD target.
    if args.profile in ("gym_discovery", "lab_discovery"):
        return

    control = env_cfg.control
    reward = env_cfg.rewards

    if args.profile == "safety_initial_repro":
        # Match the saved safety-initial environment at full curriculum.  The
        # training profile itself remains a raw-target policy.
        env_cfg.scene.env_spacing = 2.5
        env_cfg.terrain.env_spacing = 2.5
        control.clip_joint_targets = False
        control.joint_target_limit_margin = 0.0
        control.joint_target_margin_curriculum = False
        control.enable_joint_position_termination = False
        control.soft_velocity_limit = 24.0
        control.velocity_termination_start_ratio = 1.25
        control.flip_velocity_termination_ratio = 1.5
        control.velocity_termination_ratio = 1.25
        reward.recovery_success_time = 1.40
        reward.recovery_hold_time = 0.0
        reward.recovery_min_feet_contact_count = 3
        reward.use_peak_contact_forces = False
        return

    control.clip_joint_targets = True
    control.joint_target_limit_margin = 0.08
    control.joint_target_margin_curriculum = False
    control.enable_joint_position_termination = True
    reward.recovery_success_time = 2.0

    if args.profile == "safety_targets":
        control.joint_position_termination_start_excess = 0.16
        control.joint_position_termination_excess = 0.02
        control.velocity_termination_start_ratio = 3.0
        control.flip_velocity_termination_ratio = 3.0
        control.velocity_termination_ratio = 3.0
        return
    if args.profile == "hardware_targets":
        control.joint_position_termination_start_excess = 0.01
        control.joint_position_termination_excess = 0.01
        control.velocity_termination_start_ratio = 1.25
        control.flip_velocity_termination_ratio = 1.5
        control.velocity_termination_ratio = 1.25
        reward.recovery_success_time = 1.40
        reward.recovery_hold_time = 0.0
        reward.recovery_min_feet_contact_count = 3
        reward.use_peak_contact_forces = False
        return

    control.joint_position_termination_start_excess = 0.04
    control.joint_position_termination_excess = 0.005
    control.soft_velocity_limit = 20.0
    control.velocity_termination_start_ratio = 1.50
    control.flip_velocity_termination_ratio = 1.05
    control.velocity_termination_ratio = 1.05
    reward.landing_impact_start = 1.00
    reward.landing_force_threshold = 200.0
    reward.rear_landing_force_threshold = 150.0

    if args.profile == "hardware_velocity":
        control.joint_position_termination_start_excess = 0.005
        control.soft_velocity_limit = 20.0
        control.velocity_termination_start_ratio = 1.0
        control.flip_velocity_termination_ratio = 1.0
        control.velocity_termination_ratio = 1.0
        reward.landing_impact_start = 1.40
        reward.landing_force_threshold = 350.0
        reward.rear_landing_force_threshold = 350.0
        reward.recovery_success_time = 1.40
        reward.recovery_hold_time = 0.0
        reward.recovery_min_feet_contact_count = 3
        reward.use_peak_contact_forces = False


def main():
    if not args.checkpoint:
        raise ValueError("--checkpoint=/path/to/model.pt is required")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg = Go2BackflipEnvCfg() if args.randomized else Go2BackflipPlayEnvCfg()
    configure_play_profile(env_cfg)
    agent_cfg = Go2BackflipPPORunnerCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device

    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = BackflipOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    if args.export:
        export_dir = checkpoint.parent / "exported"
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.onnx")
        print(f"[INFO] Exported actor to: {export_dir}")

    obs = env.get_observations()
    step_count = 0
    while simulation_app.is_running() and (args.steps < 0 or step_count < args.steps):
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        step_count += 1
        sleep_time = env.unwrapped.step_dt - (time.time() - start)
        if args.real_time and sleep_time > 0:
            time.sleep(sleep_time)
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
