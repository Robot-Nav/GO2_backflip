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


def main():
    if not args.checkpoint:
        raise ValueError("--checkpoint=/path/to/model.pt is required")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg = Go2BackflipEnvCfg() if args.randomized else Go2BackflipPlayEnvCfg()
    if args.randomized:
        # Replay a randomized policy under the final safety envelope rather
        # than silently restarting the training curriculum at zero.
        env_cfg.rewards.safety_curriculum_start = 1.0
        env_cfg.rewards.safety_curriculum_warmup_steps = 0
        env_cfg.rewards.safety_curriculum_ramp_steps = 1
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
