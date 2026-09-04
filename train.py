"""Train the Isaac Lab Go2 backflip task with the safety-initial setup by default."""

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=1, help="Environment and PPO random seed.")
parser.add_argument(
    "--max_iterations", type=int, default=None,
    help="Number of PPO updates; defaults to the selected training profile.",
)
parser.add_argument("--run_name", type=str, default="", help="Optional suffix for the run directory.")
parser.add_argument(
    "--trainer",
    choices=("gym_ppo", "rsl_rl"),
    default="rsl_rl",
    help="PPO implementation: safety-initial RSL-RL (default) or original Gym AsymmetricPPO.",
)
parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL model checkpoint to resume.")
parser.add_argument(
    "--reset_curriculum",
    action="store_true",
    help="On resume, restart the safety curriculum instead of restoring its checkpoint progress.",
)
parser.add_argument(
    "--min_action_std",
    type=float,
    default=None,
    help="Override the PPO action-noise lower bound.",
)
parser.add_argument(
    "--max_action_std",
    type=float,
    default=None,
    help="Override the PPO action-noise upper bound.",
)
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
    help="Training profile; defaults to the saved safety-initial configuration.",
)
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    help="Load actor/critic weights but initialize a fresh optimizer (required by safety stages).",
)
parser.add_argument(
    "--safety_finetune",
    action="store_true",
    help="Deprecated alias for --profile safety_landing.",
)
parser.add_argument(
    "--flip_velocity_termination_ratio",
    type=float,
    default=None,
    help="Joint-speed termination ratio used before the landing phase.",
)
parser.add_argument(
    "--velocity_termination_ratio",
    type=float,
    default=None,
    help="Final joint-speed termination ratio used during recovery.",
)
parser.add_argument(
    "--safety_warmup_iterations",
    type=int,
    default=None,
    help="PPO updates to hold safety penalties at their initial scale.",
)
parser.add_argument(
    "--safety_ramp_iterations",
    type=int,
    default=None,
    help="PPO updates over which safety penalties ramp to full scale.",
)
parser.add_argument(
    "--validate_only",
    action="store_true",
    help="Validate task/agent configuration and imports without creating a PhysX scene.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# The lightweight Isaac Lab headless experience does not include the URDF
# importer, while this task converts its local Go2 URDF on first launch.
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
    """Return the exact dictionary consumed by the original BackflipRunner."""
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


def configure_gym_discovery(env_cfg, agent_cfg) -> int:
    """Strict numerical reproduction of the original Isaac Gym discovery task."""
    control = env_cfg.control
    reward = env_cfg.rewards
    control.clip_joint_targets = False
    control.joint_target_limit_margin = 0.0
    control.joint_target_margin_curriculum = False
    control.enable_joint_position_termination = False
    control.soft_velocity_limit = 24.0
    control.velocity_termination_start_ratio = 3.0
    control.flip_velocity_termination_ratio = 1.05
    control.velocity_termination_ratio = 1.05
    # Original Gym success is a single eligible sample after 1.40 s.
    reward.recovery_success_time = 1.40
    reward.recovery_hold_time = 0.0
    reward.landing_impact_start = reward.landing_start
    reward.landing_force_threshold = 350.0
    reward.rear_landing_force_threshold = 350.0
    reward.use_peak_contact_forces = False
    reward.safety_curriculum_start = 0.05
    reward.safety_curriculum_warmup_steps = 250 * agent_cfg.num_steps_per_env
    reward.safety_curriculum_ramp_steps = 1250 * agent_cfg.num_steps_per_env
    # Freeze every original scale here rather than inheriting a value from a
    # deployment profile.  The safe-* and rear-landing terms did not exist in
    # the Gym objective and must be exactly inactive during discovery.
    reward.scales.update(
        {
            "ang_vel_y": 5.0,
            "ang_vel_z": -1.0,
            "lin_vel_z": 20.0,
            "orientation_control": -1.0,
            "feet_height_before_backflip": -30.0,
            "height_control": -10.0,
            "default_pose": -5.0,
            "actions_symmetry": -0.1,
            "gravity_y": -10.0,
            "feet_distance": -1.0,
            "action_rate": -0.01,
            "action_jerk": -0.02,
            "dof_vel_limits": -5.0,
            "rotation_progress": 50.0,
            "rotation_target": 0.0,
            "rotation_overrun": 0.0,
            "flip_completion": 500.0,
            "flip_success": 500.0,
            "safe_flip_speed": 0.0,
            "safe_flip_target": 0.0,
            "safe_flip_position": 0.0,
            "safe_flip_landing": 0.0,
            "recovery_upright": 5.0,
            "recovery_height": 3.0,
            "recovery_default_pose": 2.0,
            "recovery_still": 2.0,
            "recovery_feet_contact": 2.0,
            "recovery_contact_balance": 0.0,
            "rear_leg_action_rate": -0.03,
            "rear_leg_symmetry": -0.5,
            "recovery_dof_velocity": -0.5,
            "head_clearance": -5.0,
            "undesired_body_contact": -10.0,
            "dof_pos_limits": -5.0,
            "joint_target_limits": -0.5,
            "head_contact": -20.0,
            "landing_impact": -3.0,
            "rear_landing_impact": 0.0,
        }
    )
    agent_cfg.min_action_std = 0.35
    agent_cfg.max_action_std = 1.50
    agent_cfg.algorithm.learning_rate = 1.0e-3
    agent_cfg.algorithm.entropy_coef = 0.005
    return 5000


def configure_safety_targets(env_cfg, agent_cfg) -> int:
    """First fine-tuning stage: fix target/position safety without landing pressure."""
    configure_gym_discovery(env_cfg, agent_cfg)
    control = env_cfg.control
    reward = env_cfg.rewards
    control.clip_joint_targets = True
    control.joint_target_limit_margin = 0.08
    control.joint_target_margin_curriculum = True
    control.enable_joint_position_termination = True
    control.joint_position_termination_start_excess = 0.16
    control.joint_position_termination_excess = 0.02
    # Keep the Gym velocity envelope in this stage.  Target clipping and the
    # position guard are the only newly tightened mechanisms.
    control.velocity_termination_start_ratio = 3.0
    control.flip_velocity_termination_ratio = 3.0
    control.velocity_termination_ratio = 3.0
    reward.recovery_success_time = 2.0
    reward.recovery_hold_time = 0.20
    reward.recovery_min_feet_contact_count = 4
    reward.use_peak_contact_forces = True
    reward.safety_curriculum_start = 0.10
    reward.safety_curriculum_warmup_steps = 0
    reward.safety_curriculum_ramp_steps = 1200 * agent_cfg.num_steps_per_env
    reward.scales.update(
        {
            "dof_pos_limits": -10.0,
            "joint_target_limits": -10.0,
            "safe_flip_target": 800.0,
            "safe_flip_position": 300.0,
            # This is intentionally introduced only after a reliable Gym-like
            # flip exists.  It makes both axle pairs participate in the
            # recovery rather than rewarding a rear-feet-only touchdown.
            "recovery_feet_contact": 6.0,
            "recovery_contact_balance": 8.0,
        }
    )
    agent_cfg.min_action_std = 0.20
    agent_cfg.max_action_std = 0.35
    agent_cfg.algorithm.learning_rate = 2.0e-4
    agent_cfg.algorithm.entropy_coef = 0.001
    return 1200


def configure_lab_discovery(env_cfg, agent_cfg) -> int:
    """Gym task/PPO with only the single-turn correction for Lab discovery.

    The strict Gym schedule reaches its 1.05 speed-reset threshold at PPO
    iteration 1500.  On Isaac Sim 5.1 the policy has not yet reached the
    completion event then, so nearly every sample terminates mid-air.  Do not
    cut the airborne trajectory short; however, unlike the original objective,
    a second half-turn must never be rewarded as more progress than one flip.
    """
    configure_gym_discovery(env_cfg, agent_cfg)
    control = env_cfg.control
    reward = env_cfg.rewards
    # Keep the original Gym raw PD target in the discovery stage.  Safety
    # target clipping belongs to stage 2; applying it here made the actor rely
    # on a controller saturation that is absent from the original Gym task.
    control.clip_joint_targets = False
    control.joint_target_limit_margin = 0.0
    control.joint_target_margin_curriculum = False
    # Do not cut off the only trajectory that can discover the completion
    # reward. Recovery is still governed by the gradually tightened global
    # threshold, so high-speed landing remains unattractive.
    control.flip_velocity_termination_ratio = 2.0
    # One-turn objective and a small tolerance for contact dynamics.  The
    # maximum angle remains observable, but only rotation up to 2*pi earns
    # progress reward; a policy past 2*pi+0.30 receives a continuous penalty
    # and cannot obtain the success event beyond 2*pi+0.35.
    one_turn = 2.0 * torch.pi
    reward.rotation_reward_cap = one_turn
    reward.rotation_target_angle = one_turn
    reward.rotation_target_start_time = 1.00
    reward.rotation_target_width = 0.28
    reward.rotation_overrun_start = one_turn + 0.30
    reward.rotation_overrun_width = 0.35
    # Preserve the original Gym completion/success entry thresholds.  The new
    # upper success window is sufficient to reject a second half-turn without
    # making the initial discovery event needlessly sparse.
    reward.flip_completion_angle = 5.50
    reward.flip_success_angle = 5.80
    reward.flip_success_max_angle = one_turn + 0.35
    # Restore the original Gym contact sampling and success event.  Four-foot
    # hold, contact balance, and peak-contact penalties are stage-2 concerns;
    # they distorted the initial discovery landing and are deliberately off.
    reward.recovery_success_time = 1.40
    reward.recovery_hold_time = 0.0
    reward.recovery_min_feet_contact_count = 3
    reward.use_peak_contact_forces = False
    # Isaac Sim needs an actual discovery window before the Gym safety terms
    # become strong.  The strict 250/1250 Gym schedule reaches full pressure
    # while this simulator is still at a half-flip (about 3.7 rad).
    reward.safety_curriculum_warmup_steps = 1000 * agent_cfg.num_steps_per_env
    reward.safety_curriculum_ramp_steps = 1500 * agent_cfg.num_steps_per_env
    reward.scales.update(
        {
            "rotation_target": 30.0,
            "rotation_overrun": -80.0,
            "flip_completion": 500.0,
            "flip_success": 500.0,
            "recovery_upright": 5.0,
            "recovery_height": 3.0,
            "recovery_default_pose": 2.0,
            "recovery_still": 2.0,
            "recovery_feet_contact": 2.0,
            "recovery_contact_balance": 0.0,
            "rear_landing_impact": 0.0,
        }
    )
    agent_cfg.min_action_std = 0.40
    # Reserve a longer from-scratch window for the single-turn constraint and
    # the delayed Isaac Sim safety curriculum.
    return 6000


def configure_safety_initial_repro(env_cfg, agent_cfg) -> int:
    """Reproduce the saved safety-initial configuration, except self-collision.

    Its saved env/agent YAML is the source of truth.  The historical run used
    3800 additional updates only because it started at iteration 1200; a new
    scratch run uses the equivalent total horizon of 5000 updates.
    """
    configure_gym_discovery(env_cfg, agent_cfg)
    control = env_cfg.control
    reward = env_cfg.rewards

    env_cfg.scene.env_spacing = 2.5
    env_cfg.terrain.env_spacing = 2.5
    # Explicitly retain the one requested deviation from the saved YAML.
    env_cfg.robot.spawn.self_collision = True
    env_cfg.robot.spawn.articulation_props.enabled_self_collisions = True

    control.clip_joint_targets = False
    control.joint_target_limit_margin = 0.0
    control.joint_target_margin_curriculum = False
    control.enable_joint_position_termination = False
    control.soft_velocity_limit = 24.0
    control.velocity_termination_start_ratio = 3.0
    control.flip_velocity_termination_ratio = 1.5
    control.velocity_termination_ratio = 1.25

    reward.recovery_success_time = 1.40
    reward.recovery_hold_time = 0.0
    reward.recovery_min_feet_contact_count = 3
    reward.use_peak_contact_forces = False
    reward.safety_curriculum_start = 0.05
    # Saved YAML: 4800/48000 control steps, i.e. 200/2000 PPO updates.
    reward.safety_curriculum_warmup_steps = 200 * agent_cfg.num_steps_per_env
    reward.safety_curriculum_ramp_steps = 2000 * agent_cfg.num_steps_per_env

    # The previous successful scratch discovery maintained an actual action
    # standard deviation around 0.44--0.47 while crossing the half-flip
    # plateau.  Keep this run from collapsing to 0.40 before that transition.
    agent_cfg.min_action_std = 0.45
    agent_cfg.max_action_std = 1.50
    agent_cfg.algorithm.learning_rate = 1.0e-3
    agent_cfg.algorithm.entropy_coef = 0.005
    return 5000


def configure_hardware_targets(env_cfg, agent_cfg) -> int:
    """Fine-tune the successful initial policy for target and position safety."""
    configure_safety_initial_repro(env_cfg, agent_cfg)
    control = env_cfg.control
    reward = env_cfg.rewards

    # The successful ONNX already passes MuJoCo with this margin.  Introduce
    # it gradually in training so raw targets stop relying on joint stops.
    control.clip_joint_targets = True
    control.joint_target_limit_margin = 0.08
    control.joint_target_margin_curriculum = True
    control.enable_joint_position_termination = True
    control.joint_position_termination_start_excess = 0.16
    control.joint_position_termination_excess = 0.01

    reward.safety_curriculum_start = 0.05
    reward.safety_curriculum_warmup_steps = 0
    reward.safety_curriculum_ramp_steps = 1200 * agent_cfg.num_steps_per_env
    reward.scales.update(
        {
            "dof_pos_limits": -15.0,
            "joint_target_limits": -12.0,
            "safe_flip_target": 500.0,
            "safe_flip_position": 400.0,
        }
    )
    agent_cfg.min_action_std = 0.15
    agent_cfg.max_action_std = 0.30
    agent_cfg.algorithm.learning_rate = 2.0e-4
    agent_cfg.algorithm.entropy_coef = 0.001
    return 1200


def configure_hardware_velocity(env_cfg, agent_cfg) -> int:
    """Fine-tune target-safe policy below the 30-rad/s hardware limit."""
    configure_hardware_targets(env_cfg, agent_cfg)
    control = env_cfg.control
    reward = env_cfg.rewards

    control.joint_target_margin_curriculum = False
    control.soft_velocity_limit = 20.0
    control.velocity_termination_start_ratio = 1.50
    # With a 1.0 floor, the common threshold smoothly moves 1.50 -> 1.00
    # in both airborne and recovery phases instead of tightening instantly.
    control.flip_velocity_termination_ratio = 1.00
    control.velocity_termination_ratio = 1.00
    control.joint_position_termination_start_excess = 0.03
    control.joint_position_termination_excess = 0.005

    reward.safe_speed_ratio = 0.95
    reward.safety_curriculum_start = 0.05
    reward.safety_curriculum_warmup_steps = 0
    reward.safety_curriculum_ramp_steps = 1500 * agent_cfg.num_steps_per_env
    reward.scales.update(
        {
            "action_jerk": -0.04,
            "dof_vel_limits": -15.0,
            "recovery_dof_velocity": -1.5,
            "dof_pos_limits": -20.0,
            "joint_target_limits": -15.0,
            "safe_flip_speed": 500.0,
            "safe_flip_target": 400.0,
            "safe_flip_position": 400.0,
        }
    )
    agent_cfg.min_action_std = 0.10
    agent_cfg.max_action_std = 0.25
    agent_cfg.algorithm.learning_rate = 1.0e-4
    agent_cfg.algorithm.entropy_coef = 5.0e-4
    return 1500


def configure_safety_landing(env_cfg, agent_cfg) -> int:
    """Second fine-tuning stage: retain safe targets and tighten speed/landing."""
    configure_safety_targets(env_cfg, agent_cfg)
    control = env_cfg.control
    reward = env_cfg.rewards
    control.joint_target_margin_curriculum = False
    control.joint_position_termination_start_excess = 0.04
    control.joint_position_termination_excess = 0.005
    control.soft_velocity_limit = 20.0
    control.velocity_termination_start_ratio = 1.50
    control.flip_velocity_termination_ratio = 1.05
    control.velocity_termination_ratio = 1.05
    reward.landing_impact_start = 1.00
    reward.landing_force_threshold = 200.0
    reward.rear_landing_force_threshold = 150.0
    reward.safety_curriculum_start = 0.25
    reward.safety_curriculum_warmup_steps = 0
    reward.safety_curriculum_ramp_steps = 1200 * agent_cfg.num_steps_per_env
    reward.scales.update(
        {
            "action_jerk": -0.03,
            "dof_vel_limits": -12.0,
            "safe_flip_speed": 300.0,
            "safe_flip_target": 600.0,
            "safe_flip_position": 300.0,
            "safe_flip_landing": 300.0,
            "rear_leg_action_rate": -0.05,
            "recovery_dof_velocity": -1.0,
            "dof_pos_limits": -15.0,
            "joint_target_limits": -10.0,
            "landing_impact": -8.0,
            "rear_landing_impact": -12.0,
        }
    )
    return 1500


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

    profile = args.profile
    if args.safety_finetune:
        if profile not in ("gym_discovery", "lab_discovery"):
            raise ValueError("--safety_finetune cannot be combined with --profile")
        profile = "safety_landing"

    if profile == "gym_discovery":
        default_iterations = configure_gym_discovery(env_cfg, agent_cfg)
    elif profile == "lab_discovery":
        default_iterations = configure_lab_discovery(env_cfg, agent_cfg)
    elif profile == "safety_targets":
        default_iterations = configure_safety_targets(env_cfg, agent_cfg)
    elif profile == "safety_landing":
        default_iterations = configure_safety_landing(env_cfg, agent_cfg)
    elif profile == "safety_initial_repro":
        default_iterations = configure_safety_initial_repro(env_cfg, agent_cfg)
    elif profile == "hardware_targets":
        default_iterations = configure_hardware_targets(env_cfg, agent_cfg)
    else:
        default_iterations = configure_hardware_velocity(env_cfg, agent_cfg)
    agent_cfg.max_iterations = (
        args.max_iterations if args.max_iterations is not None else default_iterations
    )

    if profile in (
        "safety_targets", "safety_landing", "hardware_targets", "hardware_velocity"
    ) and not args.resume:
        raise ValueError(f"--profile {profile} requires --resume --checkpoint <successful model>")
    if profile in ("safety_initial_repro", "hardware_targets", "hardware_velocity") and args.trainer != "rsl_rl":
        raise ValueError(f"--profile {profile} requires --trainer rsl_rl")
    # A checkpoint continuation of either discovery profile must retain the
    # PPO optimizer and its adaptive state.  Only a transition into a new
    # safety objective intentionally starts a fresh optimizer.
    reset_optimizer = args.reset_optimizer or profile in (
        "safety_targets", "safety_landing", "hardware_targets", "hardware_velocity"
    )

    if args.min_action_std is not None:
        agent_cfg.min_action_std = args.min_action_std
    if args.max_action_std is not None:
        agent_cfg.max_action_std = args.max_action_std
    if args.flip_velocity_termination_ratio is not None:
        env_cfg.control.flip_velocity_termination_ratio = args.flip_velocity_termination_ratio
    if args.velocity_termination_ratio is not None:
        env_cfg.control.velocity_termination_ratio = args.velocity_termination_ratio
    if args.safety_warmup_iterations is not None:
        env_cfg.rewards.safety_curriculum_warmup_steps = (
            args.safety_warmup_iterations * agent_cfg.num_steps_per_env
        )
    if args.safety_ramp_iterations is not None:
        env_cfg.rewards.safety_curriculum_ramp_steps = (
            args.safety_ramp_iterations * agent_cfg.num_steps_per_env
        )

    if env_cfg.control.flip_velocity_termination_ratio < env_cfg.control.velocity_termination_ratio:
        raise ValueError(
            "--flip_velocity_termination_ratio must be greater than or equal to "
            "--velocity_termination_ratio"
        )
    if agent_cfg.min_action_std <= 0.0 or agent_cfg.min_action_std > agent_cfg.max_action_std:
        raise ValueError("action std bounds must satisfy 0 < min_action_std <= max_action_std")
    if env_cfg.rewards.recovery_success_time + env_cfg.rewards.recovery_hold_time >= env_cfg.episode_length_s:
        raise ValueError("stable recovery window must finish before the episode timeout")
    if not 1 <= env_cfg.rewards.recovery_min_feet_contact_count <= 4:
        raise ValueError("recovery_min_feet_contact_count must be in [1, 4]")
    if env_cfg.rewards.rotation_reward_cap <= 0.0:
        raise ValueError("rotation_reward_cap must be positive")
    if (
        env_cfg.rewards.rotation_target_width <= 0.0
        or env_cfg.rewards.rotation_overrun_width <= 0.0
    ):
        raise ValueError("rotation target/overrun widths must be positive")
    if env_cfg.rewards.flip_success_max_angle < env_cfg.rewards.flip_success_angle:
        raise ValueError("flip success maximum angle must exceed its minimum angle")
    if min(
        env_cfg.rewards.safe_speed_width,
        env_cfg.rewards.safe_target_excess_width,
        env_cfg.rewards.safe_position_excess_width,
        env_cfg.rewards.safe_rear_force_width,
    ) <= 0.0:
        raise ValueError("safe-event quality widths must all be positive")
    if not (
        0.0 <= env_cfg.control.joint_position_termination_excess
        <= env_cfg.control.joint_position_termination_start_excess
    ):
        raise ValueError("joint-position termination excess must tighten from start to final")

    print(
        "[INFO] Backflip training schedule: "
        f"velocity_start_ratio={env_cfg.control.velocity_termination_start_ratio:.2f}, "
        f"flip_velocity_ratio={env_cfg.control.flip_velocity_termination_ratio:.2f}, "
        f"recovery_velocity_ratio={env_cfg.control.velocity_termination_ratio:.2f}, "
        f"position_excess=[{env_cfg.control.joint_position_termination_start_excess:.3f}, "
        f"{env_cfg.control.joint_position_termination_excess:.3f}] rad, "
        f"safety_warmup={env_cfg.rewards.safety_curriculum_warmup_steps // agent_cfg.num_steps_per_env} iters, "
        f"safety_ramp={env_cfg.rewards.safety_curriculum_ramp_steps // agent_cfg.num_steps_per_env} iters, "
        f"action_std=[{agent_cfg.min_action_std:.2f}, {agent_cfg.max_action_std:.2f}], "
        f"profile={profile}, trainer={args.trainer}, reset_optimizer={reset_optimizer}"
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
    env_cfg.log_dir = str(log_dir)
    os.makedirs(log_dir / "params", exist_ok=True)
    print(f"[INFO] Logging experiment to: {log_dir}")

    if args.resume and not args.checkpoint:
        raise ValueError("--resume requires --checkpoint=/path/to/model.pt")

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    if args.trainer == "gym_ppo":
        env = GymBackflipVecEnvAdapter(raw_env)
        runner = BackflipRunner(
            env, gym_ppo_cfg(agent_cfg), log_dir=str(log_dir), device=agent_cfg.device
        )
    else:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = BackflipOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
    if args.resume:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        print(
            f"[INFO] Resuming from: {checkpoint} "
            f"(optimizer={'reset' if reset_optimizer else 'restored'})"
        )
        runner.load(str(checkpoint), load_optimizer=not reset_optimizer)
        # ``common_step_counter`` belongs to the Isaac Lab environment rather
        # than the PPO checkpoint.  Restore it from the PPO iteration so a
        # resumed Lab-adapted discovery run does not silently restart its
        # safety curriculum at the weak initial scale.
        # A continuation of discovery must retain its progress.  A new safety
        # stage, however, must begin its own gentle target/landing curriculum;
        # otherwise a model trained for thousands of discovery updates would
        # enter the strict limits on its very first fine-tuning sample.
        if not args.reset_curriculum and profile in ("gym_discovery", "lab_discovery"):
            env.unwrapped.common_step_counter = (
                runner.current_learning_iteration * agent_cfg.num_steps_per_env
            )
            print(
                "[INFO] Restored safety-curriculum progress from checkpoint: "
                f"iteration={runner.current_learning_iteration}, "
                f"control_steps={env.unwrapped.common_step_counter}"
            )

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        # Isaac Sim's shutdown may otherwise hide the exception raised while
        # constructing a scene. Print it before closing the application.
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    if exit_code:
        sys.exit(exit_code)
