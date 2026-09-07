# PPO-Backflip

Phase-conditioned backflip control for the Unitree Go2 quadruped, trained with
asymmetric Proximal Policy Optimization (PPO) and deployed to the real robot.


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)]()
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-lightgrey)]()
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-orange)]()
[![MuJoCo](https://img.shields.io/badge/MuJoCo-sim2sim-brightgreen)]()
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-9cf)]()

---


**中文**：[👉 README_CN ](https://github.com/Robot-Nav/GO2_backflip/blob/PPO-backflip/README_CN.md)

**Getting Started Guide**：[👉 Getting Started Guide ](https://github.com/Robot-Nav/GO2_backflip/blob/PPO-backflip/%E8%BF%90%E8%A1%8C%E6%8C%87%E5%8D%97.md)


---

真机演示：


https://github.com/user-attachments/assets/d54488e1-97bc-4082-8ada-9ad9b79c1633




---
## Overview

This repository trains a 12-DoF Unitree Go2 to perform a single backflip and
recover to a stable stance. The task is formulated as a **phase-conditioned
control problem**: the policy observes a time-based phase signal and must
synchronize take-off, rotation, landing, and recovery inside a 2.0 s window.

The policy is trained with **asymmetric actor-critic PPO**. The actor only sees
deployable sensor readings (60-dim), while the critic additionally observes
simulator ground-truth randomization parameters (165-dim). A **safety
curriculum** progressively tightens joint-speed and joint-limit penalties so
that the final policy respects hardware limits.

| Module | Purpose |
| --- | --- |
| `isaaclab_backflip/` | Isaac Lab / Isaac Sim 5.1 training environment, rewards, control, and PPO config |
| `rl/` | Custom asymmetric PPO implementation (actor-critic / storage / runner) for the legacy Gym trainer |
| `legged_gym/` | Legacy Isaac Gym task source reference |
| `mujoco/` | MuJoCo sim2sim replay and ONNX export tooling |
| `deploy_real/` | Real-robot deployment layer (500 Hz low-level control and safety) |
| `resources/` | Unitree Go2 / dog URDF, meshes, and MuJoCo assets |

Real-robot demo:

---

## Method

### Control setup

The policy runs at 50 Hz (`decimation = 4`, physics step `0.005 s`, episode
length `3.0 s`). Each action is a 12-dim target-position offset from the default
pose, scaled by `action_scale = 0.5`, and tracked by a PD controller:

```text
target = action_scale * action + default_dof_pos + motor_offset
tau    = kp * (target - dof_pos) - kd * dof_vel
```

The PD target is clamped to 95% of the URDF joint range and the action itself
is clipped to `[-8, 8]`. With `kp = 40.0 N·m/rad` and
`kd = 1.0 N·m·s/rad`, the commanded torque is
saturated by the Go2HV torque-speed envelope (20.2 N·m driving / 23.4 N·m braking
below 13.5 rad/s, linearly derated to zero at 30 rad/s).

### Observation and action spaces

The **actor observation** (60-dim) is composed of deployable quantities only:

```text
delayed_sensor (30) + actions (12) + last_actions (12) + phase_features (6) = 60
```

- `delayed_sensor`: base angular velocity (3), projected gravity (3),
  `dof_pos - default_dof_pos` (12), `dof_vel` (12).
- `phase_features`: the phase-conditioning signal

```text
phi = pi * min(episode_time, 2.0) / 2.0
[sin(phi), cos(phi), sin(phi/2), cos(phi/2), sin(phi/4), cos(phi/4)]
```

The **critic observation** (165-dim) adds simulator ground truth:

```text
state_privileged (64) + dynamics_privileged (101) = 165
```

`dynamics_privileged` exposes torque/velocity scales, limb mass/inertia scales,
action/observation delays, contact parameters, base mass/CoM, PD gains, and
motor offsets to the critic.

### PPO

The policy is a diagonal Gaussian. The network is an MLP with hidden sizes
`[512, 256, 128]` and ELU activations for both actor and critic. The standard
deviation is a learnable scalar, initialized to `1.0` and clamped to
`[0.35, 1.50]` after every optimizer step.

Advantages are estimated with GAE:

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
```

The clipped surrogate objective is used:

```text
r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t))
L_CLIP     = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
```

with a clipped value loss and an entropy bonus:

```text
L = L_CLIP + value_loss_coef * L_V - entropy_coef * H[pi]
```

The learning rate is adjusted adaptively from the mean KL divergence with
`desired_kl = 0.01`.

| Hyperparameter | Value |
| --- | --- |
| Rollout steps / env | 24 |
| Learning epochs / update | 5 |
| Mini-batches | 4 |
| Clip ratio `eps` | 0.2 |
| Discount `gamma` | 0.99 |
| GAE `lam` | 0.95 |
| Entropy coefficient | 0.005 |
| Learning rate | `1e-3` (adaptive) |
| Max grad norm | 1.0 |
| Action std clamp | `[0.35, 1.50]` |

### Reward design

The reward is a weighted sum of shaping terms organized around the flip phases
(take-off, rotation, landing, recovery), sparse event bonuses for completing a
full rotation and a stable landing, and safety penalties. Representative terms
include pitch-rate reward `-omega_y`, upward-velocity reward, rotation progress,
flip-completion/success events, upright/height/pose recovery shaping, and
penalties for head contact, joint-limit excess, and joint overspeed.

### Domain randomization

| Parameter | Range |
| --- | --- |
| Friction | `[0.5, 1.25]` |
| Restitution | `[0.0, 0.05]` |
| Contact offset | `[0.0075, 0.0125]` |
| Rest offset | `[-0.001, 0.001]` |
| Base mass | `[-0.5, 1.0]` kg added |
| Base CoM | `[-0.015, 0.015]` m |
| Limb mass scale | `[0.95, 1.05]` |
| Limb inertia jitter | `[0.90, 1.10]` |
| Torque scale | `[0.75, 1.00]` |
| Motor velocity scale | `[0.85, 1.00]` |
| PD gain scale | `[0.8, 1.2]` |
| Motor offset | `[-0.02, 0.02]` |

### Safety curriculum

Safety penalties and termination thresholds share a curriculum factor:

```text
scale = start + (1 - start) * clamp((step - warmup) / ramp, 0, 1)
```

The first 1000 PPO updates retain the original objective (`start = 0`), followed
by a 3000-update ramp and 1000 updates under the complete safety envelope. The
joint-speed termination ratio tightens from `1.5` to `1.0`, while permitted
hard-limit position margin tightens from `0.30 rad` to zero. Non-foot body
contact becomes terminal once the curriculum reaches full strength.

---

## Dependencies

- **Training:** Isaac Sim 5.1, Isaac Lab, `isaaclab-rl` (RSL-RL), PyTorch,
  `gymnasium`. Install the task in the Isaac Lab environment with
  `pip install -e . --no-deps`.
- **Sim2sim:** `mujoco`, `onnxruntime`, `pynput`.
- **Deployment:** Unitree `unitree_sdk2_python`.

```bash
conda activate env_isaaclab
cd PPO-backflip
python -m pip install -e . --no-deps
```

---

## Usage

### Train

```bash
conda activate env_isaaclab
cd PPO-backflip

# Smoke test: 16 environments, one PPO update
python train.py --headless --device cuda:0 --num_envs 16 \
  --trainer rsl_rl \
  --max_iterations 1 --run_name smoke

# Train the fine-tuned Gym-equivalent task from scratch
python train.py --headless --device cuda:0 --num_envs 4096 \
  --seed 1 --trainer rsl_rl \
  --max_iterations 5000 --run_name gym_finetuned
```

Checkpoints and TensorBoard logs are written to
`logs/rsl_rl/go2_backflip/<timestamp>_<run_name>/`.

### Replay and export

```bash
python play.py --device cuda:0 --num_envs 1 \
  --checkpoint logs/rsl_rl/go2_backflip/<run>/model_<N>.pt

# Headless replay + ONNX export
python play.py --headless --device cuda:0 --num_envs 1 --steps 1 --export \
  --checkpoint logs/rsl_rl/go2_backflip/<run>/model_<N>.pt
```

### MuJoCo sim2sim

```bash
conda activate unitree_mujoco
cd PPO-backflip

python mujoco/go2/play_onnx.py \
  --joint-target-margin 0.0 \
  --onnx logs/rsl_rl/go2_backflip/<run>/exported/policy.onnx \
  --mjcf mujoco/go2/go2_isaac.xml
```

### Real-robot deployment

See [deploy_real/README.md](deploy_real/README.md) for the 500 Hz controller,
state machine, and safety checks.

---

## Hardware safety

A backflip can damage equipment and injure people. Use a safety harness or
overhead rope, clear the motion plane, keep an operator on the emergency stop,
and always validate in MuJoCo first. Passing the sim2sim check is not
authorization to run on hardware.

---
## Acknowledgements

https://github.com/uwvwko-zzz/uw-backflip Thanks to the awesome person for open-sourcing the gym project, which I migrated to the lab.

---
## License

[MIT](LICENSE)
