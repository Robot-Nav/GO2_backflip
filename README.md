# TS-backflip

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-orange.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.x-lightgrey.svg)](https://developer.nvidia.com/isaac-sim)
[![arXiv](https://img.shields.io/badge/arXiv-2409.15755-b31b1b.svg)](https://arxiv.org/abs/2409.15755)

</div>

TS-backflip is a teacher-student reinforcement learning framework for training acrobatic skills on the Unitree GO2 quadruped inside Isaac Lab. The method decomposes each skill into a sequence of stages, trains a teacher policy with constrained multi-objective PPO (CoMoPPO) under full state access, and then distills a student policy that runs from partial observations.

## Overview

Training a backflip from scratch with a single dense reward is hard: the robot has to learn when to crouch, when to jump, how to rotate, and how to land, all at once. TS-backflip addresses this by shaping the reward per stage. The skill is split into ordered phases (for example `stand -> sit -> jump -> air -> land`), and each phase activates a different subset of reward terms. A one-hot stage signal tracks the current phase and is fed to both the actor and the critics.

On top of the stage-wise rewards, the teacher is trained with a constrained multi-objective objective. Rewards and costs are treated as separate multi-dimensional signals: rewards are combined through a preference vector, while costs act as soft constraints that only penalize the policy once their discounted return exceeds a threshold. This keeps safety limits (joint position, velocity, torque, body contact) in check without manually tuning a single scalar reward.

The final policy is a student network trained by behavior cloning from the teacher. It observes only a history-stacked partial observation, so it is closer to what a real deployment would have access to.

This repository is built on the Stage-Wise CMORL method (see [Acknowledgments](#acknowledgments)).

## Method

### 1. Stage-wise task decomposition

A task is divided into $K$ stages. At each control step the environment emits a one-hot stage vector

$$p_t \in \{0, 1\}^K, \quad \sum_k p_{t,k} = 1.$$

Each reward term $i$ is defined as a stage-conditioned sum of per-stage shaping functions $\phi_{i,k}$:

$$r_{t,i} = \sum_{k=1}^{K} p_{t,k}\,\phi_{i,k}(s_t),$$

where $s_t$ is the privileged state (base velocity, angular velocity, center-of-mass height, foot contacts, gravity, friction, restitution). Stage transitions are deterministic rules such as "leave the ground", "pass half a turn", or "touch down again".

### 2. Constrained multi-objective PPO (teacher, CoMoPPO)

The teacher learns from the tuple $(o_t, s_t, p_t, a_t, r_t, c_t)$, where $r_t \in \mathbb{R}^M$ is the reward vector and $c_t \in \mathbb{R}^N$ is the cost vector.

**Value estimation.** A reward critic $V_r$ and a cost critic $V_c$ regress bootstrapped returns with smooth L1 loss:

$$\mathcal{L}^r = \mathbb{E}_t\left[\mathrm{smoothL1}\big(V_r(o_t, s_t, p_t), R_t\big)\right],$$

$$\mathcal{L}^c = \mathbb{E}_t\left[\mathrm{smoothL1}\big(V_c(o_t, s_t, p_t), C_t\big)\right],$$

where $R_t$ and $C_t$ are GAE targets computed independently for the reward and cost channels.

**Advantage reduction.** Reward advantages are reduced to a scalar with a preference vector $\omega \in \mathbb{R}^M$:

$$\hat{A}_t^{\mathrm{red}} = \sum_{i=1}^{M} \omega_i \left(\hat{A}_{t,i} - \bar{A}_i\right),$$

where $\bar{A}_i$ is the batch mean of the $i$-th advantage. The result is standardized. For each cost $j$ whose current discounted return $D_j$ exceeds its threshold $d_j$, the corresponding cost advantage is subtracted:

$$\hat{A}_t^{\mathrm{red}} \leftarrow \hat{A}_t^{\mathrm{red}} - \kappa \sum_{j:\, D_j > d_j} \hat{A}^c_{t,j},$$

followed by another standardization. Here $\kappa$ is the constraint coefficient.

**Policy update.** The actor is updated with the clipped PPO surrogate:

$$\mathcal{L}^{\mathrm{actor}} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t^{\mathrm{red}},\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t^{\mathrm{red}}\right)\right],$$

where $\rho_t$ is the probability ratio between the new and old policy. An adaptive KL bound adjusts the learning rate and clip range.

**Symmetry constraint.** The policy is regularized toward left-right symmetry:

$$\mathcal{L}^{\mathrm{sym}} = \mathbb{E}_t\left[\left\| \mu(o_t^{\mathrm{sym}}, s_t^{\mathrm{sym}}, p_t) - S_a\,\mu(o_t, s_t, p_t) \right\|\right],$$

where $S_a$ is the action symmetry matrix and $o_t^{\mathrm{sym}}, s_t^{\mathrm{sym}}$ are the mirrored observation and state. The term is added to the actor loss when it exceeds a threshold.

### 3. Teacher-student distillation

The student is a Gaussian policy trained by behavior cloning. At each step the teacher provides a target action, and the student minimizes the mean squared error:

$$\mathcal{L}^{\mathrm{student}} = \mathbb{E}_t\left[\left\| \mu^{\mathrm{student}}(o_t) - a_t^{\mathrm{teacher}} \right\|^2\right].$$

The student shares the observation normalization statistics with the teacher but only sees the partial, history-stacked observation $o_t$.

## Repository layout

```
TS-backflip/
├── main_isaaclab.py              # teacher (CoMoPPO) train/test entry point
├── main_student_isaaclab.py      # student distillation train/test entry point
├── algos/
│   ├── common/                   # base actor/critic/network classes
│   ├── comoppo/                  # constrained multi-objective PPO (teacher)
│   │   ├── agent.py
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── normalizer.py
│   │   ├── storage.py
│   │   └── go2_*.yaml            # teacher algorithm configs
│   └── student/                  # behavior-cloning student
│       ├── agent.py
│       ├── normalizer.py
│       ├── storage.py
│       └── go2_backflip.yaml
├── isaaclab_envs/                # GO2 Isaac Lab environments
│   ├── go2_env.py                # shared GO2 environment base
│   ├── go2_backflip.py
│   ├── go2_sideflip.py
│   ├── go2_sideroll.py
│   └── go2_twohand.py
├── tasks/                        # task configs and reward/cost definition
│   ├── go2_backflip.yaml
│   ├── go2_sideflip.yaml
│   ├── go2_sideroll.yaml
│   ├── go2_twohand.yaml
│   └── README.md
└── utils/                        # logging, slack, environment wrapper
```

## Tasks

| Task | Stages | Rewards | Costs |
|------|--------|---------|-------|
| Backflip | stand -> sit -> jump -> air -> land | 5 | 5 |
| Sideflip | stand -> sit -> jump -> air -> land | 5 | 5 |
| Sideroll | stand -> sit -> roll_half -> roll_full -> recover | 6 | 5 |
| Two-hand walk | stand -> tilt -> walk | 6 | 6 |

The exact reward and cost functions are documented in [tasks/README.md](tasks/README.md).

## Requirements

- Ubuntu with an NVIDIA GPU (CUDA)
- [Isaac Sim](https://developer.nvidia.com/isaac-sim) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- Python 3.10
- PyTorch (matching the Isaac Lab version)
- numpy
- pandas
- scipy
- ruamel.yaml
- requests (optional, Slack notifications)
- wandb (optional, logging)

## Setup

1. Install Isaac Sim and Isaac Lab following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. Clone the repository and install the Python dependencies:

```bash
git clone <your-repo-url>
cd TS-backflip
pip install numpy pandas scipy ruamel.yaml requests wandb
```

3. The GO2 robot model is loaded from `isaaclab_assets.robots.unitree.UNITREE_GO2_CFG`, which ships with Isaac Lab. No additional robot description is required.

## Training

A teacher must be trained first. The student loads the teacher checkpoint.

### Teacher (CoMoPPO)

```bash
python main_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/comoppo/go2_backflip.yaml \
    --headless --seed 1
```

Replace `go2_backflip` with `go2_sideflip`, `go2_sideroll`, or `go2_twohand` for the other tasks. Add `--wandb` to log to Weights & Biases.

### Student (distillation, backflip)

```bash
python main_student_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/student/go2_backflip.yaml \
    --headless --seed 1
```

The student config in `algos/student/go2_backflip.yaml` specifies which teacher checkpoint to load.

## Evaluation

Drop `--headless` to open the Isaac Sim viewer, and pass `--test` with the saved checkpoint iteration:

```bash
# teacher
python main_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/comoppo/go2_backflip.yaml \
    --test --seed 1 --model_num 1000

# student
python main_student_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/student/go2_backflip.yaml \
    --test --seed 1 --model_num 1000
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

This project is built on and inspired by the following work:

> **Stage-Wise Reward Shaping for Acrobatic Robots: A Constrained Multi-Objective Reinforcement Learning Approach**  
> [arXiv:2409.15755](https://arxiv.org/abs/2409.15755)

We thank the authors for releasing their method and code, which served as the starting point for the GO2 / Isaac Lab re-implementation in this repository.
