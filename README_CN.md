# PPO-Backflip

基于非对称近端策略优化（PPO）训练、并部署到真实机器人的宇树 Go2 四足机器人
相位条件后空翻控制。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)]()
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-lightgrey)]()
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-orange)]()
[![MuJoCo](https://img.shields.io/badge/MuJoCo-sim2sim-brightgreen)]()
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-9cf)]()

---

## 项目简介

本仓库训练一个 12 自由度的宇树 Go2，使其完成一次后空翻并恢复稳定站立。任务被
建模为**相位条件控制问题**：策略观测一个基于时间的相位信号，并在 2.0 s 的时间
窗口内同步完成起跳、翻转、落地与恢复。

策略使用**非对称 actor-critic PPO**训练。actor 只接收可部署的传感器观测（60 维），
critic 额外接收仿真器独有的随机化真值（165 维）。**安全课程**会逐步收紧关节速度
与关节限位惩罚，使最终策略满足硬件约束。

| 模块 | 用途 |
| --- | --- |
| `isaaclab_backflip/` | Isaac Lab / Isaac Sim 5.1 训练环境、奖励、控制与 PPO 配置 |
| `rl/` | 自定义非对称 PPO 实现（actor-critic / storage / runner），供旧版 Gym 训练器使用 |
| `legged_gym/` | 旧版 Isaac Gym 任务源码参考 |
| `mujoco/` | MuJoCo sim2sim 回放与 ONNX 导出工具 |
| `deploy_real/` | 真机部署层（500 Hz 底层控制与安全保护） |
| `resources/` | 宇树 Go2 / dog 的 URDF、网格与 MuJoCo 资产 |

真机演示视频：

---

## 方法

### 控制框架

策略以 50 Hz 运行（`decimation = 4`，物理步长 `0.005 s`，回合时长 `3.0 s`）。
每个动作是相对于默认姿态的 12 维目标位置偏移，经 `action_scale = 0.5` 缩放后
由 PD 控制器跟踪：

```text
target = action_scale * action + default_dof_pos + motor_offset
tau    = kp * (target - dof_pos) - kd * dof_vel
```

其中 `kp = 40.0 N·m/rad`、`kd = 1.0 N·m·s/rad`。指令力矩由 Go2HV 扭矩-速度包络
饱和（低于 13.5 rad/s 时驱动/制动峰值分别为 20.2 / 23.4 N·m，到 30 rad/s 线性
降为零）。

### 观测与动作空间

**actor 观测**（60 维）仅由可部署量组成：

```text
延迟传感器 (30) + 动作 (12) + 上一动作 (12) + 相位特征 (6) = 60
```

- `延迟传感器`：机身角速度（3）、投影重力（3）、`dof_pos - default_dof_pos`（12）、
  `dof_vel`（12）。
- `相位特征`：相位条件信号

```text
phi = pi * min(episode_time, 2.0) / 2.0
[sin(phi), cos(phi), sin(phi/2), cos(phi/2), sin(phi/4), cos(phi/4)]
```

**critic 观测**（165 维）额外加入仿真器真值：

```text
状态真值 (64) + 动力学真值 (101) = 165
```

`动力学真值` 向 critic 暴露扭矩/速度缩放、肢体质量/惯量缩放、动作与观测延迟、
接触参数、机身质量/质心、PD 增益以及电机零位。

### PPO

策略为对角高斯分布，actor 与 critic 均为隐藏层 `[512, 256, 128]`、ELU 激活的
MLP。标准差为可学习标量，初始化为 `1.0`，每次优化器更新后裁剪到
`[0.35, 1.50]`。

优势估计使用 GAE：

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
```

策略目标采用 clip 裁剪：

```text
r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t))
L_CLIP     = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
```

配合裁剪后的价值损失与熵正则项：

```text
L = L_CLIP + value_loss_coef * L_V - entropy_coef * H[pi]
```

学习率根据平均 KL 散度自适应调整，目标 `desired_kl = 0.01`。

| 超参数 | 数值 |
| --- | --- |
| 每个环境 rollout 步数 | 24 |
| 每次更新学习轮数 | 5 |
| mini-batch 数量 | 4 |
| 裁剪系数 `eps` | 0.2 |
| 折扣因子 `gamma` | 0.99 |
| GAE `lam` | 0.95 |
| 熵系数 | 0.005 |
| 学习率 | `1e-3`（自适应） |
| 最大梯度范数 | 1.0 |
| 动作标准差裁剪 | `[0.35, 1.50]` |

### 奖励设计

奖励为多项加权和，围绕翻转相位（起跳、翻转、落地、恢复）组织塑形项，并包含完成
整圈翻转与稳定落地的稀疏事件奖励以及安全惩罚。典型项包括俯仰角速度奖励
`-omega_y`、向上速度奖励、翻转进度、翻转完成/成功事件、回正/高度/姿态恢复塑形，
以及头部触地、关节限位越界与关节超速惩罚。

### 域随机化

| 参数 | 范围 |
| --- | --- |
| 摩擦系数 | `[0.5, 1.25]` |
| 恢复系数 | `[0.0, 0.05]` |
| 接触偏移 | `[0.0075, 0.0125]` |
| 静止偏移 | `[-0.001, 0.001]` |
| 机身质量 | 附加 `[-0.5, 1.0]` kg |
| 机身质心 | `[-0.01, 0.01]` m |
| 肢体质量缩放 | `[0.95, 1.05]` |
| 肢体惯量抖动 | `[0.90, 1.10]` |
| 扭矩缩放 | `[0.80, 1.00]` |
| 电机速度缩放 | `[0.85, 1.00]` |
| PD 增益缩放 | `[0.8, 1.2]` |
| 电机零位 | `[-0.02, 0.02]` |

### 安全课程

安全惩罚与终止阈值共享一个课程缩放因子：

```text
scale = start + (1 - start) * clamp((step - warmup) / ramp, 0, 1)
```

发现阶段先以较弱的安全压力开始（`start = 0.05`），让 PPO 先找到后空翻动作，再在
数千次更新内平滑加强到全强度。关节速度终止比从 `3.0`（约 90 rad/s）收紧到
`1.05`（约 31.5 rad/s），使最终策略保持在硬件包络内。

---

## 依赖

- **训练：** Isaac Sim 5.1、Isaac Lab、`isaaclab-rl`（RSL-RL）、PyTorch、
  `gymnasium`。在 Isaac Lab 环境中执行 `pip install -e . --no-deps` 安装本任务。
- **Sim2sim：** `mujoco`、`onnxruntime`、`pynput`。
- **部署：** 宇树 `unitree_sdk2_python`。

```bash
conda activate env_isaaclab
cd PPO-backflip
python -m pip install -e . --no-deps
```

---

## 使用说明

### 训练

```bash
conda activate env_isaaclab
cd PPO-backflip

# 冒烟测试：16 个环境，执行 1 次 PPO 更新
python train.py --headless --device cuda:0 --num_envs 16 \
  --trainer rsl_rl --profile safety_initial_repro \
  --max_iterations 1 --run_name smoke

# 从零完整复现 safety-initial 策略
python train.py --headless --device cuda:0 --num_envs 4096 \
  --seed 1 --trainer rsl_rl \
  --profile safety_initial_repro --max_iterations 5000 \
  --run_name safety_initial_repro
```

checkpoint 与 TensorBoard 日志写入
`logs/rsl_rl/go2_backflip/<时间戳>_<run_name>/`。

可用 profile：`gym_discovery`、`lab_discovery`、`safety_initial_repro`（默认）、
`safety_targets`、`safety_landing`、`hardware_targets`、`hardware_velocity`。
其中 `hardware_*` profile 需要配合 `--resume --checkpoint` 使用。

### 回放与导出

```bash
python play.py --device cuda:0 --num_envs 1 \
  --profile safety_initial_repro \
  --checkpoint logs/rsl_rl/go2_backflip/<run>/model_<N>.pt

# 无窗口回放并导出 ONNX
python play.py --headless --device cuda:0 --num_envs 1 --steps 1 --export \
  --profile safety_initial_repro \
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

### 真机部署

500 Hz 控制器、状态机与安全检查详见
[deploy_real/README.md](deploy_real/README.md)。

---

## 硬件安全

后空翻可能损坏设备并造成人身伤害。请使用安全绳或顶部保护架、清空运动平面、安排
专人值守急停，并始终先在 MuJoCo 中验证。通过 sim2sim 检查并不等于可以在真机上
运行。

---

## 许可证

[MIT](LICENSE)
