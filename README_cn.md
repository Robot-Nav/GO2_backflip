# TS-backflip

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-orange.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.x-lightgrey.svg)](https://developer.nvidia.com/isaac-sim)
[![arXiv](https://img.shields.io/badge/arXiv-2409.15755-b31b1b.svg)](https://arxiv.org/abs/2409.15755)

</div>

TS-backflip 是一套基于教师-学生蒸馏的强化学习框架，用于在 Isaac Lab 中训练宇树 GO2 四足机器人的特技动作。整体思路是：把每个动作拆成若干阶段，用约束多目标 PPO（CoMoPPO）在完整状态条件下训练教师策略，再把教师策略蒸馏成一个只用局部观测就能运行的学生策略。

## 项目简介

从零开始用一个稠密奖励训练后空翻是很困难的：机器人需要同时学会什么时候下蹲、什么时候起跳、如何翻转、如何落地。TS-backflip 的做法是按阶段设计奖励。动作被拆成有序的阶段（例如 `站立 -> 下蹲 -> 起跳 -> 腾空 -> 落地`），每个阶段只激活一组对应的奖励项。一个 one-hot 的阶段信号用于标记当前阶段，并同时输入给 actor 和 critic。

在分阶段奖励之上，教师使用约束多目标目标函数训练。奖励和成本被当作两个独立的多维信号：奖励通过偏好向量合成，成本则作为软约束——只有当某个成本的折扣回报超过阈值时才惩罚策略。这样可以在不手动调一个标量奖励的前提下，把关节位置、速度、力矩、身体接触等安全约束控制在合理范围内。

最终交付的策略是一个学生网络，通过行为克隆从教师策略中蒸馏得到。学生只观测带历史堆叠的局部观测，更接近真实部署时的信息条件。

本项目基于 Stage-Wise CMORL 方法（见[致谢](#致谢)）。

## 算法原理

### 1. 分阶段任务分解

任务被划分为 $K$ 个阶段。每个控制步环境输出一个 one-hot 阶段向量

$$p_t \in \{0, 1\}^K, \quad \sum_k p_{t,k} = 1.$$

每个奖励项 $i$ 定义为按阶段加权求和的形式，$\phi_{i,k}$ 是第 $k$ 阶段的奖励整形函数：

$$r_{t,i} = \sum_{k=1}^{K} p_{t,k}\,\phi_{i,k}(s_t),$$

其中 $s_t$ 是特权状态（基座线速度、角速度、质心高度、足端接触、重力、摩擦、恢复系数等）。阶段切换是确定性规则，例如“离地”“转过半圈”“重新触地”等。

### 2. 约束多目标 PPO（教师，CoMoPPO）

教师从元组 $(o_t, s_t, p_t, a_t, r_t, c_t)$ 中学习，其中 $r_t \in \mathbb{R}^M$ 是奖励向量，$c_t \in \mathbb{R}^N$ 是成本向量。

**价值估计。** 奖励 critic $V_r$ 和成本 critic $V_c$ 分别用 smooth L1 损失回归 bootstrap 回报：

$$\mathcal{L}^r = \mathbb{E}_t\left[\mathrm{smoothL1}\big(V_r(o_t, s_t, p_t), R_t\big)\right],$$

$$\mathcal{L}^c = \mathbb{E}_t\left[\mathrm{smoothL1}\big(V_c(o_t, s_t, p_t), C_t\big)\right],$$

其中 $R_t$、$C_t$ 分别是奖励通道和成本通道独立计算的 GAE 目标。

**优势合成。** 奖励优势通过偏好向量 $\omega \in \mathbb{R}^M$ 合成标量：

$$\hat{A}_t^{\mathrm{red}} = \sum_{i=1}^{M} \omega_i \left(\hat{A}_{t,i} - \bar{A}_i\right),$$

其中 $\bar{A}_i$ 是第 $i$ 维优势的批次均值，合成结果做标准化。对于每个当前折扣回报 $D_j$ 超过阈值 $d_j$ 的成本 $j$，从合成优势中减去对应的成本优势：

$$\hat{A}_t^{\mathrm{red}} \leftarrow \hat{A}_t^{\mathrm{red}} - \kappa \sum_{j:\, D_j > d_j} \hat{A}^c_{t,j},$$

随后再做一次标准化。$\kappa$ 是约束系数。

**策略更新。** actor 使用 PPO 截断代理目标更新：

$$\mathcal{L}^{\mathrm{actor}} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t^{\mathrm{red}},\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t^{\mathrm{red}}\right)\right],$$

其中 $\rho_t$ 是新旧策略的概率比，并通过自适应 KL 上限调整学习率和 clip 范围。

**对称性约束。** 策略被正则化到左右对称：

$$\mathcal{L}^{\mathrm{sym}} = \mathbb{E}_t\left[\left\| \mu(o_t^{\mathrm{sym}}, s_t^{\mathrm{sym}}, p_t) - S_a\,\mu(o_t, s_t, p_t) \right\|\right],$$

其中 $S_a$ 是动作对称矩阵，$o_t^{\mathrm{sym}}, s_t^{\mathrm{sym}}$ 是镜像后的观测和状态。当该项超过阈值时加入 actor 损失。

### 3. 教师-学生蒸馏

学生是一个通过行为克隆训练的高斯策略。每个时间步教师给出目标动作，学生最小化均方误差：

$$\mathcal{L}^{\mathrm{student}} = \mathbb{E}_t\left[\left\| \mu^{\mathrm{student}}(o_t) - a_t^{\mathrm{teacher}} \right\|^2\right].$$

学生与教师共享观测归一化统计量，但只接收带历史堆叠的局部观测 $o_t$。

## 项目结构

```
TS-backflip/
├── main_isaaclab.py              # 教师（CoMoPPO）训练/测试入口
├── main_student_isaaclab.py      # 学生蒸馏训练/测试入口
├── algos/
│   ├── common/                   # actor/critic/网络基类
│   ├── comoppo/                  # 约束多目标 PPO（教师）
│   │   ├── agent.py
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── normalizer.py
│   │   ├── storage.py
│   │   └── go2_*.yaml            # 教师算法配置
│   └── student/                  # 行为克隆学生
│       ├── agent.py
│       ├── normalizer.py
│       ├── storage.py
│       └── go2_backflip.yaml
├── isaaclab_envs/                # GO2 Isaac Lab 环境
│   ├── go2_env.py                # GO2 环境公共基类
│   ├── go2_backflip.py
│   ├── go2_sideflip.py
│   ├── go2_sideroll.py
│   └── go2_twohand.py
├── tasks/                        # 任务配置与奖励/成本定义
│   ├── go2_backflip.yaml
│   ├── go2_sideflip.yaml
│   ├── go2_sideroll.yaml
│   ├── go2_twohand.yaml
│   └── README.md
└── utils/                        # 日志、slack、环境包装
```

## 任务列表

| 任务 | 阶段 | 奖励数 | 成本数 |
|------|--------|---------|-------|
| 后空翻 | 站立 -> 下蹲 -> 起跳 -> 腾空 -> 落地 | 5 | 5 |
| 侧空翻 | 站立 -> 下蹲 -> 起跳 -> 腾空 -> 落地 | 5 | 5 |
| 侧滚翻 | 站立 -> 蹲坐 -> 半滚 -> 全滚 -> 恢复 | 6 | 5 |
| 双手撑地行走 | 站立 -> 倾斜 -> 行走 | 6 | 6 |

具体的奖励与成本函数定义见 [tasks/README.md](tasks/README.md)。

## 依赖库

- Ubuntu + NVIDIA GPU（CUDA）
- [Isaac Sim](https://developer.nvidia.com/isaac-sim) 与 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- Python 3.10
- PyTorch（版本需与 Isaac Lab 匹配）
- numpy
- pandas
- scipy
- ruamel.yaml
- requests（可选，Slack 通知）
- wandb（可选，训练日志）

## 环境配置

1. 按照[官方文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Sim 与 Isaac Lab。

2. 克隆仓库并安装 Python 依赖：

```bash
git clone <你的仓库地址>
cd TS-backflip
pip install numpy pandas scipy ruamel.yaml requests wandb
```

3. GO2 机器人模型由 `isaaclab_assets.robots.unitree.UNITREE_GO2_CFG` 提供，该库随 Isaac Lab 一起安装，无需额外配置机器人描述文件。

## 训练

需要先训练教师，学生训练时会加载教师检查点。

### 教师（CoMoPPO）

```bash
python main_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/comoppo/go2_backflip.yaml \
    --headless --seed 1
```

训练其他任务时，把 `go2_backflip` 换成 `go2_sideflip`、`go2_sideroll` 或 `go2_twohand`。需要记录到 wandb 时加上 `--wandb`。

### 学生（蒸馏，后空翻）

```bash
python main_student_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/student/go2_backflip.yaml \
    --headless --seed 1
```

学生配置 `algos/student/go2_backflip.yaml` 中指定了要加载的教师检查点。

## 测试

去掉 `--headless` 可打开 Isaac Sim 可视化窗口，并用 `--test` 指定已保存的检查点迭代次数：

```bash
# 教师
python main_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/comoppo/go2_backflip.yaml \
    --test --seed 1 --model_num 1000

# 学生
python main_student_isaaclab.py \
    --task_cfg_path tasks/go2_backflip.yaml \
    --algo_cfg_path algos/student/go2_backflip.yaml \
    --test --seed 1 --model_num 1000
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

本项目基于并参考了以下工作：

> **Stage-Wise Reward Shaping for Acrobatic Robots: A Constrained Multi-Objective Reinforcement Learning Approach**  
> [arXiv:2409.15755](https://arxiv.org/abs/2409.15755)

感谢原作者公开方法与代码，为本仓库的 GO2 / Isaac Lab 复现提供了起点。
