"""基于 IsaacLab 的 Unitree GO2 直接工作流环境基类。

将原 IsaacGym/IsaacGymEnvs 的 ``VecTask`` 环境移植到 IsaacLab 的 ``DirectRLEnv``，
对外保持与原环境一致的接口（属性名、``reset``/``step`` 返回值），从而可无缝复用
``algos``（CoMoPPO/Student）与 ``utils`` 中的训练代码。

说明：
    - 观测/状态/奖励/成本/阶段的语义与原实现完全一致。
    - 仅将 IsaacGym 的底层 API 替换为 IsaacLab 的 ``Articulation``/``ContactSensor`` 等。
    - 机器人资产为 Unitree GO2（go2.usd）。
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    """GO2 环境的 IsaacLab 配置，额外挂载机器人/地形/接触传感器配置。"""

    robot: ArticulationCfg = None
    terrain: TerrainImporterCfg = None
    contact_sensor: ContactSensorCfg = None


def _to_builtin(obj):
    """递归将 ruamel.yaml 的 ScalarFloat/CommentedSeq 等转为纯 Python 类型。

    ruamel 解析出的浮点数是 ``ScalarFloat``（继承 ``float`` 但带 ``__dict__``），
    IsaacLab 的 ``class_to_dict`` 会将其递归转成 dict，导致 ``gravity``/``dt`` 等
    配置值被错误序列化，因此统一在此处归一化。
    """
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return obj


def torch_rand_float(low: float, high: float, shape: tuple, device: torch.device) -> torch.Tensor:
    """在 [low, high) 上均匀采样，替代 ``isaacgym.torch_utils.torch_rand_float``。"""
    return (high - low) * torch.rand(shape, device=device) + low


def compute_observations(body_orns, dof_pos, dof_vel, prev_actions, commands):
    """拼接原始观测（未做历史堆叠）。"""
    return torch.cat([body_orns, dof_pos, dof_vel, prev_actions, commands], dim=-1)


def compute_states(base_quats, base_lin_vels, base_ang_vels, base_pos,
                   foot_contact_forces, calf_contact_forces, gravity,
                   friction_coeffs, restitution_coeffs, stages):
    """拼接状态（线性速度、角速度、质心高度、足端接触、重力、摩擦、恢复系数、阶段）。"""
    bb_lin_vels = quat_rotate_inverse(base_quats, base_lin_vels)
    bb_ang_vels = quat_rotate_inverse(base_quats, base_ang_vels)
    com_height = base_pos[:, 2:3]
    foot_contacts = ((torch.norm(foot_contact_forces, dim=2) > 1.0)
                     | (torch.norm(calf_contact_forces, dim=2) > 1.0)).type(torch.float)
    gravities = gravity.unsqueeze(0).repeat(base_quats.shape[0], 1)
    return torch.cat([bb_lin_vels, bb_ang_vels, com_height, foot_contacts,
                      gravities, friction_coeffs, restitution_coeffs, stages], dim=-1)


class Go2Env(DirectRLEnv):
    """Unitree GO2 后空翻类任务的环境基类。

    子类需要实现：
        - ``raw_obs_dim``：单步原始观测维度。
        - ``_init_task_buffers``：任务特有的缓冲区。
        - ``_reset_idx_task``：任务特有的复位逻辑。
        - ``_compute_observation``：单步原始观测计算。
        - ``_compute_rewards_costs``：奖励、成本、阶段更新与失败判定。
    """

    cfg: Go2EnvCfg
    raw_obs_dim: int = 42

    def __init__(self, cfg: dict, device: str = "cuda:0", render_mode: str | None = None, seed: int = 1):
        # 解析原始 yaml 配置（保持与原环境相同的读法）
        cfg = _to_builtin(cfg)
        self.task_cfg = cfg
        env_cfg_dict = cfg["env"]
        sim_cfg_dict = cfg["sim"]

        self.sim_dt = sim_cfg_dict["sim_dt"]
        self.control_dt = sim_cfg_dict["con_dt"]
        self.decimation = int(self.control_dt / self.sim_dt + 0.5)
        self.history_len = env_cfg_dict["history_len"]
        self.num_envs_cfg = env_cfg_dict["num_envs"]
        self.env_spacing = env_cfg_dict["env_spacing"]

        self.reward_names = env_cfg_dict["reward_names"]
        self.cost_names = env_cfg_dict["cost_names"]
        self.stage_names = env_cfg_dict["stage_names"]
        self.num_rewards = len(self.reward_names)
        self.num_costs = len(self.cost_names)
        self.num_stages = len(self.stage_names)
        self.num_acts = 12

        # 观测/状态维度（与原实现一致）
        self.num_obs = self.raw_obs_dim * self.history_len
        self.num_states = 3 + 3 + 1 + 4 + 3 + 1 + 1 + self.num_stages

        # 控制参数
        self.stiffness = env_cfg_dict["control"]["stiffness"]
        self.damping = env_cfg_dict["control"]["damping"]
        self.action_scale = env_cfg_dict["control"]["action_scale"]
        self.action_smooth_weight = env_cfg_dict["control"]["action_smooth_weight"]
        self.clip_actions = 1.0

        # 随机化参数
        rnd = env_cfg_dict["randomize"]
        self.is_randomized = rnd["is_randomized"]
        self.rand_period_motor_strength_s = rnd["rand_period_motor_strength_s"]
        self.rand_period_gravity_s = rnd["rand_period_gravity_s"]
        self.rand_period_motor_strength = int(self.rand_period_motor_strength_s / self.control_dt + 0.5)
        self.rand_period_gravity = int(self.rand_period_gravity_s / self.control_dt + 0.5)
        self.rand_range_body_mass = rnd["rand_range_body_mass"]
        self.rand_range_com_pos_x = rnd["rand_range_com_pos_x"]
        self.rand_range_com_pos_y = rnd["rand_range_com_pos_y"]
        self.rand_range_com_pos_z = rnd["rand_range_com_pos_z"]
        self.rand_range_dof_pos = rnd["rand_range_init_dof_pos"]
        self.rand_range_root_vel = rnd["rand_range_init_root_vel"]
        self.rand_range_motor_strength = rnd["rand_range_motor_strength"]
        self.rand_range_gravity = rnd["rand_range_gravity"]
        self.rand_range_friction = rnd["rand_range_friction"]
        self.rand_range_restitution = rnd["rand_range_restitution"]
        self.rand_range_motor_offset = rnd["rand_range_motor_offset"]
        self.noise_range_dof_pos = rnd["noise_range_dof_pos"]
        self.noise_range_dof_vel = rnd["noise_range_dof_vel"]
        self.noise_range_body_orn = rnd["noise_range_body_orn"]
        self.n_lag_action_steps = rnd["n_lag_action_steps"]
        self.n_lag_imu_steps = rnd["n_lag_imu_steps"]

        # 默认关节位置与初始位姿
        self.named_default_joint_positions = env_cfg_dict["default_joint_positions"]
        init_pose = env_cfg_dict["init_base_pose"]
        init_quat_xyzw = init_pose["quat"]  # [x, y, z, w] -> 转成 IsaacLab 的 [w, x, y, z]
        init_quat_wxyz = [init_quat_xyzw[3], init_quat_xyzw[0], init_quat_xyzw[1], init_quat_xyzw[2]]
        base_init_state = (list(init_pose["pos"]) + init_quat_wxyz
                           + list(init_pose["lin_vel"]) + list(init_pose["ang_vel"]))
        self.base_init_state = torch.tensor(base_init_state, dtype=torch.float32)

        # 构建 IsaacLab 环境配置
        env_cfg = self._build_env_cfg(device, seed)
        super().__init__(env_cfg, render_mode=render_mode)

        # 与旧接口对齐：rl_device（最大步数/控制频率由 DirectRLEnv 属性提供）
        self.rl_device = self.device
        # IsaacGym 的 progress_buf 对应 IsaacLab 的 episode_length_buf，保留旧名便于复用子类逻辑
        self.progress_buf = self.episode_length_buf

        # 机器人数据引用（保持旧变量名）
        self.num_dofs = self.robot.num_joints
        self.num_bodies = self.robot.num_bodies
        self.dof_names = list(self.robot.joint_names)
        self.link_names = list(self.robot.data.body_names)

        # 数据缓冲
        self.rew_buf = torch.zeros((self.num_envs, self.num_rewards), dtype=torch.float32, device=self.device)
        self.cost_buf = torch.zeros((self.num_envs, self.num_costs), dtype=torch.float32, device=self.device)
        self.stage_buf = torch.zeros((self.num_envs, self.num_stages), dtype=torch.float32, device=self.device)
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), dtype=torch.float32, device=self.device)

        # 世界坐标轴
        self.world_x = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.world_y = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.world_z = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.world_x[:, 0] = 1.0
        self.world_y[:, 1] = 1.0
        self.world_z[:, 2] = 1.0

        # 控制内部变量
        self.joint_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        self.prev_joint_targets = torch.zeros_like(self.joint_targets)
        self.prev_prev_joint_targets = torch.zeros_like(self.joint_targets)
        self.motor_strengths = torch.ones((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        self.motor_offsets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        self.lag_joint_target_buffer = [torch.zeros_like(self.joint_targets) for _ in range(self.n_lag_action_steps + 1)]
        self.lag_imu_buffer = [torch.zeros_like(self.world_z) for _ in range(self.n_lag_imu_steps + 1)]
        self.lag_joint_target_idx = 0
        self.lag_imu_idx = 0
        self.gravity = torch.tensor(sim_cfg_dict["gravity"], dtype=torch.float32, device=self.device)

        # 观测噪声估计值
        self.est_base_body_orns = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.est_dof_positions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        self.est_dof_velocities = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)

        # 计时缓冲（所有任务共用）
        self.start_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # 摩擦/恢复系数（作为状态特征）
        self._setup_friction_restitution()

        # 默认关节位置
        self.default_dof_positions = self.robot.data.default_joint_pos.clone()

        # 关节限位（位置取 URDF 硬限位，速度/力矩取学习配置与执行器限位的最小值）
        dof_pos_lower = self.robot.data.joint_pos_limits[0, :, 0]
        dof_pos_upper = self.robot.data.joint_pos_limits[0, :, 1]
        learn_lower, learn_upper = self._parse_learn_pos_limits(env_cfg_dict["learn"])
        self.dof_pos_lower_limits = torch.maximum(learn_lower, dof_pos_lower)
        self.dof_pos_upper_limits = torch.minimum(learn_upper, dof_pos_upper)
        self.dof_vel_upper_limits = torch.minimum(
            torch.tensor(env_cfg_dict["learn"]["joint_vel_upper"], dtype=torch.float32, device=self.device),
            self.robot.data.joint_vel_limits[0])
        self.dof_torques_upper_limits = torch.minimum(
            torch.tensor(env_cfg_dict["learn"]["joint_torque_upper"], dtype=torch.float32, device=self.device),
            self.robot.data.joint_effort_limits[0])

        # 身体索引（通过接触传感器名称查找）
        self.foot_indices = self._resolve_body_ids(["FR_foot", "FL_foot", "RR_foot", "RL_foot"], fallback_pattern=".*_foot")
        self.calf_indices = self._resolve_body_ids([], fallback_pattern=".*_calf")
        self.terminate_touch_indices = self._resolve_body_ids(["base", "trunk", "base_link"], fallback_pattern=None)
        hip_ids = self._resolve_body_ids([], fallback_pattern=".*_hip")
        self.terminate_touch_indices = torch.cat([self.terminate_touch_indices, hip_ids])
        thigh_ids = self._resolve_body_ids([], fallback_pattern=".*_thigh")
        self.undesired_touch_indices = torch.cat([thigh_ids, self.calf_indices])

        # 对称矩阵
        self._build_sym_matrices()

        # 任务特有缓冲区
        self._init_task_buffers()

    # ------------------------------------------------------------------ #
    # 配置构建
    # ------------------------------------------------------------------ #
    def _build_env_cfg(self, device: str, seed: int) -> Go2EnvCfg:
        sim_cfg_dict = self.task_cfg["sim"]
        env_cfg_dict = self.task_cfg["env"]
        ground_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0)

        robot_cfg = UNITREE_GO2_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(env_cfg_dict["init_base_pose"]["pos"]),
                joint_pos={k: v for k, v in self.named_default_joint_positions.items()},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                    stiffness=0.0,
                    damping=0.0,
                    effort_limit=23.5,
                    velocity_limit=30.0,
                ),
            },
        )

        return Go2EnvCfg(
            seed=seed,
            decimation=self.decimation,
            episode_length_s=env_cfg_dict["learn"]["episode_length_s"],
            observation_space=self.num_obs,
            action_space=self.num_acts,
            state_space=self.num_states,
            sim=SimulationCfg(
                device=device,
                dt=self.sim_dt,
                render_interval=self.decimation,
                gravity=tuple(sim_cfg_dict["gravity"]),
                physics_material=ground_material,
            ),
            scene=InteractiveSceneCfg(num_envs=self.num_envs_cfg, env_spacing=self.env_spacing, replicate_physics=True),
            terrain=TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=ground_material,
                debug_vis=False,
            ),
            robot=robot_cfg,
            contact_sensor=ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*", history_length=0, update_period=0.0
            ),
        )

    def _parse_learn_pos_limits(self, learn: dict):
        """解析关节位置约束。

        支持两种配置格式：
            - 分关节字典（hip/thigh/calf_joint_limit）
            - 扁平列表（joint_pos_lower / joint_pos_upper）
        """
        if "joint_pos_lower" in learn and "joint_pos_upper" in learn:
            lower = torch.tensor(learn["joint_pos_lower"], dtype=torch.float32, device=self.device)
            upper = torch.tensor(learn["joint_pos_upper"], dtype=torch.float32, device=self.device)
            return lower, upper
        lower, upper = [], []
        for joint_name in ["hip", "thigh", "calf"]:
            jd = learn[f"{joint_name}_joint_limit"]
            lower.append(jd.get("lower", -np.inf))
            upper.append(jd.get("upper", np.inf))
        lower = torch.tensor(lower * 4, dtype=torch.float32, device=self.device)
        upper = torch.tensor(upper * 4, dtype=torch.float32, device=self.device)
        return lower, upper

    def _resolve_body_ids(self, candidates: list[str], fallback_pattern: str | None):
        """按名称查找身体索引；候选未命中时回退到正则匹配。"""
        def _try(pattern):
            try:
                idx, _ = self.contact_sensor.find_bodies(pattern)
                return list(idx)
            except ValueError:
                return []

        ids = []
        for name in candidates:
            ids += _try(name)
        if len(ids) == 0 and fallback_pattern is not None:
            ids = _try(fallback_pattern)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _build_sym_matrices(self):
        # 关节对称矩阵（前后腿互换、左右腿互换）
        self.joint_sym_mat = torch.zeros((self.num_dofs, self.num_dofs), device=self.device, dtype=torch.float32)
        self.joint_sym_mat[:3, 3:6] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[0, 3] = -1.0
        self.joint_sym_mat[3:6, :3] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[3, 0] = -1.0
        self.joint_sym_mat[6:9, 9:12] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[6, 9] = -1.0
        self.joint_sym_mat[9:12, 6:9] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[9, 6] = -1.0

        # 观测对称矩阵
        self.obs_sym_mat = torch.zeros((self.num_obs, self.num_obs), device=self.device, dtype=torch.float32)
        raw_obs_sym_mat = self._build_raw_obs_sym_mat()
        for i in range(self.history_len):
            self.obs_sym_mat[(self.raw_obs_dim * i):(self.raw_obs_dim * (i + 1)),
                             (self.raw_obs_dim * i):(self.raw_obs_dim * (i + 1))] = raw_obs_sym_mat.clone()

        # 状态对称矩阵（默认实现；twohand 无 stage 差异，保持一致即可）
        self.state_sym_mat = torch.eye(self.num_states - self.num_stages, device=self.device, dtype=torch.float32)
        self.state_sym_mat[1, 1] = -1.0
        self.state_sym_mat[3, 3] = -1.0
        self.state_sym_mat[5, 5] = -1.0
        self.state_sym_mat[7:11, 7:11] = 0
        self.state_sym_mat[7, 8] = 1.0
        self.state_sym_mat[8, 7] = 1.0
        self.state_sym_mat[9, 10] = 1.0
        self.state_sym_mat[10, 9] = 1.0
        self.state_sym_mat[12, 12] = -1.0

    def _build_raw_obs_sym_mat(self):
        raw = torch.eye(self.raw_obs_dim, device=self.device, dtype=torch.float32)
        raw[1, 1] = -1.0
        for i in range(3):
            sl = slice(3 + self.num_dofs * i, 3 + self.num_dofs * (i + 1))
            raw[sl, sl] = self.joint_sym_mat.clone()
        raw[3 + 3 * self.num_dofs:, 3 + 3 * self.num_dofs:] = torch.eye(3, device=self.device, dtype=torch.float32)
        return raw

    # ------------------------------------------------------------------ #
    # 场景构建
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # 克隆并复制环境
        self.scene.clone_environments(copy_from_source=False)
        # 加光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # 复位
    # ------------------------------------------------------------------ #
    def reset(self, is_uniform_rollout: bool = True):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if is_uniform_rollout:
            self.progress_buf[:] = torch.randint_like(self.progress_buf, low=0, high=self.max_episode_length)
        return {"obs": self.obs_buf, "states": self.states_buf}

    def reset_idx(self, env_ids):
        env_ids = env_ids.to(dtype=torch.long)

        # 复位关节位置/速度
        if self.is_randomized:
            positions_offset = torch_rand_float(
                self.rand_range_dof_pos[0], self.rand_range_dof_pos[1], (len(env_ids), self.num_dofs), self.device)
        else:
            positions_offset = torch.ones((len(env_ids), self.num_dofs), dtype=torch.float32, device=self.device)
        joint_pos = self.default_dof_positions[env_ids] * positions_offset
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 复位根部状态（相对环境原点）
        root_state = self.base_init_state.repeat(len(env_ids), 1).to(self.device)
        root_state[:, :3] += self.scene.env_origins[env_ids]
        if self.is_randomized:
            root_state[:, 7:10] += torch_rand_float(
                self.rand_range_root_vel[0], self.rand_range_root_vel[1], (len(env_ids), 3), self.device)
            root_state[:, 10:13] += torch_rand_float(
                self.rand_range_root_vel[0], self.rand_range_root_vel[1], (len(env_ids), 3), self.device)
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # 随机化
        if self.is_randomized:
            self.randomize(env_ids)

        # 将状态写入仿真并刷新
        self.scene.write_data_to_sim()
        self.sim.forward()

        # 复位内部变量
        self.joint_targets[env_ids] = self.dof_positions[env_ids] + self.motor_offsets[env_ids]
        self.prev_joint_targets[env_ids] = self.joint_targets[env_ids].clone()
        self.prev_prev_joint_targets[env_ids] = self.joint_targets[env_ids].clone()
        self.prev_actions[env_ids] = (self.joint_targets[env_ids] - self.default_dof_positions[env_ids]) / self.action_scale

        # 复位公共缓冲
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.stage_buf[env_ids] = 0.0
        self.stage_buf[env_ids, 0] = 1.0
        self.start_time_buf[env_ids] = torch_rand_float(0.0, 5.0, (len(env_ids), 1), self.device).squeeze()
        for i in range(len(self.lag_joint_target_buffer)):
            self.lag_joint_target_buffer[i][env_ids, :] = self.joint_targets[env_ids]
        for i in range(len(self.lag_imu_buffer)):
            self.lag_imu_buffer[i][env_ids, :] = self.est_base_body_orns[env_ids]

        # 估计观测（含噪声）
        self.est_base_body_orns[env_ids] = quat_rotate_inverse(self.base_quaternions[env_ids], self.world_z[env_ids])
        self.est_dof_positions[env_ids] = self.dof_positions[env_ids] + self.motor_offsets[env_ids]
        self.est_dof_velocities[env_ids] = self.dof_velocities[env_ids]
        if self.is_randomized:
            self.est_base_body_orns[env_ids] += torch_rand_float(
                -self.noise_range_body_orn, self.noise_range_body_orn, (len(env_ids), 3), self.device)
            self.est_base_body_orns[env_ids] /= torch.norm(self.est_base_body_orns[env_ids], dim=-1, keepdim=True)
            self.est_dof_positions[env_ids] += torch_rand_float(
                -self.noise_range_dof_pos, self.noise_range_dof_pos, (len(env_ids), self.num_dofs), self.device)
            self.est_dof_velocities[env_ids] += torch_rand_float(
                -self.noise_range_dof_vel, self.noise_range_dof_vel, (len(env_ids), self.num_dofs), self.device)

        # 任务特有复位
        self._reset_idx_task(env_ids)

        # 复位观测
        obs = self._compute_observation(env_ids)
        for history_idx in range(self.history_len):
            self.obs_buf[env_ids, history_idx * self.raw_obs_dim:(history_idx + 1) * self.raw_obs_dim] = obs

        # 复位状态
        self.states_buf[env_ids] = self._compute_states(env_ids)

    # ------------------------------------------------------------------ #
    # 步进
    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.pre_physics_step(action_tensor)

        # 解耦步进：每步 PD 控制 + 仿真
        for _ in range(self.decimation):
            self.lag_joint_target_buffer = self.lag_joint_target_buffer[1:] + [self.joint_targets]
            joint_targets = self.lag_joint_target_buffer[self.lag_joint_target_idx]
            current_dof_positions = self.dof_positions + self.motor_offsets
            current_dof_velocities = self.dof_velocities
            torques = self.stiffness * (joint_targets - current_dof_positions) - self.damping * current_dof_velocities
            torques = torch.clamp(
                torques * self.motor_strengths,
                -self.dof_torques_upper_limits.unsqueeze(0),
                self.dof_torques_upper_limits.unsqueeze(0))
            self.robot.set_joint_effort_target(torques)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.post_physics_step()
        obs_dict = {"obs": self.obs_buf, "states": self.states_buf}
        return obs_dict, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = actions
        self.prev_prev_joint_targets[:] = self.prev_joint_targets
        self.prev_joint_targets[:] = self.joint_targets

        # 平滑 PD 目标
        self.joint_targets[:] = (self.action_smooth_weight * (actions * self.action_scale + self.default_dof_positions)
                                 + (1.0 - self.action_smooth_weight) * self.joint_targets)

    def post_physics_step(self):
        self.progress_buf += 1
        self.common_step_counter += 1

        if self.is_randomized:
            self.randomize()

        # 计算奖励/成本/阶段/失败（子类实现）
        self._compute_rewards_costs()

        # 计算复位缓冲
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length,
            torch.ones_like(self.reset_buf), self.fail_buf)

        # 估计观测（含噪声与 IMU 滞后）
        est_base_body_orns = quat_rotate_inverse(self.base_quaternions, self.world_z)
        self.est_dof_positions = self.dof_positions + self.motor_offsets
        self.est_dof_velocities = self.dof_velocities
        if self.is_randomized:
            est_base_body_orns = est_base_body_orns + torch_rand_float(
                -self.noise_range_body_orn, self.noise_range_body_orn, (self.num_envs, 3), self.device)
            est_base_body_orns = est_base_body_orns / torch.norm(est_base_body_orns, dim=-1, keepdim=True)
            self.est_dof_positions = self.est_dof_positions + torch_rand_float(
                -self.noise_range_dof_pos, self.noise_range_dof_pos, (self.num_envs, self.num_dofs), self.device)
            self.est_dof_velocities = self.est_dof_velocities + torch_rand_float(
                -self.noise_range_dof_vel, self.noise_range_dof_vel, (self.num_envs, self.num_dofs), self.device)
        self.lag_imu_buffer = self.lag_imu_buffer[1:] + [est_base_body_orns]
        self.est_base_body_orns[:] = self.lag_imu_buffer[self.lag_imu_idx]

        # 更新观测
        obs = self._compute_observation(torch.arange(self.num_envs, device=self.device))
        self.obs_buf[:, :-self.raw_obs_dim] = self.obs_buf[:, self.raw_obs_dim:].clone()
        self.obs_buf[:, -self.raw_obs_dim:] = obs

        # 更新状态
        self.states_buf[:] = self._compute_states(torch.arange(self.num_envs, device=self.device))

        # 返回额外信息
        self.extras["costs"] = self.cost_buf.clone()
        self.extras["fails"] = self.fail_buf.clone()
        self.extras["next_obs"] = self.obs_buf.clone()
        self.extras["next_states"] = self.states_buf.clone()
        self.extras["dones"] = self.reset_buf.clone()

        # 复位已终止环境
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

    # ------------------------------------------------------------------ #
    # 随机化
    # ------------------------------------------------------------------ #
    def randomize(self, env_ids=None):
        if self.common_step_counter % self.rand_period_gravity == 0:
            self.randomize_gravity()
        if env_ids is not None:
            self.randomize_motor_strength(env_ids)
            self.randomize_motor_offsets(env_ids)
        else:
            env_ids = (self.progress_buf % self.rand_period_motor_strength == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.randomize_motor_strength(env_ids)

    def randomize_motor_strength(self, env_ids):
        self.motor_strengths[env_ids] = torch_rand_float(
            self.rand_range_motor_strength[0], self.rand_range_motor_strength[1], (len(env_ids), 1), self.device)

    def randomize_motor_offsets(self, env_ids):
        self.motor_offsets[env_ids] = torch_rand_float(
            self.rand_range_motor_offset[0], self.rand_range_motor_offset[1], (len(env_ids), self.num_dofs), self.device)

    def randomize_gravity(self):
        # 注意：IsaacLab 中动态修改重力需通过 physx 接口，此处仅更新内部 gravity 张量；
        # 默认 is_randomized=false 不会触发。
        gravity_noise = np.random.rand(3) * (self.rand_range_gravity[1] - self.rand_range_gravity[0]) + self.rand_range_gravity[0]
        gravity = np.array(self.task_cfg["sim"]["gravity"]) + gravity_noise
        self.gravity[:] = torch.tensor(gravity, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    # 数据访问属性
    # ------------------------------------------------------------------ #
    @property
    def base_positions(self):
        return self.robot.data.root_pos_w

    @property
    def base_quaternions(self):
        return self.robot.data.root_quat_w

    @property
    def base_lin_vels(self):
        return self.robot.data.root_lin_vel_w

    @property
    def base_ang_vels(self):
        return self.robot.data.root_ang_vel_w

    @property
    def dof_positions(self):
        return self.robot.data.joint_pos

    @property
    def dof_velocities(self):
        return self.robot.data.joint_vel

    @property
    def dof_torques(self):
        return self.robot.data.applied_torque

    @property
    def contact_forces(self):
        return self.contact_sensor.data.net_forces_w

    def _compute_states(self, env_ids):
        foot_contact_forces = self.contact_forces[env_ids][:, self.foot_indices, :]
        calf_contact_forces = self.contact_forces[env_ids][:, self.calf_indices, :]
        return compute_states(
            self.base_quaternions[env_ids], self.base_lin_vels[env_ids], self.base_ang_vels[env_ids],
            self.base_positions[env_ids], foot_contact_forces, calf_contact_forces, self.gravity,
            self.friction_coeffs[env_ids], self.restitution_coeffs[env_ids], self.stage_buf[env_ids])

    # ------------------------------------------------------------------ #
    # 摩擦/恢复系数（子类或初始化时填充）
    # ------------------------------------------------------------------ #
    def _setup_friction_restitution(self):
        if self.is_randomized:
            self.friction_coeffs = torch_rand_float(
                self.rand_range_friction[0], self.rand_range_friction[1], (self.num_envs, 1), self.device)
            self.restitution_coeffs = torch_rand_float(
                self.rand_range_restitution[0], self.rand_range_restitution[1], (self.num_envs, 1), self.device)
        else:
            self.friction_coeffs = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
            self.restitution_coeffs = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    # 抽象钩子（子类实现）
    # ------------------------------------------------------------------ #
    def _init_task_buffers(self):
        raise NotImplementedError

    def _reset_idx_task(self, env_ids):
        raise NotImplementedError

    def _compute_observation(self, env_ids):
        raise NotImplementedError

    def _compute_rewards_costs(self):
        raise NotImplementedError
