"""Go2 双手撑地行走任务（移植到 IsaacLab）。"""

import numpy as np
import torch

from .go2_env import Go2Env, quat_rotate, quat_rotate_inverse, torch_rand_float


class Go2Twohand(Go2Env):
    raw_obs_dim = 45

    def _init_task_buffers(self):
        # 目标速度（线速度 x/y，角速度 z）
        self.target_vels = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        tvr = self.task_cfg["env"]["target_vel_ranges"]
        self.target_vel_x_range = tvr["lin_x"]
        self.target_vel_y_range = tvr["lin_y"]
        self.target_vel_z_range = tvr["ang_z"]
        # 机器人前方偏移（用于质心高度成本）
        self.robot_front = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.robot_front[:, 0] = 0.128
        # 随机化周期
        rnd = self.task_cfg["env"]["randomize"]
        self.rand_period_target_vel = int(rnd["rand_period_target_vel_s"] / self.control_dt + 0.5)
        self.rand_period_lag = int(rnd["rand_period_lag_s"] / self.control_dt + 0.5)

    def _reset_idx_task(self, env_ids):
        self.sample_target_vels(env_ids)

    def _compute_observation(self, env_ids):
        # 观测 = [目标速度(按行走阶段缩放), 姿态, 关节位置, 关节速度, 上一动作, 阶段]
        target_vels = self.target_vels[env_ids] * self.stage_buf[env_ids][:, 2:3]
        return torch.cat([
            target_vels, self.est_base_body_orns[env_ids],
            self.est_dof_positions[env_ids], self.est_dof_velocities[env_ids],
            self.prev_actions[env_ids], self.stage_buf[env_ids]], dim=-1)

    def _build_raw_obs_sym_mat(self):
        # 观测布局：目标速度(3) + 姿态(3) + 关节位置/速度/上一动作(各 12)
        raw = torch.eye(self.raw_obs_dim, device=self.device, dtype=torch.float32)
        raw[2, 2] = -1.0  # 目标角速度 z
        raw[4, 4] = -1.0  # 姿态 y
        for i in range(3):
            sl = slice(6 + self.num_dofs * i, 6 + self.num_dofs * (i + 1))
            raw[sl, sl] = self.joint_sym_mat.clone()
        raw[6 + 3 * self.num_dofs:, 6 + 3 * self.num_dofs:] = torch.eye(3, device=self.device, dtype=torch.float32)
        return raw

    def _compute_rewards_costs(self):
        # 阶段 0 站立，1 倾斜，2 行走
        # ============ 奖励 ============ #
        # 速度指令
        robot_dir_x = quat_rotate(self.base_quaternions, self.world_x + self.world_z)
        robot_dir_x[:, 2] = 0.0
        robot_dir_x /= torch.norm(robot_dir_x, dim=-1, keepdim=True)
        robot_dir_y = torch.cross(self.world_z, robot_dir_x, dim=-1)
        base_lin_vel_x = torch.sum(self.base_lin_vels * robot_dir_x, dim=-1)
        base_lin_vel_y = torch.sum(self.base_lin_vels * robot_dir_y, dim=-1)
        base_ang_vel_z = self.base_ang_vels[:, 2]
        target_vel_reward = -torch.square(base_lin_vel_x - self.stage_buf[:, 2] * self.target_vels[:, 0])
        target_vel_reward -= torch.square(base_lin_vel_y - self.stage_buf[:, 2] * self.target_vels[:, 1])
        target_vel_reward -= torch.square(base_ang_vel_z - self.stage_buf[:, 2] * self.target_vels[:, 2])
        self.rew_buf[:, 0] = target_vel_reward
        # 风格
        self.rew_buf[:, 1] = self.stage_buf[:, 0] * (-torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1))
        self.rew_buf[:, 1] += self.stage_buf[:, 1] * (-torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1))
        self.rew_buf[:, 1] += self.stage_buf[:, 2] * (-torch.square(self.dof_positions[:, 6:] - self.default_dof_positions[:, 6:]).mean(dim=-1))
        # 能量
        self.rew_buf[:, 2] = -torch.square(self.dof_torques).mean(dim=-1)
        # 平滑一阶
        self.rew_buf[:, 3] = -torch.square(self.joint_targets - self.prev_joint_targets).mean(dim=-1)
        # 平滑二阶
        self.rew_buf[:, 4] = -torch.square(self.joint_targets - 2.0 * self.prev_joint_targets + self.prev_prev_joint_targets).mean(dim=-1)
        # 身体平衡
        body_z = quat_rotate_inverse(self.base_quaternions, self.world_z)
        self.rew_buf[:, 5] = self.stage_buf[:, 0] * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 5] += self.stage_buf[:, 1] * (-torch.arccos(torch.clamp((body_z[:, 2] - body_z[:, 0]) / np.sqrt(2.0), -1.0, 1.0)))
        self.rew_buf[:, 5] += self.stage_buf[:, 2] * (-torch.arccos(torch.clamp((body_z[:, 2] - body_z[:, 0]) / np.sqrt(2.0), -1.0, 1.0)))

        # ============ 成本 ============ #
        stand_threshold = 0.025
        foot_contact_forces = self.contact_forces[:, self.foot_indices, :]
        calf_contact_forces = self.contact_forces[:, self.calf_indices, :]
        foot_contact = ((torch.norm(foot_contact_forces, dim=2) > 10.0)
                        | (torch.norm(calf_contact_forces, dim=2) > 10.0)).type(torch.float)
        stand_cost = (1.0 - foot_contact[:, 0]) * (1.0 - foot_contact[:, 1]) + (foot_contact[:, 2] + foot_contact[:, 3]) / 2.0
        self.cost_buf[:, 0] = self.stage_buf[:, 0] * stand_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 1] * stand_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 2] * stand_cost
        # 质心高度
        robot_front_positions = quat_rotate(self.base_quaternions, self.robot_front) + self.base_positions
        self.cost_buf[:, 1] = self.stage_buf[:, 0] * (self.base_positions[:, 2] < 0.3).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 1] * (self.base_positions[:, 2] < 0.3).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 2] * (robot_front_positions[:, 2] < 0.3).type(torch.float)
        # 身体接触
        self.cost_buf[:, 2] = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1).type(torch.float)
        self.cost_buf[:, 2] += torch.any(torch.norm(self.contact_forces[:, self.undesired_touch_indices, :], dim=-1) > 1.0, dim=-1).type(torch.float)
        # 关节位置/速度/力矩
        self.cost_buf[:, 3] = torch.mean(
            ((self.dof_positions < self.dof_pos_lower_limits) | (self.dof_positions > self.dof_pos_upper_limits)).to(torch.float), dim=-1)
        self.cost_buf[:, 4] = torch.mean((torch.abs(self.dof_velocities) > self.dof_vel_upper_limits).to(torch.float), dim=-1)
        self.cost_buf[:, 5] = torch.mean((torch.abs(self.dof_torques) > self.dof_torques_upper_limits).to(torch.float), dim=-1)

        # ============ 阶段更新 ============ #
        from2_to0 = torch.logical_and(
            self.stage_buf[:, 2] == 1.0, self.progress_buf * self.control_dt >= self.start_time_buf + 15.0).type(torch.float32)
        self.stage_buf[:, 2] = (1.0 - from2_to0) * self.stage_buf[:, 2]
        self.stage_buf[:, 0] = from2_to0 + (1.0 - from2_to0) * self.stage_buf[:, 0]
        from1_to2 = torch.logical_and(
            self.stage_buf[:, 1] == 1.0, self.progress_buf * self.control_dt >= self.start_time_buf + 5.0).type(torch.float)
        self.stage_buf[:, 1] = (1.0 - from1_to2) * self.stage_buf[:, 1]
        self.stage_buf[:, 2] = from1_to2 + (1.0 - from1_to2) * self.stage_buf[:, 2]
        from0_to1 = torch.logical_and(
            self.stage_buf[:, 0] == 1.0, torch.logical_and(
                self.progress_buf * self.control_dt >= self.start_time_buf,
                self.progress_buf * self.control_dt < self.start_time_buf + 15.0)).type(torch.float32)
        self.stage_buf[:, 0] = (1.0 - from0_to1) * self.stage_buf[:, 0]
        self.stage_buf[:, 1] = from0_to1 + (1.0 - from0_to1) * self.stage_buf[:, 1]

        # 失败判定
        body_contacts = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        self.fail_buf[:] = body_contacts.type(torch.long)

    def sample_target_vels(self, env_ids):
        self.target_vels[env_ids, 0] = torch_rand_float(
            self.target_vel_x_range[0], self.target_vel_x_range[1], (len(env_ids), 1), self.device).squeeze()
        self.target_vels[env_ids, 1] = torch_rand_float(
            self.target_vel_y_range[0], self.target_vel_y_range[1], (len(env_ids), 1), self.device).squeeze()
        self.target_vels[env_ids, 2] = torch_rand_float(
            self.target_vel_z_range[0], self.target_vel_z_range[1], (len(env_ids), 1), self.device).squeeze()
        # 小指令置零
        self.target_vels[env_ids] *= (torch.norm(self.target_vels[env_ids], dim=1) > 0.5).unsqueeze(1)

    def randomize(self, env_ids=None):
        super().randomize(env_ids)
        if self.common_step_counter % self.rand_period_lag == 0:
            self.lag_joint_target_idx = np.random.randint(0, len(self.lag_joint_target_buffer))
            self.lag_imu_idx = np.random.randint(0, len(self.lag_imu_buffer))
        if env_ids is not None:
            self.sample_target_vels(env_ids)
        else:
            env_ids = (self.progress_buf % self.rand_period_target_vel == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.sample_target_vels(env_ids)
