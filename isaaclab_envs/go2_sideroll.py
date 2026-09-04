"""Go2 侧滚任务（移植到 IsaacLab）。"""

import numpy as np
import torch

from .go2_env import Go2Env, compute_observations, quat_rotate, quat_rotate_inverse


class Go2Sideroll(Go2Env):
    raw_obs_dim = 42

    def _init_task_buffers(self):
        self.is_half_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_one_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 爬行姿态（阶段 1/2/3 的风格基准）
        named = self.task_cfg["env"]["crawled_joint_positions"]
        self.crawled_dof_positions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        for i, name in enumerate(self.dof_names):
            self.crawled_dof_positions[:, i] = named[name]

    def _reset_idx_task(self, env_ids):
        self.is_half_turn_buf[env_ids] = 0
        self.is_one_turn_buf[env_ids] = 0

    def _compute_observation(self, env_ids):
        commands = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        t = self.progress_buf[env_ids] * self.control_dt
        masks2 = (t >= self.start_time_buf[env_ids] + 0.2).type(torch.float32)
        masks1 = (1.0 - masks2) * (t >= self.start_time_buf[env_ids]).type(torch.float32)
        masks0 = (t < self.start_time_buf[env_ids]).type(torch.float32)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2
        return compute_observations(
            self.est_base_body_orns[env_ids], self.est_dof_positions[env_ids],
            self.est_dof_velocities[env_ids], self.prev_actions[env_ids], commands)

    def _compute_rewards_costs(self):
        # 阶段 0 站立，1 蹲坐，2 半滚，3 全滚，4 恢复
        com_height = self.base_positions[:, 2]
        self.rew_buf[:, 0] = self.stage_buf[:, 0] * (-torch.abs(com_height - 0.35))
        self.rew_buf[:, 0] += self.stage_buf[:, 1] * (-torch.abs(com_height - 0.1))
        self.rew_buf[:, 0] += self.stage_buf[:, 2] * (-torch.clamp(com_height - 0.2, min=0.0))
        self.rew_buf[:, 0] += self.stage_buf[:, 3] * (-torch.clamp(com_height - 0.2, min=0.0))
        self.rew_buf[:, 0] += self.stage_buf[:, 4] * (-torch.abs(com_height - 0.35))
        # 身体平衡
        body_z = quat_rotate_inverse(self.base_quaternions, self.world_z)
        self.rew_buf[:, 1] = self.stage_buf[:, 0] * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 1] * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 2] * (-torch.abs(torch.arccos(torch.clamp(body_z[:, 0], -1.0, 1.0)) - np.pi / 2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 3] * (-torch.abs(torch.arccos(torch.clamp(body_z[:, 0], -1.0, 1.0)) - np.pi / 2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 4] * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        # 旋转角度
        self.rew_buf[:, 2] = self.stage_buf[:, 0] * 0.0
        self.rew_buf[:, 2] += self.stage_buf[:, 1] * 0.0
        roll_angles = torch.arccos(torch.clamp(-body_z[:, 2], -1.0, 1.0))  # 目标 (0, 0, -1)
        masks = torch.logical_or(body_z[:, 1] > 0, torch.abs(roll_angles) < np.pi / 6.0).type(torch.float)
        self.rew_buf[:, 2] += self.stage_buf[:, 2] * (-masks * roll_angles + (1.0 - masks) * (roll_angles - 2.0 * np.pi))
        roll_angles = torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0))  # 目标 (0, 0, 1)
        masks = torch.logical_or(body_z[:, 1] <= 0, torch.abs(roll_angles) < np.pi / 6.0).type(torch.float)
        self.rew_buf[:, 2] += self.stage_buf[:, 3] * (-masks * roll_angles + (1.0 - masks) * (roll_angles - 2.0 * np.pi))
        self.rew_buf[:, 2] += self.stage_buf[:, 4] * 0.0
        # 基座速度
        x_dirs = quat_rotate(self.base_quaternions, self.world_x)
        x_dirs[:, 2] = 0.0
        x_dirs /= torch.norm(x_dirs, dim=-1, keepdim=True)
        base_ang_vel_x = torch.sum(self.base_ang_vels * x_dirs, dim=-1)
        base_lin_vels = quat_rotate_inverse(self.base_quaternions, self.base_lin_vels)
        base_ang_vels = quat_rotate_inverse(self.base_quaternions, self.base_ang_vels)
        vel_penalty = torch.square(base_lin_vels[:, 0]) + torch.square(base_lin_vels[:, 1]) + torch.square(base_ang_vels[:, 2])
        self.rew_buf[:, 3] = self.stage_buf[:, 0] * (-vel_penalty)
        self.rew_buf[:, 3] += self.stage_buf[:, 1] * (-vel_penalty)
        self.rew_buf[:, 3] += self.stage_buf[:, 2] * ((base_ang_vel_x < 4.0 * np.pi).type(torch.float) * base_ang_vel_x)
        self.rew_buf[:, 3] += self.stage_buf[:, 3] * ((base_ang_vel_x < 4.0 * np.pi).type(torch.float) * base_ang_vel_x)
        self.rew_buf[:, 3] += self.stage_buf[:, 4] * (-vel_penalty)
        # 能量
        self.rew_buf[:, 4] = -torch.square(self.dof_torques).mean(dim=-1)
        # 风格
        self.rew_buf[:, 5] = self.stage_buf[:, 0] * (-torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1))
        self.rew_buf[:, 5] += self.stage_buf[:, 1] * (-torch.square(self.dof_positions - self.crawled_dof_positions).mean(dim=-1))
        style_penalty = torch.square(self.dof_positions[:, 3:6] - self.crawled_dof_positions[:, 3:6]).mean(dim=-1)
        style_penalty += torch.square(self.dof_positions[:, 9:12] - self.crawled_dof_positions[:, 9:12]).mean(dim=-1)
        self.rew_buf[:, 5] += self.stage_buf[:, 2] * (-style_penalty)
        style_penalty = torch.square(self.dof_positions[:, :3] - self.crawled_dof_positions[:, :3]).mean(dim=-1)
        style_penalty += torch.square(self.dof_positions[:, 6:9] - self.crawled_dof_positions[:, 6:9]).mean(dim=-1)
        self.rew_buf[:, 5] += self.stage_buf[:, 3] * (-style_penalty)
        self.rew_buf[:, 5] += self.stage_buf[:, 4] * (-torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1))

        # ============ 成本 ============ #
        foot_contact_threshold = 0.25
        foot_contact_forces = self.contact_forces[:, self.foot_indices, :]
        calf_contact_forces = self.contact_forces[:, self.calf_indices, :]
        foot_contact = ((torch.norm(foot_contact_forces, dim=2) > 10.0)
                        | (torch.norm(calf_contact_forces, dim=2) > 10.0)).type(torch.float)
        self.cost_buf[:, 0] = self.stage_buf[:, 0] * foot_contact_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 1] * (1.0 - foot_contact.mean(dim=-1))
        self.cost_buf[:, 0] += self.stage_buf[:, 2] * foot_contact_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 3] * foot_contact_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 4] * foot_contact_threshold
        # 身体接触
        body_contact_threshold = 0.025
        term_contact = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        undesired_contact = torch.any(torch.norm(self.contact_forces[:, self.undesired_touch_indices, :], dim=-1) > 1.0, dim=-1)
        self.cost_buf[:, 1] = self.stage_buf[:, 0] * torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 1] * body_contact_threshold
        self.cost_buf[:, 1] += self.stage_buf[:, 2] * body_contact_threshold
        self.cost_buf[:, 1] += self.stage_buf[:, 3] * body_contact_threshold
        self.cost_buf[:, 1] += self.stage_buf[:, 4] * body_contact_threshold
        # 关节位置/速度/力矩
        self.cost_buf[:, 2] = torch.mean(
            ((self.dof_positions < self.dof_pos_lower_limits) | (self.dof_positions > self.dof_pos_upper_limits)).to(torch.float), dim=-1)
        self.cost_buf[:, 3] = torch.mean((torch.abs(self.dof_velocities) > self.dof_vel_upper_limits).to(torch.float), dim=-1)
        self.cost_buf[:, 4] = torch.mean((torch.abs(self.dof_torques) > self.dof_torques_upper_limits).to(torch.float), dim=-1)

        # ============ 阶段更新 ============ #
        from3_to4 = torch.logical_and(self.stage_buf[:, 3] == 1.0, self.is_one_turn_buf).type(torch.float32)
        self.stage_buf[:, 3] = (1.0 - from3_to4) * self.stage_buf[:, 3]
        self.stage_buf[:, 4] = from3_to4 + (1.0 - from3_to4) * self.stage_buf[:, 4]
        from2_to3 = torch.logical_and(self.stage_buf[:, 2] == 1.0, self.is_half_turn_buf).type(torch.float32)
        self.stage_buf[:, 2] = (1.0 - from2_to3) * self.stage_buf[:, 2]
        self.stage_buf[:, 3] = from2_to3 + (1.0 - from2_to3) * self.stage_buf[:, 3]
        from1_to2 = torch.logical_and(self.stage_buf[:, 1] == 1.0, com_height <= 0.15).type(torch.float32)
        self.stage_buf[:, 1] = (1.0 - from1_to2) * self.stage_buf[:, 1]
        self.stage_buf[:, 2] = from1_to2 + (1.0 - from1_to2) * self.stage_buf[:, 2]
        from0_to1 = torch.logical_and(
            self.stage_buf[:, 0] == 1.0, torch.logical_and(
                self.progress_buf * self.control_dt > self.start_time_buf,
                torch.logical_and(com_height >= 0.3, self.is_half_turn_buf == 0))).type(torch.float32)
        self.stage_buf[:, 0] = (1.0 - from0_to1) * self.stage_buf[:, 0]
        self.stage_buf[:, 1] = from0_to1 + (1.0 - from0_to1) * self.stage_buf[:, 1]

        # 翻滚检测
        self.is_half_turn_buf[:] = torch.logical_or(
            self.is_half_turn_buf, torch.logical_and(body_z[:, 1] < 0, body_z[:, 2] < 0)).type(torch.long)
        self.is_one_turn_buf[:] = torch.logical_or(
            self.is_one_turn_buf, torch.logical_and(
                self.is_half_turn_buf, torch.logical_and(body_z[:, 1] >= 0, body_z[:, 2] >= 0))).type(torch.long)

        # 失败判定
        body_contacts = torch.logical_and(
            self.stage_buf[:, 0] == 1.0,
            torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1))
        body_balances = torch.logical_and(self.stage_buf[:, 4] == 1.0, body_z[:, 2] < 0.5)
        self.fail_buf[:] = torch.logical_or(body_contacts, body_balances).type(torch.long)
