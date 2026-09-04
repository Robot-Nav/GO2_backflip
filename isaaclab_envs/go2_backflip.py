"""Go2 后空翻任务（移植到 IsaacLab）。"""

import numpy as np
import torch

from .go2_env import Go2Env, compute_observations, quat_rotate_inverse


class Go2Backflip(Go2Env):
    raw_obs_dim = 42

    def _init_task_buffers(self):
        # 后翻状态检测
        self.is_half_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_one_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cmd_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.land_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _reset_idx_task(self, env_ids):
        self.is_half_turn_buf[env_ids] = 0
        self.is_one_turn_buf[env_ids] = 0
        self.cmd_time_buf[env_ids] = 0.0
        self.land_time_buf[env_ids] = 0.0

    def _compute_observation(self, env_ids):
        commands = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        masks0 = (self.cmd_time_buf[env_ids] == 0).type(torch.float32)
        masks1 = (1.0 - masks0) * (self.progress_buf[env_ids] * self.control_dt < self.cmd_time_buf[env_ids] + 0.2).type(torch.float32)
        masks2 = (1.0 - masks0) * (1.0 - masks1)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2
        return compute_observations(
            self.est_base_body_orns[env_ids], self.est_dof_positions[env_ids],
            self.est_dof_velocities[env_ids], self.prev_actions[env_ids], commands)

    def _compute_rewards_costs(self):
        # 阶段 0 站立，1 下蹲，2 起跳，3 后翻，4 落地
        # ============ 奖励 ============ #
        # 质心高度
        com_height = self.base_positions[:, 2]
        self.rew_buf[:, 0] = self.stage_buf[:, 0] * (-torch.abs(com_height - 0.35))
        self.rew_buf[:, 0] += self.stage_buf[:, 1] * (-torch.abs(com_height - 0.2))
        self.rew_buf[:, 0] += self.stage_buf[:, 2] * (com_height <= 0.5) * com_height
        self.rew_buf[:, 0] += self.stage_buf[:, 3] * (com_height <= 0.5) * com_height
        self.rew_buf[:, 0] += self.stage_buf[:, 4] * (-torch.abs(com_height - 0.35))
        # 身体平衡
        body_z = quat_rotate_inverse(self.base_quaternions, self.world_z)
        self.rew_buf[:, 1] = self.stage_buf[:, 0] * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 1] * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 2] * (-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi / 2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 3] * (-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi / 2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 4] * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        # 俯仰角速度
        base_lin_vels = quat_rotate_inverse(self.base_quaternions, self.base_lin_vels)
        base_ang_vels = quat_rotate_inverse(self.base_quaternions, self.base_ang_vels)
        vel_penalty = torch.square(base_lin_vels[:, 0]) + torch.square(base_lin_vels[:, 1]) + torch.square(base_ang_vels[:, 2])
        base_ang_vel_y = base_ang_vels[:, 1]
        self.rew_buf[:, 2] = self.stage_buf[:, 0] * (-vel_penalty)
        self.rew_buf[:, 2] += self.stage_buf[:, 1] * (-vel_penalty)
        self.rew_buf[:, 2] += self.stage_buf[:, 2] * (1.0 - self.is_one_turn_buf) * (-base_ang_vel_y)
        self.rew_buf[:, 2] += self.stage_buf[:, 3] * (1.0 - self.is_one_turn_buf) * (-base_ang_vel_y)
        self.rew_buf[:, 2] += self.stage_buf[:, 4] * (-vel_penalty)
        # 能量
        self.rew_buf[:, 3] = -torch.square(self.dof_torques).mean(dim=-1)
        # 风格
        self.rew_buf[:, 4] = -torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1)

        # ============ 成本 ============ #
        # 足端接触
        foot_contact_threshold = 0.25
        foot_contact_forces = self.contact_forces[:, self.foot_indices, :]
        calf_contact_forces = self.contact_forces[:, self.calf_indices, :]
        foot_contact = ((torch.norm(foot_contact_forces, dim=2) > 10.0)
                        | (torch.norm(calf_contact_forces, dim=2) > 10.0)).type(torch.float)
        self.cost_buf[:, 0] = self.stage_buf[:, 0] * foot_contact_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 1] * foot_contact_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 2] * (1.0 - (foot_contact[:, 2] + foot_contact[:, 3]) / 2.0)
        self.cost_buf[:, 0] += self.stage_buf[:, 3] * foot_contact_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 4] * foot_contact_threshold
        # 身体接触
        term_contact = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        undesired_contact = torch.any(torch.norm(self.contact_forces[:, self.undesired_touch_indices, :], dim=-1) > 1.0, dim=-1)
        self.cost_buf[:, 1] = self.stage_buf[:, 0] * torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 1] * torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 2] * torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 3] * undesired_contact.type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 4] * undesired_contact.type(torch.float)
        # 关节位置
        self.cost_buf[:, 2] = torch.mean(
            ((self.dof_positions < self.dof_pos_lower_limits) | (self.dof_positions > self.dof_pos_upper_limits)).to(torch.float), dim=-1)
        # 关节速度
        self.cost_buf[:, 3] = torch.mean((torch.abs(self.dof_velocities) > self.dof_vel_upper_limits).to(torch.float), dim=-1)
        # 关节力矩
        self.cost_buf[:, 4] = torch.mean((torch.abs(self.dof_torques) > self.dof_torques_upper_limits).to(torch.float), dim=-1)

        # ============ 阶段更新（从 N 到 0） ============ #
        from3_to4 = torch.logical_and(
            self.stage_buf[:, 3] == 1.0, torch.logical_and(foot_contact.mean(dim=-1) > 0.0, self.is_half_turn_buf)).type(torch.float32)
        self.stage_buf[:, 3] = (1.0 - from3_to4) * self.stage_buf[:, 3]
        self.stage_buf[:, 4] = from3_to4 + (1.0 - from3_to4) * self.stage_buf[:, 4]
        from2_to3 = torch.logical_and(self.stage_buf[:, 2] == 1.0, foot_contact.mean(dim=-1) < 0.1).type(torch.float32)
        self.stage_buf[:, 2] = (1.0 - from2_to3) * self.stage_buf[:, 2]
        self.stage_buf[:, 3] = from2_to3 + (1.0 - from2_to3) * self.stage_buf[:, 3]
        from1_to2 = torch.logical_and(
            self.stage_buf[:, 1] == 1.0, torch.logical_and(com_height <= 0.25, foot_contact.mean(dim=-1) >= 0.9)).type(torch.float32)
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
            self.is_half_turn_buf, torch.logical_and(body_z[:, 0] < 0, body_z[:, 2] < 0)).type(torch.long)
        self.is_one_turn_buf[:] = torch.logical_or(
            self.is_one_turn_buf, torch.logical_and(
                self.is_half_turn_buf, torch.logical_and(body_z[:, 0] >= 0, body_z[:, 2] >= 0))).type(torch.long)
        land_masks = torch.logical_and(self.land_time_buf == 0, self.stage_buf[:, 4] == 1).type(torch.float32)
        self.land_time_buf[:] = land_masks * (self.progress_buf * self.control_dt) + (1.0 - land_masks) * self.land_time_buf
        cmd_masks = torch.logical_and(self.cmd_time_buf == 0, self.stage_buf[:, 1] == 1).type(torch.float32)
        self.cmd_time_buf[:] = cmd_masks * (self.progress_buf * self.control_dt) + (1.0 - cmd_masks) * self.cmd_time_buf

        # 失败判定
        body_contacts = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        landing_wo_turns = torch.logical_and(
            self.stage_buf[:, 3] == 1.0, torch.logical_and(foot_contact.mean(dim=-1) > 0.0, 1 - self.is_half_turn_buf))
        self.fail_buf[:] = torch.logical_or(body_contacts, landing_wo_turns).type(torch.long)
