"""Direct Isaac Lab implementation of the Unitree Go2 backflip task."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from .go2_backflip_env_cfg import Go2BackflipEnvCfg


class Go2BackflipEnv(DirectRLEnv):
    """Phase-conditioned backflip with deployable actor and privileged critic observations."""

    cfg: Go2BackflipEnvCfg
    POLICY_JOINT_NAMES = tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    )

    def __init__(self, cfg: Go2BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self._robot.num_joints != 12:
            raise RuntimeError(f"Expected a 12-DoF Go2, found {self._robot.num_joints} joints")

        self._joint_index = {name: index for index, name in enumerate(self._robot.joint_names)}
        required_joints = set(self.POLICY_JOINT_NAMES)
        missing_joints = sorted(required_joints.difference(self._joint_index))
        if missing_joints:
            raise RuntimeError(f"The imported Go2 is missing joints: {missing_joints}")
        # Isaac Sim 5.1 imports this URDF breadth-first (all hips, then all
        # thighs, then all calves). Preserve the original/deployment actor
        # interface, which is grouped by leg: FL, FR, RL, RR.
        self._policy_joint_index = {
            name: index for index, name in enumerate(self.POLICY_JOINT_NAMES)
        }
        self._policy_to_sim = torch.tensor(
            [self._joint_index[name] for name in self.POLICY_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self._sim_to_policy = torch.argsort(self._policy_to_sim)

        body_names = self._robot.body_names
        self._base_id = self._body_id_exact("base")
        self._feet_ids = self._body_ids_matching(lambda name: name.endswith("_foot"))
        self._head_ids = self._body_ids_matching(lambda name: name.startswith("Head_"))
        self._limb_ids = self._body_ids_matching(
            lambda name: any(name.endswith(f"_{part}") for part in ("hip", "thigh", "calf", "foot"))
        )
        if len(self._feet_ids) != 4 or len(self._head_ids) != 2 or len(self._limb_ids) != 16:
            raise RuntimeError(
                "The local Go2 URDF was not imported with the required fixed links. "
                f"Bodies={body_names}; feet={len(self._feet_ids)}, heads={len(self._head_ids)}, "
                f"limbs={len(self._limb_ids)}"
            )
        self._undesired_contact_ids = self._body_ids_matching(
            lambda name: name == "base"
            or name.startswith("Head_")
            or "radar" in name
            or any(name.endswith(f"_{part}") for part in ("hip", "thigh", "calf"))
        )
        sensor_body_index = {name: index for index, name in enumerate(self._contact_sensor.body_names)}
        missing_sensor_bodies = sorted(set(body_names).difference(sensor_body_index))
        if missing_sensor_bodies:
            raise RuntimeError(f"Contact sensor is missing robot bodies: {missing_sensor_bodies}")
        self._feet_contact_ids = torch.tensor(
            [sensor_body_index[body_names[index]] for index in self._feet_ids.tolist()],
            dtype=torch.long,
            device=self.device,
        )
        self._rear_feet_contact_ids = torch.tensor(
            [
                sensor_body_index[name]
                for name in body_names
                if name in ("RL_foot", "RR_foot")
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._front_feet_contact_ids = torch.tensor(
            [
                sensor_body_index[name]
                for name in body_names
                if name in ("FL_foot", "FR_foot")
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._head_contact_ids = torch.tensor(
            [sensor_body_index[body_names[index]] for index in self._head_ids.tolist()],
            dtype=torch.long,
            device=self.device,
        )
        self._undesired_contact_sensor_ids = torch.tensor(
            [sensor_body_index[body_names[index]] for index in self._undesired_contact_ids.tolist()],
            dtype=torch.long,
            device=self.device,
        )
        self._rear_policy_joint_ids = torch.tensor(
            [
                self._policy_joint_index[name]
                for name in (
                    "RR_hip_joint",
                    "RR_thigh_joint",
                    "RR_calf_joint",
                    "RL_hip_joint",
                    "RL_thigh_joint",
                    "RL_calf_joint",
                )
            ],
            dtype=torch.long,
            device=self.device,
        )

        num_actions = 12
        num_envs = self.num_envs
        max_action_delay = self.cfg.control.max_action_delay_steps
        max_obs_delay = self.cfg.control.max_observation_delay_steps

        self._actions = torch.zeros(num_envs, num_actions, device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._last_actions_2 = torch.zeros_like(self._actions)
        self._action_history = torch.zeros(max_action_delay + 1, num_envs, num_actions, device=self.device)
        self._action_delay_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._obs_delay_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._slew_limited_actions = torch.zeros_like(self._actions)
        self._raw_joint_pos_target = torch.zeros_like(self._actions)
        self._joint_pos_target = torch.zeros_like(self._actions)
        self._applied_torques = torch.zeros_like(self._actions)

        self._torque_scales = torch.ones_like(self._actions)
        self._motor_velocity_scales = torch.ones_like(self._actions)
        self._joint_velocity_limits = torch.tensor(
            self.cfg.control.joint_velocity_limits, device=self.device
        ).unsqueeze(0)
        self._target_velocity_limits = torch.tensor(
            self.cfg.control.target_velocity_limits, device=self.device
        ).unsqueeze(0)
        if self._joint_velocity_limits.shape != (1, num_actions):
            raise ValueError("joint_velocity_limits must contain one value per action")
        if self._target_velocity_limits.shape != (1, num_actions):
            raise ValueError("target_velocity_limits must contain one value per action")
        if torch.any(self._joint_velocity_limits <= self.cfg.control.motor_velocity_x1):
            raise ValueError("every joint velocity limit must exceed motor_velocity_x1")
        self._p_gains = torch.full_like(self._actions, self.cfg.control.stiffness)
        self._d_gains = torch.full_like(self._actions, self.cfg.control.damping)
        self._motor_offsets = torch.zeros_like(self._actions)

        self._sensor_history = torch.zeros(max_obs_delay + 1, num_envs, 30, device=self.device)
        self._obs_delay_needs_fill = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._ang_vel_bias = torch.zeros(num_envs, 3, device=self.device)
        self._gravity_bias = torch.zeros(num_envs, 3, device=self.device)
        self._dof_pos_bias = torch.zeros(num_envs, 12, device=self.device)
        self._dof_vel_bias = torch.zeros(num_envs, 12, device=self.device)

        self._flip_angle = torch.zeros(num_envs, device=self.device)
        self._max_flip_angle = torch.zeros(num_envs, device=self.device)
        self._rotation_progress_step = torch.zeros(num_envs, device=self.device)
        self._flip_completed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._flip_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._just_completed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._just_succeeded = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._was_airborne = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._airborne_counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._head_contact_episode = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._landing_detected = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._just_landed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._landing_impact_steps_remaining = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._success_hold_counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._unsafe_episode = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._unsafe_contact = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._substep_velocity_violation = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self._substep_position_violation = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )

        # Episode diagnostics distinguish timeouts from each safety guard in
        # the RSL-RL console output.
        self._velocity_terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._position_terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._contact_terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._timed_out = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._max_joint_speed_ratio = torch.zeros(num_envs, device=self.device)
        self._velocity_termination_threshold = torch.full(
            (num_envs,), self.cfg.control.velocity_termination_start_ratio, device=self.device
        )
        self._velocity_termination_ratio = self.cfg.control.velocity_termination_start_ratio
        self._position_termination_margin = self.cfg.control.position_termination_start_margin
        self._max_landing_foot_force = torch.zeros(num_envs, device=self.device)
        self._max_rear_landing_foot_force = torch.zeros(num_envs, device=self.device)
        self._current_landing_foot_force = torch.zeros(num_envs, device=self.device)
        self._current_rear_landing_foot_force = torch.zeros(num_envs, device=self.device)
        self._max_joint_target_excess = torch.zeros(num_envs, device=self.device)
        self._max_joint_pos_excess = torch.zeros(num_envs, device=self.device)
        self._current_joint_pos_excess = torch.zeros(num_envs, device=self.device)

        self._limb_mass_scales = torch.ones(num_envs, 16, device=self.device)
        self._limb_inertia_scales = torch.ones(num_envs, 16, device=self.device)
        self._contact_friction = torch.ones(num_envs, 1, device=self.device)
        self._contact_restitution = torch.zeros(num_envs, 1, device=self.device)
        self._contact_offset = torch.full((num_envs, 1), 0.01, device=self.device)
        self._base_mass_values = torch.zeros(num_envs, 1, device=self.device)
        self._base_com_values = torch.zeros(num_envs, 3, device=self.device)

        self._randomize_rigid_properties()

        self._episode_sums = {
            name: torch.zeros(num_envs, device=self.device) for name in self.cfg.rewards.scales
        }
        self._reward_functions = {
            name: getattr(self, f"_reward_{name}") for name in self.cfg.rewards.scales
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _body_id_exact(self, body_name: str) -> int:
        try:
            return self._robot.body_names.index(body_name)
        except ValueError as exc:
            raise RuntimeError(f"Body {body_name!r} not found in {self._robot.body_names}") from exc

    def _body_ids_matching(self, predicate) -> torch.Tensor:
        ids = [index for index, name in enumerate(self._robot.body_names) if predicate(name)]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _peak_contact_force(self, sensor_body_ids: torch.Tensor) -> torch.Tensor:
        """Return the configured per-body contact-force sample."""
        if not self.cfg.rewards.use_peak_contact_forces:
            return torch.norm(
                self._contact_sensor.data.net_forces_w[:, sensor_body_ids], dim=-1
            )
        history = self._contact_sensor.data.net_forces_w_history
        if history is None:
            return torch.norm(
                self._contact_sensor.data.net_forces_w[:, sensor_body_ids], dim=-1
            )
        force_norm = torch.norm(history[:, : self.cfg.decimation, sensor_body_ids], dim=-1)
        return torch.max(force_norm, dim=1).values

    @staticmethod
    def _sample_uniform(low: float, high: float, shape: tuple[int, ...], device: str | torch.device):
        return low + (high - low) * torch.rand(shape, device=device)

    def _randomize_rigid_properties(self):
        """Apply startup randomization through Isaac Lab's PhysX tensor views."""
        rand = self.cfg.domain_rand
        env_ids_cpu = torch.arange(self.num_envs, dtype=torch.long, device="cpu")
        view = self._robot.root_physx_view

        masses = view.get_masses().clone()
        inertias = view.get_inertias().clone()
        default_masses = masses.clone()
        default_inertias = inertias.clone()

        base_added_mass = self._sample_uniform(
            *rand.added_base_mass_range, (self.num_envs,), masses.device
        )
        masses[:, self._base_id] = torch.clamp(default_masses[:, self._base_id] + base_added_mass, min=0.1)
        base_ratio = masses[:, self._base_id] / default_masses[:, self._base_id]
        inertias[:, self._base_id] = default_inertias[:, self._base_id] * base_ratio.unsqueeze(-1)

        limb_mass_scales = self._sample_uniform(
            *rand.limb_mass_scale_range, (self.num_envs, len(self._limb_ids)), masses.device
        )
        inertia_jitter = self._sample_uniform(
            *rand.limb_inertia_jitter_range, (self.num_envs, len(self._limb_ids)), inertias.device
        )
        limb_ids_cpu = self._limb_ids.cpu()
        masses[:, limb_ids_cpu] = default_masses[:, limb_ids_cpu] * limb_mass_scales
        limb_inertia_scales = limb_mass_scales * inertia_jitter
        inertias[:, limb_ids_cpu] = default_inertias[:, limb_ids_cpu] * limb_inertia_scales.unsqueeze(-1)

        view.set_masses(masses, env_ids_cpu)
        view.set_inertias(inertias, env_ids_cpu)
        self._limb_mass_scales[:] = limb_mass_scales.to(self.device)
        self._limb_inertia_scales[:] = limb_inertia_scales.to(self.device)
        self._base_mass_values[:, 0] = masses[:, self._base_id].to(self.device)

        coms = view.get_coms().clone()
        com_offsets = self._sample_uniform(
            *rand.added_base_com_range, (self.num_envs, 3), coms.device
        )
        coms[:, self._base_id, :3] += com_offsets
        view.set_coms(coms, env_ids_cpu)
        self._base_com_values[:] = coms[:, self._base_id, :3].to(self.device)

        # Isaac Gym's LeggedRobot uses 64 sampled friction buckets, then
        # assigns each environment a bucket.  Preserve that mechanism rather
        # than sampling an independent continuous value per environment.
        num_friction_buckets = 64
        friction_buckets = self._sample_uniform(
            *rand.friction_range, (num_friction_buckets, 1), "cpu"
        )
        friction_bucket_ids = torch.randint(
            0, num_friction_buckets, (self.num_envs, 1), device="cpu"
        )
        friction = friction_buckets[friction_bucket_ids.squeeze(-1)]
        restitution = self._sample_uniform(*rand.restitution_range, (self.num_envs, 1), "cpu")
        materials = view.get_material_properties().clone()
        materials[..., 0] = friction
        materials[..., 1] = friction
        materials[..., 2] = restitution
        view.set_material_properties(materials, env_ids_cpu)
        self._contact_friction[:] = friction.to(self.device)
        self._contact_restitution[:] = restitution.to(self.device)

        contact_offset = self._sample_uniform(*rand.contact_offset_range, (self.num_envs, 1), "cpu")
        rest_offset = self._sample_uniform(*rand.rest_offset_range, (self.num_envs, 1), "cpu")
        rest_offset = torch.minimum(rest_offset, contact_offset - 1.0e-4)
        contact_offsets = view.get_contact_offsets().clone()
        rest_offsets = view.get_rest_offsets().clone()
        contact_offsets[:] = contact_offset
        rest_offsets[:] = rest_offset
        view.set_contact_offsets(contact_offsets, env_ids_cpu)
        view.set_rest_offsets(rest_offsets, env_ids_cpu)
        self._contact_offset[:] = contact_offset.to(self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions[:] = torch.clamp(
            actions, -self.cfg.control.action_clip, self.cfg.control.action_clip
        )
        self._action_history = torch.roll(self._action_history, shifts=1, dims=0)
        self._action_history[0] = self._actions

    def _apply_action(self):
        # This hook runs before every 5-ms simulation step, so it can latch
        # peaks which would be invisible at the 50-Hz actor rate.
        self._record_substep_safety()
        delayed_actions = self._action_history[
            self._action_delay_steps, torch.arange(self.num_envs, device=self.device)
        ]
        max_action_delta = (
            self._target_velocity_limits
            * self._motor_velocity_scales
            * self.physics_dt
            / self.cfg.control.action_scale
        )
        action_delta = torch.clamp(
            delayed_actions - self._slew_limited_actions,
            min=-max_action_delta,
            max=max_action_delta,
        )
        self._slew_limited_actions += action_delta

        default_pos = self._robot.data.default_joint_pos[:, self._policy_to_sim]
        joint_pos = self._robot.data.joint_pos[:, self._policy_to_sim]
        joint_vel = self._robot.data.joint_vel[:, self._policy_to_sim]
        self._raw_joint_pos_target[:] = (
            delayed_actions * self.cfg.control.action_scale + default_pos + self._motor_offsets
        )
        desired_joint_pos_target = (
            self._slew_limited_actions * self.cfg.control.action_scale + default_pos + self._motor_offsets
        )
        joint_limits = self._robot.data.soft_joint_pos_limits[:, self._policy_to_sim]
        raw_target_excess = torch.maximum(
            joint_limits[:, :, 0] - self._raw_joint_pos_target,
            self._raw_joint_pos_target - joint_limits[:, :, 1],
        ).clamp(min=0.0)
        self._max_joint_target_excess = torch.maximum(
            self._max_joint_target_excess,
            torch.max(raw_target_excess, dim=1).values,
        )
        self._joint_pos_target[:] = torch.maximum(
            torch.minimum(desired_joint_pos_target, joint_limits[:, :, 1]),
            joint_limits[:, :, 0],
        )
        torques = (
            self._p_gains * (self._joint_pos_target - joint_pos)
            - self._d_gains * joint_vel
        )

        x1 = self.cfg.control.motor_velocity_x1 * self._motor_velocity_scales
        x2 = self._joint_velocity_limits * self._motor_velocity_scales
        speed = torch.abs(joint_vel)
        speed_fraction = torch.where(
            speed < x1,
            torch.ones_like(speed),
            torch.clamp((x2 - speed) / torch.clamp(x2 - x1, min=1.0e-6), 0.0, 1.0),
        )
        same_direction = joint_vel * torques > 0.0
        drive_peak = torch.full_like(torques, self.cfg.control.motor_torque_y1)
        brake_peak = torch.full_like(torques, self.cfg.control.motor_torque_y2)
        peak_torque = torch.where(same_direction, drive_peak, brake_peak) * self._torque_scales
        torque_limit = peak_torque * speed_fraction
        self._applied_torques[:] = torch.clamp(torques, min=-torque_limit, max=torque_limit)
        self._robot.set_joint_effort_target(self._applied_torques[:, self._sim_to_policy])

    def _record_substep_safety(self):
        """Latch hardware-relevant speed and position peaks at 200 Hz."""
        curriculum = self._safety_curriculum_scale()
        speed_ratio = self._joint_speed_ratio()
        self._max_joint_speed_ratio = torch.maximum(self._max_joint_speed_ratio, speed_ratio)
        velocity_ratio = self.cfg.control.velocity_termination_start_ratio + curriculum * (
            self.cfg.control.velocity_termination_ratio
            - self.cfg.control.velocity_termination_start_ratio
        )
        self._substep_velocity_violation |= speed_ratio > velocity_ratio

        joint_pos = self._robot.data.joint_pos[:, self._policy_to_sim]
        hard_limits = self._robot.data.joint_pos_limits[:, self._policy_to_sim]
        below = torch.clamp(hard_limits[:, :, 0] - joint_pos, min=0.0)
        above = torch.clamp(joint_pos - hard_limits[:, :, 1], min=0.0)
        position_excess = torch.max(below + above, dim=1).values
        self._current_joint_pos_excess[:] = position_excess
        self._max_joint_pos_excess = torch.maximum(
            self._max_joint_pos_excess, position_excess
        )
        position_margin = self.cfg.control.position_termination_start_margin + curriculum * (
            self.cfg.control.position_termination_margin
            - self.cfg.control.position_termination_start_margin
        )
        self._substep_position_violation |= position_excess > position_margin

    def _phase_features(self) -> tuple[torch.Tensor, ...]:
        phase_time = torch.clamp(
            self.episode_length_buf[:, None].float() * self.step_dt,
            max=self.cfg.rewards.phase_duration,
        )
        phase = torch.pi * phase_time / 2.0
        return (
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2.0),
            torch.cos(phase / 2.0),
            torch.sin(phase / 4.0),
            torch.cos(phase / 4.0),
        )

    def _clean_sensor_observation(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._policy_to_sim]
        default_joint_pos = self._robot.data.default_joint_pos[:, self._policy_to_sim]
        joint_vel = self._robot.data.joint_vel[:, self._policy_to_sim]
        return torch.cat(
            (
                self._robot.data.root_ang_vel_b * self.cfg.obs_scales.ang_vel,
                self._robot.data.projected_gravity_b,
                (joint_pos - default_joint_pos) * self.cfg.obs_scales.dof_pos,
                joint_vel * self.cfg.obs_scales.dof_vel,
            ),
            dim=-1,
        )

    def _yaw_frame_linear_velocity(self) -> torch.Tensor:
        yaw_quat = math_utils.yaw_quat(self._robot.data.root_link_quat_w)
        return math_utils.quat_apply_inverse(yaw_quat, self._robot.data.root_link_lin_vel_w)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        phase_features = self._phase_features()
        sensor_obs = self._clean_sensor_observation()
        actor_sensor = sensor_obs.clone()

        if self.cfg.noise.enabled:
            actor_sensor[:, 0:3] += self._ang_vel_bias * self.cfg.obs_scales.ang_vel
            actor_sensor[:, 3:6] += self._gravity_bias
            actor_sensor[:, 6:18] += self._dof_pos_bias * self.cfg.obs_scales.dof_pos
            actor_sensor[:, 18:30] += self._dof_vel_bias * self.cfg.obs_scales.dof_vel
            noise_scales = torch.tensor(
                [
                    *([self.cfg.noise.ang_vel_noise] * 3),
                    *([self.cfg.noise.gravity_noise] * 3),
                    *([self.cfg.noise.dof_pos_noise] * 12),
                    *([self.cfg.noise.dof_vel_noise] * 12),
                ],
                device=self.device,
            )
            actor_sensor += (2.0 * torch.rand_like(actor_sensor) - 1.0) * noise_scales

        self._sensor_history = torch.roll(self._sensor_history, shifts=1, dims=0)
        self._sensor_history[0] = actor_sensor
        fill_ids = self._obs_delay_needs_fill.nonzero(as_tuple=False).flatten()
        if len(fill_ids) > 0:
            self._sensor_history[:, fill_ids] = actor_sensor[fill_ids].unsqueeze(0)
            self._obs_delay_needs_fill[fill_ids] = False
        delayed_sensor = self._sensor_history[
            self._obs_delay_steps, torch.arange(self.num_envs, device=self.device)
        ]

        policy_obs = torch.cat(
            (delayed_sensor, self._actions, self._last_actions, *phase_features), dim=-1
        )
        state_privileged = torch.cat(
            (
                self._robot.data.root_link_pos_w[:, 2:3],
                self._yaw_frame_linear_velocity() * self.cfg.obs_scales.lin_vel,
                sensor_obs,
                self._actions,
                self._last_actions,
                *phase_features,
            ),
            dim=-1,
        )

        max_action_delay = max(1, self.cfg.control.max_action_delay_steps)
        max_obs_delay = max(1, self.cfg.control.max_observation_delay_steps)
        center_scale = max(abs(value) for value in self.cfg.domain_rand.added_base_com_range)
        motor_offset_scale = max(abs(value) for value in self.cfg.domain_rand.motor_offset_range)
        restitution_scale = max(self.cfg.domain_rand.restitution_range[1], 1.0e-6)
        nominal_base_mass = self._robot.data.default_mass[0, self._base_id]

        dynamics_privileged = torch.cat(
            (
                self._torque_scales,
                self._motor_velocity_scales,
                self._action_delay_steps[:, None].float() / max_action_delay,
                self._obs_delay_steps[:, None].float() / max_obs_delay,
                self._limb_mass_scales,
                self._limb_inertia_scales,
                self._contact_friction,
                self._contact_restitution / restitution_scale,
                self._contact_offset / 0.01,
                self._base_mass_values / nominal_base_mass,
                self._base_com_values / max(center_scale, 1.0e-6),
                self._p_gains / self.cfg.control.stiffness,
                self._d_gains,
                self._motor_offsets / max(motor_offset_scale, 1.0e-6),
            ),
            dim=-1,
        )
        critic_obs = torch.cat((state_privileged, dynamics_privileged), dim=-1)

        if policy_obs.shape[1] != 60 or critic_obs.shape[1] != 165:
            raise RuntimeError(
                f"Observation shape mismatch: policy={policy_obs.shape}, critic={critic_obs.shape}"
            )

        observations = {
            "policy": torch.clamp(policy_obs, -self.cfg.obs_scales.clip, self.cfg.obs_scales.clip),
            "critic": torch.clamp(critic_obs, -self.cfg.obs_scales.clip, self.cfg.obs_scales.clip),
        }
        self._last_actions_2[:] = self._last_actions
        self._last_actions[:] = self._actions
        return observations

    def _update_flip_state(self):
        # Include the state after the fourth and final physics substep.
        self._record_substep_safety()
        foot_forces = self._peak_contact_force(self._feet_contact_ids)
        self._current_landing_foot_force = torch.max(foot_forces, dim=1).values
        self._current_rear_landing_foot_force = torch.max(
            self._peak_contact_force(self._rear_feet_contact_ids), dim=1
        ).values
        feet_contact = foot_forces > self.cfg.rewards.recovery_contact_force
        feet_contact_count = torch.sum(feet_contact, dim=1)
        any_foot_contact = feet_contact_count > 0
        airborne_now = (self._time() >= self.cfg.rewards.takeoff_start) & (~any_foot_contact)
        self._airborne_counter = torch.where(
            airborne_now,
            self._airborne_counter + 1,
            torch.zeros_like(self._airborne_counter),
        )
        confirmed_takeoff = (
            (self._airborne_counter >= self.cfg.rewards.takeoff_confirm_steps)
            & (self._robot.data.root_link_pos_w[:, 2] >= self.cfg.rewards.takeoff_min_height)
        )
        self._was_airborne |= confirmed_takeoff
        self._just_landed = self._was_airborne & (~self._landing_detected) & any_foot_contact
        self._landing_detected |= self._just_landed

        head_force = torch.max(
            self._peak_contact_force(self._head_contact_ids), dim=1
        ).values
        self._head_contact_episode |= head_force > self.cfg.rewards.head_contact_success_force

        previous_max = self._max_flip_angle.clone()
        pitch_rate = torch.clamp(
            -self._robot.data.root_ang_vel_b[:, 1],
            min=-self.cfg.rewards.flip_angle_rate_clip,
            max=self.cfg.rewards.flip_angle_rate_clip,
        )
        invalid_pre_takeoff = (~self._was_airborne) & (~airborne_now)
        self._flip_angle[invalid_pre_takeoff] = 0.0
        self._max_flip_angle[invalid_pre_takeoff] = 0.0
        rotation_active = airborne_now & (~self._landing_detected) & (~self._head_contact_episode)
        self._flip_angle += pitch_rate * self.step_dt * rotation_active.float()
        self._max_flip_angle = torch.maximum(self._max_flip_angle, self._flip_angle)
        self._rotation_progress_step = torch.clamp(
            self._max_flip_angle - previous_max, min=0.0
        )
        self._just_completed = (
            (~self._flip_completed)
            & self._was_airborne
            & (self._max_flip_angle >= self.cfg.rewards.flip_completion_angle)
        )
        self._flip_completed |= self._just_completed

        impact_window = self.cfg.rewards.landing_impact_window_steps
        self._landing_impact_steps_remaining = torch.where(
            self._just_landed,
            torch.full_like(self._landing_impact_steps_remaining, impact_window),
            torch.clamp(self._landing_impact_steps_remaining - 1, min=0),
        )
        landing_active = self._landing_impact_steps_remaining > 0
        self._max_landing_foot_force = torch.maximum(
            self._max_landing_foot_force,
            self._current_landing_foot_force * landing_active,
        )
        self._max_rear_landing_foot_force = torch.maximum(
            self._max_rear_landing_foot_force,
            self._current_rear_landing_foot_force * landing_active,
        )

        undesired_forces = self._peak_contact_force(self._undesired_contact_sensor_ids)
        self._unsafe_contact = (
            torch.max(undesired_forces, dim=1).values
            > self.cfg.rewards.unsafe_body_contact_force
        )
        curriculum = self._safety_curriculum_scale()
        enforce_body_contact = (
            curriculum >= self.cfg.rewards.unsafe_contact_termination_curriculum
        )
        unsafe_now = (
            (self._unsafe_contact & enforce_body_contact)
            | self._substep_position_violation
            | self._substep_velocity_violation
            | self._head_contact_episode
        )
        self._unsafe_episode |= unsafe_now
        self._flip_success &= ~unsafe_now

        pose_error = torch.max(
            torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1
        ).values
        max_joint_speed = torch.max(torch.abs(self._robot.data.joint_vel), dim=1).values
        success_candidate = (
            self._flip_completed
            & (self._max_flip_angle >= self.cfg.rewards.flip_success_angle)
            & self._landing_detected
            & (self._time() >= self.cfg.rewards.landing_start)
            & (-self._robot.data.projected_gravity_b[:, 2] >= self.cfg.rewards.recovery_upright_cos)
            & (self._robot.data.root_link_pos_w[:, 2] >= self.cfg.rewards.recovery_min_height)
            & (torch.abs(self._robot.data.root_ang_vel_b[:, 1]) <= self.cfg.rewards.recovery_max_pitch_rate)
            & (max_joint_speed <= self.cfg.rewards.success_max_joint_speed)
            & (pose_error <= self.cfg.rewards.recovery_pose_error)
            & (feet_contact_count >= 3)
            & (~self._unsafe_episode)
        )
        self._success_hold_counter = torch.where(
            success_candidate,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        required_hold_steps = max(
            1, int(round(self.cfg.rewards.success_hold_time_s / self.step_dt))
        )
        success_now = self._success_hold_counter >= required_hold_steps
        self._just_succeeded = (~self._flip_success) & success_now
        self._flip_success |= self._just_succeeded

    def _joint_speed_ratio(self) -> torch.Tensor:
        return torch.max(
            torch.abs(self._robot.data.joint_vel[:, self._policy_to_sim])
            / torch.clamp(
                self._joint_velocity_limits * self._motor_velocity_scales,
                min=1.0,
            ),
            dim=1,
        ).values

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_flip_state()
        curriculum = self._safety_curriculum_scale()
        start_ratio = self.cfg.control.velocity_termination_start_ratio
        end_ratio = self.cfg.control.velocity_termination_ratio
        self._velocity_termination_ratio = start_ratio + (end_ratio - start_ratio) * curriculum
        position_start = self.cfg.control.position_termination_start_margin
        position_end = self.cfg.control.position_termination_margin
        self._position_termination_margin = position_start + (
            position_end - position_start
        ) * curriculum
        self._velocity_termination_threshold[:] = self._velocity_termination_ratio
        velocity_terminated = self._substep_velocity_violation.clone()
        position_terminated = self._substep_position_violation.clone()
        enforce_body_contact = (
            curriculum >= self.cfg.rewards.unsafe_contact_termination_curriculum
        )
        contact_terminated = self._unsafe_contact & enforce_body_contact
        terminated = velocity_terminated | position_terminated | contact_terminated
        # Match the original Gym task: it increments the control-step counter
        # before checking ``> max_episode_length``.  This preserves its final
        # 20-ms recovery sample instead of ending one policy step earlier.
        time_out = self.episode_length_buf > self.max_episode_length
        self._velocity_terminated[:] = velocity_terminated
        self._position_terminated[:] = position_terminated
        self._contact_terminated[:] = contact_terminated
        self._timed_out[:] = time_out
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        rewards = {
            name: function() * self.cfg.rewards.scales[name] * self.step_dt
            for name, function in self._reward_functions.items()
        }
        for name, value in rewards.items():
            self._episode_sums[name] += value
        return torch.stack(tuple(rewards.values()), dim=0).sum(dim=0)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        finished_angle = self._max_flip_angle[env_ids].clone()
        finished_success = self._flip_success[env_ids].float().clone()
        finished_unsafe = self._unsafe_episode[env_ids].float().clone()
        finished_airborne = self._was_airborne[env_ids].float().clone()
        finished_head_contact = self._head_contact_episode[env_ids].float().clone()
        finished_velocity_terminated = self._velocity_terminated[env_ids].float().clone()
        finished_position_terminated = self._position_terminated[env_ids].float().clone()
        finished_contact_terminated = self._contact_terminated[env_ids].float().clone()
        finished_timed_out = self._timed_out[env_ids].float().clone()
        finished_max_speed_ratio = self._max_joint_speed_ratio[env_ids].clone()
        finished_max_landing_force = self._max_landing_foot_force[env_ids].clone()
        finished_max_rear_landing_force = self._max_rear_landing_foot_force[env_ids].clone()
        finished_max_joint_target_excess = self._max_joint_target_excess[env_ids].clone()
        finished_max_joint_pos_excess = self._max_joint_pos_excess[env_ids].clone()
        finished_recovery_hold_time = (
            self._success_hold_counter[env_ids].float().clone() * self.step_dt
        )
        super()._reset_idx(env_ids)

        self.extras["log"] = {}
        if hasattr(self, "_episode_sums"):
            for name, episodic_sum in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{name}"] = torch.mean(episodic_sum[env_ids]) / self.max_episode_length_s
                episodic_sum[env_ids] = 0.0
            self.extras["log"]["Episode_Metric/flip_angle_rad"] = torch.mean(finished_angle)
            self.extras["log"]["Episode_Metric/flip_success"] = torch.mean(finished_success)
            self.extras["log"]["Episode_Metric/unsafe_episode"] = torch.mean(finished_unsafe)
            self.extras["log"]["Episode_Metric/airborne_episode"] = torch.mean(finished_airborne)
            self.extras["log"]["Episode_Metric/head_contact_episode"] = torch.mean(
                finished_head_contact
            )
            self.extras["log"]["Episode_Metric/max_joint_speed_ratio"] = torch.mean(
                finished_max_speed_ratio
            )
            self.extras["log"]["Episode_Metric/max_landing_foot_force_n"] = torch.mean(
                finished_max_landing_force
            )
            self.extras["log"]["Episode_Metric/max_rear_landing_foot_force_n"] = torch.mean(
                finished_max_rear_landing_force
            )
            self.extras["log"]["Episode_Metric/max_joint_target_excess_rad"] = torch.mean(
                finished_max_joint_target_excess
            )
            self.extras["log"]["Episode_Metric/max_joint_pos_excess_rad"] = torch.mean(
                finished_max_joint_pos_excess
            )
            self.extras["log"]["Episode_Metric/recovery_hold_time_s"] = torch.mean(
                finished_recovery_hold_time
            )
            self.extras["log"]["Episode_Termination/joint_velocity_rate"] = torch.mean(
                finished_velocity_terminated
            )
            self.extras["log"]["Episode_Termination/joint_position_rate"] = torch.mean(
                finished_position_terminated
            )
            self.extras["log"]["Episode_Termination/body_contact_rate"] = torch.mean(
                finished_contact_terminated
            )
            self.extras["log"]["Episode_Termination/timeout_rate"] = torch.mean(
                finished_timed_out
            )
            self.extras["log"]["Episode_Curriculum/safety_scale"] = self._safety_curriculum_scale()
            self.extras["log"]["Episode_Curriculum/velocity_termination_ratio"] = (
                self._velocity_termination_ratio
            )
            self.extras["log"]["Episode_Curriculum/position_termination_margin_rad"] = (
                self._position_termination_margin
            )

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._last_actions_2[env_ids] = 0.0
        self._action_history[:, env_ids] = 0.0
        self._slew_limited_actions[env_ids] = 0.0
        policy_joint_pos = joint_pos[:, self._policy_to_sim]
        self._raw_joint_pos_target[env_ids] = policy_joint_pos
        self._joint_pos_target[env_ids] = policy_joint_pos
        self._applied_torques[env_ids] = 0.0
        self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)

        rand = self.cfg.domain_rand
        count = len(env_ids)
        self._torque_scales[env_ids] = self._sample_uniform(
            *rand.torque_scale_range, (count, 12), self.device
        )
        self._motor_velocity_scales[env_ids] = self._sample_uniform(
            *rand.motor_velocity_scale_range, (count, 12), self.device
        )
        kp_scale = self._sample_uniform(*rand.kp_scale_range, (count, 12), self.device)
        kd_scale = self._sample_uniform(*rand.kd_scale_range, (count, 12), self.device)
        self._p_gains[env_ids] = self.cfg.control.stiffness * kp_scale
        self._d_gains[env_ids] = self.cfg.control.damping * kd_scale
        self._motor_offsets[env_ids] = self._sample_uniform(
            *rand.motor_offset_range, (count, 12), self.device
        )
        if self.cfg.control.fixed_action_delay_steps is None:
            self._action_delay_steps[env_ids] = torch.randint(
                0, self.cfg.control.max_action_delay_steps + 1, (count,), device=self.device
            )
        else:
            self._action_delay_steps[env_ids] = self.cfg.control.fixed_action_delay_steps
        if self.cfg.control.fixed_observation_delay_steps is None:
            self._obs_delay_steps[env_ids] = torch.randint(
                0, self.cfg.control.max_observation_delay_steps + 1, (count,), device=self.device
            )
        else:
            self._obs_delay_steps[env_ids] = self.cfg.control.fixed_observation_delay_steps
        self._sensor_history[:, env_ids] = 0.0
        self._obs_delay_needs_fill[env_ids] = True

        if self.cfg.noise.enabled:
            for buffer, value_range in (
                (self._ang_vel_bias, self.cfg.noise.ang_vel_bias_range),
                (self._gravity_bias, self.cfg.noise.gravity_bias_range),
                (self._dof_pos_bias, self.cfg.noise.dof_pos_bias_range),
                (self._dof_vel_bias, self.cfg.noise.dof_vel_bias_range),
            ):
                buffer[env_ids] = self._sample_uniform(
                    *value_range, (count, buffer.shape[1]), self.device
                )
        else:
            self._ang_vel_bias[env_ids] = 0.0
            self._gravity_bias[env_ids] = 0.0
            self._dof_pos_bias[env_ids] = 0.0
            self._dof_vel_bias[env_ids] = 0.0

        self._flip_angle[env_ids] = 0.0
        self._max_flip_angle[env_ids] = 0.0
        self._rotation_progress_step[env_ids] = 0.0
        self._flip_completed[env_ids] = False
        self._flip_success[env_ids] = False
        self._just_completed[env_ids] = False
        self._just_succeeded[env_ids] = False
        self._was_airborne[env_ids] = False
        self._airborne_counter[env_ids] = 0
        self._head_contact_episode[env_ids] = False
        self._landing_detected[env_ids] = False
        self._just_landed[env_ids] = False
        self._landing_impact_steps_remaining[env_ids] = 0
        self._success_hold_counter[env_ids] = 0
        self._unsafe_episode[env_ids] = False
        self._unsafe_contact[env_ids] = False
        self._substep_velocity_violation[env_ids] = False
        self._substep_position_violation[env_ids] = False
        self._velocity_terminated[env_ids] = False
        self._position_terminated[env_ids] = False
        self._contact_terminated[env_ids] = False
        self._timed_out[env_ids] = False
        self._max_joint_speed_ratio[env_ids] = 0.0
        self._max_landing_foot_force[env_ids] = 0.0
        self._max_rear_landing_foot_force[env_ids] = 0.0
        self._current_landing_foot_force[env_ids] = 0.0
        self._current_rear_landing_foot_force[env_ids] = 0.0
        self._max_joint_target_excess[env_ids] = 0.0
        self._max_joint_pos_excess[env_ids] = 0.0
        self._current_joint_pos_excess[env_ids] = 0.0
        self._velocity_termination_threshold[env_ids] = (
            self.cfg.control.velocity_termination_start_ratio
        )

    def _time(self) -> torch.Tensor:
        return self.episode_length_buf.float() * self.step_dt

    def _safety_curriculum_scale(self) -> float:
        warmup = self.cfg.rewards.safety_curriculum_warmup_steps
        ramp = max(1, self.cfg.rewards.safety_curriculum_ramp_steps)
        progress = min(max((self.common_step_counter - warmup) / ramp, 0.0), 1.0)
        start = self.cfg.rewards.safety_curriculum_start
        return start + (1.0 - start) * progress

    def _reward_termination(self):
        terminated = (
            self._velocity_terminated
            | self._position_terminated
            | self._contact_terminated
        )
        return (terminated & (~self._timed_out)).float()

    def _reward_ang_vel_y(self):
        value = torch.clamp(
            -self._robot.data.root_ang_vel_b[:, 1],
            -self.cfg.rewards.max_pitch_rate,
            self.cfg.rewards.max_pitch_rate,
        )
        active = (self._time() > self.cfg.rewards.takeoff_start) & (
            self._time() < self.cfg.rewards.rotation_end
        )
        return value * active

    def _reward_ang_vel_z(self):
        return torch.abs(self._robot.data.root_ang_vel_b[:, 2])

    def _reward_lin_vel_z(self):
        value = torch.clamp(
            self._robot.data.root_link_lin_vel_w[:, 2], max=self.cfg.rewards.max_upward_velocity
        )
        active = (self._time() > self.cfg.rewards.takeoff_start) & (
            self._time() < self.cfg.rewards.takeoff_end
        )
        return value * active

    def _reward_orientation_control(self):
        phase = torch.clamp(self._time() - self.cfg.rewards.takeoff_start, 0.0, 0.5)
        desired_angle = -4.0 * torch.pi * phase
        axis = torch.zeros(self.num_envs, 3, device=self.device)
        axis[:, 1] = 1.0
        desired_quat = math_utils.quat_from_angle_axis(desired_angle, axis)
        gravity = torch.zeros(self.num_envs, 3, device=self.device)
        gravity[:, 2] = -1.0
        desired_gravity = math_utils.quat_apply_inverse(desired_quat, gravity)
        return torch.square(self._robot.data.projected_gravity_b - desired_gravity).sum(dim=1)

    def _reward_feet_height_before_backflip(self):
        height = torch.clamp(self._robot.data.body_link_pos_w[:, self._feet_ids, 2] - 0.02, min=0.0)
        return height.sum(dim=1) * (self._time() < self.cfg.rewards.takeoff_start)

    def _reward_height_control(self):
        value = torch.square(self.cfg.rewards.target_height - self._robot.data.root_link_pos_w[:, 2])
        active = (self._time() < 0.4) | self._landing_detected
        return value * active

    def _reward_default_pose(self):
        error = torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos).sum(dim=1)
        active = (self._time() < self.cfg.rewards.takeoff_start) | self._landing_detected
        return error * active

    def _reward_head_clearance(self):
        min_head_z = torch.min(self._robot.data.body_link_pos_w[:, self._head_ids, 2], dim=1).values
        shortfall = torch.clamp(self.cfg.rewards.min_head_center_height - min_head_z, min=0.0)
        return self._safety_curriculum_scale() * torch.square(
            shortfall / self.cfg.rewards.min_head_center_height
        )

    def _reward_head_contact(self):
        forces = self._peak_contact_force(self._head_contact_ids)
        normalized = torch.clamp(
            torch.max(forces, dim=1).values / self.cfg.rewards.head_contact_force,
            min=0.0,
            max=self.cfg.rewards.max_body_contact_penalty,
        )
        return self._safety_curriculum_scale() * normalized

    def _reward_landing_impact(self):
        peak_force = self._current_landing_foot_force
        active = self._landing_impact_steps_remaining > 0
        excess = torch.clamp(
            (peak_force - self.cfg.rewards.landing_force_threshold)
            / self.cfg.rewards.landing_force_threshold,
            min=0.0,
            max=self.cfg.rewards.max_landing_impact_penalty,
        )
        return (
            self._safety_curriculum_scale()
            * torch.square(excess)
            * active
        )

    def _reward_actions_symmetry(self):
        pairs = (
            ("FR_hip_joint", "FL_hip_joint", 1.0),
            ("FR_thigh_joint", "FL_thigh_joint", -1.0),
            ("FR_calf_joint", "FL_calf_joint", -1.0),
            ("RR_hip_joint", "RL_hip_joint", 1.0),
            ("RR_thigh_joint", "RL_thigh_joint", -1.0),
            ("RR_calf_joint", "RL_calf_joint", -1.0),
        )
        value = torch.zeros(self.num_envs, device=self.device)
        for right, left, sign in pairs:
            value += torch.square(
                self._actions[:, self._policy_joint_index[right]]
                + sign * self._actions[:, self._policy_joint_index[left]]
            )
        return value

    def _reward_gravity_y(self):
        return torch.square(self._robot.data.projected_gravity_b[:, 1])

    def _reward_feet_distance(self):
        relative = self._robot.data.body_link_pos_w[:, self._feet_ids] - self._robot.data.root_link_pos_w[:, None]
        quat = self._robot.data.root_link_quat_w[:, None].expand(-1, len(self._feet_ids), -1).reshape(-1, 4)
        body_feet = math_utils.quat_apply_inverse(quat, relative.reshape(-1, 3)).reshape(self.num_envs, -1, 3)
        return torch.square(body_feet[:, :, 1]).sum(dim=1)

    def _reward_action_rate(self):
        return torch.square(self._actions - self._last_actions).sum(dim=1)

    def _reward_action_jerk(self):
        second_difference = self._actions - 2.0 * self._last_actions + self._last_actions_2
        return self._safety_curriculum_scale() * torch.square(second_difference).sum(dim=1)

    def _reward_dof_vel_limits(self):
        effective_limit = (
            self._joint_velocity_limits
            * self.cfg.rewards.soft_dof_vel_limit
            * self._motor_velocity_scales
        )
        relative_excess = torch.clamp(
            torch.abs(self._robot.data.joint_vel[:, self._policy_to_sim]) / effective_limit - 1.0,
            min=0.0,
            max=4.0,
        )
        return self._safety_curriculum_scale() * torch.square(relative_excess).sum(dim=1)

    def _reward_rotation_progress(self):
        return self._rotation_progress_step

    def _reward_flip_completion(self):
        return self._just_completed.float()

    def _reward_flip_success(self):
        return self._just_succeeded.float()

    def _recovery_active(self):
        return self._flip_completed & self._landing_detected

    def _reward_recovery_upright(self):
        upright_error = torch.square(self._robot.data.projected_gravity_b[:, :2]).sum(dim=1)
        return torch.exp(-4.0 * upright_error) * self._recovery_active()

    def _reward_recovery_height(self):
        error = torch.square(
            (self._robot.data.root_link_pos_w[:, 2] - self.cfg.rewards.target_height) / 0.10
        )
        return torch.exp(-error) * self._recovery_active()

    def _reward_recovery_default_pose(self):
        mean_error = torch.square(
            self._robot.data.joint_pos - self._robot.data.default_joint_pos
        ).mean(dim=1)
        return torch.exp(-mean_error / 0.09) * self._recovery_active()

    def _reward_recovery_still(self):
        motion = torch.square(self._robot.data.root_ang_vel_b).sum(dim=1) + 0.25 * torch.square(
            self._yaw_frame_linear_velocity()
        ).sum(dim=1)
        return torch.exp(-0.5 * motion) * self._recovery_active()

    def _reward_recovery_feet_contact(self):
        contact_fraction = torch.mean(
            (
                self._peak_contact_force(self._feet_contact_ids)
                > self.cfg.rewards.recovery_contact_force
            ).float(),
            dim=1,
        )
        return contact_fraction * self._recovery_active()

    def _reward_rear_leg_action_rate(self):
        delta = (
            self._actions[:, self._rear_policy_joint_ids]
            - self._last_actions[:, self._rear_policy_joint_ids]
        )
        return (
            self._safety_curriculum_scale()
            * torch.square(delta).sum(dim=1)
            * (self._time() > self.cfg.rewards.takeoff_start)
        )

    def _reward_rear_leg_symmetry(self):
        pos = self._robot.data.joint_pos
        hip = pos[:, self._joint_index["RR_hip_joint"]] + pos[:, self._joint_index["RL_hip_joint"]]
        thigh = pos[:, self._joint_index["RR_thigh_joint"]] - pos[:, self._joint_index["RL_thigh_joint"]]
        calf = pos[:, self._joint_index["RR_calf_joint"]] - pos[:, self._joint_index["RL_calf_joint"]]
        return (
            self._safety_curriculum_scale()
            * (torch.square(hip) + torch.square(thigh) + torch.square(calf))
            * (self._time() > self.cfg.rewards.takeoff_start)
        )

    def _reward_recovery_dof_velocity(self):
        normalized = torch.clamp(torch.abs(self._robot.data.joint_vel) / 10.0, max=2.0)
        return (
            self._safety_curriculum_scale()
            * torch.square(normalized).sum(dim=1)
            * (self._time() >= self.cfg.rewards.recovery_velocity_start)
        )

    def _reward_undesired_body_contact(self):
        forces = self._peak_contact_force(self._undesired_contact_sensor_ids)
        normalized = torch.clamp(
            forces / self.cfg.rewards.body_contact_force,
            min=0.0,
            max=self.cfg.rewards.max_body_contact_penalty,
        )
        return self._safety_curriculum_scale() * torch.max(normalized, dim=1).values

    def _reward_dof_pos_limits(self):
        limits = self._robot.data.soft_joint_pos_limits
        below = -(self._robot.data.joint_pos - limits[:, :, 0]).clamp(max=0.0)
        above = (self._robot.data.joint_pos - limits[:, :, 1]).clamp(min=0.0)
        return self._safety_curriculum_scale() * (below + above).sum(dim=1)

    def _reward_joint_target_limits(self):
        limits = self._robot.data.soft_joint_pos_limits[:, self._policy_to_sim]
        below = torch.clamp(
            limits[:, :, 0] - self._raw_joint_pos_target, min=0.0
        )
        above = torch.clamp(
            self._raw_joint_pos_target - limits[:, :, 1], min=0.0
        )
        return self._safety_curriculum_scale() * torch.square(below + above).sum(dim=1)
