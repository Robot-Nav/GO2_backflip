"""Configuration for the phase-conditioned Unitree Go2 backflip task."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GO2_URDF = PROJECT_ROOT / "resources" / "robots" / "go2" / "urdf" / "go2_description.urdf"


@configclass
class ControlCfg:
    action_scale: float = 0.5
    stiffness: float = 40.0
    damping: float = 1.0
    max_action_delay_steps: int = 2
    max_observation_delay_steps: int = 2
    fixed_action_delay_steps: int | None = None
    fixed_observation_delay_steps: int | None = None
    motor_velocity_x1: float = 13.5
    motor_torque_y1: float = 20.2
    motor_torque_y2: float = 23.4
    # Go2HV no-load and deployable PD-target slew speeds, in policy joint order.
    joint_velocity_limits: tuple[float, ...] = (30.0,) * 12
    target_velocity_limits: tuple[float, ...] = (13.5,) * 12
    velocity_termination_start_ratio: float = 1.50
    velocity_termination_ratio: float = 1.00
    # Actual position is checked against the hard URDF range. The permitted
    # overshoot tightens to zero with the safety curriculum.
    position_termination_start_margin: float = 0.30
    position_termination_margin: float = 0.0
    position_validation_tolerance: float = 1.0e-3
    action_clip: float = 8.0


@configclass
class DomainRandomizationCfg:
    friction_range: tuple[float, float] = (0.5, 1.25)
    restitution_range: tuple[float, float] = (0.0, 0.05)
    contact_offset_range: tuple[float, float] = (0.0075, 0.0125)
    rest_offset_range: tuple[float, float] = (-0.001, 0.001)
    added_base_mass_range: tuple[float, float] = (-0.5, 1.0)
    added_base_com_range: tuple[float, float] = (-0.015, 0.015)
    limb_mass_scale_range: tuple[float, float] = (0.95, 1.05)
    limb_inertia_jitter_range: tuple[float, float] = (0.90, 1.10)
    torque_scale_range: tuple[float, float] = (0.75, 1.00)
    motor_velocity_scale_range: tuple[float, float] = (0.85, 1.00)
    kp_scale_range: tuple[float, float] = (0.8, 1.2)
    kd_scale_range: tuple[float, float] = (0.8, 1.2)
    motor_offset_range: tuple[float, float] = (-0.02, 0.02)


@configclass
class NoiseCfg:
    enabled: bool = True
    ang_vel_bias_range: tuple[float, float] = (-0.05, 0.05)
    gravity_bias_range: tuple[float, float] = (-0.01, 0.01)
    dof_pos_bias_range: tuple[float, float] = (-0.01, 0.01)
    dof_vel_bias_range: tuple[float, float] = (-0.20, 0.20)
    ang_vel_noise: float = 0.025
    gravity_noise: float = 0.01
    dof_pos_noise: float = 0.01
    dof_vel_noise: float = 0.025


@configclass
class ObservationScaleCfg:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    clip: float = 100.0


@configclass
class RewardCfg:
    soft_dof_pos_limit: float = 0.95
    soft_dof_vel_limit: float = 0.80
    target_height: float = 0.30
    takeoff_start: float = 0.50
    takeoff_end: float = 0.75
    rotation_end: float = 1.00
    landing_start: float = 1.40
    recovery_velocity_start: float = 1.00
    landing_impact_window_steps: int = 6
    success_hold_time_s: float = 0.20
    success_max_joint_speed: float = 2.0
    unsafe_body_contact_force: float = 100.0
    unsafe_contact_termination_curriculum: float = 1.00
    max_pitch_rate: float = 7.2
    max_upward_velocity: float = 3.0
    phase_duration: float = 2.0
    flip_angle_rate_clip: float = 20.0
    flip_completion_angle: float = 5.50
    flip_success_angle: float = 5.80
    recovery_upright_cos: float = 0.90
    recovery_min_height: float = 0.24
    recovery_max_pitch_rate: float = 2.0
    recovery_pose_error: float = 0.30
    recovery_contact_force: float = 5.0
    takeoff_confirm_steps: int = 2
    takeoff_min_height: float = 0.25
    head_contact_success_force: float = 1.0
    use_peak_contact_forces: bool = False
    body_contact_force: float = 100.0
    max_body_contact_penalty: float = 5.0
    min_head_center_height: float = 0.08
    head_contact_force: float = 40.0
    landing_force_threshold: float = 200.0
    max_landing_impact_penalty: float = 5.0
    # 1000 PPO iterations of discovery, a 3000-iteration ramp, then 1000
    # iterations under the complete safety envelope (24 steps/update).
    safety_curriculum_start: float = 0.0
    safety_curriculum_warmup_steps: int = 24000
    safety_curriculum_ramp_steps: int = 72000

    scales: dict[str, float] = {
        "termination": -1000.0,
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
        "dof_vel_limits": -20.0,
        "rotation_progress": 50.0,
        "flip_completion": 500.0,
        "flip_success": 500.0,
        "recovery_upright": 5.0,
        "recovery_height": 3.0,
        "recovery_default_pose": 2.0,
        "recovery_still": 2.0,
        "recovery_feet_contact": 2.0,
        "rear_leg_action_rate": -0.03,
        "rear_leg_symmetry": -0.5,
        "recovery_dof_velocity": -2.0,
        "head_clearance": -5.0,
        "undesired_body_contact": -10.0,
        "dof_pos_limits": -20.0,
        "joint_target_limits": -10.0,
        "head_contact": -20.0,
        "landing_impact": -10.0,
    }


@configclass
class Go2BackflipEnvCfg(DirectRLEnvCfg):
    """Isaac Lab equivalent of ``Go2BackflipCfg`` from the Isaac Gym project."""

    seed = 1
    decimation = 4
    episode_length_s = 3.0
    is_finite_horizon = False
    action_space = 12
    observation_space = 60
    state_space = 165

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(GO2_URDF),
            usd_dir="/tmp/uw_backflip_isaaclab/go2_usd",
            fix_base=False,
            link_density=0.001,
            merge_fixed_joints=True,
            make_instanceable=True,
            # Go2BackflipCfg.asset.self_collisions = 0 in Isaac Gym means
            # ``disable_self_collisions=False``.  Keep self contacts enabled
            # for the discovery task instead of inheriting the usual Go2
            # locomotion setting.
            self_collision=True,
            replace_cylinders_with_capsules=True,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.32),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={
                "FR_hip_joint": 0.0,
                "FR_thigh_joint": 0.8,
                "FR_calf_joint": -1.5,
                "FL_hip_joint": 0.0,
                "FL_thigh_joint": 0.8,
                "FL_calf_joint": -1.5,
                "RR_hip_joint": 0.0,
                "RR_thigh_joint": 1.0,
                "RR_calf_joint": -1.5,
                "RL_hip_joint": 0.0,
                "RL_thigh_joint": 1.0,
                "RL_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=1.0e6,
                effort_limit_sim=1.0e6,
                velocity_limit=1000.0,
                velocity_limit_sim=1000.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
            )
        },
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.005,
        history_length=4,
        track_air_time=False,
    )

    control: ControlCfg = ControlCfg()
    domain_rand: DomainRandomizationCfg = DomainRandomizationCfg()
    noise: NoiseCfg = NoiseCfg()
    obs_scales: ObservationScaleCfg = ObservationScaleCfg()
    rewards: RewardCfg = RewardCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=(-2.5, -2.5, 1.8),
        lookat=(0.0, 0.0, 0.35),
        origin_type="env",
        env_index=0,
    )


@configclass
class Go2BackflipPlayEnvCfg(Go2BackflipEnvCfg):
    """Nominal deterministic replay of the fine-tuned Gym task."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True
    )
    noise: NoiseCfg = NoiseCfg(enabled=False)
    control: ControlCfg = ControlCfg(
        fixed_action_delay_steps=1,
        fixed_observation_delay_steps=0,
    )
    rewards: RewardCfg = RewardCfg(
        safety_curriculum_start=1.0,
        safety_curriculum_warmup_steps=0,
        safety_curriculum_ramp_steps=1,
    )
    domain_rand: DomainRandomizationCfg = DomainRandomizationCfg(
        friction_range=(1.0, 1.0),
        restitution_range=(0.0, 0.0),
        contact_offset_range=(0.01, 0.01),
        rest_offset_range=(0.0, 0.0),
        added_base_mass_range=(0.0, 0.0),
        added_base_com_range=(0.0, 0.0),
        limb_mass_scale_range=(1.0, 1.0),
        limb_inertia_jitter_range=(1.0, 1.0),
        torque_scale_range=(1.0, 1.0),
        motor_velocity_scale_range=(1.0, 1.0),
        kp_scale_range=(1.0, 1.0),
        kd_scale_range=(1.0, 1.0),
        motor_offset_range=(0.0, 0.0),
    )
