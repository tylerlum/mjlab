"""Manager-based SimToolReal environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_scene_to_default, time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.simtoolreal import mdp
from mjlab.tasks.simtoolreal.assets import (
  get_goal_cfg,
  get_iiwa_sharpa_cfg,
  get_object_cfg,
  get_table_cfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


def make_simtoolreal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the SimToolReal KUKA+Sharpa training environment.

  This is the manager-based MJLab/Warp port of the IsaacGym MDP, using the source
  URDF assets directly instead of the browser-demo XML. The object sampler uses
  separate per-world box geoms for the handle and head; cylinder handles from the
  source distribution are approximated by cuboidal handles.
  """
  simtoolreal_obs = ObservationTermCfg(
    func=mdp.simtoolreal_observation,
    params={
      "use_object_state_delay_noise": not play,
      "object_state_delay_max": 10,
      "object_state_xyz_noise_std": 0.0 if play else 0.01,
      "object_state_rotation_noise_degrees": 0.0 if play else 5.0,
      "object_scale_noise_multiplier_range": (1.0, 1.0),
      "joint_velocity_obs_noise_std": 0.0 if play else 0.1,
    },
    clip=(-10.0, 10.0),
    delay_min_lag=0,
    delay_max_lag=0 if play else 3,
  )
  observations = {
    "actor": ObservationGroupCfg(
      {"simtoolreal": simtoolreal_obs},
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg({"simtoolreal": simtoolreal_obs}),
  }

  object_geom_cfg = SceneEntityCfg(
    "object", geom_names=("object_handle_geom", "object_head_geom")
  )
  goal_geom_cfg = SceneEntityCfg(
    "goal", geom_names=("goal_handle_geom", "goal_head_geom")
  )
  events = {
    "reset_scene_to_default": EventTermCfg(
      func=reset_scene_to_default,
      mode="reset",
    ),
    "randomize_object_size": EventTermCfg(
      func=mdp.randomize_handle_head_equivalent_size,
      mode="reset",
      params={"asset_cfg": object_geom_cfg, "goal_asset_cfg": goal_geom_cfg},
    ),
    "reset_object": EventTermCfg(
      func=mdp.reset_object_uniform,
      mode="reset",
      params={
        "x_range": (0.0, 0.0),
        "y_range": (0.05, 0.05),
        "table_reset_z": 0.38,
        "table_reset_z_range": 0.0 if play else 0.01,
        "table_object_z_offset": 0.25,
        "reset_position_noise_x": 0.0 if play else 0.1,
        "reset_position_noise_y": 0.0 if play else 0.1,
        "reset_position_noise_z": 0.0 if play else 0.02,
        "randomize_object_rotation": not play,
        "object_scale_noise_multiplier_range": (1.0, 1.0),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_robot_joints_simtoolreal,
      mode="reset",
      params={
        "arm_pos_noise": 0.0 if play else 0.1,
        "finger_pos_noise": 0.0 if play else 0.1,
        "vel_noise": 0.0 if play else 0.5,
      },
    ),
    "reset_goal": EventTermCfg(
      func=mdp.reset_goal_uniform,
      mode="reset",
    ),
    "reset_task_state": EventTermCfg(
      func=mdp.reset_simtoolreal_state,
      mode="reset",
    ),
    "cache_prev_targets": EventTermCfg(
      func=mdp.cache_prev_targets,
      mode="step",
    ),
    "random_object_perturbations": EventTermCfg(
      func=mdp.apply_random_object_perturbations,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("object",)),
        "force_scale": 0.0 if play else 20.0,
        "torque_scale": 0.0 if play else 2.0,
        "force_decay": 0.0,
        "torque_decay": 0.0,
        "force_decay_interval": 0.08,
        "torque_decay_interval": 0.08,
        "lin_vel_impulse_scale": 0.0,
        "ang_vel_impulse_scale": 0.0,
      },
    ),
    "reset_successful_goals": EventTermCfg(
      func=mdp.reset_successful_goals,
      mode="step",
    ),
  }

  rewards = {
    "fingertip_delta": RewardTermCfg(func=mdp.fingertip_delta_reward, weight=50.0),
    "lift": RewardTermCfg(func=mdp.lifting_reward, weight=20.0),
    "lift_bonus": RewardTermCfg(func=mdp.lifting_bonus_reward, weight=1.0),
    "keypoint_delta": RewardTermCfg(func=mdp.keypoint_delta_reward, weight=200.0),
    "success": RewardTermCfg(func=mdp.success_bonus, weight=1.0),
    "kuka_action_penalty": RewardTermCfg(func=mdp.kuka_action_penalty, weight=1.0),
    "hand_action_penalty": RewardTermCfg(func=mdp.hand_action_penalty, weight=1.0),
    "object_velocity": RewardTermCfg(func=mdp.object_velocity_penalty, weight=0.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "object_fell": TerminationTermCfg(func=mdp.object_fell),
    "hand_far_from_object": TerminationTermCfg(func=mdp.hand_far_from_object),
    "max_successes": TerminationTermCfg(
      func=mdp.max_consecutive_successes_reached,
      params={"max_consecutive_successes": 50},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": get_iiwa_sharpa_cfg(),
        "object": get_object_cfg(),
        "goal": get_goal_cfg(),
        "table": get_table_cfg(),
      },
      num_envs=1 if play else 1024,
      env_spacing=2.0,
      extent=2.0,
    ),
    observations=observations,
    actions={
      "joint_pos": mdp.SimToolRealJointPositionActionCfg(
        entity_name="robot",
        use_action_delay=not play,
        action_delay_max=3,
      ),
    },
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="iiwa14_link_7",
      distance=1.3,
      elevation=-25.0,
      azimuth=135.0,
      width=1280,
      height=720,
    ),
    sim=SimulationCfg(
      nconmax=96,
      njmax=512,
      mujoco=MujocoCfg(
        timestep=1.0 / 120.0,
        integrator="implicitfast",
        cone="elliptic",
        impratio=10.0,
        iterations=20,
        ls_iterations=20,
      ),
    ),
    decimation=2,
    episode_length_s=1.0e9 if play else 10.0,
    scale_rewards_by_dt=False,
  )
