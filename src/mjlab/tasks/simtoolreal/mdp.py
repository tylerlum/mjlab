"""Manager-based MDP pieces for the SimToolReal MJLab port."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.simtoolreal.assets import JOINT_NAMES
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

N_ACT = 29
N_OBS = 140
N_STATE = 162
OBJECT_BASE_SIZE = 0.04
OBJECT_KEYPOINT_SIGNS = torch.tensor(
  [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]]
)
FIXED_SIZE = torch.tensor([0.141, 0.03025, 0.0271])
PALM_OFFSET = torch.tensor([0.0, -0.02, 0.16])
FINGERTIP_OFFSET = torch.tensor([0.02, 0.002, 0.0])
FINGERTIP_BODIES = (
  "left_index_DP",
  "left_middle_DP",
  "left_ring_DP",
  "left_thumb_DP",
  "left_pinky_DP",
)
OBJECT_GEOM_CFG = SceneEntityCfg(
  "object", geom_names=("object_handle_geom", "object_head_geom")
)
OBJECT_HANDLE_GEOM_CFG = SceneEntityCfg("object", geom_names=("object_handle_geom",))
OBJECT_HEAD_GEOM_CFG = SceneEntityCfg("object", geom_names=("object_head_geom",))
OBJECT_BODY_CFG = SceneEntityCfg("object", body_names=("object",))
TABLE_BODY_CFG = SceneEntityCfg("table", body_names=("table_object",))
GOAL_GEOM_CFG = SceneEntityCfg(
  "goal", geom_names=("goal_handle_geom", "goal_head_geom")
)
GOAL_HANDLE_GEOM_CFG = SceneEntityCfg("goal", geom_names=("goal_handle_geom",))
GOAL_HEAD_GEOM_CFG = SceneEntityCfg("goal", geom_names=("goal_head_geom",))
ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=JOINT_NAMES)

HANDLE_HEAD_DISTRIBUTIONS = (
  ((0.15, 0.02, 0.015), (0.30, 0.04, 0.03), (0.02, 0.05, 0.02), (0.06, 0.12, 0.06)),
  ((0.15, 0.015), (0.30, 0.03), (0.02, 0.05, 0.02), (0.06, 0.12, 0.06)),
  ((0.07, 0.025, 0.025), (0.12, 0.04, 0.04), (0.07, 0.01, 0.01), (0.15, 0.015, 0.015)),
  ((0.07, 0.025), (0.12, 0.04), (0.07, 0.01, 0.01), (0.15, 0.015, 0.015)),
  ((0.075, 0.015), (0.15, 0.03), (0.01, 0.005, 0.005), (0.03, 0.01, 0.01)),
  ((0.10, 0.0125, 0.006), (0.20, 0.025, 0.025), (0.05, 0.03, 0.01), (0.15, 0.07, 0.03)),
  ((0.10, 0.0125), (0.20, 0.025), (0.05, 0.03, 0.01), (0.15, 0.07, 0.03)),
  ((0.07, 0.02, 0.02), (0.15, 0.07, 0.07), None, None),
  ((0.05, 0.01, 0.01), (0.20, 0.04, 0.03), (0.05, 0.03, 0.03), (0.12, 0.05, 0.08)),
  ((0.05, 0.01), (0.20, 0.03), (0.05, 0.03, 0.03), (0.12, 0.05, 0.08)),
  ((0.05, 0.01, 0.01), (0.20, 0.04, 0.03), (0.05, 0.05, 0.02), (0.12, 0.12, 0.04)),
  ((0.05, 0.01), (0.20, 0.03), (0.05, 0.05, 0.02), (0.12, 0.12, 0.04)),
)
LOW_DENSITY_RANGE = (300.0, 600.0)
HIGH_DENSITY_RANGE = (800.0, 2000.0)
HANDLE_HEAD_DENSITY_RANGES = (
  (*LOW_DENSITY_RANGE, *HIGH_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *HIGH_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *HIGH_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *HIGH_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *LOW_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *LOW_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *LOW_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, 0.0, 0.0),
  (*LOW_DENSITY_RANGE, *LOW_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *LOW_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *LOW_DENSITY_RANGE),
  (*LOW_DENSITY_RANGE, *LOW_DENSITY_RANGE),
)

Q_LOWER = torch.tensor(
  [
    -2.9671,
    -2.0944,
    -2.9671,
    -2.0944,
    -2.9671,
    -2.0944,
    -3.0543,
    -0.1745,
    -0.3491,
    -0.5236,
    -0.3491,
    0.0,
    -0.1745,
    -0.0349,
    0.0,
    0.0,
    -0.1745,
    -0.0349,
    0.0,
    0.0,
    -0.1745,
    -0.0349,
    0.0,
    0.0,
    0.0,
    -0.1745,
    -0.0349,
    0.0,
    0.0,
  ]
)
Q_UPPER = torch.tensor(
  [
    2.9671,
    2.0944,
    2.9671,
    2.0944,
    2.9671,
    2.0944,
    3.0543,
    1.9199,
    0.1309,
    1.3963,
    0.3491,
    1.7453,
    1.5708,
    0.0349,
    1.7453,
    1.3963,
    1.5708,
    0.0349,
    1.7453,
    1.3963,
    1.5708,
    0.0349,
    1.7453,
    1.3963,
    0.2618,
    1.5708,
    0.0349,
    1.7453,
    1.3963,
  ]
)


def _state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  if not hasattr(env, "_simtoolreal_state"):
    env._simtoolreal_state = {  # type: ignore[attr-defined]
      "object_scales": torch.tensor(
        [5.0, 0.75, 0.75], device=env.device, dtype=torch.float32
      ).repeat(env.num_envs, 1),
      "handle_lengths": torch.tensor(
        [0.15, 0.025, 0.025], device=env.device, dtype=torch.float32
      ).repeat(env.num_envs, 1),
      "head_lengths": torch.tensor(
        [0.05, 0.05, 0.03], device=env.device, dtype=torch.float32
      ).repeat(env.num_envs, 1),
      "closest_keypoint_max_dist": torch.full(
        (env.num_envs,), float("inf"), device=env.device
      ),
      "closest_keypoint_max_dist_fixed_size": torch.full(
        (env.num_envs,), float("inf"), device=env.device
      ),
      "closest_fingertip_dist": torch.full(
        (env.num_envs, len(FINGERTIP_BODIES)), float("inf"), device=env.device
      ),
      "furthest_hand_dist": torch.zeros(env.num_envs, device=env.device),
      "near_goal_steps": torch.zeros(env.num_envs, device=env.device),
      "reset_goal_buf": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
      "successes": torch.zeros(env.num_envs, device=env.device),
      "prev_episode_successes": torch.zeros(env.num_envs, device=env.device),
      "lifted_object": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
      "just_lifted_object": torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
      ),
      "initial_object_z": torch.full((env.num_envs,), 0.545, device=env.device),
      "object_masses": torch.full((env.num_envs,), 0.08, device=env.device),
      "table_reset_z": torch.full((env.num_envs,), 0.38, device=env.device),
      "object_scale_noise_multiplier": torch.ones(
        env.num_envs, 3, device=env.device
      ),
      "rb_forces": torch.zeros(env.num_envs, 1, 3, device=env.device),
      "rb_torques": torch.zeros(env.num_envs, 1, 3, device=env.device),
      "random_force_prob": _log_uniform(env, 0.001, 0.1, (env.num_envs,)),
      "random_torque_prob": _log_uniform(env, 0.001, 0.1, (env.num_envs,)),
      "random_lin_vel_impulse_prob": _log_uniform(env, 0.001, 0.1, (env.num_envs,)),
      "random_ang_vel_impulse_prob": _log_uniform(env, 0.001, 0.1, (env.num_envs,)),
      "object_state_queue": None,
    }
  return env._simtoolreal_state  # type: ignore[attr-defined]


def _log_uniform(
  env: ManagerBasedRlEnv,
  min_value: float,
  max_value: float,
  shape: tuple[int, ...],
) -> torch.Tensor:
  return torch.exp(
    torch.empty(shape, device=env.device).uniform_(
      math.log(min_value), math.log(max_value)
    )
  )


def _update_queue(queue: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
  queue[:, 1:] = queue[:, :-1].clone()
  queue[:, 0] = current
  return queue


def _random_quat(env: ManagerBasedRlEnv, n: int) -> torch.Tensor:
  uvw = torch.rand((n, 3), device=env.device)
  qx = torch.sqrt(1.0 - uvw[:, 0]) * torch.sin(2.0 * math.pi * uvw[:, 1])
  qy = torch.sqrt(1.0 - uvw[:, 0]) * torch.cos(2.0 * math.pi * uvw[:, 1])
  qz = torch.sqrt(uvw[:, 0]) * torch.sin(2.0 * math.pi * uvw[:, 2])
  qw = torch.sqrt(uvw[:, 0]) * torch.cos(2.0 * math.pi * uvw[:, 2])
  return torch.stack([qw, qx, qy, qz], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  w1, x1, y1, z1 = q1.unbind(dim=-1)
  w2, x2, y2, z2 = q2.unbind(dim=-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
  return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _random_small_quat(env: ManagerBasedRlEnv, n: int, max_angle: float) -> torch.Tensor:
  axis = torch.randn((n, 3), device=env.device)
  axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-8)
  angle = torch.empty(n, device=env.device).uniform_(-max_angle, max_angle)
  half_angle = 0.5 * angle
  return torch.cat(
    [torch.cos(half_angle)[:, None], axis * torch.sin(half_angle)[:, None]],
    dim=-1,
  )


def _q_limits(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  return Q_LOWER.to(env.device), Q_UPPER.to(env.device)


def _robot(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["robot"]


def _object(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["object"]


def _goal(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["goal"]


def _table(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["table"]


@dataclass(kw_only=True)
class SimToolRealJointPositionActionCfg(ActionTermCfg):
  hand_moving_average: float = 0.1
  arm_moving_average: float = 0.1
  arm_speed_scale: float = 1.5
  use_action_delay: bool = True
  action_delay_max: int = 3

  def build(self, env: ManagerBasedRlEnv) -> "SimToolRealJointPositionAction":
    return SimToolRealJointPositionAction(self, env)


class SimToolRealJointPositionAction(ActionTerm):
  cfg: SimToolRealJointPositionActionCfg

  def __init__(self, cfg: SimToolRealJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    joint_ids, joint_names = self._entity.find_joints(JOINT_NAMES, preserve_order=True)
    if tuple(joint_names) != JOINT_NAMES:
      raise RuntimeError(f"Unexpected SimToolReal joint order: {joint_names}")
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros(self.num_envs, N_ACT, device=self.device)
    self._delayed_actions = torch.zeros_like(self._raw_actions)
    self._action_queue = torch.zeros(
      self.num_envs, max(1, cfg.action_delay_max), N_ACT, device=self.device
    )
    self.prev_targets = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    self.targets = self.prev_targets.clone()

  @property
  def action_dim(self) -> int:
    return N_ACT

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def applied_action(self) -> torch.Tensor:
    return self._delayed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = torch.clamp(actions.to(self.device), -1.0, 1.0)
    _update_queue(self._action_queue, self._raw_actions)
    if self.cfg.use_action_delay and self.cfg.action_delay_max > 1:
      delay_ids = torch.randint(
        0, self.cfg.action_delay_max, (self.num_envs,), device=self.device
      )
      actions_for_targets = self._action_queue[
        torch.arange(self.num_envs, device=self.device), delay_ids
      ]
    else:
      actions_for_targets = self._raw_actions
    self._delayed_actions[:] = actions_for_targets
    q_lower, q_upper = _q_limits(self._env)
    targets = self.prev_targets.clone()

    hand_targets = 0.5 * (actions_for_targets[:, 7:] + 1.0) * (
      q_upper[7:] - q_lower[7:]
    ) + q_lower[7:]
    targets[:, 7:] = (
      self.cfg.hand_moving_average * hand_targets
      + (1.0 - self.cfg.hand_moving_average) * self.prev_targets[:, 7:]
    )
    targets[:, 7:] = torch.clamp(targets[:, 7:], q_lower[7:], q_upper[7:])

    arm_targets = (
      self.prev_targets[:, :7]
      + self.cfg.arm_speed_scale * self._env.step_dt * actions_for_targets[:, :7]
    )
    arm_targets = torch.clamp(arm_targets, q_lower[:7], q_upper[:7])
    targets[:, :7] = (
      self.cfg.arm_moving_average * arm_targets
      + (1.0 - self.cfg.arm_moving_average) * self.prev_targets[:, :7]
    )
    self.targets[:] = targets

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self.targets, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._delayed_actions[env_ids] = 0.0
    self._action_queue[env_ids] = 0.0
    joint_pos = self._entity.data.joint_pos[:, self._joint_ids]
    self.prev_targets[env_ids] = joint_pos[env_ids]
    self.targets[env_ids] = joint_pos[env_ids]


def cache_prev_targets(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  del env_ids
  term = env.action_manager.get_term("joint_pos")
  if isinstance(term, SimToolRealJointPositionAction):
    term.prev_targets[:] = term.targets


def reset_simtoolreal_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  state = _state(env)
  object_entity = _object(env)
  object_pose = object_entity.data.root_link_pose_w
  state["initial_object_z"][env_ids] = object_pose[env_ids, 2]
  state["closest_keypoint_max_dist"][env_ids] = float("inf")
  state["closest_keypoint_max_dist_fixed_size"][env_ids] = float("inf")
  state["closest_fingertip_dist"][env_ids] = float("inf")
  state["furthest_hand_dist"][env_ids] = 0.0
  state["near_goal_steps"][env_ids] = 0.0
  state["reset_goal_buf"][env_ids] = False
  state["prev_episode_successes"][env_ids] = state["successes"][env_ids]
  state["successes"][env_ids] = 0.0
  state["lifted_object"][env_ids] = False
  state["just_lifted_object"][env_ids] = False
  state["rb_forces"][env_ids] = 0.0
  state["rb_torques"][env_ids] = 0.0
  state["random_force_prob"][env_ids] = _log_uniform(env, 0.001, 0.1, (len(env_ids),))
  state["random_torque_prob"][env_ids] = _log_uniform(env, 0.001, 0.1, (len(env_ids),))
  state["random_lin_vel_impulse_prob"][env_ids] = _log_uniform(
    env, 0.001, 0.1, (len(env_ids),)
  )
  state["random_ang_vel_impulse_prob"][env_ids] = _log_uniform(
    env, 0.001, 0.1, (len(env_ids),)
  )
  if state["object_state_queue"] is not None:
    pose_vel = torch.cat(
      [
        object_pose[env_ids, :7],
        object_entity.data.root_link_lin_vel_w[env_ids],
        object_entity.data.root_link_ang_vel_w[env_ids],
      ],
      dim=-1,
    )
    state["object_state_queue"][env_ids] = pose_vel[:, None, :]


@requires_model_fields("body_pos", recompute=RecomputeLevel.set_const_0)
def reset_object_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  x_range: tuple[float, float] = (-0.04, 0.04),
  y_range: tuple[float, float] = (0.02, 0.08),
  z: float | None = None,
  table_reset_z: float = 0.38,
  table_reset_z_range: float = 0.01,
  table_object_z_offset: float = 0.25,
  reset_position_noise_x: float = 0.1,
  reset_position_noise_y: float = 0.1,
  reset_position_noise_z: float = 0.02,
  randomize_object_rotation: bool = True,
  object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0),
  table_cfg: SceneEntityCfg = TABLE_BODY_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  obj = _object(env)
  table = _table(env)
  state = _state(env)
  table_z = table_reset_z + torch.empty(len(env_ids), device=env.device).uniform_(
    -table_reset_z_range, table_reset_z_range
  )
  table_body_ids = table.indexing.body_ids[table_cfg.body_ids]
  env.sim.model.body_pos[env_ids[:, None], table_body_ids, 2] = table_z[:, None]
  state["table_reset_z"][env_ids] = table_z

  pos = torch.zeros((len(env_ids), 3), device=env.device)
  pos[:, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*x_range)
  pos[:, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*y_range)
  if z is None:
    pos[:, 2] = table_z + table_object_z_offset
  else:
    pos[:, 2] = z
  rand = torch.empty((len(env_ids), 3), device=env.device).uniform_(-1.0, 1.0)
  pos[:, 0] += reset_position_noise_x * rand[:, 0]
  pos[:, 1] += reset_position_noise_y * rand[:, 1]
  pos[:, 2] += reset_position_noise_z * rand[:, 2]
  pos += env.scene.env_origins[env_ids]
  if randomize_object_rotation:
    quat = _random_quat(env, len(env_ids))
  else:
    quat = torch.zeros((len(env_ids), 4), device=env.device)
    quat[:, 0] = 1.0
  obj.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  obj.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)
  noise_min, noise_max = object_scale_noise_multiplier_range
  state["object_scale_noise_multiplier"][env_ids] = torch.empty(
    (len(env_ids), 3), device=env.device
  ).uniform_(noise_min, noise_max)


def reset_robot_joints_simtoolreal(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  arm_pos_noise: float = 0.1,
  finger_pos_noise: float = 0.1,
  vel_noise: float = 0.5,
  asset_cfg: SceneEntityCfg = ROBOT_JOINT_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  robot = _robot(env)
  q_lower, q_upper = _q_limits(env)
  joint_ids = robot.indexing.joint_ids[asset_cfg.joint_ids]
  default = robot.data.default_joint_pos[env_ids][:, joint_ids].clone()
  delta_min = q_lower - default
  delta_max = q_upper - default
  rand = torch.rand((len(env_ids), N_ACT), device=env.device)
  rand_delta = delta_min + (delta_max - delta_min) * rand
  noise_coeff = torch.empty(N_ACT, device=env.device)
  noise_coeff[:7] = arm_pos_noise
  noise_coeff[7:] = finger_pos_noise
  joint_pos = torch.clamp(default + noise_coeff * rand_delta, q_lower, q_upper)
  joint_vel = torch.empty((len(env_ids), N_ACT), device=env.device).uniform_(
    -vel_noise, vel_noise
  )
  robot.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    joint_ids=joint_ids,
    env_ids=env_ids,
  )
  term = env.action_manager.get_term("joint_pos")
  if isinstance(term, SimToolRealJointPositionAction):
    term.prev_targets[env_ids] = joint_pos
    term.targets[env_ids] = joint_pos


def reset_goal_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  x_range: tuple[float, float] = (-0.35, 0.35),
  y_range: tuple[float, float] = (-0.2, 0.2),
  z_range: tuple[float, float] = (0.6, 0.95),
  is_first_goal: bool = True,
  goal_sampling_type: str = "delta",
  delta_goal_distance: float = 0.1,
  delta_rotation_degrees: float = 90.0,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  goal = _goal(env)
  if not is_first_goal and goal_sampling_type == "delta":
    pos = goal.data.root_link_pos_w[env_ids].clone() - env.scene.env_origins[env_ids]
    delta = torch.empty((len(env_ids), 3), device=env.device).uniform_(
      -delta_goal_distance, delta_goal_distance
    )
    pos = pos + delta
    mins = torch.tensor([x_range[0], y_range[0], z_range[0]], device=env.device)
    maxs = torch.tensor([x_range[1], y_range[1], z_range[1]], device=env.device)
    pos = torch.clamp(pos, mins, maxs)
    delta_quat = _random_small_quat(
      env, len(env_ids), math.radians(delta_rotation_degrees)
    )
    quat = _quat_normalize(_quat_mul(goal.data.root_link_quat_w[env_ids], delta_quat))
  else:
    pos = torch.zeros((len(env_ids), 3), device=env.device)
    pos[:, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*x_range)
    pos[:, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*y_range)
    pos[:, 2] = torch.empty(len(env_ids), device=env.device).uniform_(*z_range)
    roll = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
    pitch = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
    quat = quat_from_euler_xyz(roll, pitch, yaw)
  min_z = (
    _object(env).data.root_link_pos_w[env_ids, 2]
    - env.scene.env_origins[env_ids, 2]
    - 0.05
    + 0.15
  )
  if is_first_goal or goal_sampling_type not in {"delta", "coin_flip"}:
    pos[:, 2] = torch.maximum(pos[:, 2], min_z)
  pos += env.scene.env_origins[env_ids]
  goal.write_mocap_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)


def sample_handle_head_lengths(
  env: ManagerBasedRlEnv, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
  handle_lengths, head_lengths, _, _ = sample_handle_head_properties(env, n)
  return handle_lengths, head_lengths


def sample_handle_head_properties(
  env: ManagerBasedRlEnv, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  handle_lengths = torch.zeros((n, 3), device=env.device)
  head_lengths = torch.zeros((n, 3), device=env.device)
  handle_densities = torch.zeros(n, device=env.device)
  head_densities = torch.zeros(n, device=env.device)
  choices = torch.randint(0, len(HANDLE_HEAD_DISTRIBUTIONS), (n,), device=env.device)
  for i, (h_min, h_max, head_min, head_max) in enumerate(HANDLE_HEAD_DISTRIBUTIONS):
    mask = choices == i
    if not mask.any():
      continue
    count = int(mask.sum().item())
    h_min_t = torch.tensor(h_min, device=env.device, dtype=torch.float32)
    h_max_t = torch.tensor(h_max, device=env.device, dtype=torch.float32)
    handle = torch.rand((count, len(h_min)), device=env.device) * (h_max_t - h_min_t) + h_min_t
    if handle.shape[1] == 2:
      handle = torch.stack([handle[:, 0], handle[:, 1], handle[:, 1]], dim=-1)
    handle_lengths[mask] = handle
    h_density_min, h_density_max, head_density_min, head_density_max = (
      HANDLE_HEAD_DENSITY_RANGES[i]
    )
    handle_densities[mask] = torch.empty(count, device=env.device).uniform_(
      h_density_min, h_density_max
    )
    if head_min is None or head_max is None:
      head_lengths[mask] = 0.0
      head_densities[mask] = 0.0
    else:
      head_min_t = torch.tensor(head_min, device=env.device, dtype=torch.float32)
      head_max_t = torch.tensor(head_max, device=env.device, dtype=torch.float32)
      head_lengths[mask] = (
        torch.rand((count, 3), device=env.device) * (head_max_t - head_min_t)
        + head_min_t
      )
      head_densities[mask] = torch.empty(count, device=env.device).uniform_(
        head_density_min, head_density_max
      )
  return handle_lengths, head_lengths, handle_densities, head_densities


def sample_handle_head_equivalent_lengths(env: ManagerBasedRlEnv, n: int) -> torch.Tensor:
  handle_lengths, head_lengths = sample_handle_head_lengths(env, n)
  return torch.stack(
    [
      handle_lengths[:, 0] + head_lengths[:, 0],
      torch.maximum(handle_lengths[:, 1], head_lengths[:, 1]),
      torch.maximum(handle_lengths[:, 2], head_lengths[:, 2]),
    ],
    dim=-1,
  )


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def randomize_simple_cuboid_size(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  min_lengths: tuple[float, float, float] = (0.10, 0.03, 0.03),
  max_lengths: tuple[float, float, float] = (0.25, 0.07, 0.07),
  asset_cfg: SceneEntityCfg = OBJECT_GEOM_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  state = _state(env)
  dr.geom_size(
    env,
    env_ids=env_ids,
    ranges={
      0: (0.5 * min_lengths[0], 0.5 * max_lengths[0]),
      1: (0.5 * min_lengths[1], 0.5 * max_lengths[1]),
      2: (0.5 * min_lengths[2], 0.5 * max_lengths[2]),
    },
    operation="abs",
    axes=[0, 1, 2],
    asset_cfg=asset_cfg,
  )
  obj = _object(env)
  geom_ids = obj.indexing.geom_ids[asset_cfg.geom_ids]
  lengths = 2.0 * env.sim.model.geom_size[env_ids[:, None], geom_ids, :3]
  lengths = lengths.max(dim=1).values
  state["object_scales"][env_ids] = lengths / OBJECT_BASE_SIZE


@requires_model_fields(
  "geom_size",
  "geom_pos",
  "geom_rbound",
  "geom_aabb",
  "body_mass",
  "body_ipos",
  "body_inertia",
  recompute=RecomputeLevel.set_const,
)
def randomize_handle_head_equivalent_size(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  density: float | None = None,
  asset_cfg: SceneEntityCfg = OBJECT_GEOM_CFG,
  goal_asset_cfg: SceneEntityCfg = GOAL_GEOM_CFG,
  body_cfg: SceneEntityCfg = OBJECT_BODY_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  handle_lengths, head_lengths, handle_densities, head_densities = (
    sample_handle_head_properties(env, len(env_ids))
  )
  if density is not None:
    handle_densities[:] = density
    head_densities = torch.where(
      head_lengths[:, 0] > 0.0,
      torch.full_like(head_densities, density),
      torch.zeros_like(head_densities),
    )
  lengths = torch.stack(
    [
      handle_lengths[:, 0] + head_lengths[:, 0],
      torch.maximum(handle_lengths[:, 1], head_lengths[:, 1]),
      torch.maximum(handle_lengths[:, 2], head_lengths[:, 2]),
    ],
    dim=-1,
  )
  min_geom_size = torch.full_like(head_lengths, 1.0e-4)
  head_geom_lengths = torch.where(head_lengths > 0.0, head_lengths, min_geom_size)
  handle_centers = torch.zeros_like(handle_lengths)
  head_centers = torch.zeros_like(head_lengths)
  handle_centers[:, 0] = -0.5 * lengths[:, 0] + 0.5 * handle_lengths[:, 0]
  head_centers[:, 0] = 0.5 * lengths[:, 0] - 0.5 * head_geom_lengths[:, 0]

  obj = _object(env)
  goal = _goal(env)
  geom_ids = obj.indexing.geom_ids[asset_cfg.geom_ids]
  goal_geom_ids = goal.indexing.geom_ids[goal_asset_cfg.geom_ids]
  env_grid = env_ids[:, None]
  geom_lengths = torch.stack([handle_lengths, head_geom_lengths], dim=1)
  geom_centers = torch.stack([handle_centers, head_centers], dim=1)
  env.sim.model.geom_size[env_grid, geom_ids, :3] = 0.5 * geom_lengths
  env.sim.model.geom_pos[env_grid, geom_ids, :3] = geom_centers
  env.sim.model.geom_size[env_grid, goal_geom_ids, :3] = 0.5 * geom_lengths
  env.sim.model.geom_pos[env_grid, goal_geom_ids, :3] = geom_centers
  from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds

  _recompute_geom_bounds(env, env_ids=env_ids, asset_cfg=asset_cfg)
  _recompute_geom_bounds(env, env_ids=env_ids, asset_cfg=goal_asset_cfg)

  body_ids = obj.indexing.body_ids[body_cfg.body_ids]
  handle_mass = handle_densities * torch.prod(handle_lengths, dim=-1)
  head_mass = head_densities * torch.prod(head_lengths, dim=-1)
  mass = (handle_mass + head_mass).clamp_min(1.0e-6)
  com_x = (handle_mass * handle_centers[:, 0] + head_mass * head_centers[:, 0]) / mass
  handle_dx = handle_centers[:, 0] - com_x
  head_dx = head_centers[:, 0] - com_x
  handle_inertia = torch.stack(
    [
      handle_mass * (handle_lengths[:, 1] ** 2 + handle_lengths[:, 2] ** 2) / 12.0,
      handle_mass * (handle_lengths[:, 0] ** 2 + handle_lengths[:, 2] ** 2) / 12.0
      + handle_mass * handle_dx**2,
      handle_mass * (handle_lengths[:, 0] ** 2 + handle_lengths[:, 1] ** 2) / 12.0
      + handle_mass * handle_dx**2,
    ],
    dim=-1,
  )
  head_inertia = torch.stack(
    [
      head_mass * (head_lengths[:, 1] ** 2 + head_lengths[:, 2] ** 2) / 12.0,
      head_mass * (head_lengths[:, 0] ** 2 + head_lengths[:, 2] ** 2) / 12.0
      + head_mass * head_dx**2,
      head_mass * (head_lengths[:, 0] ** 2 + head_lengths[:, 1] ** 2) / 12.0
      + head_mass * head_dx**2,
    ],
    dim=-1,
  )
  inertia = (handle_inertia + head_inertia).clamp_min(1.0e-8)
  env.sim.model.body_mass[env_ids[:, None], body_ids] = mass[:, None]
  env.sim.model.body_ipos[env_ids[:, None], body_ids, :] = 0.0
  env.sim.model.body_ipos[env_ids[:, None], body_ids, 0] = com_x[:, None]
  env.sim.model.body_inertia[env_ids[:, None], body_ids] = inertia[:, None, :]
  state = _state(env)
  state["handle_lengths"][env_ids] = handle_lengths
  state["head_lengths"][env_ids] = head_lengths
  state["object_scales"][env_ids] = lengths / OBJECT_BASE_SIZE
  state["object_masses"][env_ids] = mass


def randomize_handle_head_size(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  density: float | None = None,
  asset_cfg: SceneEntityCfg = OBJECT_GEOM_CFG,
  goal_asset_cfg: SceneEntityCfg = GOAL_GEOM_CFG,
  body_cfg: SceneEntityCfg = OBJECT_BODY_CFG,
) -> None:
  randomize_handle_head_equivalent_size(
    env,
    env_ids,
    density=density,
    asset_cfg=asset_cfg,
    goal_asset_cfg=goal_asset_cfg,
    body_cfg=body_cfg,
  )


def _body_pose(
  entity: Entity, body_names: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
  local_ids, _ = entity.find_bodies(body_names, preserve_order=True)
  body_ids = entity.indexing.body_ids[local_ids]
  pos = entity.data.data.xpos[:, body_ids]
  quat = entity.data.data.xquat[:, body_ids]
  return pos, quat


def _body_velocity(entity: Entity, body_names: tuple[str, ...]) -> torch.Tensor:
  local_ids, _ = entity.find_bodies(body_names, preserve_order=True)
  body_ids = entity.indexing.body_ids[local_ids]
  return entity.data.body_link_vel_w[:, body_ids]


def _simtoolreal_kinematics(
  env: ManagerBasedRlEnv,
  use_object_state_delay_noise: bool = True,
  object_state_delay_max: int = 10,
  object_state_xyz_noise_std: float = 0.01,
  object_state_rotation_noise_degrees: float = 5.0,
  object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0),
  joint_velocity_obs_noise_std: float = 0.1,
  fixed_size_keypoint_reward: bool = True,
) -> dict[str, torch.Tensor]:
  robot = _robot(env)
  obj = _object(env)
  goal = _goal(env)
  state = _state(env)
  q_lower, q_upper = _q_limits(env)
  action = env.action_manager.get_term("joint_pos")
  prev_targets = (
    action.prev_targets
    if isinstance(action, SimToolRealJointPositionAction)
    else robot.data.default_joint_pos
  )

  palm_pos_raw, palm_quat = _body_pose(robot, ("iiwa14_link_7",))
  palm_pos_raw = palm_pos_raw[:, 0]
  palm_quat = palm_quat[:, 0]
  palm_offset = PALM_OFFSET.to(env.device).repeat(env.num_envs, 1)
  palm_pos = palm_pos_raw + quat_apply(palm_quat, palm_offset)

  fingertip_pos, fingertip_quat = _body_pose(robot, FINGERTIP_BODIES)
  fingertip_offset = FINGERTIP_OFFSET.to(env.device).repeat(env.num_envs, 1)
  fingertip_offset = fingertip_offset[:, None, :].repeat(1, len(FINGERTIP_BODIES), 1)
  fingertip_pos = fingertip_pos + quat_apply(fingertip_quat, fingertip_offset)
  fingertip_rel_palm = fingertip_pos - palm_pos[:, None, :]

  object_pos = obj.data.root_link_pos_w
  object_quat = obj.data.root_link_quat_w
  object_lin_vel = obj.data.root_link_lin_vel_w
  object_ang_vel = obj.data.root_link_ang_vel_w
  if use_object_state_delay_noise and object_state_delay_max > 1:
    pose_vel = torch.cat(
      [object_pos, object_quat, object_lin_vel, object_ang_vel], dim=-1
    )
    queue = state.get("object_state_queue")
    if queue is None or queue.shape != (env.num_envs, object_state_delay_max, 13):
      queue = pose_vel[:, None, :].repeat(1, object_state_delay_max, 1)
      state["object_state_queue"] = queue
    _update_queue(queue, pose_vel)
    delay_ids = torch.randint(
      0, object_state_delay_max, (env.num_envs,), device=env.device
    )
    observed = queue[torch.arange(env.num_envs, device=env.device), delay_ids].clone()
    object_pos = observed[:, :3]
    object_quat = observed[:, 3:7]
    object_lin_vel = observed[:, 7:10]
    object_ang_vel = observed[:, 10:13]
    object_pos = object_pos + torch.randn_like(object_pos) * object_state_xyz_noise_std
    max_angle = math.radians(object_state_rotation_noise_degrees)
    object_quat = _quat_normalize(
      _quat_mul(_random_small_quat(env, env.num_envs, max_angle), object_quat)
    )
  goal_pos = goal.data.root_link_pos_w
  goal_quat = goal.data.root_link_quat_w

  signs = OBJECT_KEYPOINT_SIGNS.to(env.device)
  offsets = signs[None] * (OBJECT_BASE_SIZE * 1.5 * 0.5)
  scales = state["object_scales"]
  if use_object_state_delay_noise:
    del object_scale_noise_multiplier_range
    scales = scales * state["object_scale_noise_multiplier"]
  offsets = offsets * scales[:, None, :]
  object_keypoints = object_pos[:, None, :] + quat_apply(
    object_quat[:, None, :].repeat(1, 4, 1), offsets
  )
  goal_keypoints = goal_pos[:, None, :] + quat_apply(
    goal_quat[:, None, :].repeat(1, 4, 1), offsets
  )
  fixed_offsets = signs[None] * (FIXED_SIZE.to(env.device) * 1.5 * 0.5)
  fixed_offsets = fixed_offsets.repeat(env.num_envs, 1, 1)
  object_keypoints_fixed = object_pos[:, None, :] + quat_apply(
    object_quat[:, None, :].repeat(1, 4, 1), fixed_offsets
  )
  goal_keypoints_fixed = goal_pos[:, None, :] + quat_apply(
    goal_quat[:, None, :].repeat(1, 4, 1), fixed_offsets
  )
  keypoints_rel_goal = object_keypoints - goal_keypoints
  keypoints_rel_goal_fixed = object_keypoints_fixed - goal_keypoints_fixed
  keypoint_dist = torch.linalg.norm(keypoints_rel_goal, dim=-1)
  keypoint_max_dist = keypoint_dist.max(dim=-1).values
  keypoint_max_dist_fixed = torch.linalg.norm(keypoints_rel_goal_fixed, dim=-1).max(
    dim=-1
  ).values
  fingertip_dist = torch.linalg.norm(fingertip_pos - object_pos[:, None, :], dim=-1)

  joint_vel = robot.data.joint_vel[:, :N_ACT]
  if joint_velocity_obs_noise_std > 0.0:
    joint_vel = joint_vel + torch.randn_like(joint_vel) * joint_velocity_obs_noise_std

  return {
    "joint_pos_unscaled": (2.0 * robot.data.joint_pos[:, :N_ACT] - q_upper - q_lower)
    / (q_upper - q_lower),
    "joint_vel": joint_vel,
    "prev_targets": prev_targets,
    "palm_pos": palm_pos,
    "palm_quat": palm_quat,
    "object_quat": object_quat,
    "fingertip_rel_palm": fingertip_rel_palm.reshape(env.num_envs, -1),
    "keypoints_rel_palm": (object_keypoints - palm_pos[:, None, :]).reshape(
      env.num_envs, -1
    ),
    "keypoints_rel_goal": keypoints_rel_goal.reshape(env.num_envs, -1),
    "object_scales": scales,
    "keypoint_max_dist": keypoint_max_dist,
    "keypoint_max_dist_fixed": keypoint_max_dist_fixed
    if fixed_size_keypoint_reward
    else keypoint_max_dist,
    "fingertip_dist": fingertip_dist,
    "object_pos": object_pos,
    "object_lin_vel": object_lin_vel,
  "object_ang_vel": object_ang_vel,
  }


def simtoolreal_observation(
  env: ManagerBasedRlEnv,
  use_object_state_delay_noise: bool = True,
  object_state_delay_max: int = 10,
  object_state_xyz_noise_std: float = 0.01,
  object_state_rotation_noise_degrees: float = 5.0,
  object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0),
  joint_velocity_obs_noise_std: float = 0.1,
) -> torch.Tensor:
  kin = _simtoolreal_kinematics(
    env,
    use_object_state_delay_noise=use_object_state_delay_noise,
    object_state_delay_max=object_state_delay_max,
    object_state_xyz_noise_std=object_state_xyz_noise_std,
    object_state_rotation_noise_degrees=object_state_rotation_noise_degrees,
    object_scale_noise_multiplier_range=object_scale_noise_multiplier_range,
    joint_velocity_obs_noise_std=joint_velocity_obs_noise_std,
  )
  return torch.cat(
    [
      kin["joint_pos_unscaled"],
      kin["joint_vel"],
      kin["prev_targets"],
      kin["palm_pos"],
      kin["palm_quat"][:, [1, 2, 3, 0]],
      kin["object_quat"][:, [1, 2, 3, 0]],
      kin["fingertip_rel_palm"],
      kin["keypoints_rel_palm"],
      kin["keypoints_rel_goal"],
      kin["object_scales"],
    ],
    dim=-1,
  )


def simtoolreal_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Source-style asymmetric critic state for SimToolReal.

  This mirrors the IsaacGym ``stateList`` default: clean actor observation fields
  plus palm/object velocities, reward-progress bookkeeping, and success state.
  """
  kin = _simtoolreal_kinematics(
    env,
    use_object_state_delay_noise=False,
    joint_velocity_obs_noise_std=0.0,
  )
  state = _state(env)
  palm_vel = _body_velocity(_robot(env), ("iiwa14_link_7",))[:, 0]
  reward_buf = getattr(env, "reward_buf", None)
  if reward_buf is None:
    reward_buf = torch.zeros(env.num_envs, device=env.device)
  closest_keypoint = torch.where(
    torch.isinf(state["closest_keypoint_max_dist_fixed_size"]),
    kin["keypoint_max_dist_fixed"],
    state["closest_keypoint_max_dist_fixed_size"],
  )
  closest_fingertip = torch.where(
    torch.isinf(state["closest_fingertip_dist"]),
    kin["fingertip_dist"],
    state["closest_fingertip_dist"],
  )
  return torch.cat(
    [
      kin["joint_pos_unscaled"],
      kin["joint_vel"],
      kin["prev_targets"],
      kin["palm_pos"],
      kin["palm_quat"][:, [1, 2, 3, 0]],
      palm_vel,
      kin["object_quat"][:, [1, 2, 3, 0]],
      torch.cat([kin["object_lin_vel"], kin["object_ang_vel"]], dim=-1),
      kin["fingertip_rel_palm"],
      kin["keypoints_rel_palm"],
      kin["keypoints_rel_goal"],
      kin["object_scales"],
      closest_keypoint.unsqueeze(-1),
      closest_fingertip,
      state["lifted_object"].float().unsqueeze(-1),
      torch.log(env.episode_length_buf.float().unsqueeze(-1) / 10.0 + 1.0),
      torch.log(state["successes"].unsqueeze(-1) + 1.0),
      0.01 * reward_buf.unsqueeze(-1),
    ],
    dim=-1,
  )


def keypoint_delta_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env, use_object_state_delay_noise=False)
  state = _state(env)
  z_lift = 0.05 + kin["object_pos"][:, 2] - state["initial_object_z"]
  lifted = state["lifted_object"] | (z_lift > 0.15)
  state["lifted_object"][:] = lifted
  closest = state["closest_keypoint_max_dist"]
  closest_fixed = state["closest_keypoint_max_dist_fixed_size"]
  first_sample = torch.isinf(closest)
  first_fixed = torch.isinf(closest_fixed)
  delta_fixed = torch.where(
    first_fixed,
    torch.zeros_like(closest_fixed),
    torch.clamp(closest_fixed - kin["keypoint_max_dist_fixed"], min=0.0),
  )
  closest[:] = torch.where(
    first_sample,
    kin["keypoint_max_dist"],
    torch.minimum(closest, kin["keypoint_max_dist"]),
  )
  closest_fixed[:] = torch.where(
    first_fixed,
    kin["keypoint_max_dist_fixed"],
    torch.minimum(closest_fixed, kin["keypoint_max_dist_fixed"]),
  )
  return delta_fixed * lifted


def fingertip_delta_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env, use_object_state_delay_noise=False)
  state = _state(env)
  lifted = state["lifted_object"]
  closest = state["closest_fingertip_dist"]
  first_sample = torch.isinf(closest)
  delta = torch.where(
    first_sample,
    torch.zeros_like(closest),
    torch.clamp(closest - kin["fingertip_dist"], min=0.0),
  )
  state["closest_fingertip_dist"][:] = torch.where(
    first_sample,
    kin["fingertip_dist"],
    torch.minimum(closest, kin["fingertip_dist"]),
  )
  return delta.sum(dim=-1) * (~lifted)


def lifting_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env, use_object_state_delay_noise=False)
  state = _state(env)
  z_lift = 0.05 + kin["object_pos"][:, 2] - state["initial_object_z"]
  was_lifted = state["lifted_object"].clone()
  lifted = was_lifted | (z_lift > 0.15)
  state["just_lifted_object"][:] = lifted & ~was_lifted
  state["lifted_object"][:] = lifted
  return torch.clamp(z_lift, 0.0, 0.5) * (~lifted)


def lifting_bonus_reward(env: ManagerBasedRlEnv, bonus: float = 300.0) -> torch.Tensor:
  return bonus * _state(env)["just_lifted_object"].float()


def success_bonus(
  env: ManagerBasedRlEnv,
  tolerance: float = 0.075,
  keypoint_scale: float = 1.5,
  steps: int = 10,
  reach_goal_bonus: float = 1000.0,
) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env, use_object_state_delay_noise=False)
  state = _state(env)
  near = kin["keypoint_max_dist_fixed"] <= tolerance * keypoint_scale
  state["near_goal_steps"] += near.float()
  is_success = state["near_goal_steps"] >= steps
  state["successes"] += is_success.float()
  state["reset_goal_buf"] |= is_success
  return near.float() * (reach_goal_bonus / steps)


def reset_successful_goals(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  max_consecutive_successes: int = 50,
) -> None:
  del env_ids
  state = _state(env)
  should_reset_goal = state["reset_goal_buf"] & (
    state["successes"] < max_consecutive_successes
  )
  reset_goal_env_ids = should_reset_goal.nonzero(as_tuple=False).squeeze(-1)
  state["reset_goal_buf"] &= state["successes"] >= max_consecutive_successes
  if len(reset_goal_env_ids) == 0:
    return
  reset_goal_uniform(env, reset_goal_env_ids, is_first_goal=False)
  state["reset_goal_buf"][reset_goal_env_ids] = False
  state["near_goal_steps"][reset_goal_env_ids] = 0.0
  state["closest_keypoint_max_dist"][reset_goal_env_ids] = float("inf")
  state["closest_keypoint_max_dist_fixed_size"][reset_goal_env_ids] = float("inf")
  env.episode_length_buf[reset_goal_env_ids] = 0
  env.scene.write_data_to_sim()
  env.sim.forward()


def kuka_action_penalty(
  env: ManagerBasedRlEnv, scale: float = 0.03
) -> torch.Tensor:
  return -scale * torch.sum(torch.abs(_robot(env).data.joint_vel[:, :7]), dim=-1)


def hand_action_penalty(
  env: ManagerBasedRlEnv, scale: float = 0.003
) -> torch.Tensor:
  return -scale * torch.sum(torch.abs(_robot(env).data.joint_vel[:, 7:N_ACT]), dim=-1)


def object_velocity_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env, use_object_state_delay_noise=False)
  return -torch.sum(kin["object_lin_vel"] ** 2, dim=-1) - torch.sum(
    kin["object_ang_vel"] ** 2, dim=-1
  )


def object_fell(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _object(env).data.root_link_pos_w[:, 2] < 0.1


def hand_far_from_object(env: ManagerBasedRlEnv, threshold: float = 1.5) -> torch.Tensor:
  robot = _robot(env)
  obj = _object(env)
  fingertip_pos, fingertip_quat = _body_pose(robot, FINGERTIP_BODIES)
  offset = FINGERTIP_OFFSET.to(env.device).repeat(env.num_envs, 1)
  offset = offset[:, None, :].repeat(1, len(FINGERTIP_BODIES), 1)
  fingertip_pos = fingertip_pos + quat_apply(fingertip_quat, offset)
  dist = torch.linalg.norm(fingertip_pos - obj.data.root_link_pos_w[:, None, :], dim=-1)
  return dist.max(dim=-1).values > threshold


def goal_reached(
  env: ManagerBasedRlEnv, tolerance: float = 0.075, steps: int = 10
) -> torch.Tensor:
  del tolerance
  state = _state(env)
  return state["near_goal_steps"] >= steps


def max_consecutive_successes_reached(
  env: ManagerBasedRlEnv,
  max_consecutive_successes: int = 50,
) -> torch.Tensor:
  return _state(env)["successes"] >= max_consecutive_successes


def apply_random_object_perturbations(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  force_scale: float = 20.0,
  torque_scale: float = 2.0,
  force_decay: float = 0.0,
  torque_decay: float = 0.0,
  force_decay_interval: float = 0.08,
  torque_decay_interval: float = 0.08,
  force_only_when_lifted: bool = True,
  torque_only_when_lifted: bool = True,
  lin_vel_impulse_scale: float = 0.0,
  ang_vel_impulse_scale: float = 0.0,
  lin_vel_impulse_only_when_lifted: bool = True,
  ang_vel_impulse_only_when_lifted: bool = True,
  asset_cfg: SceneEntityCfg = OBJECT_BODY_CFG,
) -> None:
  del env_ids
  state = _state(env)
  obj = _object(env)
  lifted = state["lifted_object"].float()[:, None, None]
  body_ids = asset_cfg.body_ids

  force_decay_factor = force_decay ** (env.step_dt / force_decay_interval)
  torque_decay_factor = torque_decay ** (env.step_dt / torque_decay_interval)
  state["rb_forces"] *= force_decay_factor
  state["rb_torques"] *= torque_decay_factor

  if force_scale > 0.0:
    force_mask = torch.rand(env.num_envs, device=env.device) < state["random_force_prob"]
    if force_mask.any():
      state["rb_forces"][force_mask] = (
        torch.randn((int(force_mask.sum().item()), 1, 3), device=env.device)
        * force_scale
        * state["object_masses"][force_mask, None, None]
      )
    if force_only_when_lifted:
      state["rb_forces"] *= lifted

  if torque_scale > 0.0:
    torque_mask = (
      torch.rand(env.num_envs, device=env.device) < state["random_torque_prob"]
    )
    if torque_mask.any():
      state["rb_torques"][torque_mask] = (
        torch.randn((int(torque_mask.sum().item()), 1, 3), device=env.device)
        * torque_scale
        * state["object_masses"][torque_mask, None, None]
      )
    if torque_only_when_lifted:
      state["rb_torques"] *= lifted

  obj.write_external_wrench_to_sim(
    state["rb_forces"], state["rb_torques"], body_ids=body_ids
  )

  velocity = torch.cat([obj.data.root_link_lin_vel_w, obj.data.root_link_ang_vel_w], dim=-1)
  if lin_vel_impulse_scale > 0.0:
    lin_mask = (
      torch.rand(env.num_envs, device=env.device)
      < state["random_lin_vel_impulse_prob"]
    )
    if lin_vel_impulse_only_when_lifted:
      lin_mask &= state["lifted_object"]
    velocity[lin_mask, :3] = (
      torch.randn((int(lin_mask.sum().item()), 3), device=env.device)
      * lin_vel_impulse_scale
    )
  if ang_vel_impulse_scale > 0.0:
    ang_mask = (
      torch.rand(env.num_envs, device=env.device)
      < state["random_ang_vel_impulse_prob"]
    )
    if ang_vel_impulse_only_when_lifted:
      ang_mask &= state["lifted_object"]
    velocity[ang_mask, 3:6] = (
      torch.randn((int(ang_mask.sum().item()), 3), device=env.device)
      * ang_vel_impulse_scale
    )
  if lin_vel_impulse_scale > 0.0 or ang_vel_impulse_scale > 0.0:
    obj.write_root_link_velocity_to_sim(velocity)
