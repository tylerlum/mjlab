"""Manager-based MDP pieces for the SimToolReal MJLab port."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.simtoolreal.assets import JOINT_NAMES
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

N_ACT = 29
N_OBS = 140
OBJECT_BASE_SIZE = 0.04
OBJECT_KEYPOINT_SIGNS = torch.tensor(
  [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]]
)
PALM_OFFSET = torch.tensor([0.0, -0.02, 0.16])
FINGERTIP_OFFSET = torch.tensor([0.02, 0.002, 0.0])
FINGERTIP_BODIES = (
  "left_index_DP",
  "left_middle_DP",
  "left_ring_DP",
  "left_thumb_DP",
  "left_pinky_DP",
)
OBJECT_GEOM_CFG = SceneEntityCfg("object", geom_names=("object_geom",))

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
      "closest_keypoint_max_dist": torch.full(
        (env.num_envs,), float("inf"), device=env.device
      ),
      "near_goal_steps": torch.zeros(env.num_envs, device=env.device),
      "lifted_object": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
      "initial_object_z": torch.full((env.num_envs,), 0.545, device=env.device),
    }
  return env._simtoolreal_state  # type: ignore[attr-defined]


def _q_limits(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  return Q_LOWER.to(env.device), Q_UPPER.to(env.device)


def _robot(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["robot"]


def _object(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["object"]


def _goal(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["goal"]


@dataclass(kw_only=True)
class SimToolRealJointPositionActionCfg(ActionTermCfg):
  hand_moving_average: float = 0.1
  arm_moving_average: float = 0.1
  arm_speed_scale: float = 1.5

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
    self.prev_targets = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    self.targets = self.prev_targets.clone()

  @property
  def action_dim(self) -> int:
    return N_ACT

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = torch.clamp(actions.to(self.device), -1.0, 1.0)
    q_lower, q_upper = _q_limits(self._env)
    targets = self.prev_targets.clone()

    hand_targets = 0.5 * (self._raw_actions[:, 7:] + 1.0) * (
      q_upper[7:] - q_lower[7:]
    ) + q_lower[7:]
    targets[:, 7:] = (
      self.cfg.hand_moving_average * hand_targets
      + (1.0 - self.cfg.hand_moving_average) * self.prev_targets[:, 7:]
    )
    targets[:, 7:] = torch.clamp(targets[:, 7:], q_lower[7:], q_upper[7:])

    arm_targets = (
      self.prev_targets[:, :7]
      + self.cfg.arm_speed_scale * self._env.step_dt * self._raw_actions[:, :7]
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
    default = self._entity.data.default_joint_pos[:, self._joint_ids]
    self.prev_targets[env_ids] = default[env_ids]
    self.targets[env_ids] = default[env_ids]


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
  state["near_goal_steps"][env_ids] = 0.0
  state["lifted_object"][env_ids] = False


def reset_object_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  x_range: tuple[float, float] = (-0.04, 0.04),
  y_range: tuple[float, float] = (0.02, 0.08),
  z: float | None = None,
  table_surface_z: float = 0.53,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  obj = _object(env)
  state = _state(env)
  pos = torch.zeros((len(env_ids), 3), device=env.device)
  pos[:, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*x_range)
  pos[:, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*y_range)
  if z is None:
    object_lengths = OBJECT_BASE_SIZE * state["object_scales"][env_ids]
    pos[:, 2] = table_surface_z + 0.5 * object_lengths[:, 2] + 0.002
  else:
    pos[:, 2] = z
  pos += env.scene.env_origins[env_ids]
  quat = torch.zeros((len(env_ids), 4), device=env.device)
  yaw = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
  quat[:] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
  obj.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  obj.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def reset_goal_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  x_range: tuple[float, float] = (-0.12, 0.12),
  y_range: tuple[float, float] = (-0.02, 0.18),
  z_range: tuple[float, float] = (0.68, 0.95),
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  goal = _goal(env)
  pos = torch.zeros((len(env_ids), 3), device=env.device)
  pos[:, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*x_range)
  pos[:, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*y_range)
  pos[:, 2] = torch.empty(len(env_ids), device=env.device).uniform_(*z_range)
  pos += env.scene.env_origins[env_ids]
  roll = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
  pitch = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
  yaw = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  goal.write_mocap_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)


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
  lengths = 2.0 * env.sim.model.geom_size[env_ids[:, None], geom_ids, :3].squeeze(1)
  state["object_scales"][env_ids] = lengths / OBJECT_BASE_SIZE


def _body_pose(
  entity: Entity, body_names: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
  local_ids, _ = entity.find_bodies(body_names, preserve_order=True)
  body_ids = entity.indexing.body_ids[local_ids]
  pos = entity.data.data.xpos[:, body_ids]
  quat = entity.data.data.xquat[:, body_ids]
  return pos, quat


def _simtoolreal_kinematics(
  env: ManagerBasedRlEnv,
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
  goal_pos = goal.data.root_link_pos_w
  goal_quat = goal.data.root_link_quat_w

  signs = OBJECT_KEYPOINT_SIGNS.to(env.device)
  offsets = signs[None] * (OBJECT_BASE_SIZE * 1.5 * 0.5)
  offsets = offsets * state["object_scales"][:, None, :]
  object_keypoints = object_pos[:, None, :] + quat_apply(
    object_quat[:, None, :].repeat(1, 4, 1), offsets
  )
  goal_keypoints = goal_pos[:, None, :] + quat_apply(
    goal_quat[:, None, :].repeat(1, 4, 1), offsets
  )
  keypoints_rel_goal = object_keypoints - goal_keypoints
  keypoint_dist = torch.linalg.norm(keypoints_rel_goal, dim=-1)
  keypoint_max_dist = keypoint_dist.max(dim=-1).values

  return {
    "joint_pos_unscaled": (2.0 * robot.data.joint_pos[:, :N_ACT] - q_upper - q_lower)
    / (q_upper - q_lower),
    "joint_vel": robot.data.joint_vel[:, :N_ACT],
    "prev_targets": prev_targets,
    "palm_pos": palm_pos,
    "palm_quat": palm_quat,
    "object_quat": object_quat,
    "fingertip_rel_palm": fingertip_rel_palm.reshape(env.num_envs, -1),
    "keypoints_rel_palm": (object_keypoints - palm_pos[:, None, :]).reshape(
      env.num_envs, -1
    ),
    "keypoints_rel_goal": keypoints_rel_goal.reshape(env.num_envs, -1),
    "object_scales": state["object_scales"],
    "keypoint_max_dist": keypoint_max_dist,
    "object_pos": object_pos,
    "object_lin_vel": obj.data.root_link_lin_vel_w,
    "object_ang_vel": obj.data.root_link_ang_vel_w,
  }


def simtoolreal_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env)
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


def keypoint_delta_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env)
  state = _state(env)
  z_lift = 0.05 + kin["object_pos"][:, 2] - state["initial_object_z"]
  lifted = state["lifted_object"] | (z_lift > 0.15)
  state["lifted_object"][:] = lifted
  closest = state["closest_keypoint_max_dist"]
  first_sample = torch.isinf(closest)
  delta = torch.where(
    first_sample,
    torch.zeros_like(closest),
    torch.clamp(closest - kin["keypoint_max_dist"], min=0.0),
  )
  closest[:] = torch.where(
    first_sample,
    kin["keypoint_max_dist"],
    torch.minimum(closest, kin["keypoint_max_dist"]),
  )
  return delta * lifted


def lifting_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env)
  state = _state(env)
  z_lift = 0.05 + kin["object_pos"][:, 2] - state["initial_object_z"]
  lifted = state["lifted_object"] | (z_lift > 0.15)
  return torch.clamp(z_lift, 0.0, 0.5) * (~lifted)


def success_bonus(
  env: ManagerBasedRlEnv, tolerance: float = 0.01, steps: int = 5
) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env)
  state = _state(env)
  near = kin["keypoint_max_dist"] <= tolerance
  state["near_goal_steps"][:] = torch.where(
    near, state["near_goal_steps"] + 1.0, torch.zeros_like(state["near_goal_steps"])
  )
  return (state["near_goal_steps"] >= steps).float()


def object_velocity_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  kin = _simtoolreal_kinematics(env)
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
  env: ManagerBasedRlEnv, tolerance: float = 0.01, steps: int = 5
) -> torch.Tensor:
  del tolerance
  state = _state(env)
  return state["near_goal_steps"] >= steps
