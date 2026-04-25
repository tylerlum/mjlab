"""Single-environment SimToolReal MuJoCo baseline.

This is not the final MJLab Warp/vectorized training environment.  It provides a
small, testable API around the public browser-demo MuJoCo scene so the MDP port
has a reference implementation for observations, action filtering, resets, and
reward diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mjlab.tasks.simtoolreal.policy import (
  DEFAULT_QPOS,
  N_ACT,
  N_OBS,
  BodyCache,
  SimToolRealOnnxPolicy,
  keypoints,
  normalized_action_to_targets,
  reset_object_and_goal,
)


@dataclass
class SimToolRealBrowserEnvCfg:
  scene_path: Path
  policy_path: Path | None = None
  control_dt: float = 1.0 / 60.0
  episode_length_steps: int = 1200
  success_tolerance: float = 0.01
  success_steps: int = 5
  lifting_bonus_threshold: float = 0.15
  lifting_reward_scale: float = 3.0
  keypoint_reward_scale: float = 60.0
  reach_goal_bonus: float = 250.0
  object_lin_vel_penalty_scale: float = 0.0
  object_ang_vel_penalty_scale: float = 0.0
  info: dict[str, Any] = field(default_factory=dict)


class SimToolRealBrowserEnv:
  """Gym-like single MuJoCo environment for SimToolReal parity work."""

  def __init__(self, cfg: SimToolRealBrowserEnvCfg):
    self.cfg = cfg
    self.model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
    self.data = mujoco.MjData(self.model)
    self.policy = (
      SimToolRealOnnxPolicy(self.model, cfg.policy_path)
      if cfg.policy_path is not None
      else None
    )
    self.cache = self.policy.cache if self.policy is not None else self._build_cache()
    self.policy_decimation = max(1, round(cfg.control_dt / self.model.opt.timestep))
    self.prev_targets = DEFAULT_QPOS.copy()
    self.latest_action = np.zeros(N_ACT, dtype=np.float32)
    self.elapsed_steps = 0
    self.near_goal_steps = 0
    self.lifted_object = False
    self.closest_keypoint_max_dist = np.inf
    self.initial_object_z = 0.0

  def _build_cache(self) -> BodyCache:
    return SimToolRealOnnxPolicy._build_cache(self.model)

  def reset(self) -> np.ndarray:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:N_ACT] = DEFAULT_QPOS
    self.data.ctrl[:N_ACT] = DEFAULT_QPOS
    self.data.qvel[:N_ACT] = 0.0
    self.prev_targets = DEFAULT_QPOS.copy()
    self.latest_action.fill(0.0)
    self.elapsed_steps = 0
    self.near_goal_steps = 0
    self.lifted_object = False
    self.closest_keypoint_max_dist = np.inf
    if self.policy is not None:
      self.policy.reset(self.data)
      self.prev_targets = self.policy.prev_targets.copy()
    reset_object_and_goal(self.model, self.data, self.cache)
    self.initial_object_z = float(self.object_pos[2])
    self.closest_keypoint_max_dist = self.keypoint_max_dist()
    return self.observation()

  def observation(self) -> np.ndarray:
    if self.policy is not None:
      return self.policy.observation(self.data)
    # The no-policy mode is intended for stepping external actions.  Reuse the
    # policy object for observation construction only if callers provided ONNX.
    raise RuntimeError("Observation construction currently requires policy_path")

  def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    self.latest_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    self.prev_targets = normalized_action_to_targets(self.latest_action, self.prev_targets)
    for _ in range(self.policy_decimation):
      self.data.ctrl[:N_ACT] = self.prev_targets
      mujoco.mj_step(self.model, self.data)
    self.elapsed_steps += 1
    reward, info = self.compute_reward()
    done = bool(info["success"] or self.elapsed_steps >= self.cfg.episode_length_steps)
    return self.observation(), reward, done, info

  def step_pretrained_policy(
    self,
  ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    if self.policy is None:
      raise RuntimeError("step_pretrained_policy requires policy_path")
    action = self.policy.step_policy(self.data)
    self.latest_action = action.copy()
    self.prev_targets = self.policy.prev_targets.copy()
    for _ in range(self.policy_decimation):
      self.data.ctrl[:N_ACT] = self.prev_targets
      mujoco.mj_step(self.model, self.data)
    self.elapsed_steps += 1
    reward, info = self.compute_reward()
    done = bool(info["success"] or self.elapsed_steps >= self.cfg.episode_length_steps)
    return self.observation(), reward, done, info

  @property
  def object_pos(self) -> np.ndarray:
    obj = self.cache.object_qpos_adr
    return self.data.qpos[obj : obj + 3].astype(np.float32)

  @property
  def object_quat_xyzw(self) -> np.ndarray:
    obj = self.cache.object_qpos_adr
    return self.data.qpos[obj + 3 : obj + 7].astype(np.float32)[[1, 2, 3, 0]]

  @property
  def goal_pos(self) -> np.ndarray:
    return self.data.mocap_pos[self.cache.goal_mocap_id].astype(np.float32)

  @property
  def goal_quat_xyzw(self) -> np.ndarray:
    quat_wxyz = self.data.mocap_quat[self.cache.goal_mocap_id].astype(np.float32)
    return quat_wxyz[[1, 2, 3, 0]]

  def keypoint_max_dist(self) -> float:
    object_points = keypoints(self.object_pos, self.object_quat_xyzw)
    goal_points = keypoints(self.goal_pos, self.goal_quat_xyzw)
    return float(np.linalg.norm(object_points - goal_points, axis=-1).max())

  def compute_reward(self) -> tuple[float, dict[str, Any]]:
    z_lift = 0.05 + float(self.object_pos[2]) - self.initial_object_z
    lifted_object = self.lifted_object or z_lift > self.cfg.lifting_bonus_threshold
    just_lifted = lifted_object and not self.lifted_object
    self.lifted_object = lifted_object

    keypoint_max_dist = self.keypoint_max_dist()
    keypoint_delta = max(self.closest_keypoint_max_dist - keypoint_max_dist, 0.0)
    self.closest_keypoint_max_dist = min(
      self.closest_keypoint_max_dist, keypoint_max_dist
    )

    near_goal = keypoint_max_dist <= self.cfg.success_tolerance
    self.near_goal_steps = self.near_goal_steps + 1 if near_goal else 0
    success = self.near_goal_steps >= self.cfg.success_steps

    lifting_reward = max(min(z_lift, 0.5), 0.0) * (not lifted_object)
    lift_bonus = self.cfg.reach_goal_bonus if just_lifted else 0.0
    keypoint_reward = keypoint_delta * lifted_object
    success_bonus = self.cfg.reach_goal_bonus if success else 0.0

    object_joint = self.model.body_jntadr[self.model.body_rootid[self.cache.object_body]]
    object_dof = self.model.jnt_dofadr[object_joint]
    object_lin_vel = self.data.qvel[object_dof : object_dof + 3]
    object_ang_vel = self.data.qvel[object_dof + 3 : object_dof + 6]
    lin_vel_penalty = -float(np.square(object_lin_vel).sum())
    ang_vel_penalty = -float(np.square(object_ang_vel).sum())

    reward = (
      self.cfg.lifting_reward_scale * lifting_reward
      + lift_bonus
      + self.cfg.keypoint_reward_scale * keypoint_reward
      + success_bonus
      + self.cfg.object_lin_vel_penalty_scale * lin_vel_penalty
      + self.cfg.object_ang_vel_penalty_scale * ang_vel_penalty
    )
    info = {
      "success": success,
      "near_goal": near_goal,
      "near_goal_steps": self.near_goal_steps,
      "lifted_object": lifted_object,
      "keypoint_max_dist": keypoint_max_dist,
      "z_lift": z_lift,
      "lifting_reward": lifting_reward,
      "keypoint_reward": keypoint_reward,
      "lin_vel_penalty": lin_vel_penalty,
      "ang_vel_penalty": ang_vel_penalty,
    }
    return float(reward), info


__all__ = [
  "N_ACT",
  "N_OBS",
  "SimToolRealBrowserEnv",
  "SimToolRealBrowserEnvCfg",
]
