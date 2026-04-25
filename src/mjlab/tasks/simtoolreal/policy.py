"""SimToolReal MuJoCo policy utilities.

This module mirrors the public SimToolReal browser demo's MuJoCo/ONNX inference
path.  It is intentionally separate from the training MDP so we can keep a
known-good pretrained-policy baseline while porting the environment managers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

N_ACT = 29
N_OBS = 140
LSTM_SIZE = 1024

DEFAULT_QPOS = np.array(
  [-1.571, 1.39647, 0.0, 1.55053, 0.0, 1.485, 1.308] + [0.0] * 22,
  dtype=np.float32,
)

Q_LOWER = np.array(
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
    -0.1745,
    -0.0349,
    0.0,
    0.0,
    0.0,
  ],
  dtype=np.float32,
)
Q_UPPER = np.array(
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
    1.5708,
    0.0349,
    1.7453,
    1.3963,
    0.2618,
  ],
  dtype=np.float32,
)

PALM_OFFSET = np.array([0.0, -0.02, 0.16], dtype=np.float32)
FINGERTIP_OFFSET = np.array([0.02, 0.002, 0.0], dtype=np.float32)
OBJECT_SCALES = np.array([5.0, 0.75, 0.5], dtype=np.float32)
OBJECT_BASE_SIZE = 0.04
KEYPOINT_SCALE = 1.5
KEYPOINT_SIGNS = np.array(
  [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]],
  dtype=np.float32,
)
FINGERTIP_BODIES = (
  "palmleft_index_DP",
  "palmleft_middle_DP",
  "palmleft_ring_DP",
  "palmleft_thumb_DP",
  "palmleft_pinky_DP",
)


def quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  qvec = np.array([x, y, z], dtype=np.float32)
  return (
    vec * (2.0 * w * w - 1.0)
    + np.cross(qvec, vec) * (2.0 * w)
    + qvec * (2.0 * np.dot(qvec, vec))
  )


def keypoints(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
  quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
  half_size = OBJECT_BASE_SIZE * KEYPOINT_SCALE * 0.5
  points = []
  for sign in KEYPOINT_SIGNS:
    points.append(pos + quat_rotate_wxyz(quat_wxyz, sign * half_size * OBJECT_SCALES))
  return np.asarray(points, dtype=np.float32)


def normalized_action_to_targets(
  action: np.ndarray, prev_targets: np.ndarray
) -> np.ndarray:
  """Convert SimToolReal normalized action to MuJoCo position-control targets."""
  action = np.asarray(action, dtype=np.float32).reshape(N_ACT)
  prev_targets = np.asarray(prev_targets, dtype=np.float32).reshape(N_ACT)

  targets = np.zeros(N_ACT, dtype=np.float32)
  targets[7:N_ACT] = 0.5 * (action[7:N_ACT] + 1.0) * (
    Q_UPPER[7:N_ACT] - Q_LOWER[7:N_ACT]
  ) + Q_LOWER[7:N_ACT]
  targets[7:N_ACT] = 0.1 * targets[7:N_ACT] + 0.9 * prev_targets[7:N_ACT]
  targets[7:N_ACT] = np.clip(targets[7:N_ACT], Q_LOWER[7:N_ACT], Q_UPPER[7:N_ACT])

  targets[:7] = prev_targets[:7] + 1.5 * (1.0 / 60.0) * action[:7]
  targets[:7] = np.clip(targets[:7], Q_LOWER[:7], Q_UPPER[:7])
  targets[:7] = 0.1 * targets[:7] + 0.9 * prev_targets[:7]
  return targets


@dataclass
class BodyCache:
  link7: int
  fingertips: list[int]
  object_body: int
  object_qpos_adr: int
  goal_mocap_id: int


class SimToolRealOnnxPolicy:
  """Stateful ONNX policy wrapper for the SimToolReal browser policy."""

  def __init__(self, model: mujoco.MjModel, policy_path: Path):
    self.model = model
    self.cache = self._build_cache(model)
    self.session = ort.InferenceSession(
      str(policy_path), providers=["CPUExecutionProvider"]
    )
    self.h = np.zeros((1, 1, LSTM_SIZE), dtype=np.float32)
    self.c = np.zeros((1, 1, LSTM_SIZE), dtype=np.float32)
    self.prev_targets = DEFAULT_QPOS.copy()
    self.latest_action = np.zeros(N_ACT, dtype=np.float32)

  @staticmethod
  def _build_cache(model: mujoco.MjModel) -> BodyCache:
    link7 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_joint = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "object_free_joint"
    )
    goal_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_object")
    missing = {
      "link7": link7,
      "object": object_body,
      "object_free_joint": object_joint,
      "goal_object": goal_body,
    }
    bad = [name for name, idx in missing.items() if idx < 0]
    if bad:
      raise RuntimeError(f"MuJoCo scene is missing required names: {bad}")
    fingertips = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
      for name in FINGERTIP_BODIES
    ]
    if any(idx < 0 for idx in fingertips):
      raise RuntimeError(f"MuJoCo scene is missing fingertip bodies: {FINGERTIP_BODIES}")
    return BodyCache(
      link7=link7,
      fingertips=fingertips,
      object_body=object_body,
      object_qpos_adr=int(model.jnt_qposadr[object_joint]),
      goal_mocap_id=int(model.body_mocapid[goal_body]),
    )

  def reset(self, data: mujoco.MjData) -> None:
    self.h.fill(0.0)
    self.c.fill(0.0)
    self.prev_targets = DEFAULT_QPOS.copy()
    self.latest_action.fill(0.0)
    data.qpos[:N_ACT] = DEFAULT_QPOS
    data.ctrl[:N_ACT] = DEFAULT_QPOS
    data.qvel[:N_ACT] = 0.0
    mujoco.mj_forward(self.model, data)

  def observation(self, data: mujoco.MjData) -> np.ndarray:
    obs = np.zeros(N_OBS, dtype=np.float32)
    i = 0
    q = data.qpos[:N_ACT].astype(np.float32)
    obs[i : i + N_ACT] = (2.0 * q - Q_UPPER - Q_LOWER) / (Q_UPPER - Q_LOWER)
    i += N_ACT
    obs[i : i + N_ACT] = data.qvel[:N_ACT]
    i += N_ACT
    obs[i : i + N_ACT] = self.prev_targets
    i += N_ACT

    link7_quat = data.xquat[self.cache.link7].astype(np.float32)
    link7_pos = data.xpos[self.cache.link7].astype(np.float32)
    palm_pos = link7_pos + quat_rotate_wxyz(link7_quat, PALM_OFFSET)
    obs[i : i + 3] = palm_pos
    i += 3
    obs[i : i + 4] = link7_quat[[1, 2, 3, 0]]
    i += 4

    obj_adr = self.cache.object_qpos_adr
    object_pos = data.qpos[obj_adr : obj_adr + 3].astype(np.float32)
    object_quat_wxyz = data.qpos[obj_adr + 3 : obj_adr + 7].astype(np.float32)
    object_quat_xyzw = object_quat_wxyz[[1, 2, 3, 0]]
    obs[i : i + 4] = object_quat_xyzw
    i += 4

    for body_id in self.cache.fingertips:
      fingertip_pos = data.xpos[body_id].astype(np.float32)
      fingertip_quat = data.xquat[body_id].astype(np.float32)
      obs[i : i + 3] = (
        fingertip_pos + quat_rotate_wxyz(fingertip_quat, FINGERTIP_OFFSET) - palm_pos
      )
      i += 3

    object_keypoints = keypoints(object_pos, object_quat_xyzw)
    for point in object_keypoints:
      obs[i : i + 3] = point - palm_pos
      i += 3

    goal_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32)
    goal_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if self.cache.goal_mocap_id >= 0:
      goal_pos = data.mocap_pos[self.cache.goal_mocap_id].astype(np.float32)
      goal_quat_wxyz = data.mocap_quat[self.cache.goal_mocap_id].astype(np.float32)
      goal_quat_xyzw = goal_quat_wxyz[[1, 2, 3, 0]]
    goal_keypoints = keypoints(goal_pos, goal_quat_xyzw)
    for point, goal_point in zip(object_keypoints, goal_keypoints, strict=True):
      obs[i : i + 3] = point - goal_point
      i += 3

    obs[i : i + 3] = OBJECT_SCALES
    i += 3
    if i != N_OBS:
      raise RuntimeError(f"Observation length mismatch: wrote {i}, expected {N_OBS}")
    return obs

  def step_policy(self, data: mujoco.MjData) -> np.ndarray:
    obs = np.zeros((1, N_OBS + 1), dtype=np.float32)
    obs[0, :N_OBS] = self.observation(data)
    obs[0, N_OBS] = 50.0
    outputs = self.session.run(
      None,
      {
        "obs": obs,
        "h_in": self.h,
        "c_in": self.c,
      },
    )
    output_names = [out.name for out in self.session.get_outputs()]
    out = dict(zip(output_names, outputs, strict=True))
    self.latest_action = np.clip(out["mu"].reshape(-1)[:N_ACT], -1.0, 1.0)
    self.h = out["h_out"].astype(np.float32)
    self.c = out["c_out"].astype(np.float32)
    self._compute_joint_targets()
    return self.latest_action

  def _compute_joint_targets(self) -> None:
    self.prev_targets = normalized_action_to_targets(
      self.latest_action, self.prev_targets
    )

  def apply_control(self, data: mujoco.MjData) -> None:
    data.ctrl[:N_ACT] = self.prev_targets


def reset_object_and_goal(
  model: mujoco.MjModel, data: mujoco.MjData, cache: BodyCache
) -> None:
  obj = cache.object_qpos_adr
  data.qpos[obj : obj + 7] = np.array([0.0, 0.05, 0.545, 1.0, 0.0, 0.0, 0.0])
  object_root = model.body_rootid[cache.object_body]
  object_joint = model.body_jntadr[object_root]
  object_dof = model.jnt_dofadr[object_joint]
  data.qvel[object_dof : object_dof + 6] = 0.0
  if cache.goal_mocap_id >= 0:
    data.mocap_pos[cache.goal_mocap_id] = np.array([0.0, 0.05, 0.78])
    data.mocap_quat[cache.goal_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
  mujoco.mj_forward(model, data)


def sim_step(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  policy: SimToolRealOnnxPolicy,
  step: int,
  policy_decimation: int,
) -> None:
  if step % policy_decimation == 0:
    policy.step_policy(data)
  policy.apply_control(data)
  mujoco.mj_step(model, data)


def print_rollout_summary(
  data: mujoco.MjData, policy: SimToolRealOnnxPolicy, steps: int, policy_decimation: int
) -> None:
  obj = policy.cache.object_qpos_adr
  object_pos = data.qpos[obj : obj + 3]
  goal_pos = data.mocap_pos[policy.cache.goal_mocap_id]
  distance = float(np.linalg.norm(object_pos - goal_pos))
  print(f"steps={steps}")
  print(f"policy_decimation={policy_decimation}")
  print(f"object_pos={object_pos.tolist()}")
  print(f"goal_pos={goal_pos.tolist()}")
  print(f"object_goal_distance={distance:.6f}")
  print(
    f"last_action_minmax=({policy.latest_action.min():.6f}, "
    f"{policy.latest_action.max():.6f})"
  )
