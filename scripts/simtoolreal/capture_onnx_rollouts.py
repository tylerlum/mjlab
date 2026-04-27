"""Capture SimToolReal website-ONNX policy rollouts as interactive HTML files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from mjlab.tasks.simtoolreal import SimToolRealOnnxPolicy
from mjlab.tasks.simtoolreal import policy as policy_module
from mjlab.tasks.simtoolreal.assets import JOINT_NAMES, ROBOT_URDF
from mjlab.tasks.simtoolreal.interactive_viewer import create_html, make_embedded_robot

MJLAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SITE_ROOT = MJLAB_ROOT.parent / "simtoolreal.github.io"


@dataclass(frozen=True)
class ObjectVariant:
  name: str
  handle: tuple[float, float, float]
  head: tuple[float, float, float]
  goal: tuple[float, float, float]

  @property
  def bbox(self) -> tuple[float, float, float]:
    return (
      self.handle[0] + self.head[0],
      max(self.handle[1], self.head[1]),
      max(self.handle[2], self.head[2]),
    )


VARIANTS = (
  ObjectVariant("hammer_box", (0.20, 0.03, 0.02), (0.05, 0.09, 0.04), (0.00, 0.05, 0.82)),
  ObjectVariant("screwdriver_box", (0.10, 0.035, 0.035), (0.12, 0.012, 0.012), (0.08, -0.02, 0.78)),
  ObjectVariant("spatula_box", (0.16, 0.02, 0.012), (0.10, 0.06, 0.02), (-0.08, 0.08, 0.80)),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--scene",
    type=Path,
    default=DEFAULT_SITE_ROOT / "mujoco_wasm/assets/scenes/iiwa_sharpa.xml",
  )
  parser.add_argument(
    "--policy",
    type=Path,
    default=DEFAULT_SITE_ROOT / "mujoco_wasm/dist-desktop/policy_iiwa_sharpa.onnx",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/simtoolreal_pretrained_rollouts"),
  )
  parser.add_argument("--policy-steps", type=int, default=360)
  return parser.parse_args()


def _name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
  idx = mujoco.mj_name2id(model, obj_type, name)
  if idx < 0:
    raise RuntimeError(f"Missing MuJoCo {obj_type.name}: {name}")
  return int(idx)


def _pose_xyzw(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
  return np.concatenate(
    [pos.astype(np.float32), quat_wxyz[[1, 2, 3, 0]].astype(np.float32)]
  )


def _box_urdf(name: str, variant: ObjectVariant | None = None) -> str:
  if variant is None:
    return f"""<robot name="{name}">
  <link name="{name}">
    <visual><geometry><box size="0.475 0.4 0.3"/></geometry></visual>
  </link>
</robot>"""
  handle = variant.handle
  head = variant.head
  total = variant.bbox[0]
  handle_x = -0.5 * total + 0.5 * handle[0]
  head_x = 0.5 * total - 0.5 * head[0]
  return f"""<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="{handle_x} 0 0"/>
      <geometry><box size="{handle[0]} {handle[1]} {handle[2]}"/></geometry>
    </visual>
    <visual>
      <origin xyz="{head_x} 0 0"/>
      <geometry><box size="{head[0]} {head[1]} {head[2]}"/></geometry>
    </visual>
  </link>
</robot>"""


def _set_variant(model: mujoco.MjModel, variant: ObjectVariant) -> None:
  handle_ids = [
    _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_handle_geom"),
    _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_handle_geom"),
  ]
  head_ids = [
    _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_head_geom"),
    _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_head_geom"),
  ]
  total = variant.bbox[0]
  handle_center_x = -0.5 * total + 0.5 * variant.handle[0]
  head_center_x = 0.5 * total - 0.5 * variant.head[0]
  for geom_id in handle_ids:
    model.geom_size[geom_id, :3] = 0.5 * np.asarray(variant.handle, dtype=np.float64)
    model.geom_pos[geom_id, :3] = (handle_center_x, 0.0, 0.0)
  for geom_id in head_ids:
    # The browser XML uses capsule heads.  Keep the type, but approximate the
    # requested cuboidal head by matching x half-length and the larger yz radius.
    radius = 0.5 * max(variant.head[1], variant.head[2])
    model.geom_size[geom_id, 0] = radius
    model.geom_size[geom_id, 1] = max(0.5 * variant.head[0] - radius, 1.0e-4)
    model.geom_pos[geom_id, :3] = (head_center_x, 0.0, 0.0)
  policy_module.OBJECT_SCALES[:] = np.asarray(variant.bbox, dtype=np.float32) / policy_module.OBJECT_BASE_SIZE


def _reset_object_goal(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  policy: SimToolRealOnnxPolicy,
  variant: ObjectVariant,
) -> None:
  obj = policy.cache.object_qpos_adr
  data.qpos[obj : obj + 7] = np.array([0.0, 0.05, 0.63, 1.0, 0.0, 0.0, 0.0])
  object_root = model.body_rootid[policy.cache.object_body]
  object_joint = model.body_jntadr[object_root]
  object_dof = model.jnt_dofadr[object_joint]
  data.qvel[object_dof : object_dof + 6] = 0.0
  data.mocap_pos[policy.cache.goal_mocap_id] = np.asarray(variant.goal)
  data.mocap_quat[policy.cache.goal_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
  mujoco.mj_forward(model, data)


def _rollout_variant(
  scene: Path,
  policy_path: Path,
  output_dir: Path,
  variant: ObjectVariant,
  policy_steps: int,
) -> Path:
  model = mujoco.MjModel.from_xml_path(str(scene))
  _set_variant(model, variant)
  data = mujoco.MjData(model)
  policy = SimToolRealOnnxPolicy(model, policy_path)
  policy.reset(data)
  _reset_object_goal(model, data, policy, variant)
  policy_decimation = max(1, round((1.0 / 60.0) / model.opt.timestep))

  object_body = policy.cache.object_body
  goal_body = _name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_object")
  table_body = _name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
  robot_joint_pos = []
  robot_base_pose = []
  object_pose = []
  goal_pose = []
  table_pose = []
  for _ in range(policy_steps):
    policy.step_policy(data)
    for _ in range(policy_decimation):
      policy.apply_control(data)
      mujoco.mj_step(model, data)
    robot_joint_pos.append(data.qpos[: policy_module.N_ACT].copy())
    robot_base_pose.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    object_pose.append(_pose_xyzw(data.xpos[object_body], data.xquat[object_body]))
    goal_pose.append(_pose_xyzw(data.xpos[goal_body], data.xquat[goal_body]))
    table_pose.append(_pose_xyzw(data.xpos[table_body], data.xquat[table_body]))

  robots = [
    make_embedded_robot(
      name="robot",
      urdf_text=ROBOT_URDF.read_text(encoding="utf-8"),
      animated=True,
    ),
    make_embedded_robot(
      name="object",
      urdf_text=_box_urdf("object", variant),
      color_override=(0.1, 0.45, 0.95),
    ),
    make_embedded_robot(
      name="goal",
      urdf_text=_box_urdf("goal", variant),
      color_override=(0.1, 0.8, 0.25),
    ),
    make_embedded_robot(
      name="table",
      urdf_text=_box_urdf("table"),
      color_override=(0.6, 0.6, 0.6),
    ),
  ]
  html = create_html(
    joint_names=list(JOINT_NAMES),
    robot_joint_positions=np.asarray(robot_joint_pos),
    robots=robots,
    object_poses={
      "object": np.asarray(object_pose),
      "goal": np.asarray(goal_pose),
      "table": np.asarray(table_pose),
    },
    robot_base_poses=np.asarray(robot_base_pose),
    dt=1.0 / 60.0,
    robot_name="robot",
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  path = output_dir / f"{timestamp}_{variant.name}_pretrained_onnx.html"
  path.write_text(html, encoding="utf-8")
  distance = float(np.linalg.norm(data.xpos[object_body] - data.xpos[goal_body]))
  print(f"{variant.name}: saved {path} final_object_goal_distance={distance:.4f}")
  return path


def main() -> None:
  args = parse_args()
  if not args.scene.exists():
    raise FileNotFoundError(f"Scene XML not found: {args.scene}")
  if not args.policy.exists():
    raise FileNotFoundError(f"Policy ONNX not found: {args.policy}")
  paths = [
    _rollout_variant(
      args.scene,
      args.policy,
      args.output_dir,
      variant,
      policy_steps=args.policy_steps,
    )
    for variant in VARIANTS
  ]
  print("Saved HTML rollouts:")
  for path in paths:
    print(path)


if __name__ == "__main__":
  main()
