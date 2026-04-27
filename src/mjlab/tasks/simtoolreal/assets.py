"""Asset builders for the SimToolReal MJLab training port."""

from __future__ import annotations

import base64
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

SIMTOOLREAL_ROOT = Path(__file__).resolve().parents[5] / "simtoolreal"
SIMTOOLREAL_ASSET_ROOT = SIMTOOLREAL_ROOT / "assets/urdf/kuka_sharpa_description"
ROBOT_URDF = SIMTOOLREAL_ASSET_ROOT / "iiwa14_left_sharpa_adjusted_restricted.urdf"
ROBOT_MESH_CACHE = Path(
  os.environ.get("MJLAB_SIMTOOLREAL_MESH_CACHE", "/tmp/mjlab_simtoolreal_meshes")
)

JOINT_NAMES = (
  "iiwa14_joint_1",
  "iiwa14_joint_2",
  "iiwa14_joint_3",
  "iiwa14_joint_4",
  "iiwa14_joint_5",
  "iiwa14_joint_6",
  "iiwa14_joint_7",
  "left_1_thumb_CMC_FE",
  "left_thumb_CMC_AA",
  "left_thumb_MCP_FE",
  "left_thumb_MCP_AA",
  "left_thumb_IP",
  "left_2_index_MCP_FE",
  "left_index_MCP_AA",
  "left_index_PIP",
  "left_index_DIP",
  "left_3_middle_MCP_FE",
  "left_middle_MCP_AA",
  "left_middle_PIP",
  "left_middle_DIP",
  "left_4_ring_MCP_FE",
  "left_ring_MCP_AA",
  "left_ring_PIP",
  "left_ring_DIP",
  "left_5_pinky_CMC",
  "left_pinky_MCP_FE",
  "left_pinky_MCP_AA",
  "left_pinky_PIP",
  "left_pinky_DIP",
)

DEFAULT_JOINT_POS = {
  "iiwa14_joint_1": -1.571,
  "iiwa14_joint_2": 1.39647,
  "iiwa14_joint_3": 0.0,
  "iiwa14_joint_4": 1.55053,
  "iiwa14_joint_5": 0.0,
  "iiwa14_joint_6": 1.485,
  "iiwa14_joint_7": 1.308,
  ".*": 0.0,
}


def _resolve_robot_mesh_paths(spec: mujoco.MjSpec) -> None:
  """Patch MuJoCo's URDF-imported basename mesh paths to absolute source paths."""
  for mesh in spec.meshes:
    filename = Path(mesh.file).name
    if filename.startswith("link_"):
      path = SIMTOOLREAL_ASSET_ROOT / "new_iiwa14_meshes/collision" / filename
    else:
      path = SIMTOOLREAL_ASSET_ROOT / "left_sharpa_meshes" / filename
    if path.exists():
      mesh.file = str(path)


def _prepare_robot_mesh_cache() -> Path:
  """Make MuJoCo's basename-only URDF mesh import resolvable at compile time."""
  ROBOT_MESH_CACHE.mkdir(parents=True, exist_ok=True)
  mesh_dirs = (
    SIMTOOLREAL_ASSET_ROOT / "new_iiwa14_meshes/collision",
    SIMTOOLREAL_ASSET_ROOT / "left_sharpa_meshes",
  )
  for mesh_dir in mesh_dirs:
    for mesh_path in mesh_dir.glob("*.[Ss][Tt][Ll]"):
      cache_path = ROBOT_MESH_CACHE / mesh_path.name
      if cache_path.exists():
        continue
      try:
        cache_path.symlink_to(mesh_path)
      except OSError:
        shutil.copy2(mesh_path, cache_path)
  return ROBOT_MESH_CACHE


def read_robot_urdf_for_viewer() -> str:
  """Read the robot URDF with mesh files embedded for standalone HTML viewers."""
  root = ET.fromstring(ROBOT_URDF.read_text(encoding="utf-8"))
  for mesh in root.findall(".//mesh"):
    filename = mesh.attrib.get("filename")
    if not filename:
      continue
    path = SIMTOOLREAL_ASSET_ROOT / filename
    if not path.exists():
      continue
    suffix = path.suffix.lower()
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mesh.attrib["filename"] = (
      f"data:application/octet-stream;ext={suffix};base64,{encoded}"
    )
  return ET.tostring(root, encoding="unicode")


def get_iiwa_sharpa_spec() -> mujoco.MjSpec:
  if not ROBOT_URDF.exists():
    raise FileNotFoundError(
      f"SimToolReal robot URDF not found at {ROBOT_URDF}. "
      "Clone simtoolreal next to mjlab or vendor the asset before training."
    )
  spec = mujoco.MjSpec.from_file(str(ROBOT_URDF))
  spec.modelfiledir = str(_prepare_robot_mesh_cache())
  _resolve_robot_mesh_paths(spec)
  return spec


def get_iiwa_sharpa_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_iiwa_sharpa_spec,
    init_state=EntityCfg.InitialStateCfg(joint_pos=DEFAULT_JOINT_POS),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=JOINT_NAMES[:7],
          stiffness=600.0,
          damping=25.0,
          effort_limit=300.0,
        ),
        BuiltinPositionActuatorCfg(
          target_names_expr=JOINT_NAMES[7:],
          stiffness=8.0,
          damping=0.3,
          effort_limit=4.0,
        ),
      ),
    ),
    sort_actuators=True,
  )


def get_object_spec(
  handle_half_size: tuple[float, float, float] = (0.075, 0.0125, 0.0125),
  head_half_size: tuple[float, float, float] = (0.025, 0.025, 0.015),
  mass: float = 0.08,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_free_joint")
  body.add_geom(
    name="object_handle_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(-0.025, 0.0, 0.0),
    size=handle_half_size,
    mass=0.75 * mass,
    rgba=(0.45, 0.45, 0.45, 1.0),
    condim=6,
  )
  body.add_geom(
    name="object_head_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.075, 0.0, 0.0),
    size=head_half_size,
    mass=0.25 * mass,
    rgba=(0.45, 0.45, 0.45, 1.0),
    condim=6,
  )
  return spec


def get_object_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_object_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.05, 0.545),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )


def get_goal_spec(
  handle_half_size: tuple[float, float, float] = (0.075, 0.0125, 0.0125),
  head_half_size: tuple[float, float, float] = (0.025, 0.025, 0.015),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal_object", mocap=True)
  body.add_geom(
    name="goal_handle_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(-0.025, 0.0, 0.0),
    size=handle_half_size,
    rgba=(0.1, 0.8, 0.2, 0.35),
    contype=0,
    conaffinity=0,
  )
  body.add_geom(
    name="goal_head_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.075, 0.0, 0.0),
    size=head_half_size,
    rgba=(0.1, 0.8, 0.2, 0.35),
    contype=0,
    conaffinity=0,
  )
  return spec


def get_goal_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_goal_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.05, 0.78),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )


def get_table_spec(
  half_size: tuple[float, float, float] = (0.2375, 0.2, 0.15),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table_object")
  body.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half_size,
    mass=500.0,
    rgba=(0.82, 0.56, 0.35, 1.0),
    friction=(1.0, 0.005, 0.0001),
    condim=6,
  )
  return spec


def get_table_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_table_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.05, 0.38),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )
