"""Asset builders for the SimToolReal MJLab training port."""

from __future__ import annotations

import base64
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import (
  EntityArticulationInfoCfg,
  EntityCfg,
  VariantCfg,
  VariantEntityCfg,
)

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
  "iiwa14_joint_2": 1.571,
  "iiwa14_joint_3": 0.0,
  "iiwa14_joint_4": 1.376,
  "iiwa14_joint_5": 0.0,
  "iiwa14_joint_6": 1.485,
  "iiwa14_joint_7": 1.308,
  ".*": 0.0,
}

DEFAULT_ASSET_FRICTION = 1.0
FINGERTIP_FRICTION = 1.5
DEFAULT_GEOM_FRICTION = (DEFAULT_ASSET_FRICTION, 0.005, 0.0001)
FINGERTIP_GEOM_FRICTION = (FINGERTIP_FRICTION, 0.005, 0.0001)
CYLINDER_X_QUAT = (0.7071067811865476, 0.0, 0.7071067811865475, 0.0)
MESH_OBJECT_VARIANTS = (
  (
    "cuboid_hammer",
    "box",
    (0.165, 0.030, 0.026),
    (0.085, 0.055, 0.060),
    450.0,
    1200.0,
  ),
  (
    "cylinder_hammer",
    "cylinder",
    (0.165, 0.030, 0.030),
    (0.085, 0.055, 0.060),
    450.0,
    1200.0,
  ),
  (
    "cuboid_marker",
    "box",
    (0.130, 0.035, 0.032),
    (0.0, 0.0, 0.0),
    450.0,
    0.0,
  ),
  (
    "cylinder_marker",
    "cylinder",
    (0.130, 0.028, 0.028),
    (0.0, 0.0, 0.0),
    450.0,
    0.0,
  ),
)
SIMPLE_MESH_OBJECT_VARIANTS = (
  (
    "simple_cuboid",
    "box",
    (0.180, 0.050, 0.050),
    (0.0, 0.0, 0.0),
    450.0,
    0.0,
  ),
  (
    "simple_cylinder",
    "cylinder",
    (0.180, 0.050, 0.050),
    (0.0, 0.0, 0.0),
    450.0,
    0.0,
  ),
)
ALL_MESH_OBJECT_VARIANTS = MESH_OBJECT_VARIANTS + SIMPLE_MESH_OBJECT_VARIANTS
ObjectMeshVariant = tuple[
  str,
  str,
  tuple[float, float, float],
  tuple[float, float, float],
  float,
  float,
]
MESH_OBJECT_VARIANT_BY_NAME = {
  name: (name, shape, handle, head, handle_density, head_density)
  for name, shape, handle, head, handle_density, head_density in ALL_MESH_OBJECT_VARIANTS
}


def register_object_mesh_variant(
  name: str,
  shape: str,
  handle_lengths: tuple[float, float, float],
  head_lengths: tuple[float, float, float],
  handle_density: float,
  head_density: float,
) -> str:
  """Register a runtime object mesh variant for eval/capture jobs."""
  if shape not in ("box", "cylinder"):
    raise ValueError(f"Unsupported mesh variant shape: {shape}")
  MESH_OBJECT_VARIANT_BY_NAME[name] = (
    name,
    shape,
    tuple(float(v) for v in handle_lengths),
    tuple(float(v) for v in head_lengths),
    float(handle_density),
    float(head_density),
  )
  return name


FINGERTIP_LINK_NAMES = (
  "left_index_DP",
  "left_middle_DP",
  "left_ring_DP",
  "left_thumb_DP",
  "left_pinky_DP",
)
FINGERTIP_MESH_NAMES = (
  "left_DP",
  "left_thumb_DP",
  "elastomer",
  "thumb_elastomer",
)

KUKA_STIFFNESSES = (600.0, 600.0, 500.0, 400.0, 200.0, 200.0, 200.0)
KUKA_DAMPINGS = (
  27.027026473513512,
  27.027026473513512,
  24.672186769721083,
  22.067474708266914,
  9.752538131173853,
  9.147747263670984,
  9.147747263670984,
)
KUKA_EFFORTS = (300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0)

HAND_STIFFNESSES = (
  6.95,
  13.2,
  4.76,
  6.62,
  0.9,
  4.76,
  6.62,
  0.9,
  0.9,
  4.76,
  6.62,
  0.9,
  0.9,
  4.76,
  6.62,
  0.9,
  0.9,
  1.38,
  4.76,
  6.62,
  0.9,
  0.9,
)
HAND_DAMPINGS = (
  0.28676845,
  0.40845109,
  0.20394083,
  0.24044435,
  0.04190723,
  0.20859232,
  0.24595532,
  0.04243185,
  0.03504461,
  0.2085923,
  0.24595532,
  0.04243185,
  0.03504461,
  0.20859226,
  0.24595528,
  0.04243183,
  0.0350446,
  0.02782345,
  0.20859229,
  0.24595528,
  0.04243183,
  0.0350446,
)
HAND_ARMATURES = (
  0.0032,
  0.0032,
  0.00265,
  0.00265,
  0.0006,
  0.00265,
  0.00265,
  0.0006,
  0.00042,
  0.00265,
  0.00265,
  0.0006,
  0.00042,
  0.00265,
  0.00265,
  0.0006,
  0.00042,
  0.00012,
  0.00265,
  0.00265,
  0.0006,
  0.00042,
)
HAND_FRICTIONLOSSES = (
  0.132,
  0.132,
  0.07456,
  0.07456,
  0.01276,
  0.07456,
  0.07456,
  0.01276,
  0.00378738,
  0.07456,
  0.07456,
  0.01276,
  0.00378738,
  0.07456,
  0.07456,
  0.01276,
  0.00378738,
  0.012,
  0.07456,
  0.07456,
  0.01276,
  0.00378738,
)
HAND_VISCOUS_DAMPINGS = (
  4.2e-05,
  4.2e-05,
  2.38e-05,
  2.38e-05,
  4.06e-06,
  2.38e-05,
  2.38e-05,
  4.06e-06,
  1.21e-06,
  2.38e-05,
  2.38e-05,
  4.06e-06,
  1.21e-06,
  2.38e-05,
  2.38e-05,
  4.06e-06,
  1.21e-06,
  4.2e-05,
  2.38e-05,
  2.38e-05,
  4.06e-06,
  1.21e-06,
)


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


def _apply_robot_friction_overrides(spec: mujoco.MjSpec) -> None:
  """Mirror SimToolReal's default/fingertip asset friction split on URDF geoms."""
  fingertip_meshes = set(FINGERTIP_LINK_NAMES) | set(FINGERTIP_MESH_NAMES)
  for geom in spec.geoms:
    is_fingertip = geom.meshname in fingertip_meshes
    friction = FINGERTIP_GEOM_FRICTION if is_fingertip else DEFAULT_GEOM_FRICTION
    geom.friction = friction
    if is_fingertip:
      geom.condim = 6


def _apply_robot_gravity_compensation(spec: mujoco.MjSpec) -> None:
  """Apply body-level MuJoCo gravity compensation to the robot only."""
  for body in spec.bodies[1:]:
    body.gravcomp = 1.0


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
  _apply_robot_friction_overrides(spec)
  _apply_robot_gravity_compensation(spec)
  return spec


def get_iiwa_sharpa_cfg() -> EntityCfg:
  actuators = tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=stiffness,
      damping=damping,
      effort_limit=effort,
    )
    for joint_name, stiffness, damping, effort in zip(
      JOINT_NAMES[:7], KUKA_STIFFNESSES, KUKA_DAMPINGS, KUKA_EFFORTS, strict=True
    )
  ) + tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=stiffness,
      damping=damping,
      armature=armature,
      frictionloss=frictionloss,
      viscous_damping=viscous_damping,
    )
    for joint_name, stiffness, damping, armature, frictionloss, viscous_damping in zip(
      JOINT_NAMES[7:],
      HAND_STIFFNESSES,
      HAND_DAMPINGS,
      HAND_ARMATURES,
      HAND_FRICTIONLOSSES,
      HAND_VISCOUS_DAMPINGS,
      strict=True,
    )
  )
  return EntityCfg(
    spec_fn=get_iiwa_sharpa_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.8, 0.0),
      joint_pos=DEFAULT_JOINT_POS,
    ),
    articulation=EntityArticulationInfoCfg(actuators=actuators),
    sort_actuators=True,
  )


def get_object_spec(
  handle_half_size: tuple[float, float, float] = (0.075, 0.0125, 0.0125),
  head_half_size: tuple[float, float, float] = (0.025, 0.025, 0.015),
  mass: float = 0.08,
  handle_shape: str = "box",
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_free_joint")
  handle_type = (
    mujoco.mjtGeom.mjGEOM_CYLINDER
    if handle_shape == "cylinder"
    else mujoco.mjtGeom.mjGEOM_BOX
  )
  handle_size = (
    (handle_half_size[1], handle_half_size[0], 0.0)
    if handle_shape == "cylinder"
    else handle_half_size
  )
  body.add_geom(
    name="object_handle_geom",
    type=handle_type,
    pos=(0.0, 0.0, 0.0),
    quat=CYLINDER_X_QUAT if handle_shape == "cylinder" else (1.0, 0.0, 0.0, 0.0),
    size=handle_size,
    mass=0.75 * mass,
    rgba=(0.45, 0.45, 0.45, 1.0),
    friction=DEFAULT_GEOM_FRICTION,
    condim=6,
  )
  body.add_geom(
    name="object_head_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(handle_half_size[0] + head_half_size[0], 0.0, 0.0),
    size=head_half_size,
    mass=0.25 * mass,
    rgba=(0.45, 0.45, 0.45, 1.0),
    friction=DEFAULT_GEOM_FRICTION,
    condim=6,
  )
  return spec


def _trimesh_to_mujoco_mesh(
  spec: mujoco.MjSpec, name: str, mesh: trimesh.Trimesh
) -> None:
  mj_mesh = spec.add_mesh()
  mj_mesh.name = name
  mj_mesh.uservert = np.asarray(mesh.vertices, dtype=np.float64).flatten().tolist()
  mj_mesh.userface = np.asarray(mesh.faces, dtype=np.int32).flatten().tolist()


def _box_mesh(extents: tuple[float, float, float]) -> trimesh.Trimesh:
  return trimesh.creation.box(extents=extents)


def _cylinder_x_mesh(length: float, diameter: float) -> trimesh.Trimesh:
  mesh = trimesh.creation.cylinder(radius=0.5 * diameter, height=length, sections=32)
  transform = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0])
  mesh.apply_transform(transform)
  return mesh


def get_object_mesh_variant_spec(
  handle_shape: str,
  handle_lengths: tuple[float, float, float],
  head_lengths: tuple[float, float, float],
  handle_density: float,
  head_density: float,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  has_head = head_lengths[0] > 1.0e-5
  head_center_x = 0.5 * handle_lengths[0] + 0.5 * max(head_lengths[0], 1.0e-4)

  handle_mesh = (
    _cylinder_x_mesh(handle_lengths[0], handle_lengths[1])
    if handle_shape == "cylinder"
    else _box_mesh(handle_lengths)
  )
  _trimesh_to_mujoco_mesh(spec, "handle_mesh", handle_mesh)
  _trimesh_to_mujoco_mesh(
    spec, "head_mesh", _box_mesh(tuple(max(v, 1.0e-4) for v in head_lengths))
  )

  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_free_joint")
  body.add_geom(
    name="object_handle_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="handle_mesh",
    pos=(0.0, 0.0, 0.0),
    density=handle_density,
    rgba=(0.45, 0.45, 0.45, 1.0),
    friction=DEFAULT_GEOM_FRICTION,
    condim=6,
  )
  body.add_geom(
    name="object_head_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="head_mesh",
    pos=(head_center_x, 0.0, 0.0),
    density=head_density if has_head else 1.0,
    rgba=(0.45, 0.45, 0.45, 1.0),
    friction=DEFAULT_GEOM_FRICTION,
    condim=6,
    contype=1 if has_head else 0,
    conaffinity=1 if has_head else 0,
  )
  return spec


def get_object_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_object_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.63),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )


def get_object_mesh_variant_cfg(
  variant_names: tuple[str, ...] | None = None,
) -> VariantEntityCfg:
  selected_variants = (
    MESH_OBJECT_VARIANTS
    if variant_names is None
    else tuple(MESH_OBJECT_VARIANT_BY_NAME[name] for name in variant_names)
  )
  return VariantEntityCfg(
    variants={
      name: VariantCfg(
        spec_fn=(
          lambda shape=shape,
          handle=handle,
          head=head,
          handle_density=handle_density,
          head_density=head_density: (
            get_object_mesh_variant_spec(
              handle_shape=shape,
              handle_lengths=handle,
              head_lengths=head,
              handle_density=handle_density,
              head_density=head_density,
            )
          )
        ),
        weight=1.0,
      )
      for name, shape, handle, head, handle_density, head_density in selected_variants
    },
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.63),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )


def get_goal_spec(
  handle_half_size: tuple[float, float, float] = (0.075, 0.0125, 0.0125),
  head_half_size: tuple[float, float, float] = (0.025, 0.025, 0.015),
  handle_shape: str = "box",
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal_object", mocap=True)
  handle_type = (
    mujoco.mjtGeom.mjGEOM_CYLINDER
    if handle_shape == "cylinder"
    else mujoco.mjtGeom.mjGEOM_BOX
  )
  handle_size = (
    (handle_half_size[1], handle_half_size[0], 0.0)
    if handle_shape == "cylinder"
    else handle_half_size
  )
  body.add_geom(
    name="goal_handle_geom",
    type=handle_type,
    pos=(0.0, 0.0, 0.0),
    quat=CYLINDER_X_QUAT if handle_shape == "cylinder" else (1.0, 0.0, 0.0, 0.0),
    size=handle_size,
    rgba=(0.1, 0.8, 0.2, 0.35),
    contype=0,
    conaffinity=0,
  )
  body.add_geom(
    name="goal_head_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(handle_half_size[0] + head_half_size[0], 0.0, 0.0),
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
      pos=(0.0, 0.0, 0.78),
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
    friction=DEFAULT_GEOM_FRICTION,
    condim=6,
  )
  return spec


def get_table_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_table_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.38),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )
