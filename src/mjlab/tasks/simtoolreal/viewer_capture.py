"""Pose-based SimToolReal viewer capture for training runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.simtoolreal.assets import JOINT_NAMES, ROBOT_URDF
from mjlab.tasks.simtoolreal.mdp import N_ACT, OBJECT_BASE_SIZE, _state


@dataclass(kw_only=True)
class SimToolRealViewerCaptureCfg:
  enabled: bool = False
  output_dir: Path = Path("videos")
  capture_freq: int = 6000
  capture_len: int = 600
  wandb_key: str = "interactive_viewer"
  log_to_wandb: bool = True


class SimToolRealViewerCaptureWrapper:
  """Wrap an MJLab env and periodically log a SimToolReal interactive HTML clip."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: SimToolRealViewerCaptureCfg,
  ) -> None:
    self.env = env
    self.cfg = cfg
    self.step_count = 0
    self._frames: list[dict[str, np.ndarray]] | None = None
    self._armed = False
    self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  def reset(self, *args, **kwargs):
    result = self.env.reset(*args, **kwargs)
    if self.cfg.enabled and self._armed and self._frames is None:
      self._frames = []
      self._armed = False
    return result

  def step(self, action):
    result = self.env.step(action)
    if self.cfg.enabled:
      self._maybe_capture()
    self.step_count += 1
    return result

  def close(self) -> None:
    self.env.close()

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env

  def _maybe_capture(self) -> None:
    if self._frames is None and self.step_count % self.cfg.capture_freq == 0:
      self._armed = True
      self._frames = []
    if self._frames is None:
      return

    self._frames.append(self._capture_frame())
    if len(self._frames) >= self.cfg.capture_len:
      self._finalize_capture()

  def _pose_xyzw(self, pose_wxyz: torch.Tensor) -> np.ndarray:
    pose = pose_wxyz.detach().cpu().numpy().astype(np.float32)
    return np.concatenate([pose[:3], pose[[4, 5, 6, 3]]], axis=0)

  def _capture_frame(self) -> dict[str, np.ndarray]:
    robot = self.env.scene["robot"]
    obj = self.env.scene["object"]
    goal = self.env.scene["goal"]
    table = self.env.scene["table"]
    return {
      "robot_joint_pos": robot.data.joint_pos[0, :N_ACT].detach().cpu().numpy(),
      "robot_base_pose": self._pose_xyzw(robot.data.root_link_pose_w[0]),
      "object_pose": self._pose_xyzw(obj.data.root_link_pose_w[0]),
      "goal_pose": self._pose_xyzw(goal.data.root_link_pose_w[0]),
      "table_pose": self._pose_xyzw(table.data.root_link_pose_w[0]),
    }

  def _box_urdf(self, name: str, size: tuple[float, float, float]) -> str:
    sx, sy, sz = size
    return f"""<robot name="{name}">
  <link name="{name}">
    <visual><geometry><box size="{sx} {sy} {sz}"/></geometry></visual>
    <collision><geometry><box size="{sx} {sy} {sz}"/></geometry></collision>
  </link>
</robot>"""

  def _handle_head_urdf(
    self,
    name: str,
    handle_size: tuple[float, float, float],
    head_size: tuple[float, float, float],
  ) -> str:
    hx, hy, hz = handle_size
    tx, ty, tz = head_size
    if tx <= 1.0e-5:
      return self._box_urdf(name, handle_size)
    total_x = hx + tx
    handle_x = -0.5 * total_x + 0.5 * hx
    head_x = 0.5 * total_x - 0.5 * tx
    return f"""<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="{handle_x} 0 0"/>
      <geometry><box size="{hx} {hy} {hz}"/></geometry>
    </visual>
    <visual>
      <origin xyz="{head_x} 0 0"/>
      <geometry><box size="{tx} {ty} {tz}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{handle_x} 0 0"/>
      <geometry><box size="{hx} {hy} {hz}"/></geometry>
    </collision>
    <collision>
      <origin xyz="{head_x} 0 0"/>
      <geometry><box size="{tx} {ty} {tz}"/></geometry>
    </collision>
  </link>
</robot>"""

  def _finalize_capture(self) -> None:
    assert self._frames is not None
    from mjlab.tasks.simtoolreal.interactive_viewer import (
      create_html,
      make_embedded_robot,
    )

    robot_urdf = Path(ROBOT_URDF).read_text(encoding="utf-8")
    object_size = tuple(
      (OBJECT_BASE_SIZE * _state(self.env)["object_scales"][0]).detach().cpu().tolist()
    )
    sim_state = _state(self.env)
    handle_size = tuple(sim_state["handle_lengths"][0].detach().cpu().tolist())
    head_size = tuple(sim_state["head_lengths"][0].detach().cpu().tolist())
    object_urdf = (
      self._handle_head_urdf("object", handle_size, head_size)
      if any(x > 1.0e-5 for x in head_size)
      else self._box_urdf("object", object_size)
    )
    goal_urdf = (
      self._handle_head_urdf("goal", handle_size, head_size)
      if any(x > 1.0e-5 for x in head_size)
      else self._box_urdf("goal", object_size)
    )
    robots = [
      make_embedded_robot(
        name="robot",
        urdf_text=robot_urdf,
        animated=True,
      ),
      make_embedded_robot(
        name="object",
        urdf_text=object_urdf,
        animated=False,
        color_override=(0.1, 0.45, 0.95),
      ),
      make_embedded_robot(
        name="goal",
        urdf_text=goal_urdf,
        animated=False,
        color_override=(0.1, 0.8, 0.25),
      ),
      make_embedded_robot(
        name="table",
        urdf_text=self._box_urdf("table", (0.8, 1.0, 0.02)),
        animated=False,
        color_override=(0.6, 0.6, 0.6),
      ),
    ]
    html = create_html(
      joint_names=list(JOINT_NAMES),
      robot_joint_positions=np.stack([f["robot_joint_pos"] for f in self._frames]),
      robots=robots,
      object_poses={
        "object": np.stack([f["object_pose"] for f in self._frames]),
        "goal": np.stack([f["goal_pose"] for f in self._frames]),
        "table": np.stack([f["table_pose"] for f in self._frames]),
      },
      robot_base_poses=np.stack([f["robot_base_pose"] for f in self._frames]),
      dt=float(self.env.step_dt),
      robot_name="robot",
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = self.cfg.output_dir / f"{timestamp}_viewer_{self.step_count}.html"
    path.write_text(html, encoding="utf-8")
    if self.cfg.log_to_wandb:
      try:
        import wandb

        if wandb.run is not None:
          wandb.log({self.cfg.wandb_key: wandb.Html(html)}, step=self.step_count)
      except Exception as exc:  # pragma: no cover - best-effort logging.
        print(f"[SimToolReal] Viewer W&B logging skipped: {exc}")
    print(f"[SimToolReal] Saved interactive viewer capture to {path}")
    self._frames = None
    self._armed = False
