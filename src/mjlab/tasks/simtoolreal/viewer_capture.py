"""Pose-based SimToolReal viewer capture for training runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.simtoolreal.assets import JOINT_NAMES, read_robot_urdf_for_viewer
from mjlab.tasks.simtoolreal.mdp import N_ACT, _state


@dataclass(kw_only=True)
class SimToolRealViewerCaptureCfg:
  enabled: bool = False
  output_dir: Path = Path("videos")
  capture_freq: int = 6000
  capture_len: int = 600
  env_index: int = 0
  wandb_key: str = "interactive_viewer"
  log_to_wandb: bool = True
  github_raw_base: str = "https://raw.githubusercontent.com/tylerlum/simtoolreal/main/"
  url_check: str = "warn"
  object_urdf_relpath: str | None = None
  default_show_frame_axes: bool = False


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
    env_index = self.cfg.env_index
    env_origin = (
      self.env.scene.env_origins[env_index].detach().cpu().numpy().astype(np.float32)
    )

    def local_pose(pose_wxyz: torch.Tensor) -> np.ndarray:
      pose = self._pose_xyzw(pose_wxyz)
      pose[:3] -= env_origin
      return pose

    return {
      "robot_joint_pos": robot.data.joint_pos[env_index, :N_ACT].detach().cpu().numpy(),
      "robot_base_pose": local_pose(robot.data.root_link_pose_w[env_index]),
      "object_pose": local_pose(obj.data.root_link_pose_w[env_index]),
      "goal_pose": local_pose(goal.data.root_link_pose_w[env_index]),
      "table_pose": local_pose(table.data.root_link_pose_w[env_index]),
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
    handle_is_cylinder: bool = False,
  ) -> str:
    hx, hy, hz = handle_size
    tx, ty, tz = head_size
    handle_geometry = (
      f'<cylinder radius="{0.5 * hy}" length="{hx}"/>'
      if handle_is_cylinder
      else f'<box size="{hx} {hy} {hz}"/>'
    )
    handle_origin = (
      f'<origin xyz="0 0 0" rpy="0 {np.pi / 2.0} 0"/>'
      if handle_is_cylinder
      else '<origin xyz="0 0 0"/>'
    )
    if tx <= 1.0e-5:
      return f"""<robot name="{name}">
  <link name="{name}">
    <visual>
      {handle_origin}
      <geometry>{handle_geometry}</geometry>
    </visual>
    <collision>
      {handle_origin}
      <geometry>{handle_geometry}</geometry>
    </collision>
  </link>
</robot>"""
    head_x = 0.5 * hx + 0.5 * tx
    handle_origin = (
      f'<origin xyz="0 0 0" rpy="0 {np.pi / 2.0} 0"/>'
      if handle_is_cylinder
      else '<origin xyz="0 0 0"/>'
    )
    return f"""<robot name="{name}">
  <link name="{name}">
    <visual>
      {handle_origin}
      <geometry>{handle_geometry}</geometry>
    </visual>
    <visual>
      <origin xyz="{head_x} 0 0"/>
      <geometry><box size="{tx} {ty} {tz}"/></geometry>
    </visual>
    <collision>
      {handle_origin}
      <geometry>{handle_geometry}</geometry>
    </collision>
    <collision>
      <origin xyz="{head_x} 0 0"/>
      <geometry><box size="{tx} {ty} {tz}"/></geometry>
    </collision>
  </link>
</robot>"""

  def _github_raw_url(self, relpath: str) -> str:
    base = self.cfg.github_raw_base.strip()
    if not base:
      base = "https://raw.githubusercontent.com/tylerlum/simtoolreal/main/"
    return base.rstrip("/") + "/" + relpath.lstrip("/")

  @staticmethod
  def _check_viewer_urls(urls: list[str], url_check: str) -> set[str]:
    """Validate URL-backed viewer assets with IsaacGym-style warn/error/skip modes."""
    import time
    import urllib.request

    mode = url_check.lower()
    if mode not in {"warn", "error", "skip"}:
      raise ValueError(f"url_check must be one of warn/error/skip, got {url_check!r}")
    failed: set[str] = set()
    if mode == "skip":
      print(
        "[SimToolReal viewer] URL check skipped. "
        "URL-backed mesh objects may fail to load in the browser."
      )
      return failed

    for url in dict.fromkeys(urls):
      print(f"[SimToolReal viewer] URL check ({mode}): {url}")
      start = time.monotonic()
      try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
      except Exception as exc:
        failed.add(url)
        msg = (
          "\n" + "=" * 72 + "\n"
          "[SimToolReal viewer] URL CHECK FAILED\n"
          f"  URL   : {url}\n"
          f"  Error : {exc}\n"
          "  Common fixes: push the commit containing the asset, set "
          "github_raw_base to a reachable branch/commit, or set url_check='skip'.\n"
          + "="
          * 72
        )
        if mode == "error":
          raise ValueError(msg) from exc
        print(msg)
      else:
        print(f"[SimToolReal viewer]   passed in {time.monotonic() - start:.2f}s")
    return failed

  def _make_object_viewer_robots(
    self,
    make_embedded_robot,
    make_url_robot,
    object_urdf: str,
  ) -> tuple[dict, dict]:
    """Use inline primitive URDFs by default; use GitHub raw URLs for mesh objects."""
    if self.cfg.object_urdf_relpath is None:
      print(
        "[SimToolReal viewer] Using embedded primitive handle/head URDF for "
        "object and goal; no GitHub object URL needed."
      )
      return (
        make_embedded_robot(
          name="object",
          urdf_text=object_urdf,
          animated=False,
          color_override=(0.1, 0.45, 0.95),
        ),
        make_embedded_robot(
          name="goal",
          urdf_text=object_urdf,
          animated=False,
          color_override=(0.1, 0.8, 0.25),
        ),
      )

    object_url = self._github_raw_url(self.cfg.object_urdf_relpath)
    print(
      "[SimToolReal viewer] Using URL-backed mesh object for object and goal:\n"
      f"  {object_url}"
    )
    failed_urls = self._check_viewer_urls([object_url], self.cfg.url_check)
    if object_url in failed_urls:
      fallback_url = self._github_raw_url(
        "assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf"
      )
      print(
        "[SimToolReal viewer] Falling back to claw_hammer visual because the "
        "configured object URL failed. Object shape will be wrong until the "
        "URL is fixed.\n"
        f"  failed   : {object_url}\n"
        f"  fallback : {fallback_url}"
      )
      object_url = fallback_url
    return (
      make_url_robot(
        name="object",
        urdf_url=object_url,
        animated=False,
        color_override=(0.1, 0.45, 0.95),
      ),
      make_url_robot(
        name="goal",
        urdf_url=object_url,
        animated=False,
        color_override=(0.1, 0.8, 0.25),
      ),
    )

  def _finalize_capture(self) -> None:
    assert self._frames is not None
    from mjlab.tasks.simtoolreal.interactive_viewer import (
      create_html,
      make_embedded_robot,
      make_url_robot,
    )

    robot_urdf = read_robot_urdf_for_viewer()
    sim_state = _state(self.env)
    env_index = self.cfg.env_index
    handle_size = tuple(sim_state["handle_lengths"][env_index].detach().cpu().tolist())
    head_size = tuple(sim_state["head_lengths"][env_index].detach().cpu().tolist())
    handle_is_cylinder = bool(
      sim_state["handle_is_cylinder"][env_index].detach().cpu().item()
    )
    object_urdf = self._handle_head_urdf(
      "object", handle_size, head_size, handle_is_cylinder
    )
    object_robot, goal_robot = self._make_object_viewer_robots(
      make_embedded_robot,
      make_url_robot,
      object_urdf,
    )
    robots = [
      make_embedded_robot(
        name="robot",
        urdf_text=robot_urdf,
        animated=True,
      ),
      object_robot,
      goal_robot,
      make_embedded_robot(
        name="table",
        urdf_text=self._box_urdf("table", (0.475, 0.4, 0.3)),
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
      default_show_frame_axes=self.cfg.default_show_frame_axes,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = (
      self.cfg.output_dir / f"{timestamp}_env{env_index}_viewer_{self.step_count}.html"
    )
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
