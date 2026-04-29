"""Capture private SimToolReal rl_games checkpoint rollouts in MJLab."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from gym import spaces
from omegaconf import OmegaConf

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.simtoolreal import N_ACT, N_OBS
from mjlab.tasks.simtoolreal.assets import JOINT_NAMES, read_robot_urdf_for_viewer
from mjlab.tasks.simtoolreal.env_cfg import make_simtoolreal_env_cfg
from mjlab.tasks.simtoolreal.interactive_viewer import create_html, make_embedded_robot
from mjlab.tasks.simtoolreal.mdp import _state
from mjlab.tasks.simtoolreal.viewer_capture import (
  SimToolRealViewerCaptureCfg,
  SimToolRealViewerCaptureWrapper,
)
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner

TASK_ID = "Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid"
DEFAULT_POLICY_DIR = Path(
  "/home/tylerlum/github_repos/simtoolreal_private/pretrained_policy"
)


class _DummyRlGamesEnv:
  """Small env stub used only to construct the rl_games player."""

  observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(N_OBS,),
    dtype=np.float32,
  )
  action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ACT,), dtype=np.float32)

  def __init__(self, num_envs: int = 1) -> None:
    self.num_envs = num_envs

  def get_env_info(self) -> dict[str, Any]:
    return {
      "observation_space": self.observation_space,
      "action_space": self.action_space,
      "agents": 1,
      "value_size": 1,
    }

  def set_env_state(self, *args: Any, **kwargs: Any) -> None:
    del args, kwargs


class PrivatePretrainedPolicy:
  """rl_games player for ``simtoolreal_private/pretrained_policy/model.pth``."""

  def __init__(
    self,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    num_envs: int = 1,
  ) -> None:
    self.device = device
    self.player = self._load_player(config_path, checkpoint_path, num_envs)

  def reset(self) -> None:
    self.player.reset()

  @torch.inference_mode()
  def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    if obs.shape[-1] != N_OBS:
      raise ValueError(
        f"Expected actor observation shape (*, {N_OBS}), got {obs.shape}"
      )
    obs = obs.to(self.device)
    sapg_conditioning = torch.full(
      (obs.shape[0], 1),
      50.0,
      dtype=obs.dtype,
      device=obs.device,
    )
    policy_obs = torch.cat([obs, sapg_conditioning], dim=-1)
    action = self.player.get_action(policy_obs, is_deterministic=deterministic)
    return action.reshape(-1, N_ACT)

  def _load_player(self, config_path: Path, checkpoint_path: Path, num_envs: int):
    cfg = _read_private_rl_games_cfg(config_path)
    env_configurations.register(
      "rlgpu",
      {
        "env_creator": lambda **kwargs: _DummyRlGamesEnv(num_envs),
        "vecenv_type": "RLGPU",
      },
    )
    train_cfg = cfg["train"]
    train_cfg["load_path"] = str(checkpoint_path)
    train_cfg["params"]["config"]["device"] = self.device
    train_cfg["params"]["config"]["device_name"] = self.device
    train_cfg["params"]["config"].setdefault("player", {})["device_name"] = self.device
    runner = Runner()
    runner.load(train_cfg)
    player = runner.create_player()
    player.init_rnn()
    player.has_batch_dimension = True
    player.restore(str(checkpoint_path))
    return player


def _read_private_rl_games_cfg(config_path: Path) -> dict[str, Any]:
  def eval_resolver(expr: str) -> Any:
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307

  def resolve_default(default: Any, value: Any) -> Any:
    return default if value in (None, "", "null") else value

  resolvers = {
    "eval": eval_resolver,
    "eq": lambda left, right: left == right,
    "if": lambda condition, true_value, false_value: (
      true_value if condition else false_value
    ),
    "resolve_default": resolve_default,
  }
  for name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(name):
      OmegaConf.register_new_resolver(name, resolver)
  with config_path.open("r", encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)
  return OmegaConf.to_container(OmegaConf.create(raw_cfg), resolve=True)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-rollouts", type=int, default=3)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=360)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--object-mesh-variants",
    action="store_true",
    help="Use upstream MJLab per-world mesh variants for mixed cuboid/cylinder objects.",
  )
  parser.add_argument(
    "--capture-all-envs",
    action="store_true",
    help="Write one HTML trajectory for every parallel env.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/simtoolreal_private_pretrained_rollouts"),
  )
  parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample actions instead of using deterministic policy means.",
  )
  return parser.parse_args()


def _make_env(
  device: str,
  output_dir: Path,
  steps: int,
  num_envs: int,
  object_mesh_variants: bool,
) -> SimToolRealViewerCaptureWrapper:
  env_cfg = (
    make_simtoolreal_env_cfg(play=True, object_mesh_variants=True)
    if object_mesh_variants
    else load_env_cfg(TASK_ID, play=True)
  )
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  return SimToolRealViewerCaptureWrapper(
    env,
    SimToolRealViewerCaptureCfg(
      enabled=False,
      output_dir=output_dir,
      capture_freq=steps + 1,
      capture_len=steps,
      log_to_wandb=False,
    ),
  )


def _object_summary(env: ManagerBasedRlEnv) -> str:
  sim_state = _state(env)
  handle = sim_state["handle_lengths"][0].detach().cpu().numpy()
  head = sim_state["head_lengths"][0].detach().cpu().numpy()
  scale = sim_state["object_scales"][0].detach().cpu().numpy()
  return (
    f"handle={np.array2string(handle, precision=4)} "
    f"head={np.array2string(head, precision=4)} "
    f"scale={np.array2string(scale, precision=4)}"
  )


def _pose_xyzw(pose_wxyz: torch.Tensor) -> np.ndarray:
  pose = pose_wxyz.detach().cpu().numpy().astype(np.float32)
  return np.concatenate([pose[:3], pose[[4, 5, 6, 3]]], axis=0)


def _box_urdf(name: str, size: tuple[float, float, float]) -> str:
  sx, sy, sz = size
  return f"""<robot name="{name}">
  <link name="{name}">
    <visual><geometry><box size="{sx} {sy} {sz}"/></geometry></visual>
    <collision><geometry><box size="{sx} {sy} {sz}"/></geometry></collision>
  </link>
</robot>"""


def _handle_head_urdf(
  name: str,
  handle_size: tuple[float, float, float],
  head_size: tuple[float, float, float],
  handle_is_cylinder: bool,
) -> str:
  hx, hy, hz = handle_size
  tx, ty, tz = head_size
  geometry = (
    f'<cylinder radius="{0.5 * hy}" length="{hx}"/>'
    if handle_is_cylinder
    else f'<box size="{hx} {hy} {hz}"/>'
  )
  total_x = hx + max(tx, 0.0)
  handle_x = -0.5 * total_x + 0.5 * hx if tx > 1.0e-5 else 0.0
  handle_origin = (
    f'<origin xyz="{handle_x} 0 0" rpy="0 {np.pi / 2.0} 0"/>'
    if handle_is_cylinder
    else f'<origin xyz="{handle_x} 0 0"/>'
  )
  if tx <= 1.0e-5:
    return f"""<robot name="{name}">
  <link name="{name}">
    <visual>{handle_origin}<geometry>{geometry}</geometry></visual>
    <collision>{handle_origin}<geometry>{geometry}</geometry></collision>
  </link>
</robot>"""
  head_x = 0.5 * total_x - 0.5 * tx
  return f"""<robot name="{name}">
  <link name="{name}">
    <visual>{handle_origin}<geometry>{geometry}</geometry></visual>
    <visual><origin xyz="{head_x} 0 0"/><geometry><box size="{tx} {ty} {tz}"/></geometry></visual>
    <collision>{handle_origin}<geometry>{geometry}</geometry></collision>
    <collision><origin xyz="{head_x} 0 0"/><geometry><box size="{tx} {ty} {tz}"/></geometry></collision>
  </link>
</robot>"""


def _capture_frame(env: ManagerBasedRlEnv, env_idx: int) -> dict[str, np.ndarray]:
  robot = env.scene["robot"]
  obj = env.scene["object"]
  goal = env.scene["goal"]
  table = env.scene["table"]
  return {
    "robot_joint_pos": robot.data.joint_pos[env_idx, :N_ACT].detach().cpu().numpy(),
    "robot_base_pose": _pose_xyzw(robot.data.root_link_pose_w[env_idx]),
    "object_pose": _pose_xyzw(obj.data.root_link_pose_w[env_idx]),
    "goal_pose": _pose_xyzw(goal.data.root_link_pose_w[env_idx]),
    "table_pose": _pose_xyzw(table.data.root_link_pose_w[env_idx]),
  }


def _write_html(
  env: ManagerBasedRlEnv,
  frames: list[dict[str, np.ndarray]],
  output_dir: Path,
  rollout_idx: int,
  env_idx: int,
) -> Path:
  sim_state = _state(env)
  handle_size = tuple(sim_state["handle_lengths"][env_idx].detach().cpu().tolist())
  head_size = tuple(sim_state["head_lengths"][env_idx].detach().cpu().tolist())
  handle_is_cylinder = bool(
    sim_state["handle_is_cylinder"][env_idx].detach().cpu().item()
  )
  object_urdf = _handle_head_urdf(
    "object", handle_size, head_size, handle_is_cylinder
  )
  goal_urdf = _handle_head_urdf("goal", handle_size, head_size, handle_is_cylinder)
  robots = [
    make_embedded_robot(
      name="robot", urdf_text=read_robot_urdf_for_viewer(), animated=True
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
      urdf_text=_box_urdf("table", (0.475, 0.4, 0.3)),
      animated=False,
      color_override=(0.6, 0.6, 0.6),
    ),
  ]
  html = create_html(
    joint_names=list(JOINT_NAMES),
    robot_joint_positions=np.stack([f["robot_joint_pos"] for f in frames]),
    robots=robots,
    object_poses={
      "object": np.stack([f["object_pose"] for f in frames]),
      "goal": np.stack([f["goal_pose"] for f in frames]),
      "table": np.stack([f["table_pose"] for f in frames]),
    },
    robot_base_poses=np.stack([f["robot_base_pose"] for f in frames]),
    dt=float(env.step_dt),
    robot_name="robot",
  )
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  shape = "cylinder" if handle_is_cylinder else "cuboid"
  path = output_dir / f"{timestamp}_rollout{rollout_idx}_env{env_idx}_{shape}.html"
  path.write_text(html, encoding="utf-8")
  return path


def _run_one(
  policy: PrivatePretrainedPolicy,
  rollout_idx: int,
  args: argparse.Namespace,
) -> None:
  env = _make_env(
    args.device,
    args.output_dir,
    args.steps,
    args.num_envs,
    args.object_mesh_variants,
  )
  try:
    obs, _ = env.reset(seed=args.seed + rollout_idx)
    policy.reset()
    obs, _, _, _, _ = env.step(torch.zeros((args.num_envs, N_ACT), device=env.device))
    print(
      f"rollout={rollout_idx} seed={args.seed + rollout_idx} "
      f"{_object_summary(env.unwrapped)}"
    )
    capture_env_ids = (
      list(range(args.num_envs)) if args.capture_all_envs else [env.cfg.env_index]
    )
    frames = {
      env_idx: [_capture_frame(env.unwrapped, env_idx)] for env_idx in capture_env_ids
    }
    final_reward = 0.0
    done_steps = 0
    for _ in range(args.steps):
      action = policy.act(obs["actor"], deterministic=not args.stochastic)
      obs, rew, terminated, truncated, _ = env.step(action.to(env.device))
      final_reward = float(rew[0].detach().cpu())
      done_steps += int(bool((terminated | truncated)[0].detach().cpu()))
      for env_idx in capture_env_ids:
        frames[env_idx].append(_capture_frame(env.unwrapped, env_idx))
    print(f"rollout={rollout_idx} final_reward={final_reward:.4f} resets={done_steps}")
    for env_idx, env_frames in frames.items():
      path = _write_html(env.unwrapped, env_frames, args.output_dir, rollout_idx, env_idx)
      print(f"rollout={rollout_idx} env={env_idx} html={path}")
  finally:
    env.close()


def main() -> None:
  args = parse_args()
  config_path = args.policy_dir / "config.yaml"
  checkpoint_path = args.policy_dir / "model.pth"
  if not config_path.exists():
    raise FileNotFoundError(f"Missing private pretrained config: {config_path}")
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Missing private pretrained checkpoint: {checkpoint_path}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  policy = PrivatePretrainedPolicy(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    device=args.device,
    num_envs=args.num_envs,
  )
  for rollout_idx in range(args.num_rollouts):
    _run_one(policy, rollout_idx, args)
  print(f"Saved MJLab private-checkpoint HTML rollouts under {args.output_dir}")


if __name__ == "__main__":
  main()
