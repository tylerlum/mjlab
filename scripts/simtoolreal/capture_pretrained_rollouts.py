"""Capture private SimToolReal rl_games checkpoint rollouts in MJLab."""

from __future__ import annotations

import argparse
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
  num_envs = 1

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
  ) -> None:
    self.device = device
    self.player = self._load_player(config_path, checkpoint_path)

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

  def _load_player(self, config_path: Path, checkpoint_path: Path):
    cfg = _read_private_rl_games_cfg(config_path)
    env_configurations.register(
      "rlgpu",
      {"env_creator": lambda **kwargs: _DummyRlGamesEnv(), "vecenv_type": "RLGPU"},
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
  parser.add_argument("--steps", type=int, default=360)
  parser.add_argument("--seed", type=int, default=0)
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
  device: str, output_dir: Path, steps: int
) -> SimToolRealViewerCaptureWrapper:
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  return SimToolRealViewerCaptureWrapper(
    env,
    SimToolRealViewerCaptureCfg(
      enabled=True,
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


def _run_one(
  policy: PrivatePretrainedPolicy,
  rollout_idx: int,
  args: argparse.Namespace,
) -> None:
  env = _make_env(args.device, args.output_dir, args.steps)
  try:
    obs, _ = env.reset(seed=args.seed + rollout_idx)
    policy.reset()
    print(
      f"rollout={rollout_idx} seed={args.seed + rollout_idx} "
      f"{_object_summary(env.unwrapped)}"
    )
    final_reward = 0.0
    done_steps = 0
    for _ in range(args.steps):
      action = policy.act(obs["actor"], deterministic=not args.stochastic)
      obs, rew, terminated, truncated, _ = env.step(action.to(env.device))
      final_reward = float(rew[0].detach().cpu())
      done_steps += int(bool((terminated | truncated)[0].detach().cpu()))
    print(f"rollout={rollout_idx} final_reward={final_reward:.4f} resets={done_steps}")
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
  )
  for rollout_idx in range(args.num_rollouts):
    _run_one(policy, rollout_idx, args)
  print(f"Saved MJLab private-checkpoint HTML rollouts under {args.output_dir}")


if __name__ == "__main__":
  main()
