"""Train SimToolReal with selectable RSL-RL, simple_rl, or rl_games backends."""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from dataclasses import dataclass, field, replace

import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.simtoolreal_backend_cfg import (
  BackendName,
  MjlabRlGamesVecEnv,
  MjlabSimpleRlWrapper,
  SimToolRealAltRunnerCfg,
  make_rl_games_config,
  make_simple_rl_configs,
)

TASK_ID = "Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid"


@dataclass(frozen=True)
class SimToolRealTrainCli:
  backend: BackendName = "rsl_rl"
  """Training backend to use."""

  num_envs: int = 1024
  """Number of vectorized MJLab environments."""

  device: str = "cuda:0"
  """Device for MJLab/Warp and the alternate RL backends."""

  alt: SimToolRealAltRunnerCfg = field(default_factory=SimToolRealAltRunnerCfg)
  """Config used when backend is simple_rl or rl_games."""


def _run_rsl_rl(cfg: SimToolRealTrainCli) -> None:
  from mjlab.scripts.train import TrainConfig, launch_training

  args = TrainConfig.from_task(TASK_ID)
  args.env.scene.num_envs = cfg.num_envs
  launch_training(task_id=TASK_ID, args=args)


def _normalize_alt_cfg(cfg: SimToolRealTrainCli) -> SimToolRealTrainCli:
  alt = replace(cfg.alt, backend=cfg.backend)
  num_envs = cfg.num_envs
  if cfg.backend in ("simple_rl", "rl_games") and alt.algorithm == "sapg":
    remainder = num_envs % alt.sapg_blocks
    if remainder:
      num_envs += alt.sapg_blocks - remainder
      print(
        f"[SimToolReal] Adjusted num_envs to {num_envs} so SAPG can split "
        f"evenly across {alt.sapg_blocks} blocks."
      )
  return replace(cfg, num_envs=num_envs, alt=alt)


def _make_env(cfg: SimToolRealTrainCli) -> ManagerBasedRlEnv:
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs
  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)


def _run_simple_rl(cfg: SimToolRealTrainCli) -> None:
  from simple_rl.agent import Agent

  alt = cfg.alt
  alt.experiment_dir.mkdir(parents=True, exist_ok=True)
  env = _make_env(cfg)
  wrapper = MjlabSimpleRlWrapper(env)
  ppo_cfg, network_cfg = make_simple_rl_configs(alt, num_envs=env.num_envs)
  ppo_cfg.device = cfg.device
  agent = Agent(
    experiment_dir=alt.experiment_dir / f"simple_rl_{alt.algorithm}",
    ppo_config=ppo_cfg,
    network_config=network_cfg,
    env=wrapper,
  )
  try:
    agent.train()
  finally:
    wrapper.close()


def _run_rl_games(cfg: SimToolRealTrainCli) -> None:
  from rl_games.torch_runner import Runner

  alt = cfg.alt
  alt.experiment_dir.mkdir(parents=True, exist_ok=True)
  env = _make_env(cfg)
  vec_env = MjlabRlGamesVecEnv(env)
  runner_cfg = make_rl_games_config(alt, num_envs=env.num_envs, device=cfg.device)
  runner = Runner()
  runner.load(runner_cfg)
  runner.params["config"]["vec_env"] = vec_env
  try:
    runner.run({"train": True, "play": False, "checkpoint": None, "sigma": None})
  finally:
    vec_env.close()


def main() -> None:
  warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.*` is deprecated.*",
    category=FutureWarning,
  )
  cfg = _normalize_alt_cfg(tyro.cli(SimToolRealTrainCli))
  os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
  os.environ.setdefault("ONNXRUNTIME_LOG_SEVERITY_LEVEL", "3")
  with contextlib.redirect_stderr(io.StringIO()):
    import mjlab.tasks  # noqa: F401

  if cfg.backend == "rsl_rl":
    _run_rsl_rl(cfg)
  elif cfg.backend == "simple_rl":
    _run_simple_rl(cfg)
  elif cfg.backend == "rl_games":
    _run_rl_games(cfg)
  else:
    raise ValueError(f"Unsupported backend: {cfg.backend}")


if __name__ == "__main__":
  main()
