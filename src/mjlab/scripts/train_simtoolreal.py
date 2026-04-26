"""Train SimToolReal with selectable RSL-RL, simple_rl, or rl_games backends."""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime

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
from mjlab.tasks.simtoolreal.viewer_capture import (
  SimToolRealViewerCaptureCfg,
  SimToolRealViewerCaptureWrapper,
)
from mjlab.utils.wrappers import VideoRecorder

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


def _maybe_start_wandb(cfg: SimToolRealTrainCli) -> None:
  alt = cfg.alt
  if not alt.wandb_activate:
    return
  import wandb

  if wandb.run is not None:
    return
  run_name = alt.wandb_name or (
    f"simtoolreal_mjlab_{cfg.backend}_{alt.algorithm}_"
    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
  )
  wandb.init(
    project=alt.wandb_project,
    entity=alt.wandb_entity,
    group=alt.wandb_group,
    name=run_name,
    config={
      "backend": cfg.backend,
      "algorithm": alt.algorithm,
      "num_envs": cfg.num_envs,
      "device": cfg.device,
      "alt": alt.__dict__,
    },
  )


def _maybe_finish_wandb(cfg: SimToolRealTrainCli) -> None:
  if not cfg.alt.wandb_activate:
    return
  import wandb

  if wandb.run is not None:
    wandb.finish()


def _make_env(cfg: SimToolRealTrainCli) -> ManagerBasedRlEnv:
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(
    cfg=env_cfg,
    device=cfg.device,
    render_mode="rgb_array" if cfg.alt.capture_video else None,
  )
  if cfg.alt.capture_video:
    env = VideoRecorder(
      env,
      video_folder=cfg.alt.experiment_dir / "videos" / "train",
      step_trigger=lambda step: step % cfg.alt.capture_video_freq == 0,
      video_length=cfg.alt.capture_video_len,
      disable_logger=True,
    )
  if cfg.alt.capture_viewer:
    env = SimToolRealViewerCaptureWrapper(
      env,
      SimToolRealViewerCaptureCfg(
        enabled=True,
        output_dir=cfg.alt.experiment_dir / "videos" / "train",
        capture_freq=cfg.alt.capture_viewer_freq,
        capture_len=cfg.alt.capture_viewer_len,
        log_to_wandb=cfg.alt.wandb_activate,
      ),
    )
  return env


def _run_simple_rl(cfg: SimToolRealTrainCli) -> None:
  from simple_rl.agent import Agent

  alt = cfg.alt
  alt.experiment_dir.mkdir(parents=True, exist_ok=True)
  _maybe_start_wandb(cfg)
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
    _maybe_finish_wandb(cfg)


def _run_rl_games(cfg: SimToolRealTrainCli) -> None:
  from rl_games.torch_runner import Runner

  alt = cfg.alt
  alt.experiment_dir.mkdir(parents=True, exist_ok=True)
  _maybe_start_wandb(cfg)
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
    _maybe_finish_wandb(cfg)


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
