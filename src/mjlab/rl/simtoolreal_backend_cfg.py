"""Backend configs and wrappers for SimToolReal training runners."""
# ruff: noqa: E402, I001

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from pathlib import Path
from typing import Any, Literal

with contextlib.redirect_stderr(io.StringIO()):
  import gym
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

BackendName = Literal["rsl_rl", "simple_rl", "rl_games"]
AlgorithmName = Literal["ppo", "sapg"]
N_ACT = 29
N_OBS = 140
N_STATE = 162


@dataclass(kw_only=True)
class SimToolRealAltRunnerCfg:
  """Shared config for the vendored simple_rl and rl_games runner paths."""

  backend: BackendName = "rsl_rl"
  algorithm: AlgorithmName = "ppo"
  experiment_dir: Path = Path("logs/simtoolreal_alt")
  max_epochs: int = 1_000_000
  max_frames: int = 100_000_000_000_000
  horizon_length: int = 16
  minibatch_size: int | None = None
  mini_epochs: int | None = None
  learning_rate: float = 1.0e-4
  gamma: float = 0.99
  tau: float = 0.95
  e_clip: float = 0.1
  grad_norm: float = 1.0
  critic_coef: float = 4.0
  entropy_coef: float = 0.0
  reward_scale: float = 0.01
  bounds_loss_coef: float = 0.0001
  mixed_precision: bool = True
  normalize_input: bool = True
  normalize_value: bool = True
  normalize_advantage: bool = True
  lr_schedule: Literal["adaptive", "linear", "none"] = "adaptive"
  schedule_type: Literal["legacy", "standard"] = "standard"
  kl_threshold: float = 0.016
  save_best_after: int = 100
  save_frequency: int = 3000
  games_to_track: int = 3000
  use_asymmetric_critic: bool = True
  use_lstm: bool = True
  sapg_blocks: int = 6
  sapg_conditioning_dim: int = 32
  sapg_use_others_experience: bool = True
  sapg_off_policy_ratio: float = 1.0
  sapg_entropy_coef_scale: float = 0.005
  wandb_activate: bool = False
  wandb_project: str = "simtoolreal_mjlab"
  wandb_entity: str | None = None
  wandb_group: str | None = None
  wandb_name: str | None = None
  capture_viewer: bool = False
  capture_viewer_freq: int = 6000
  capture_viewer_len: int = 600
  capture_video: bool = False
  capture_video_freq: int = 6000
  capture_video_len: int = 600


def _gym_box(shape: tuple[int, ...], low: float, high: float) -> gym.spaces.Box:
  return gym.spaces.Box(
    low=np.full(shape, low, dtype=np.float32),
    high=np.full(shape, high, dtype=np.float32),
    dtype=np.float32,
  )


def _unwrap_env(env: Any) -> ManagerBasedRlEnv:
  while hasattr(env, "env"):
    next_env = env.env
    if next_env is env or not hasattr(next_env, "step"):
      break
    env = next_env
  return env


def _finite_progress(value: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
  return torch.where(torch.isinf(value), fallback, value)


def _simtoolreal_infos(
  env: Any,
  extras: dict,
  rew: torch.Tensor,
  truncated: torch.Tensor,
) -> dict:
  """Build the SimToolReal info keys expected by the copied trainer code."""
  from mjlab.tasks.simtoolreal import mdp

  infos = dict(extras)
  base_env = _unwrap_env(env)
  state = mdp._state(base_env)
  kin = mdp._simtoolreal_kinematics(base_env, use_object_state_delay_noise=False)
  closest_keypoint = _finite_progress(
    state["closest_keypoint_max_dist_fixed_size"], kin["keypoint_max_dist_fixed"]
  )
  closest_fingertip = _finite_progress(
    state["closest_fingertip_dist"], kin["fingertip_dist"]
  )

  reward_terms = {}
  if hasattr(base_env.reward_manager, "active_terms"):
    for idx, name in enumerate(base_env.reward_manager.active_terms):
      reward_terms[name] = base_env.reward_manager._step_reward[:, idx].detach().clone()

  infos.update(
    {
      "time_outs": truncated,
      "successes": state["successes"].detach().clone(),
      "true_objective": state["successes"].detach().clone(),
      "closest_keypoint_max_dist": closest_keypoint.detach().clone(),
      "closest_fingertip_dist": closest_fingertip.detach().clone(),
      "keypoint_max_dist": kin["keypoint_max_dist_fixed"].detach().clone(),
      "near_goal": state["near_goal"].float().detach().clone(),
      "near_goal_steps": state["near_goal_steps"].detach().clone(),
      "lifted_object": state["lifted_object"].float().detach().clone(),
      "reward": rew.detach().clone(),
      "episode_cumulative": {"reward": rew.detach().clone(), **reward_terms},
    }
  )
  return infos


class MjlabSimpleRlWrapper:
  """Expose a manager-based MJLab env through simple_rl's VecTask-like API."""

  def __init__(self, env: ManagerBasedRlEnv, obs_group: str = "actor") -> None:
    self.env = env
    self.obs_group = obs_group
    self.state_group = "critic"
    self.num_envs = env.num_envs
    self.num_states = N_STATE
    self.device = torch.device(env.device)
    self.observation_space = _gym_box((N_OBS,), -np.inf, np.inf)
    self.state_space = _gym_box((N_STATE,), -np.inf, np.inf)
    self.action_space = _gym_box((N_ACT,), -1.0, 1.0)

  def get_env_info(self) -> dict:
    return {
      "observation_space": self.observation_space,
      "state_space": self.state_space,
      "action_space": self.action_space,
      "agents": 1,
      "value_size": 1,
    }

  def _obs_dict(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"obs": obs[self.obs_group], "states": obs[self.state_group]}

  def reset(self) -> dict[str, torch.Tensor]:
    obs, _ = self.env.reset()
    return self._obs_dict(obs)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    obs, rew, terminated, truncated, extras = self.env.step(actions.to(self.env.device))
    infos = _simtoolreal_infos(self.env, extras, rew, truncated)
    return self._obs_dict(obs), rew, terminated | truncated, infos

  def set_train_info(self, env_frames: int, *args, **kwargs) -> None:
    del env_frames, args, kwargs

  def get_env_state(self) -> None:
    return None

  def set_env_state(self, env_state: object) -> None:
    del env_state

  def close(self) -> None:
    self.env.close()


class MjlabRlGamesVecEnv:
  """Expose a manager-based MJLab env through rl_games' IVecEnv API."""

  class _EnvDevice:
    def __init__(self, device: str) -> None:
      self.device = device

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    obs_group: str = "actor",
    state_group: str = "critic",
  ) -> None:
    from rl_games.common.ivecenv import IVecEnv

    if not isinstance(self, IVecEnv):
      pass
    self.unwrapped = env
    self.env = self._EnvDevice(env.device)
    self.obs_group = obs_group
    self.state_group = state_group
    self.num_envs = env.num_envs
    self.observation_space = _gym_box((N_OBS,), -np.inf, np.inf)
    self.state_space = _gym_box((N_STATE,), -np.inf, np.inf)
    self.action_space = _gym_box((N_ACT,), -1.0, 1.0)

  def _obs_dict(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"obs": obs[self.obs_group], "states": obs[self.state_group]}

  def reset(self) -> dict[str, torch.Tensor]:
    obs, _ = self.unwrapped.reset()
    return self._obs_dict(obs)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    obs, rew, terminated, truncated, extras = self.unwrapped.step(
      actions.to(self.unwrapped.device)
    )
    infos = _simtoolreal_infos(self.unwrapped, extras, rew, truncated)
    return self._obs_dict(obs), rew, terminated | truncated, infos

  def get_env_info(self) -> dict:
    return {
      "action_space": self.action_space,
      "observation_space": self.observation_space,
      "state_space": self.state_space,
      "agents": 1,
      "value_size": 1,
    }

  def get_number_of_agents(self) -> int:
    return 1

  def set_train_info(self, env_frames: int, *args, **kwargs) -> None:
    del env_frames, args, kwargs

  def get_env_state(self) -> None:
    return None

  def set_env_state(self, env_state: object) -> None:
    del env_state

  def close(self) -> None:
    self.unwrapped.close()


def _default_minibatch_size(cfg: SimToolRealAltRunnerCfg) -> int:
  if cfg.algorithm == "sapg":
    return 16_384
  if cfg.use_lstm and cfg.use_asymmetric_critic:
    return 98_304
  return 32_768


def _default_mini_epochs(cfg: SimToolRealAltRunnerCfg) -> int:
  if cfg.algorithm == "sapg":
    return 2
  if cfg.use_lstm and cfg.use_asymmetric_critic:
    return 2
  return 4


def make_simple_rl_configs(
  cfg: SimToolRealAltRunnerCfg,
  num_envs: int,
) -> tuple[object, object]:
  from simple_rl.agent import PpoConfig, SapgConfig
  from simple_rl.utils.asymmetric_critic import AsymmetricCriticConfig
  from simple_rl.utils.network import MlpConfig, NetworkConfig, RnnConfig
  from simple_rl.utils.rewards_shaper import RewardsShaperParams

  sapg = None
  if cfg.algorithm == "sapg":
    sapg = SapgConfig(
      num_conditionings=cfg.sapg_blocks,
      conditioning_dim=cfg.sapg_conditioning_dim,
      use_others_experience=cfg.sapg_use_others_experience,
      off_policy_ratio=int(cfg.sapg_off_policy_ratio),
      use_entropy_bonus=True,
      entropy_coef_scale=cfg.sapg_entropy_coef_scale,
    )

  minibatch_size = cfg.minibatch_size or _default_minibatch_size(cfg)
  mini_epochs = cfg.mini_epochs or _default_mini_epochs(cfg)
  lr_schedule = None if cfg.lr_schedule == "none" else cfg.lr_schedule
  network = NetworkConfig(
    mlp=MlpConfig(units=(1024, 1024, 512, 512)),
    rnn=RnnConfig(
      name="lstm",
      units=1024,
      layers=1,
      before_mlp=True,
      layer_norm=True,
    )
    if cfg.use_lstm
    else None,
  )
  asymmetric_critic = None
  if cfg.use_asymmetric_critic:
    asymmetric_minibatch_size = cfg.minibatch_size or _default_minibatch_size(cfg)
    asymmetric_critic = AsymmetricCriticConfig(
      name="asymmetric_critic",
      learning_rate=cfg.learning_rate,
      mini_epochs=mini_epochs,
      normalize_input=cfg.normalize_input,
      truncate_grads=True,
      minibatch_size=asymmetric_minibatch_size,
      lr_schedule=None,
      network=NetworkConfig(
        mlp=MlpConfig(units=(1024, 1024, 512, 512)),
        asymmetric_critic=True,
      ),
      grad_norm=cfg.grad_norm,
      e_clip=cfg.e_clip,
    )
  ppo = PpoConfig(
    num_actors=num_envs,
    learning_rate=cfg.learning_rate,
    entropy_coef=cfg.entropy_coef,
    horizon_length=cfg.horizon_length,
    normalize_advantage=cfg.normalize_advantage,
    normalize_input=cfg.normalize_input,
    grad_norm=cfg.grad_norm,
    critic_coef=cfg.critic_coef,
    gamma=cfg.gamma,
    tau=cfg.tau,
    reward_shaper=RewardsShaperParams(scale_value=cfg.reward_scale),
    mini_epochs=mini_epochs,
    e_clip=cfg.e_clip,
    device="cuda:0",
    minibatch_size=minibatch_size,
    max_epochs=cfg.max_epochs,
    max_frames=cfg.max_frames,
    seq_length=cfg.horizon_length,
    normalize_value=cfg.normalize_value,
    truncate_grads=True,
    mixed_precision=cfg.mixed_precision,
    bounds_loss_coef=cfg.bounds_loss_coef,
    lr_schedule=lr_schedule,
    schedule_type=cfg.schedule_type,
    kl_threshold=cfg.kl_threshold if lr_schedule == "adaptive" else None,
    save_frequency=cfg.save_frequency,
    save_best_after=cfg.save_best_after,
    games_to_track=cfg.games_to_track,
    asymmetric_critic=asymmetric_critic,
    sapg=sapg,
    print_stats=True,
  )
  return ppo, network


def make_rl_games_config(
  cfg: SimToolRealAltRunnerCfg,
  num_envs: int,
  device: str,
) -> dict:
  minibatch_size = cfg.minibatch_size or _default_minibatch_size(cfg)
  mini_epochs = cfg.mini_epochs or _default_mini_epochs(cfg)
  expl_type = "none"
  fixed_sigma = "fixed"
  block_size = num_envs
  use_others_experience = "none"
  expl_reward_type = "rnd"
  expl_reward_coef_scale = 1.0
  if cfg.algorithm == "sapg":
    expl_type = "mixed_expl_learn_param"
    fixed_sigma = "coef_cond"
    block_size = num_envs // cfg.sapg_blocks
    use_others_experience = "lf" if cfg.sapg_use_others_experience else "none"
    expl_reward_type = "entropy"
    expl_reward_coef_scale = cfg.sapg_entropy_coef_scale

  lr_schedule = None if cfg.lr_schedule == "none" else cfg.lr_schedule
  network = {
    "name": "actor_critic",
    "separate": False,
    "space": {
      "continuous": {
        "mu_activation": "None",
        "sigma_activation": "None",
        "mu_init": {"name": "default"},
        "sigma_init": {"name": "const_initializer", "val": 0},
        "fixed_sigma": fixed_sigma,
      }
    },
    "mlp": {
      "units": [1024, 1024, 512, 512],
      "activation": "elu",
      "d2rl": False,
      "initializer": {"name": "default"},
      "regularizer": {"name": "None"},
    },
  }
  if cfg.use_lstm:
    network["rnn"] = {
      "name": "lstm",
      "units": 1024,
      "layers": 1,
      "before_mlp": True,
      "layer_norm": True,
    }

  central_value_config = None
  if cfg.use_asymmetric_critic:
    central_value_minibatch_size = cfg.minibatch_size or _default_minibatch_size(cfg)
    central_value_config = {
      "minibatch_size": central_value_minibatch_size,
      "mini_epochs": 2,
      "learning_rate": cfg.learning_rate,
      "kl_threshold": cfg.kl_threshold,
      "clip_value": True,
      "normalize_input": cfg.normalize_input,
      "truncate_grads": True,
      "network": {
        "name": "actor_critic",
        "central_value": True,
        "mlp": {
          "units": [1024, 1024, 512, 512],
          "activation": "elu",
          "d2rl": False,
          "initializer": {"name": "default"},
          "regularizer": {"name": "None"},
        },
      },
    }

  return {
    "params": {
      "seed": 42,
      "algo": {"name": "a2c_continuous"},
      "model": {"name": "continuous_a2c_logstd"},
      "network": network,
      "config": {
        "name": f"simtoolreal_mjlab_{cfg.algorithm}",
        "train_dir": str(cfg.experiment_dir / "rl_games"),
        "device_name": device,
        "device": device,
        "env_name": "mjlab",
        "network_path": str(cfg.experiment_dir / "rl_games_nn"),
        "log_path": str(cfg.experiment_dir / "rl_games_log"),
        "ppo": True,
        "mixed_precision": cfg.mixed_precision,
        "normalize_input": cfg.normalize_input,
        "normalize_value": cfg.normalize_value,
        "normalize_advantage": cfg.normalize_advantage,
        "reward_shaper": {"scale_value": cfg.reward_scale},
        "num_actors": num_envs,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "learning_rate": cfg.learning_rate,
        "lr_schedule": lr_schedule,
        "kl_threshold": cfg.kl_threshold,
        "clip_value": True,
        "bounds_loss_coef": cfg.bounds_loss_coef,
        "schedule_type": cfg.schedule_type,
        "entropy_coef": cfg.entropy_coef,
        "e_clip": cfg.e_clip,
        "minibatch_size": minibatch_size,
        "mini_epochs": mini_epochs,
        "critic_coef": cfg.critic_coef,
        "grad_norm": cfg.grad_norm,
        "truncate_grads": True,
        "horizon_length": cfg.horizon_length,
        "seq_length": cfg.horizon_length,
        "max_epochs": cfg.max_epochs,
        "max_frames": cfg.max_frames,
        "score_to_win": 1_000_000,
        "save_best_after": cfg.save_best_after,
        "save_frequency": cfg.save_frequency,
        "print_stats": True,
        "use_others_experience": use_others_experience,
        "off_policy_ratio": cfg.sapg_off_policy_ratio,
        "expl_type": expl_type,
        "expl_reward_coef_embd_size": cfg.sapg_conditioning_dim,
        "expl_reward_coef_scale": expl_reward_coef_scale,
        "expl_reward_type": expl_reward_type,
        "expl_coef_block_size": block_size,
        "good_reset_boundary": 0,
        "central_value_config": central_value_config,
      },
    }
  }
