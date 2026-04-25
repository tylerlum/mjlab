"""Backend configs and wrappers for SimToolReal training runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gym
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.simtoolreal.mdp import N_ACT, N_OBS

BackendName = Literal["rsl_rl", "simple_rl", "rl_games"]
AlgorithmName = Literal["ppo", "sapg"]


@dataclass(kw_only=True)
class SimToolRealAltRunnerCfg:
  """Shared config for the vendored simple_rl and rl_games runner paths."""

  backend: BackendName = "rsl_rl"
  algorithm: AlgorithmName = "ppo"
  experiment_dir: Path = Path("logs/simtoolreal_alt")
  max_epochs: int = 1
  horizon_length: int = 32
  minibatch_size: int | None = None
  mini_epochs: int = 5
  learning_rate: float = 3.0e-4
  gamma: float = 0.995
  tau: float = 0.95
  e_clip: float = 0.2
  grad_norm: float = 1.0
  critic_coef: float = 1.0
  entropy_coef: float = 0.005
  sapg_blocks: int = 6
  sapg_conditioning_dim: int = 32
  sapg_use_others_experience: bool = True


def _gym_box(shape: tuple[int, ...], low: float, high: float) -> gym.spaces.Box:
  return gym.spaces.Box(
    low=np.full(shape, low, dtype=np.float32),
    high=np.full(shape, high, dtype=np.float32),
    dtype=np.float32,
  )


class MjlabSimpleRlWrapper:
  """Expose a manager-based MJLab env through simple_rl's VecTask-like API."""

  def __init__(self, env: ManagerBasedRlEnv, obs_group: str = "actor") -> None:
    self.env = env
    self.obs_group = obs_group
    self.num_envs = env.num_envs
    self.num_states = 0
    self.device = torch.device(env.device)
    self.observation_space = _gym_box((N_OBS,), -np.inf, np.inf)
    self.action_space = _gym_box((N_ACT,), -1.0, 1.0)

  def get_env_info(self) -> dict:
    return {
      "observation_space": self.observation_space,
      "action_space": self.action_space,
      "agents": 1,
      "value_size": 1,
    }

  def reset(self) -> torch.Tensor:
    obs, _ = self.env.reset()
    return obs[self.obs_group]

  def step(
    self, actions: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    obs, rew, terminated, truncated, extras = self.env.step(actions.to(self.env.device))
    infos = dict(extras)
    infos["time_outs"] = truncated
    return obs[self.obs_group], rew, terminated | truncated, infos

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
    infos = dict(extras)
    infos["time_outs"] = truncated
    return self._obs_dict(obs), rew, terminated | truncated, infos

  def get_env_info(self) -> dict:
    return {
      "action_space": self.action_space,
      "observation_space": self.observation_space,
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


def make_simple_rl_configs(
  cfg: SimToolRealAltRunnerCfg,
  num_envs: int,
) -> tuple[object, object]:
  from simple_rl.agent import PpoConfig, SapgConfig
  from simple_rl.utils.network import MlpConfig, NetworkConfig
  from simple_rl.utils.rewards_shaper import RewardsShaperParams

  sapg = None
  if cfg.algorithm == "sapg":
    sapg = SapgConfig(
      num_conditionings=cfg.sapg_blocks,
      conditioning_dim=cfg.sapg_conditioning_dim,
      use_others_experience=cfg.sapg_use_others_experience,
      entropy_coef_scale=cfg.entropy_coef,
    )

  minibatch_size = cfg.minibatch_size or num_envs * cfg.horizon_length // 4
  ppo = PpoConfig(
    num_actors=num_envs,
    learning_rate=cfg.learning_rate,
    entropy_coef=0.0 if sapg is not None else cfg.entropy_coef,
    horizon_length=cfg.horizon_length,
    normalize_advantage=True,
    normalize_input=True,
    grad_norm=cfg.grad_norm,
    critic_coef=cfg.critic_coef,
    gamma=cfg.gamma,
    tau=cfg.tau,
    reward_shaper=RewardsShaperParams(scale_value=1.0),
    mini_epochs=cfg.mini_epochs,
    e_clip=cfg.e_clip,
    device="cuda:0",
    minibatch_size=minibatch_size,
    max_epochs=cfg.max_epochs,
    seq_length=cfg.horizon_length,
    normalize_value=True,
    truncate_grads=True,
    bounds_loss_coef=0.0,
    sapg=sapg,
    print_stats=True,
  )
  network = NetworkConfig(mlp=MlpConfig(units=(512, 256, 128)))
  return ppo, network


def make_rl_games_config(
  cfg: SimToolRealAltRunnerCfg,
  num_envs: int,
  device: str,
) -> dict:
  minibatch_size = cfg.minibatch_size or num_envs * cfg.horizon_length // 4
  expl_type = "none"
  fixed_sigma = "fixed"
  block_size = num_envs
  if cfg.algorithm == "sapg":
    expl_type = "mixed_expl_learn_param"
    fixed_sigma = "coef_cond"
    block_size = num_envs // cfg.sapg_blocks

  return {
    "params": {
      "seed": 42,
      "algo": {"name": "a2c_continuous"},
      "model": {"name": "continuous_a2c_logstd"},
      "network": {
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
          "units": [512, 256, 128],
          "activation": "elu",
          "d2rl": False,
          "initializer": {"name": "default"},
          "regularizer": {"name": "None"},
        },
      },
      "config": {
        "name": f"simtoolreal_mjlab_{cfg.algorithm}",
        "device_name": device,
        "env_name": "mjlab",
        "network_path": str(cfg.experiment_dir / "rl_games_nn"),
        "log_path": str(cfg.experiment_dir / "rl_games_log"),
        "ppo": True,
        "mixed_precision": False,
        "normalize_input": True,
        "normalize_value": True,
        "normalize_advantage": True,
        "reward_shaper": {"scale_value": 1.0},
        "num_actors": num_envs,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "learning_rate": cfg.learning_rate,
        "lr_schedule": None,
        "clip_value": True,
        "bounds_loss_coef": 0.0,
        "schedule_type": "legacy",
        "entropy_coef": 0.0 if cfg.algorithm == "sapg" else cfg.entropy_coef,
        "e_clip": cfg.e_clip,
        "minibatch_size": minibatch_size,
        "mini_epochs": cfg.mini_epochs,
        "critic_coef": cfg.critic_coef,
        "grad_norm": cfg.grad_norm,
        "truncate_grads": True,
        "horizon_length": cfg.horizon_length,
        "seq_length": cfg.horizon_length,
        "max_epochs": cfg.max_epochs,
        "score_to_win": 1_000_000,
        "save_best_after": 100,
        "save_frequency": 0,
        "print_stats": True,
        "use_others_experience": "none",
        "off_policy_ratio": 1.0,
        "expl_type": expl_type,
        "expl_reward_coef_embd_size": cfg.sapg_conditioning_dim,
        "expl_reward_coef_scale": cfg.entropy_coef,
        "expl_reward_type": "none",
        "expl_coef_block_size": block_size,
        "good_reset_boundary": 0,
      },
    }
  }
