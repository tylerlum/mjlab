from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.simtoolreal import mdp
from mjlab.tasks.simtoolreal.assets import ROBOT_URDF
from mjlab.tasks.simtoolreal.mdp import N_ACT, N_OBS

TASK_ID = "Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid"

pytestmark = pytest.mark.skipif(
  not Path(ROBOT_URDF).exists(),
  reason="SimToolReal robot assets are not available locally",
)


def test_simtoolreal_task_registered() -> None:
  assert TASK_ID in list_tasks()


def test_simtoolreal_manager_env_reset_step_smoke() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 2

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, N_OBS)
    assert torch.isfinite(obs["actor"]).all()
    assert env.action_manager.total_action_dim == N_ACT

    action = torch.zeros((env.num_envs, N_ACT), device=env.device)
    obs, reward, terminated, truncated, _ = env.step(action)

    assert obs["actor"].shape == (2, N_OBS)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()
    assert not terminated.any()
    assert not truncated.any()
  finally:
    env.close()


def test_simtoolreal_training_cfg_matches_source_cadence_and_randomization() -> None:
  cfg = load_env_cfg(TASK_ID, play=False)

  assert cfg.sim.mujoco.timestep == pytest.approx(1.0 / 120.0)
  assert cfg.decimation == 2
  assert cfg.episode_length_s == pytest.approx(10.0)
  assert cfg.observations["actor"].terms["simtoolreal"].delay_max_lag == 3
  assert cfg.actions["joint_pos"].use_action_delay
  assert cfg.actions["joint_pos"].action_delay_max == 3
  assert cfg.events["randomize_object_size"].func is mdp.randomize_handle_head_equivalent_size
  assert cfg.events["random_object_perturbations"].func is mdp.apply_random_object_perturbations
  assert "fingertip_delta" in cfg.rewards


def test_simtoolreal_play_cfg_disables_training_noise() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  obs_term = cfg.observations["actor"].terms["simtoolreal"]

  assert cfg.scene.num_envs == 1
  assert obs_term.delay_max_lag == 0
  assert not obs_term.params["use_object_state_delay_noise"]
  assert obs_term.params["joint_velocity_obs_noise_std"] == 0.0
  assert not cfg.actions["joint_pos"].use_action_delay
