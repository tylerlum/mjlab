from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
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
