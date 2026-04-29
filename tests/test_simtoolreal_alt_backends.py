from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.simtoolreal_backend_cfg import (
  N_STATE,
  MjlabRlGamesVecEnv,
  MjlabSimpleRlWrapper,
  SimToolRealAltRunnerCfg,
  make_rl_games_config,
  make_simple_rl_configs,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.simtoolreal import mdp
from mjlab.tasks.simtoolreal.assets import ROBOT_URDF
from mjlab.tasks.simtoolreal.mdp import N_ACT

TASK_ID = "Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid"


def test_simtoolreal_simple_rl_configs_build_for_ppo_and_sapg() -> None:
  ppo_cfg, network_cfg = make_simple_rl_configs(
    SimToolRealAltRunnerCfg(algorithm="ppo", horizon_length=8, max_epochs=1),
    num_envs=8,
  )
  assert ppo_cfg.sapg is None
  assert ppo_cfg.seq_length == 8
  assert tuple(network_cfg.mlp.units) == (1024, 1024, 512, 512)
  assert network_cfg.rnn is not None
  assert network_cfg.rnn.name == "lstm"
  assert ppo_cfg.reward_shaper.scale_value == 0.01
  assert ppo_cfg.minibatch_size == 98_304
  assert ppo_cfg.mini_epochs == 2
  assert ppo_cfg.asymmetric_critic is not None
  assert ppo_cfg.asymmetric_critic.minibatch_size == 98_304
  assert N_STATE == 162

  sapg_cfg, _ = make_simple_rl_configs(
    SimToolRealAltRunnerCfg(algorithm="sapg", horizon_length=8, max_epochs=1),
    num_envs=12,
  )
  assert sapg_cfg.sapg is not None
  assert sapg_cfg.sapg.num_conditionings == 6
  assert sapg_cfg.sapg.entropy_coef_scale == 0.005
  assert sapg_cfg.entropy_coef == 0.0


def test_simtoolreal_rl_games_configs_build_for_ppo_and_sapg() -> None:
  ppo = make_rl_games_config(
    SimToolRealAltRunnerCfg(algorithm="ppo", horizon_length=8, max_epochs=1),
    num_envs=8,
    device="cuda:0",
  )
  assert ppo["params"]["config"]["expl_type"] == "none"
  assert ppo["params"]["config"]["device"] == "cuda:0"
  assert ppo["params"]["config"]["device_name"] == "cuda:0"
  assert ppo["params"]["config"]["train_dir"].endswith("logs/simtoolreal_alt/rl_games")
  assert ppo["params"]["config"]["horizon_length"] == 8
  assert ppo["params"]["config"]["reward_shaper"]["scale_value"] == 0.01
  assert ppo["params"]["config"]["minibatch_size"] == 98_304
  assert ppo["params"]["config"]["mini_epochs"] == 2
  assert ppo["params"]["network"]["mlp"]["units"] == [1024, 1024, 512, 512]
  assert ppo["params"]["network"]["rnn"]["name"] == "lstm"
  assert ppo["params"]["config"]["central_value_config"] is not None
  assert ppo["params"]["config"]["central_value_config"]["minibatch_size"] == 98_304

  sapg = make_rl_games_config(
    SimToolRealAltRunnerCfg(algorithm="sapg", horizon_length=8, max_epochs=1),
    num_envs=12,
    device="cuda:0",
  )
  assert sapg["params"]["config"]["expl_type"] == "mixed_expl_learn_param"
  assert sapg["params"]["config"]["expl_coef_block_size"] == 2
  assert sapg["params"]["config"]["use_others_experience"] == "lf"
  assert sapg["params"]["config"]["expl_reward_type"] == "entropy"
  assert sapg["params"]["config"]["expl_reward_coef_scale"] == 0.005
  assert (
    sapg["params"]["network"]["space"]["continuous"]["fixed_sigma"] == "coef_cond"
  )


@pytest.mark.skipif(
  not Path(ROBOT_URDF).exists(),
  reason="SimToolReal robot assets are not available locally",
)
def test_simtoolreal_alt_wrappers_emit_finite_task_infos() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 2

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    state = mdp._state(env)
    state["closest_keypoint_max_dist_fixed_size"][:] = float("inf")
    state["closest_fingertip_dist"][:] = float("inf")
    action = torch.zeros((env.num_envs, N_ACT), device=env.device)

    simple = MjlabSimpleRlWrapper(env)
    _, _, _, simple_infos = simple.step(action)
    assert {
      "successes",
      "success_ratio",
      "all_goals_hit_ratio",
      "true_objective",
      "closest_keypoint_max_dist",
      "current_closest_keypoint_max_dist",
      "success_tolerance",
    } <= set(simple_infos)
    assert torch.isfinite(simple_infos["closest_keypoint_max_dist"]).all()
    assert torch.isfinite(simple_infos["current_closest_keypoint_max_dist"]).all()
    assert torch.isfinite(simple_infos["closest_fingertip_dist"]).all()
    assert "episode_cumulative" in simple_infos
    assert "reward" in simple_infos["episode_cumulative"]

    rl_games = MjlabRlGamesVecEnv(env)
    _, _, _, rl_games_infos = rl_games.step(action)
    assert torch.isfinite(rl_games_infos["closest_keypoint_max_dist"]).all()
    assert torch.isfinite(rl_games_infos["closest_fingertip_dist"]).all()
  finally:
    env.close()
