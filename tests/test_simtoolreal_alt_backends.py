from __future__ import annotations

from mjlab.rl.simtoolreal_backend_cfg import (
  SimToolRealAltRunnerCfg,
  make_rl_games_config,
  make_simple_rl_configs,
)


def test_simtoolreal_simple_rl_configs_build_for_ppo_and_sapg() -> None:
  ppo_cfg, network_cfg = make_simple_rl_configs(
    SimToolRealAltRunnerCfg(algorithm="ppo", horizon_length=8, max_epochs=1),
    num_envs=8,
  )
  assert ppo_cfg.sapg is None
  assert ppo_cfg.seq_length == 8
  assert tuple(network_cfg.mlp.units) == (512, 256, 128)

  sapg_cfg, _ = make_simple_rl_configs(
    SimToolRealAltRunnerCfg(algorithm="sapg", horizon_length=8, max_epochs=1),
    num_envs=12,
  )
  assert sapg_cfg.sapg is not None
  assert sapg_cfg.sapg.num_conditionings == 6
  assert sapg_cfg.entropy_coef == 0.0


def test_simtoolreal_rl_games_configs_build_for_ppo_and_sapg() -> None:
  ppo = make_rl_games_config(
    SimToolRealAltRunnerCfg(algorithm="ppo", horizon_length=8, max_epochs=1),
    num_envs=8,
    device="cuda:0",
  )
  assert ppo["params"]["config"]["expl_type"] == "none"
  assert ppo["params"]["config"]["horizon_length"] == 8

  sapg = make_rl_games_config(
    SimToolRealAltRunnerCfg(algorithm="sapg", horizon_length=8, max_epochs=1),
    num_envs=12,
    device="cuda:0",
  )
  assert sapg["params"]["config"]["expl_type"] == "mixed_expl_learn_param"
  assert sapg["params"]["config"]["expl_coef_block_size"] == 2
  assert (
    sapg["params"]["network"]["space"]["continuous"]["fixed_sigma"] == "coef_cond"
  )
