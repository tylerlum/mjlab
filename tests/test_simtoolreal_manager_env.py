from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.simtoolreal import mdp
from mjlab.tasks.simtoolreal.assets import ROBOT_URDF
from mjlab.tasks.simtoolreal.env_cfg import make_simtoolreal_env_cfg
from mjlab.tasks.simtoolreal.mdp import N_ACT, N_OBS, N_STATE

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
    assert obs["critic"].shape == (2, N_STATE)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(obs["critic"]).all()
    assert env.action_manager.total_action_dim == N_ACT
    action_term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    state = mdp._state(env)
    torch.testing.assert_close(
      state["initial_object_z"],
      env.scene["object"].data.root_link_pos_w[:, 2],
    )
    torch.testing.assert_close(
      action_term.prev_targets,
      robot.data.joint_pos[:, action_term._joint_ids],
    )

    action = torch.zeros((env.num_envs, N_ACT), device=env.device)
    obs, reward, terminated, truncated, _ = env.step(action)

    assert obs["actor"].shape == (2, N_OBS)
    assert obs["critic"].shape == (2, N_STATE)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(obs["critic"]).all()
    assert torch.isfinite(reward).all()
    assert not terminated.any()
    assert not truncated.any()
  finally:
    env.close()


def test_simtoolreal_training_cfg_matches_source_cadence_and_randomization() -> None:
  cfg = load_env_cfg(TASK_ID, play=False)

  assert cfg.scene.num_envs == 8192
  assert cfg.scene.env_spacing == pytest.approx(1.2)
  assert cfg.sim.mujoco.timestep == pytest.approx(1.0 / 120.0)
  assert cfg.decimation == 2
  assert cfg.episode_length_s == pytest.approx(10.0)
  assert cfg.observations["actor"].terms["simtoolreal"].delay_max_lag == 3
  assert (
    cfg.observations["actor"]
    .terms["simtoolreal"]
    .params["joint_velocity_obs_noise_std"]
    == pytest.approx(0.01)
  )
  assert cfg.actions["joint_pos"].use_action_delay
  assert cfg.actions["joint_pos"].action_delay_max == 3
  assert cfg.events["randomize_object_size"].func is mdp.randomize_handle_head_equivalent_size
  assert cfg.events["reset_object"].params["table_reset_z"] == pytest.approx(0.38)
  assert cfg.events["reset_object"].params["table_reset_z_range"] == pytest.approx(0.01)
  assert cfg.events["reset_object"].params["y_range"] == pytest.approx((0.0, 0.0))
  assert cfg.events["reset_robot_joints"].func is mdp.reset_robot_joints_simtoolreal
  assert cfg.events["reset_successful_goals"].func is mdp.reset_successful_goals
  assert cfg.events["random_object_perturbations"].func is mdp.apply_random_object_perturbations
  assert cfg.events["random_object_perturbations"].params["force_scale"] == pytest.approx(2.0)
  assert cfg.events["random_object_perturbations"].params["torque_scale"] == pytest.approx(0.0)
  assert cfg.events["random_object_perturbations"].params["force_decay"] == pytest.approx(0.99)
  assert cfg.events["reset_goal"].params["y_range"] == pytest.approx((-0.1, 0.2))
  assert cfg.events["reset_goal"].params["z_range"] == pytest.approx((0.68, 1.05))
  assert "goal_reached" not in cfg.terminations
  assert cfg.terminations["max_successes"].func is mdp.max_consecutive_successes_reached
  assert "fingertip_delta" in cfg.rewards


def test_simtoolreal_play_cfg_disables_training_noise() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  obs_term = cfg.observations["actor"].terms["simtoolreal"]

  assert cfg.scene.num_envs == 1
  assert obs_term.delay_max_lag == 0
  assert not obs_term.params["use_object_state_delay_noise"]
  assert obs_term.params["joint_velocity_obs_noise_std"] == 0.0
  assert not cfg.actions["joint_pos"].use_action_delay


def test_simtoolreal_object_uses_separate_handle_and_head_geoms() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 2

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    obj = env.scene["object"]
    goal = env.scene["goal"]
    _, object_geom_names = obj.find_geoms(
      ("object_handle_geom", "object_head_geom"), preserve_order=True
    )
    _, goal_geom_names = goal.find_geoms(
      ("goal_handle_geom", "goal_head_geom"), preserve_order=True
    )
    assert tuple(object_geom_names) == ("object_handle_geom", "object_head_geom")
    assert tuple(goal_geom_names) == ("goal_handle_geom", "goal_head_geom")
    state = mdp._state(env)
    assert torch.all(state["handle_lengths"] > 0.0)
    assert torch.all(state["object_scales"][:, 0] > 0.0)
  finally:
    env.close()


def test_simtoolreal_mesh_variants_assign_cuboids_and_cylinders() -> None:
  cfg = make_simtoolreal_env_cfg(play=True, object_mesh_variants=True)
  cfg.scene.num_envs = 4

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset(seed=11)
    state = mdp._state(env)

    assert env.sim.world_to_variant["object"].detach().cpu().tolist() == [0, 1, 2, 3]
    assert state["handle_is_cylinder"].detach().cpu().tolist() == [
      False,
      True,
      False,
      True,
    ]
    goal = env.scene["goal"]
    goal_geom_ids = goal.indexing.geom_ids[mdp.GOAL_GEOM_CFG.geom_ids]
    torch.testing.assert_close(
      env.sim.model.geom_pos[:, goal_geom_ids[0], :3].detach().cpu(),
      torch.zeros(4, 3),
    )
    expected_head_x = 0.5 * (
      state["handle_lengths"][:, 0] + state["head_lengths"].clamp_min(1.0e-4)[:, 0]
    )
    torch.testing.assert_close(
      env.sim.model.geom_pos[:, goal_geom_ids[1], 0].detach().cpu(),
      expected_head_x.detach().cpu(),
    )
    assert torch.all(state["object_masses"] > 0.0)
    assert torch.isfinite(obs["actor"]).all()

    action = torch.zeros((env.num_envs, N_ACT), device=env.device)
    obs, reward, terminated, truncated, _ = env.step(action)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()
    assert not terminated.any()
    assert not truncated.any()
  finally:
    env.close()


def test_simtoolreal_success_resamples_goal_without_episode_termination() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    state = mdp._state(env)
    state["reset_goal_buf"][0] = True
    state["near_goal_steps"][0] = 10.0
    env.episode_length_buf[0] = 123
    old_goal = env.scene["goal"].data.root_link_pos_w[0].clone()
    mdp.reset_successful_goals(env, None)
    new_goal = env.scene["goal"].data.root_link_pos_w[0]
    assert not state["reset_goal_buf"][0]
    assert state["near_goal_steps"][0] == 0.0
    assert env.episode_length_buf[0] == 0
    assert not torch.allclose(old_goal, new_goal)
  finally:
    env.close()
