from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.simtoolreal import mdp
from mjlab.tasks.simtoolreal import policy as simtoolreal_policy
from mjlab.tasks.simtoolreal.assets import (
  ROBOT_URDF,
  get_iiwa_sharpa_spec,
  get_object_spec,
  get_table_spec,
)
from mjlab.tasks.simtoolreal.env_cfg import make_simtoolreal_env_cfg
from mjlab.tasks.simtoolreal.mdp import N_ACT, N_OBS, N_STATE

TASK_ID = "Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid"

pytestmark = pytest.mark.skipif(
  not Path(ROBOT_URDF).exists(),
  reason="SimToolReal robot assets are not available locally",
)


def test_simtoolreal_task_registered() -> None:
  assert TASK_ID in list_tasks()


def test_simtoolreal_pinky_limit_order_matches_policy_joint_order() -> None:
  expected_lower = torch.tensor([0.0, -0.1745, -0.0349, 0.0, 0.0])
  expected_upper = torch.tensor([0.2618, 1.5708, 0.0349, 1.7453, 1.3963])

  torch.testing.assert_close(mdp.Q_LOWER[24:29], expected_lower)
  torch.testing.assert_close(mdp.Q_UPPER[24:29], expected_upper)
  torch.testing.assert_close(
    torch.from_numpy(simtoolreal_policy.Q_LOWER[24:29]), expected_lower
  )
  torch.testing.assert_close(
    torch.from_numpy(simtoolreal_policy.Q_UPPER[24:29]), expected_upper
  )


def test_simtoolreal_contact_params_match_mujoco_reference() -> None:
  table_model = get_table_spec().compile()
  table_geom_id = table_model.geom("table_geom").id
  assert table_model.geom_friction[table_geom_id, 0] == pytest.approx(1.0)
  assert table_model.geom_friction[table_geom_id, 1] == pytest.approx(0.005)
  assert table_model.geom_friction[table_geom_id, 2] == pytest.approx(0.0001)
  assert table_model.geom_condim[table_geom_id] == 6

  object_model = get_object_spec().compile()
  for geom_name in ("object_handle_geom", "object_head_geom"):
    geom_id = object_model.geom(geom_name).id
    assert object_model.geom_friction[geom_id, 0] == pytest.approx(1.0)
    assert object_model.geom_condim[geom_id] == 6

  robot_model = get_iiwa_sharpa_spec().compile()
  fingertip_geom_ids = [
    geom_id
    for geom_id in range(robot_model.ngeom)
    if robot_model.geom_friction[geom_id, 0] == pytest.approx(1.5)
  ]
  assert fingertip_geom_ids
  for geom_id in fingertip_geom_ids:
    assert robot_model.geom_friction[geom_id, 0] == pytest.approx(1.5)
    assert robot_model.geom_condim[geom_id] == 6


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
  assert cfg.sim.mujoco.timestep == pytest.approx(1.0 / 180.0)
  assert cfg.decimation == 3
  assert cfg.episode_length_s == pytest.approx(10.0)
  assert cfg.observations["actor"].terms["simtoolreal"].delay_max_lag == 3
  assert cfg.observations["actor"].terms["simtoolreal"].params[
    "joint_velocity_obs_noise_std"
  ] == pytest.approx(0.01)
  assert cfg.actions["joint_pos"].use_action_delay
  assert cfg.actions["joint_pos"].action_delay_max == 3
  assert (
    cfg.events["randomize_object_size"].func
    is mdp.randomize_handle_head_equivalent_size
  )
  assert cfg.events["reset_object"].params["table_reset_z"] == pytest.approx(0.38)
  assert cfg.events["reset_object"].params["table_reset_z_range"] == pytest.approx(0.01)
  assert cfg.events["reset_object"].params["y_range"] == pytest.approx((0.0, 0.0))
  assert cfg.events["reset_robot_joints"].func is mdp.reset_robot_joints_simtoolreal
  assert cfg.events["reset_successful_goals"].func is mdp.reset_successful_goals
  assert (
    cfg.events["random_object_perturbations"].func
    is mdp.apply_random_object_perturbations
  )
  assert cfg.events["random_object_perturbations"].params[
    "force_scale"
  ] == pytest.approx(2.0)
  assert cfg.events["random_object_perturbations"].params[
    "torque_scale"
  ] == pytest.approx(0.0)
  assert cfg.events["random_object_perturbations"].params[
    "force_decay"
  ] == pytest.approx(0.99)
  assert cfg.events["reset_goal"].params["y_range"] == pytest.approx((-0.1, 0.2))
  assert cfg.events["reset_goal"].params["z_range"] == pytest.approx((0.68, 1.05))
  assert "goal_reached" not in cfg.terminations
  assert (
    cfg.terminations["object_dropped_after_lift"].func is mdp.object_dropped_after_lift
  )
  assert cfg.terminations["max_successes"].func is mdp.max_consecutive_successes_reached
  assert cfg.rewards["success"].params["tolerance"] == pytest.approx(0.075)
  assert cfg.terminations["success_update"].params["tolerance"] == pytest.approx(0.075)
  assert "fingertip_delta" in cfg.rewards
  assert cfg.curriculum["success_tolerance"].func is mdp.success_tolerance_curriculum


def test_simtoolreal_play_cfg_disables_training_noise() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  obs_term = cfg.observations["actor"].terms["simtoolreal"]

  assert cfg.scene.num_envs == 1
  assert obs_term.delay_max_lag == 0
  assert not obs_term.params["use_object_state_delay_noise"]
  assert obs_term.params["joint_velocity_obs_noise_std"] == 0.0
  assert not cfg.actions["joint_pos"].use_action_delay
  assert cfg.rewards["success"].params["tolerance"] == pytest.approx(0.01)
  assert cfg.terminations["success_update"].params["tolerance"] == pytest.approx(0.01)
  assert "success_tolerance" not in cfg.curriculum


def test_simtoolreal_success_tolerance_can_be_overridden() -> None:
  cfg = make_simtoolreal_env_cfg(play=True, success_tolerance=0.075)

  assert cfg.rewards["success"].params["tolerance"] == pytest.approx(0.075)
  assert cfg.terminations["success_update"].params["tolerance"] == pytest.approx(0.075)


def test_simtoolreal_object_distribution_types_config_threads_to_reset_event() -> None:
  cfg = make_simtoolreal_env_cfg(
    play=True,
    object_distribution_types=("simple_cuboid", "simple_cylinder"),
  )

  assert cfg.events["randomize_object_size"].params["distribution_types"] == (
    "simple_cuboid",
    "simple_cylinder",
  )


def test_simtoolreal_simple_mesh_variants_assign_true_cuboids_and_cylinders() -> None:
  cfg = make_simtoolreal_env_cfg(
    play=True,
    object_mesh_variants=True,
    object_mesh_variant_names=("simple_cuboid", "simple_cylinder"),
  )
  cfg.scene.num_envs = 4

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=13)
    state = mdp._state(env)

    assert env.sim.world_to_variant["object"].detach().cpu().tolist() == [0, 0, 1, 1]
    assert state["handle_is_cylinder"].detach().cpu().tolist() == [
      False,
      False,
      True,
      True,
    ]
    torch.testing.assert_close(
      state["head_lengths"],
      torch.zeros_like(state["head_lengths"]),
    )
    torch.testing.assert_close(
      state["handle_lengths"][:, 0],
      torch.full((4,), 0.18, device=env.device),
    )
  finally:
    env.close()


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


def test_simtoolreal_initial_object_scale_matches_default_handle_geometry() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    state = mdp._state(env)
    torch.testing.assert_close(
      state["object_scales"][0],
      state["handle_lengths"][0] / mdp.OBJECT_BASE_SIZE,
    )
  finally:
    env.close()


def test_simtoolreal_dropped_after_lift_matches_source_reset_rule() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    state = mdp._state(env)
    state["lifted_object"][0] = False
    state["initial_object_z"][0] = 0.63
    object_entity = env.scene["object"]
    pose = object_entity.data.root_link_pose_w.clone()
    pose[0, 2] = 0.62
    object_entity.write_root_link_pose_to_sim(pose)
    env.scene.write_data_to_sim()
    env.sim.forward()
    assert not mdp.object_dropped_after_lift(env)[0]

    state["lifted_object"][0] = True
    assert mdp.object_dropped_after_lift(env)[0]
  finally:
    env.close()


def test_simtoolreal_partial_reset_preserves_other_object_delay_queues() -> None:
  cfg = load_env_cfg(TASK_ID, play=False)
  cfg.scene.num_envs = 2

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    mdp._simtoolreal_kinematics(env, use_object_state_delay_noise=True)
    state = mdp._state(env)
    queue = state["object_state_queue"]
    assert queue is not None
    queue[1] = 123.0

    mdp.reset_simtoolreal_state(env, torch.tensor([0], device=env.device))

    torch.testing.assert_close(queue[1], torch.full_like(queue[1], 123.0))
    assert torch.isfinite(queue[0]).all()
    assert not torch.allclose(queue[0], torch.full_like(queue[0], 123.0))
  finally:
    env.close()


def test_simtoolreal_success_tolerance_curriculum_updates_reward_and_done_terms() -> (
  None
):
  cfg = load_env_cfg(TASK_ID, play=False)
  cfg.scene.num_envs = 2

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    state = mdp._state(env)
    state["prev_episode_successes"][:] = 3.0
    env.common_step_counter = 3000

    result = mdp.success_tolerance_curriculum(env, None)

    assert result["success_tolerance"] == pytest.approx(0.0675)
    assert env.reward_manager.get_term_cfg("success").params[
      "tolerance"
    ] == pytest.approx(0.0675)
    assert env.termination_manager.get_term_cfg("success_update").params[
      "tolerance"
    ] == pytest.approx(0.0675)
  finally:
    env.close()


def test_simtoolreal_object_distribution_filter_matches_source_ranges() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 16

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device)
    mdp.randomize_handle_head_equivalent_size(
      env,
      env_ids,
      distribution_types=("simple_cuboid", "simple_cylinder"),
    )
    state = mdp._state(env)
    handle_lengths = state["handle_lengths"]
    head_lengths = state["head_lengths"]

    assert torch.all(handle_lengths[:, 0] >= 0.10)
    assert torch.all(handle_lengths[:, 0] <= 0.25)
    assert torch.all(handle_lengths[:, 1] >= 0.03)
    assert torch.all(handle_lengths[:, 1] <= 0.07)
    assert torch.all(handle_lengths[:, 2] >= 0.03)
    assert torch.all(handle_lengths[:, 2] <= 0.07)
    assert torch.allclose(head_lengths, torch.zeros_like(head_lengths))
    torch.testing.assert_close(
      state["object_scales"],
      handle_lengths / mdp.OBJECT_BASE_SIZE,
    )
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
    state["closest_keypoint_max_dist"][0] = 0.25
    state["closest_keypoint_max_dist"][1:] = 0.75
    env.episode_length_buf[0] = 123
    old_goal = env.scene["goal"].data.root_link_pos_w[0].clone()
    mdp.reset_successful_goals(env, None)
    new_goal = env.scene["goal"].data.root_link_pos_w[0]
    assert not state["reset_goal_buf"][0]
    assert state["near_goal_steps"][0] == 0.0
    assert torch.isinf(state["closest_keypoint_max_dist"][0])
    torch.testing.assert_close(
      state["total_episode_closest_keypoint_max_dist"][0],
      torch.tensor(0.25, device=env.device),
    )
    assert env.episode_length_buf[0] == 0
    assert not torch.allclose(old_goal, new_goal)
  finally:
    env.close()


def test_simtoolreal_full_reset_preserves_completed_episode_stats_per_env() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 2

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    state = mdp._state(env)
    state["successes"][:] = torch.tensor([4.0, 8.0], device=env.device)
    state["true_objective"][:] = torch.tensor([1.04, 1.08], device=env.device)
    state["prev_total_episode_closest_keypoint_max_dist"][:] = torch.tensor(
      [0.8, 1.6], device=env.device
    )
    state["total_episode_closest_keypoint_max_dist"][:] = torch.tensor(
      [1.2, 2.4], device=env.device
    )

    mdp.reset_simtoolreal_state(env, torch.tensor([0], device=env.device))

    assert state["prev_episode_successes"][0] == 4.0
    assert state["successes"][0] == 0.0
    assert state["successes"][1] == 8.0
    assert state["prev_episode_true_objective"][0] == pytest.approx(1.04)
    assert state["prev_episode_closest_keypoint_max_dist"][0] == pytest.approx(0.2)
    assert state["total_episode_closest_keypoint_max_dist"][0] == 0.0
    assert state["total_episode_closest_keypoint_max_dist"][1] == pytest.approx(2.4)
  finally:
    env.close()
