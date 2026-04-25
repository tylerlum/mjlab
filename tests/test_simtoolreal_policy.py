from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.tasks.simtoolreal import (
  N_ACT,
  N_OBS,
  SimToolRealBrowserEnv,
  SimToolRealBrowserEnvCfg,
  SimToolRealOnnxPolicy,
  reset_object_and_goal,
  sim_step,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = REPO_ROOT.parent / "simtoolreal.github.io"
SCENE_PATH = SITE_ROOT / "mujoco_wasm/assets/scenes/iiwa_sharpa.xml"
POLICY_PATH = SITE_ROOT / "mujoco_wasm/dist-desktop/policy_iiwa_sharpa.onnx"


pytestmark = pytest.mark.skipif(
  not SCENE_PATH.exists() or not POLICY_PATH.exists(),
  reason="SimToolReal website MuJoCo assets are not available locally",
)


def _make_policy() -> tuple[mujoco.MjModel, mujoco.MjData, SimToolRealOnnxPolicy]:
  model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
  data = mujoco.MjData(model)
  policy = SimToolRealOnnxPolicy(model, POLICY_PATH)
  policy.reset(data)
  reset_object_and_goal(model, data, policy.cache)
  return model, data, policy


def test_simtoolreal_browser_observation_shape_and_finiteness() -> None:
  _, data, policy = _make_policy()

  obs = policy.observation(data)

  assert obs.shape == (N_OBS,)
  assert np.isfinite(obs).all()


def test_simtoolreal_onnx_policy_rollout_smoke() -> None:
  model, data, policy = _make_policy()
  policy_decimation = max(1, round((1.0 / 60.0) / model.opt.timestep))

  for step in range(120):
    sim_step(model, data, policy, step, policy_decimation)

  assert policy.latest_action.shape == (N_ACT,)
  assert np.isfinite(policy.latest_action).all()
  assert np.isfinite(data.qpos).all()


def test_simtoolreal_browser_env_pretrained_step() -> None:
  env = SimToolRealBrowserEnv(
    SimToolRealBrowserEnvCfg(scene_path=SCENE_PATH, policy_path=POLICY_PATH)
  )
  obs = env.reset()

  next_obs, reward, done, info = env.step_pretrained_policy()

  assert obs.shape == (N_OBS,)
  assert next_obs.shape == (N_OBS,)
  assert np.isfinite(next_obs).all()
  assert np.isfinite(reward)
  assert isinstance(done, bool)
  assert "keypoint_max_dist" in info
