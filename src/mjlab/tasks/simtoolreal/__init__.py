"""SimToolReal task port utilities."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.simtoolreal.browser_env import (
  SimToolRealBrowserEnv,
  SimToolRealBrowserEnvCfg,
)
from mjlab.tasks.simtoolreal.env_cfg import make_simtoolreal_env_cfg
from mjlab.tasks.simtoolreal.policy import (
  N_ACT,
  N_OBS,
  SimToolRealOnnxPolicy,
  print_rollout_summary,
  reset_object_and_goal,
  sim_step,
)
from mjlab.tasks.simtoolreal.rl_cfg import simtoolreal_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid",
  env_cfg=make_simtoolreal_env_cfg(),
  play_env_cfg=make_simtoolreal_env_cfg(play=True),
  rl_cfg=simtoolreal_ppo_runner_cfg(),
)

__all__ = [
  "N_ACT",
  "N_OBS",
  "SimToolRealBrowserEnv",
  "SimToolRealBrowserEnvCfg",
  "SimToolRealOnnxPolicy",
  "make_simtoolreal_env_cfg",
  "print_rollout_summary",
  "reset_object_and_goal",
  "sim_step",
  "simtoolreal_ppo_runner_cfg",
]
