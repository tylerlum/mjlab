"""SimToolReal task port utilities."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.simtoolreal.env_cfg import make_simtoolreal_env_cfg
from mjlab.tasks.simtoolreal.mdp import N_ACT, N_OBS
from mjlab.tasks.simtoolreal.rl_cfg import simtoolreal_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid",
  env_cfg=make_simtoolreal_env_cfg(),
  play_env_cfg=make_simtoolreal_env_cfg(play=True),
  rl_cfg=simtoolreal_ppo_runner_cfg(),
)


def __getattr__(name: str) -> object:
  if name in ("SimToolRealBrowserEnv", "SimToolRealBrowserEnvCfg"):
    from mjlab.tasks.simtoolreal.browser_env import (
      SimToolRealBrowserEnv,
      SimToolRealBrowserEnvCfg,
    )

    return {
      "SimToolRealBrowserEnv": SimToolRealBrowserEnv,
      "SimToolRealBrowserEnvCfg": SimToolRealBrowserEnvCfg,
    }[name]
  if name in (
    "SimToolRealOnnxPolicy",
    "print_rollout_summary",
    "reset_object_and_goal",
    "sim_step",
  ):
    from mjlab.tasks.simtoolreal.policy import (
      SimToolRealOnnxPolicy,
      print_rollout_summary,
      reset_object_and_goal,
      sim_step,
    )

    return {
      "SimToolRealOnnxPolicy": SimToolRealOnnxPolicy,
      "print_rollout_summary": print_rollout_summary,
      "reset_object_and_goal": reset_object_and_goal,
      "sim_step": sim_step,
    }[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
