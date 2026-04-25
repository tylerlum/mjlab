"""Run the SimToolReal pretrained ONNX policy in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from mjlab.tasks.simtoolreal import (
  SimToolRealOnnxPolicy,
  print_rollout_summary,
  reset_object_and_goal,
  sim_step,
)

MJLAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SITE_ROOT = MJLAB_ROOT.parent / "simtoolreal.github.io"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--scene",
    type=Path,
    default=DEFAULT_SITE_ROOT / "mujoco_wasm/assets/scenes/iiwa_sharpa.xml",
  )
  parser.add_argument(
    "--policy",
    type=Path,
    default=DEFAULT_SITE_ROOT / "mujoco_wasm/dist-desktop/policy_iiwa_sharpa.onnx",
  )
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Sleep in viewer mode so the simulation runs near wall-clock time.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.scene.exists():
    raise FileNotFoundError(f"Scene XML not found: {args.scene}")
  if not args.policy.exists():
    raise FileNotFoundError(f"Policy ONNX not found: {args.policy}")

  model = mujoco.MjModel.from_xml_path(str(args.scene))
  data = mujoco.MjData(model)
  policy = SimToolRealOnnxPolicy(model, args.policy)
  policy.reset(data)
  reset_object_and_goal(model, data, policy.cache)
  policy_decimation = max(1, round((1.0 / 60.0) / model.opt.timestep))

  if args.headless:
    for step in range(args.steps):
      sim_step(model, data, policy, step, policy_decimation)
    print_rollout_summary(data, policy, args.steps, policy_decimation)
    return

  print("Viewer controls: close the window to stop. The policy starts immediately.")
  with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    next_wall_time = time.perf_counter()
    while viewer.is_running() and (args.steps <= 0 or step < args.steps):
      sim_step(model, data, policy, step, policy_decimation)
      viewer.sync()
      step += 1
      if args.realtime:
        next_wall_time += model.opt.timestep
        sleep_time = next_wall_time - time.perf_counter()
        if sleep_time > 0.0:
          time.sleep(sleep_time)
        else:
          next_wall_time = time.perf_counter()
  print_rollout_summary(data, policy, step, policy_decimation)


if __name__ == "__main__":
  main()
