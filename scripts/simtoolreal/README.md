# SimToolReal MJLab Port Notes

Current branch: `2026-04-25_SimToolReal`

## Environment

MJLab was installed from this checkout with uv:

```bash
uv sync --extra cu128 --group dev
```

Use `--extra cu128` on `uv run` commands in this checkout. Running without the
extra can cause uv to swap the Torch wheel variant.

## Current Milestone

`play_onnx_mujoco.py` is a reproduction harness for the pretrained SimToolReal
browser policy. It runs from the MJLab uv environment but intentionally mirrors
the MuJoCo/WASM demo first, before introducing MJLab manager-based environment
abstractions.

Default inputs:

- Scene: `../simtoolreal.github.io/mujoco_wasm/assets/scenes/iiwa_sharpa.xml`
- Policy: `../simtoolreal.github.io/mujoco_wasm/dist-desktop/policy_iiwa_sharpa.onnx`

Run:

```bash
uv run --extra cu128 python scripts/simtoolreal/play_onnx_mujoco.py --steps 6000
```

Observed smoke-test result on 2026-04-25:

```text
steps=6000
policy_decimation=17
object_goal_distance=0.028655
last_action_minmax=(-1.000000, 1.000000)
```

The policy moves the object toward the goal, so ONNX inference, recurrent state,
observation construction, and action filtering are live. This is not yet a full
MJLab task or training environment.

The reusable implementation lives in `src/mjlab/tasks/simtoolreal/`:

- `policy.py`: browser-demo observation construction, ONNX inference, and action
  filtering.
- `browser_env.py`: single-environment MuJoCo reference API with reset, step,
  pretrained-policy stepping, and reward diagnostics. This is the parity harness
  for the later vectorized MJLab/Warp MDP.

## Important Parity Risks

- Observation ordering currently follows the WASM demo, not the IsaacGym
  `obsList` order verbatim. The deployed ONNX was built for this path, so this is
  correct for browser-policy reproduction but must be rechecked for `.pth`
  checkpoints through `rl_games` or `simple_rl`.
- The current harness uses the website MuJoCo XML directly. That is good for
  physics parity with the browser demo, but it bypasses MJLab manager-based reset,
  reward, termination, and vectorization code.
- The object is the browser demo hammer-like primitive, with fixed scales
  `[5.0, 0.75, 0.5]`. MJLab has primitive `geom_size` domain randomization, but I
  did not find an obvious same-scene, different-mesh-per-world API in this pass.
- The rollout success metric is only final object-goal position distance. The
  SimToolReal MDP also uses keypoint pose errors, lifting rewards, action
  penalties, success windows, resets, random impulses, object distributions, and
  curriculum.
- ONNX Runtime reports a GPU discovery warning on this machine but runs with the
  CPU provider. PyTorch in the uv env sees CUDA.
- Action saturation is visible in the smoke test. It may be normal for this policy
  early in the episode, but it should be compared against the browser demo and the
  original Python MuJoCo deployment before using it as a parity signal.

## Next Implementation Steps

1. Copy the browser-policy observation/action code into MJLab task modules with
   tests that compare against this harness on a saved MuJoCo state.
2. Build a manager-based `Mjlab-SimToolReal-Iiwa-Sharpa` environment around the
   same robot/object model, initially with one primitive object and no mesh
   swapping.
3. Port reset sampling, goal sampling, reward terms, success logic, and
   terminations from `simtoolreal/isaacgymenvs/tasks/simtoolreal/env.py`.
4. Decide whether training should use the original `rl_games` SAPG path or the
   cleaner `simple_rl` implementation from `simtoolreal_private`, then add a
   wrapper only after the MDP tensors match.
