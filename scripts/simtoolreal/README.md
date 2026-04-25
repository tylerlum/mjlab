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
observation construction, and action filtering are live. This remains the
single-env parity reference for the manager-based MJLab task below.

The reusable browser-reference implementation lives in
`src/mjlab/tasks/simtoolreal/`:

- `policy.py`: browser-demo observation construction, ONNX inference, and action
  filtering.
- `browser_env.py`: single-environment MuJoCo reference API with reset, step,
  pretrained-policy stepping, and reward diagnostics. This is the parity harness
  for the later vectorized MJLab/Warp MDP.

## Manager-Based MJLab/Warp Environment

The initial vectorized training task is registered as:

```text
Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid
```

Smoke test:

```bash
uv run --extra cu128 pytest -q tests/test_simtoolreal_manager_env.py
```

Training entrypoint:

```bash
uv run --extra cu128 python -m mjlab.scripts.train \
  Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid
```

Selectable backend entrypoint:

```bash
# RSL-RL through the original MJLab task registry path
uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend rsl_rl --num-envs 1024

# Vendored simple_rl PPO or SAPG
uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend simple_rl --alt.algorithm ppo --num-envs 1024
uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend simple_rl --alt.algorithm sapg --num-envs 1020

# Vendored rl_games PPO or SAPG
uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend rl_games --alt.algorithm ppo --num-envs 1024
uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend rl_games --alt.algorithm sapg --num-envs 1020
```

For SAPG, `num-envs` must be divisible by `--alt.sapg-blocks` (default 6).

What is currently ported:

- Source SimToolReal KUKA iiwa14 + left SHARPA URDF robot, not the website XML.
- Fixed narrow table primitive under the object, matching the source task's
  table dimensions and nominal surface height.
- 29-dimensional SimToolReal joint position action transform with separate arm
  relative control and hand absolute target smoothing.
- 140-dimensional observation tensor matching the browser-policy shape.
- Vectorized object and goal resets with per-env origin offsets.
- Primitive cuboid object size randomization through MJLab `dr.geom_size`, which
  also updates `geom_rbound` and `geom_aabb` for Warp.
- First-pass lifting, keypoint-progress, success, velocity, fall, distance, and
  timeout terms.
- RSL-RL PPO runner config as a runnable baseline.
- Vendored copies of `simple_rl` and the private `rl_games` fork from
  `simtoolreal_private`, with wrappers that expose the MJLab manager env to each
  trainer. Tiny PPO and SAPG smoke runs pass for both alternate backends.

## Important Parity Risks

- The manager environment observation ordering currently follows the WASM demo,
  not the IsaacGym `obsList` order verbatim. The deployed ONNX was built for this
  path, so this is correct for browser-policy reproduction but must be rechecked
  for `.pth` checkpoints through `rl_games` or `simple_rl`.
- The manager environment uses the source URDF and MJLab builtin position
  actuators. The browser harness uses the website MuJoCo XML. Joint names and
  kinematic body names now come from the IsaacGym/URDF path, but actuator gains,
  contact settings, inertias, and geom simplifications still need parity audits.
- The task currently starts with one primitive cuboid distribution from
  `simtoolreal_private`. I did not find an obvious same-scene,
  different-mesh-per-world API in this pass.
- Object scaling is implemented for primitive boxes. If mesh objects are added,
  scaling/contact bounds need to be validated separately.
- The RSL-RL config is only a baseline runner path. The vendored `rl_games` and
  `simple_rl` backends now run, but their hyperparameters are still compact
  bridge defaults rather than the full original SimToolReal YAML.
- The current harness uses the website MuJoCo XML directly. That is good for
  physics parity with the browser demo, but it bypasses MJLab manager-based reset,
  reward, termination, and vectorization code.
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

1. Audit every manager MDP tensor against `simtoolreal/isaacgymenvs/tasks/
   simtoolreal/env.py`, especially observation ordering, fingertip frames, reward
   scales, reset windows, and success/lift state transitions.
2. Add saved-state parity tests between the browser harness, source MuJoCo/URDF
   env, and manager-based MJLab env for keypoints, palm/fingertip positions, and
   action target filtering.
3. Add the original random impulses, richer object distributions, physics/domain
   randomization, and curriculum terms once the simple-cuboid MDP is stable.
4. Decide whether training should use the original `rl_games` SAPG path or the
   cleaner `simple_rl` implementation from `simtoolreal_private`, then add that
   runner after the MDP tensors match.
