# SimToolReal MJLab Port Status

Last updated: 2026-04-29  
Branch: `2026-04-29_SimToolReal`

This note summarizes the current SimToolReal-on-MJLab work: what is wired up,
how the major pieces fit together, what has been checked against IsaacGym and
the MuJoCo/WASM website demo, and the main gotchas that can still affect policy
performance.

## High-Level Shape

The MJLab port is a manager-based environment for the SimToolReal iiwa14 +
left-SHARPA object manipulation task. The goal is to reproduce the original
IsaacGym MDP and then train/evaluate through MJLab/Warp, while keeping the
website MuJoCo/WASM demo as a useful single-env reference.

The main implementation lives under:

- `src/mjlab/tasks/simtoolreal/assets.py`
- `src/mjlab/tasks/simtoolreal/env_cfg.py`
- `src/mjlab/tasks/simtoolreal/mdp.py`
- `src/mjlab/tasks/simtoolreal/object_size_distributions.py`
- `src/mjlab/tasks/simtoolreal/viewer_capture.py`
- `scripts/simtoolreal/capture_pretrained_rollouts.py`
- `src/mjlab/scripts/train_simtoolreal.py`

The registered task is:

```text
Mjlab-SimToolReal-Iiwa-Sharpa-SimpleCuboid
```

The environment uses the source SimToolReal robot URDF, not the website XML. The
website XML remains important because the browser demo is currently the best
known MuJoCo policy-running reference.

## Environment And Physics Defaults

The current manager env uses:

- Robot base pose: `pos=(0, 0.8, 0)`.
- Table pose: `pos=(0, 0, 0.38)`.
- Policy/control cadence: 60 Hz.
- Current default physics dt: `1/180`.
- Current default decimation: `3`.
- MuJoCo integrator: `implicitfast`.
- Contact cone: `elliptic`.
- `impratio=10`.
- `iterations=20`, `ls_iterations=20`.
- Table friction: `1 0.005 0.0001`.
- Object/table/fingertip-relevant contact dimensionality: `condim=6`.

The capture script can override timing for experiments:

```bash
--physics-timestep 0.0011111111111111111 --decimation 15
```

This keeps the policy dt at exactly 60 Hz while using 900 Hz physics. The
website XML itself uses `timestep="0.001"` and computes policy decimation as:

```js
round((1 / 60) / timestep) = 17
```

so browser timing is about 58.8 Hz, not exactly 60 Hz. The capture script has:

```bash
--allow-policy-dt-mismatch
```

for this browser-timing comparison.

## Robot Details

The robot asset comes from the SimToolReal source URDF:

```text
simtoolreal/assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf
```

Important robot choices:

- The policy action/observation joint order is 29 joints: 7 iiwa arm joints,
  followed by 22 SHARPA hand joints.
- IIWA position gains use the IsaacGym/browser values:
  `600, 600, 500, 400, 200, 200, 200`.
- Hand stiffness, damping, armature, frictionloss, and viscous damping are
  explicitly threaded through MJLab builtin position actuators.
- Robot body `gravcomp` is applied in the MJLab asset builder. This mirrors the
  website fix where robot bodies get gravity compensation but the object/table do
  not.

One subtle website bug was fixed separately in `simtoolreal.github.io`: the
pinky joint limit scaling in the browser bundle was stale/wrong. That affected
both joint-position normalization and action-to-target scaling. The MJLab code
uses the corrected policy-order limits.

## Object Modeling

There are two object paths:

1. **Primitive geom-size randomization path**
   - Single object structure.
   - Uses per-world `geom_size` and `geom_pos`.
   - Good for cuboid-equivalent randomization.
   - Cannot switch MuJoCo primitive `geom_type` per world.

2. **Heterogeneous mesh-variant path**
   - Uses MJLab per-world mesh variants.
   - Allows different envs to use different cuboid/cylinder/capsule-like object
     meshes in one vectorized env.
   - Current sampled rollout path uses this.

The source object size distributions were copied from SimToolReal:

```text
src/mjlab/tasks/simtoolreal/object_size_distributions.py
```

The rollout script can sample one runtime mesh variant per env from the copied
IsaacGym distributions:

```bash
--sample-object-distribution-types simple_cuboid simple_cylinder
```

The sampled dimensions and densities are printed before the env is built, and
the generated variants are registered at runtime.

### Cylinder vs Capsule

IsaacGym generated URDFs contain `<cylinder>` primitives for 2D
`(length, diameter)` handles, but object and goal assets are loaded with:

```python
object_asset_options.replace_cylinder_with_capsule = True
```

The IsaacGym inertia helper also uses capsule formulas for 2D round scales:

```python
MODE = "capsule"
```

Therefore, the MJLab MDP treats both `shape="cylinder"` and `shape="capsule"` as
round handles with capsule inertia.

The rollout script exposes:

```bash
--round-handle-geom capsule
--round-handle-geom cylinder
```

This controls the collision mesh shape in the heterogeneous mesh-variant path.
The default is `capsule`, because that better matches IsaacGym's
`replace_cylinder_with_capsule=True`.

Important caveat: MJLab's current heterogeneous variant mechanism varies mesh
assignment per world. It does not vary primitive `geom_type` per world. Directly
switching primitive cylinder/capsule/box types per env is not supported by this
path because MuJoCo `geom_type` is model-global, not per-world. A cleaner future
design could use several fixed primitive slots and per-world activation/contact
masks if MJLab/Warp supports the required fields.

### Mass, COM, And Inertia

Mesh-derived inertias were not good enough for SimToolReal parity. The MDP now
explicitly overwrites per-world object:

- `body_mass`
- `body_ipos`
- `body_inertia`

for mesh variants, using the same cuboid/capsule formulas used in the primitive
randomization path. This was important: an earlier version updated MDP state and
goal geometry but accidentally left `body_mass/body_inertia` close to
mesh-derived values.

Sanity check after the fix showed exact agreement, to displayed precision, for:

```text
box
capsule
cylinder-shaped mesh with capsule inertia
box handle + box head
```

The check compared actual MJLab `env.sim.model.body_mass/body_ipos/body_inertia`
against closed-form mass, COM, and inertia formulas.

## Website Demo Object

The website object is not the full training distribution. It is a fixed
tool-like object:

- Body origin at the handle center.
- Box handle half-size: `0.10 0.015 0.01`.
- Handle density: `400`.
- Capsule head from `0.10 -0.03 0` to `0.10 0.03 0`.
- Head radius: `0.02`.
- Head density: `300`.
- Object scale observation: `[5, 0.75, 0.5]`.

MJLab has a `website_demo` mesh variant for testing this exact object contact
geometry and scale:

```bash
--object-mesh-variants --object-mesh-variant-names website_demo
```

Matching this object alone did not close the performance gap with the website
demo, which suggests remaining differences are elsewhere: robot XML/URDF,
actuation, contact, viewer/control path, or reset/goal details.

## Observations And Actions

The actor observation is 140-dimensional. It includes the same main SimToolReal
groups used by the pretrained checkpoint:

- normalized joint positions
- joint velocities
- previous/current target terms
- keypoints relative to palm
- keypoints relative to goal
- fingertip positions relative to palm
- object scales
- closest keypoint progress term

Action dimension is 29. The action term implements SimToolReal's split behavior:

- arm actions are relative/delta target updates,
- hand actions are absolute/smoothed target updates,
- optional action delay is enabled during training and disabled in play/eval.

The capture script loads the private `rl_games` checkpoint directly from:

```text
/home/tylerlum/github_repos/simtoolreal_private/pretrained_policy
```

The policy input appends SAPG conditioning value `50.0` to the 140-d actor
observation before calling the `rl_games` player.

## Rewards And Stateful MDP Terms

The MJLab MDP ports the major IsaacGym stateful terms:

- lifting reward and lift bonus,
- pre-lift fingertip progress,
- fixed-size keypoint progress,
- success bonus,
- action penalties,
- object velocity penalty slot,
- object fell / object dropped / hand far / timeout / max-success termination.

Stateful buffers include:

- `lifted_object`
- `just_lifted_object`
- `closest_fingertip_dist`
- `closest_keypoint_max_dist`
- `closest_keypoint_max_dist_fixed_size`
- `successes`
- `near_goal`
- `near_goal_steps`
- `reset_goal_buf`
- per-episode closest-keypoint summary buffers

Sentinel initialization uses `inf` in MJLab. The reward code handles this
explicitly so the first sample does not produce an infinite reward.

## Success And Goal Resampling

This was double-checked against IsaacGym. The pretrained/default config has:

```yaml
keypointScale: 1.5
successSteps: 10
fixedSizeKeypointReward: true
successTolerance: 0.075
targetSuccessTolerance: 0.01
evalSuccessTolerance: null
forceConsecutiveNearGoalSteps: False
```

Success uses:

```text
keypoint_success_tolerance = success_tolerance * keypointScale
near_goal = keypoints_max_dist_fixed_size <= keypoint_success_tolerance
near_goal_steps += near_goal
is_success = near_goal_steps >= successSteps
```

The important detail is that default IsaacGym does **not** require 10
consecutive near-goal steps. It requires 10 accumulated near-goal env steps,
because `forceConsecutiveNearGoalSteps` is false.

The MJLab code matches this cumulative behavior.

When we run eval captures with:

```bash
--success-tolerance 0.01
```

the effective fixed-size keypoint threshold is:

```text
0.01 * 1.5 = 0.015 m
```

When we run:

```bash
--success-tolerance 0.03
```

the effective threshold is:

```text
0.03 * 1.5 = 0.045 m
```

A visual "looks at the goal" moment may not resample if the fixed-size keypoint
max distance does not cross the threshold often enough. With `0.01`, several
rollouts visually looked close but did not accumulate 10 near-goal steps.

## Training Backends

The repo has integration paths for:

- RSL-RL
- vendored `simple_rl`
- vendored private `rl_games`

Entrypoint:

```bash
uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend rsl_rl --num-envs 1024

uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend simple_rl --alt.algorithm sapg --num-envs 1024

uv run --extra cu128 python -m mjlab.scripts.train_simtoolreal \
  --backend rl_games --alt.algorithm sapg --num-envs 1024
```

Both alternate backends have smoke-test coverage. They are wired through an
MJLab-to-VecEnv adapter and use SimToolReal-shaped observation/action spaces.

The alternate backend defaults follow the original SimToolReal asymmetric
PPO/SAPG-style setup:

- `[1024, 1024, 512, 512]` MLP,
- 1024-unit LSTM,
- asymmetric critic,
- reward scale `0.01`,
- horizon/sequence length `16`,
- adaptive LR with KL threshold `0.016`,
- SAPG leader/follower entropy structure.

## Viewer And HTML Capture

The pretrained rollout capture script writes standalone interactive HTML:

```bash
uv run --extra cu128 python scripts/simtoolreal/capture_pretrained_rollouts.py \
  --device cuda:0 \
  --num-rollouts 1 \
  --num-envs 4 \
  --steps 1000 \
  --seed 37 \
  --sample-object-distribution-types simple_cuboid simple_cylinder \
  --round-handle-geom capsule \
  --capture-all-envs \
  --success-tolerance 0.03 \
  --physics-timestep 0.0011111111111111111 \
  --decimation 15 \
  --output-dir artifacts/simtoolreal_private_pretrained_rollouts_sampled_simple_distributions_1000_dt900_capsule_tol003
```

Recent generated examples:

```text
artifacts/simtoolreal_private_pretrained_rollouts_sampled_simple_distributions_1000_dt900_capsule_tol003/
artifacts/simtoolreal_private_pretrained_rollouts_sampled_simple_distributions_1000_dt900_capsule/
artifacts/simtoolreal_private_pretrained_rollouts_sampled_simple_distributions_1000_dt900_cylinder/
artifacts/simtoolreal_private_pretrained_rollouts_website_demo_1000_dt001_dec17/
```

The viewer centers each HTML in the env frame, not the global vectorized env
grid. It also supports diagnostic object/goal frame axes through the interactive
viewer template.

Viewer caveat: URDF has no standard capsule primitive. The current standalone
HTML object rendering approximates round/capsule handles as cylinders unless we
add mesh-embedded object visuals for the viewer export. The MJLab physics can
use capsule-shaped meshes, but the HTML visual may not show the rounded caps.

## Important Gotchas

### Mesh Variants Are Not Primitive Variants

MJLab's heterogeneous object path varies mesh data per world. It does not vary
MuJoCo `geom_type` per world. Attempting to directly use primitive capsule slots
inside the mesh-variant path caused variant compilation problems. For now, use
mesh-shaped capsule/cylinder variants.

### Density Can Reach Metadata Without Reaching Physics

The MDP state can show the intended density-derived mass while the compiled
physics model still uses mesh-derived mass if `body_mass/body_ipos/body_inertia`
are not explicitly overwritten. This was found and fixed for mesh variants.

### Capsule vs Cylinder Naming

`shape="cylinder"` currently means the sampled distribution was a 2D round
handle. Inertia is capsule-style to match IsaacGym. The rollout flag controls
whether the heterogeneous collision mesh is cylinder-like or capsule-like.

### Success Is Fixed-Size Keypoint Based

With `fixedSizeKeypointReward: true`, both success and keypoint reward use the
fixed-size keypoint distance, not necessarily the current sampled object's exact
visual/contact extents. This is consistent with the pretrained config and is one
reason visual judgment can be misleading.

### Website Performance Is Not Yet Fully Explained

The website policy still appears stronger than some MJLab captures. Tested
factors so far:

- higher physics rate (`1/900`, decimation `15`),
- browser timing (`0.001`, decimation `17`),
- website exact object geometry,
- capsule-like round handles,
- fixed inertia bug.

These improve confidence but do not fully explain the difference. Remaining
suspects include robot XML vs URDF details, actuator implementation, contact
model differences between MuJoCo native WASM and MuJoCo/Warp, reset/goal
distribution details, and object visual-vs-physics mismatch in the HTML export.

## Verification Commands

Focused tests:

```bash
uv run --extra cu128 ruff check \
  src/mjlab/tasks/simtoolreal/assets.py \
  src/mjlab/tasks/simtoolreal/mdp.py \
  scripts/simtoolreal/capture_pretrained_rollouts.py

uv run --extra cu128 pytest \
  tests/test_simtoolreal_manager_env.py \
  tests/test_simtoolreal_policy.py \
  tests/test_simtoolreal_alt_backends.py
```

Recent checks passed:

```text
ruff check: passed
tests/test_simtoolreal_manager_env.py tests/test_simtoolreal_policy.py: 22 passed
```

The inertia sanity check was run as an ad hoc script and should be promoted into
a focused regression test if we continue iterating on object variants.

