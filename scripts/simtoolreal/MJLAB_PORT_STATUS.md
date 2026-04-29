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

These contact settings intentionally follow the working
`simtoolreal.github.io` MuJoCo/WASM demo more closely than plain MuJoCo defaults.
The important non-default choices are elliptic cones, 6D contact, higher table
friction, `impratio > 1`, and a non-trivial solver iteration count. Early
weaker/default-ish contact setups were easier to make run, but they were risky:
the object could look like it was sinking into the table or become unstable once
the robot applied force.

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

### Physics Timestep Tradeoff

There are three timing regimes that matter:

- **60 Hz policy dt** is non-negotiable for the pretrained policy and should
  stay fixed unless intentionally testing out-of-distribution behavior.
- **`1/180` physics dt with decimation `3`** is the current MJLab default. It is
  much cheaper than 1 kHz-class physics and matches the style of other
  vectorized MJLab tasks better.
- **`1/900` physics dt with decimation `15`** is useful for fidelity/debug
  rollouts because it is closer to the 1 ms website MuJoCo demo while preserving
  exactly 60 Hz policy updates.

The website's native `0.001` dt with decimation `17` is also worth testing, but
it runs the policy at about 58.8 Hz. This is not exactly the pretrained policy's
training cadence. For strict policy reproduction, prefer exact 60 Hz control;
for website parity debugging, browser timing is a useful comparison.

Open question: we do not yet know whether `1/180` is sufficient for final
training. It is probably the right throughput default, but if the robot can
drive the object through the table, if round handles jitter badly, or if success
transfer remains noticeably below the website/MuJoCo reference, re-running
training/eval at `1/900` should be one of the first ablations.

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
   - Safest fallback if heterogeneous variants remain unstable: train on
     all-cuboid tool objects with different per-env scales, offsets, densities,
     COM, and inertia. This gives up round-handle geometry but keeps native
     MuJoCo box collision.

2. **Heterogeneous mesh-variant path**
   - Uses MJLab per-world mesh variants.
   - Allows different envs to use different cuboid/cylinder/capsule-like object
     meshes in one vectorized env.
   - Current sampled rollout path uses this.
   - Needed today for mixed cuboid and round-handle objects in one vectorized
     env.
   - These are mesh variants representing primitive-like shapes, not actual
     MuJoCo primitive `box/cylinder/capsule` geoms switched per env.

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

The distinction between "primitive object" and "mesh version of a primitive" is
important:

- Native primitives give robust primitive contacts and simple inertias, but
  `geom_type` is model-global and currently cannot be changed independently for
  each world in the vectorized model.
- Mesh variants let us choose a different object per world, but the geometry is
  compiled as a mesh. That means mass, COM, and inertia must be assigned
  explicitly, and contact stability needs empirical checks.

Because of that, there are three realistic training options:

1. Use all-cuboid objects with per-env primitive `geom_size` scaling. This is
   the most conservative physics path and may be the best first stable training
   baseline.
2. Use heterogeneous mesh variants for cuboid and round-handle objects, with
   explicit density-derived mass/COM/inertia. This is closer to SimToolReal's
   object distribution but riskier numerically.
3. Use a fixed website-style object for pretrained-policy debugging only. This
   is useful for single-object parity checks but is not the full training
   distribution.

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

Capsules are probably the better round-handle default for SimToolReal parity.
They match IsaacGym's `replace_cylinder_with_capsule=True` behavior and usually
avoid the sharper cylinder rim contact that can make table/hand interaction more
brittle. The caveat is that in the current heterogeneous path they are
capsule-shaped meshes, not native capsule geoms.

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

This is a major gotcha. It is possible for all the high-level task metadata to
look correct while the actual MuJoCo/Warp model is still integrating the wrong
rigid body. For example, the observation can report the intended handle/head
scale and density-derived mass, and the reward can use the intended keypoints,
while the physics still uses mesh-generated inertia unless the model arrays are
overwritten.

Current intended behavior:

- Sample handle/head dimensions from the SimToolReal object distribution.
- Sample handle/head densities from the same distribution logic.
- Compute handle mass and head mass from their volumes and sampled densities.
- Compute the composite COM from the handle/head masses and offsets.
- Compute the composite inertia about the composite COM.
- Write those values into MJLab/MuJoCo per-world model fields:
  `body_mass`, `body_ipos`, and `body_inertia`.

This is especially important for round handles because the mesh shape, the
IsaacGym capsule replacement, and the inertia helper can otherwise disagree. The
policy is sensitive to rotational dynamics, so "looks right in the viewer" is
not sufficient.

Sanity check after the fix showed exact agreement, to displayed precision, for:

```text
box
capsule
cylinder-shaped mesh with capsule inertia
box handle + box head
```

The check compared actual MJLab `env.sim.model.body_mass/body_ipos/body_inertia`
against closed-form mass, COM, and inertia formulas.

This should become a regression test before serious training runs. The test
should cover at least:

- simple cuboid,
- simple cylinder distribution represented as capsule inertia,
- simple cylinder distribution represented as cylinder-like collision mesh,
- website demo handle/head object,
- multiple envs with different sampled objects in one vectorized model.

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

The website object is useful because it isolates many distribution issues, but
it can hide training-distribution bugs. A policy looking good on this fixed
object does not prove that per-env scaling, density sampling, keypoint reward,
or heterogeneous object assignment are correct.

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

Another caveat: HTML export is a diagnostic view of captured MJLab states. It is
not itself the physics engine. If an object appears slightly sunk or offset in
HTML, check both possibilities:

- the underlying MJLab/Warp state really has a contact/pose issue,
- the standalone viewer export approximated the physics geometry or origin
  incorrectly.

For goal/object origin debugging, enable object and goal frame axes in the
viewer. The desired convention is that the object body origin is at the handle
center, not at the head center and not at the combined visual bounding-box
center.

## Important Gotchas

### Mesh Variants Are Not Primitive Variants

MJLab's heterogeneous object path varies mesh data per world. It does not vary
MuJoCo `geom_type` per world. Attempting to directly use primitive capsule slots
inside the mesh-variant path caused variant compilation problems. For now, use
mesh-shaped capsule/cylinder variants.

This means "different meshes per env" and "different native primitive types per
env" are different capabilities. The branch currently has the former. It should
not be assumed that a capsule-like mesh has exactly the same collision behavior
as a native capsule geom.

If training is unstable, the recommended fallback is:

```text
all-cuboid handle/head objects + per-env primitive scaling + explicit density-derived inertia
```

That would still exercise object size randomization and fixed-size keypoint
reward logic while avoiding mesh-variant contact complexity.

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

The most concrete stability observations so far:

- GPU/Warp can generate useful 4-env website-object HTML rollouts quickly.
- A 1000-step GPU website-object rollout went non-finite around step 884 in one
  test after the inertia fix. A 700-step rollout completed and produced HTML.
- CPU/Warp for this scene is not practical for quick visual checks: even a
  50-step 4-env capture spent a long time in `mjwarp.step()` solver/contact code
  before being interrupted.

This means most current viewer artifacts have effectively been GPU physics
artifacts. CPU-vs-GPU comparison remains open, but the CPU path is too slow to
use casually for 4-env HTML captures.

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
