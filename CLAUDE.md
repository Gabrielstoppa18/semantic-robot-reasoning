# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`semantic-robot-reasoning` is a master's thesis project that integrates a Large
Language Model (LLM) with computer vision to let a simulated robot interpret
natural-language instructions, reason about semantic/contextual meaning assigned
to objects, and execute the corresponding manipulation actions in simulation.

Example flow: given the instruction "the blue cube is water and the red cube is
fire, use the water to put out the fire", the system must perceive the environment,
detect and identify the cubes, infer via LLM reasoning that "extinguishing fire"
means placing the water object on top of the fire object, plan the action sequence,
and execute it on the simulated robot — without this rule being explicitly hardcoded.

## Architecture (target)

The pipeline has four stages:

1. **Perception** — camera captures the scene; object detection/segmentation
   identifies and localizes objects (bounding boxes / 3D position).
2. **Semantic grounding** — detected objects + instruction are passed to the LLM,
   which infers the contextual/semantic role of each object based on the instruction
   (not from fixed labels).
3. **Reasoning & planning** — the LLM reasons over the semantic context and the
   instruction to produce a sequence of intended actions (e.g. "pick blue cube",
   "place on red cube").
4. **Action execution** — the planned actions are translated into low-level
   commands/trajectories for the simulated robot (motion planning + control).

Each stage should be a separable module so perception, reasoning, and action can be
developed, tested, and swapped independently (e.g. swapping the detection model or
the LLM without touching the rest of the pipeline). Modules integrate directly via
CoppeliaSim's Python remote API (ZMQ remote API) — no ROS2 for now.

## Status

Perception, simulation, and action scaffolding exist; only the reasoning (LLM)
stage is not started yet (its package exists as an empty stub).

/perception    # color-based stereo XYZ localization of the cubes (perception/stereo_localization.py)
/reasoning     # not started — LLM integration, prompting, semantic grounding, planning
/action        # IK-driven pick-and-place using perceived cube positions (action/pick_and_place.py)
/simulation    # CoppeliaSim scene-building and remote API connection (simulation/build_scene.py, connection.py, geometry.py)
/tests         # unit tests for the pure-logic parts of each stage (no CoppeliaSim needed to run them)
/docs          # thesis notes, references, experiment logs (not started)

All packages are pip-installed in editable mode (see Setup below), so
`from simulation.connection import connect` works from anywhere in the repo
without path hacks. `simulation/build_scene.py` and
`perception/stereo_localization.py` additionally bootstrap `sys.path` at the
top so they also work when run directly (e.g. an IDE's "Run" button), not
just via `python -m`.

## Setup

Requires CoppeliaSim Edu installed locally (ZMQ remote API server starts
automatically with it). Python 3.10+, project uses a `.venv` + `pyproject.toml`
(no more `requirements.txt`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Common commands

```bash
python -m simulation.build_scene         # requires CoppeliaSim open on an empty scene
python -m simulation.build_scene --no-cubes --no-cameras  # robot + pedestal only, for IK/motion testing in isolation
python -m perception.stereo_localization # requires the scene above to already be built
python -m action.pick_and_place          # requires the full scene; picks up fire_cube, places it on water_cube
python -m action.test_ik_motion          # requires the robot-only scene; IK/motion smoke test, no perception/grasping
ruff check .                             # lint
ruff format .                            # format
pytest                                   # run tests (single test: pytest tests/test_geometry.py::test_normalize_unit_length)
```

## Simulation scene (`simulation/build_scene.py`)

Builds, in the currently running CoppeliaSim instance:

- `water_cube` (blue) and `fire_cube` (red) — the thesis's canonical example objects
- a UR5 arm with an RG2 gripper (loaded from CoppeliaSim's bundled model library,
  paths hardcoded to the default Windows install at
  `C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu`), mounted on a `robot_pedestal`
  stand (see below) set back from the cubes
- a fixed stereo pair of vision sensors (`left_camera`, `right_camera`) overlooking
  the workspace

`main()` takes `--no-cubes` and `--no-cameras` to build a reduced scene (just
the robot on its pedestal) for IK/motion testing in isolation, without needing
perception or grasp targets — see `action/test_ik_motion.py`.

**The UR5 is mounted on a `PEDESTAL_HEIGHT`-tall pedestal, not the floor.**
Reaching down to floor-level cubes from a floor-level base is a near-singular
"elbow at max extension" configuration -- full-pose IK (position + orientation)
was unsolvable there almost everywhere, even for positions well within the
arm's nominal reach. Raising the base gives the arm a normal "reach down and
out" geometry instead. If the cubes or robot base ever move, re-verify
`action.pick_and_place` still finds IK solutions rather than assuming reach
distance alone is what matters -- the *conditioning* of the pose matters as
much as whether it's in range.

CoppeliaSim vision sensors/cameras look along their local **+Z** axis (confirmed
empirically — this is the opposite of the common OpenGL -Z convention). See
`simulation/geometry.py::look_at_euler` for the orientation helper this relies on.

`load_robot()` also does several things to make the arm externally controllable
and IK-drivable, each discovered by trial and error against the running sim
(not documented behavior, so re-verify if the UR5 model in CoppeliaSim's
library ever changes):

- **Removes the UR5's own demo-behavior script** (`find_child_by_name(ur5, "Script")`).
  Left in place, it silently resets joint angles every simulation step, fighting
  any externally-driven control — RG2's own script is untouched and stays (it
  drives the gripper open/close, see below).
- **Sets `sim.modelproperty_not_dynamic` on the UR5 model.** Without it the base
  isn't anchored to the world and slowly falls/drifts under gravity once its
  joints are kinematic.
- **Switches the 6 arm joints to `sim.jointmode_kinematic`.** `action/pick_and_place.py`
  drives them by directly writing solved IK joint angles, not motor torques.
- **Seeds the joints to `ARM_READY_POSE_DEG`** — see the comment on that
  constant for why this exact pose (found by a small grid search, not
  hand-derived) rather than an arbitrary bent-elbow pose.
- **Creates the `ik_target` dummy and attaches an embedded IK actuation
  script to the UR5** — see "Action" below; this is what `action/pick_and_place.py`
  actually drives, not `simIK` directly.

`ARM_JOINT_SUBPATHS` (base-to-wrist) is how those 6 joints are found reliably:
UR5.ttm nests each next joint one level deeper (`/joint`, `/link/joint`, ...),
and the paths are resolved from the model's actual root alias, not a hardcoded
`/UR5`, in case a scene ever has more than one instance.

## Perception (`perception/stereo_localization.py`)

Estimates each cube's world XYZ by: color-thresholding each stereo camera's image
(HSV ranges in `COLOR_RANGES`) to get a 2D pixel centroid, casting a ray through
that pixel using camera intrinsics/extrinsics queried live from the sim (not
hardcoded), and triangulating the two rays' closest point. Validated against the
scene's ground-truth cube positions to ~1mm accuracy.

Note the rendered image is mirrored left/right relative to the camera's local +X
axis (`pixel_ray` accounts for this — see its docstring); this was confirmed
empirically, not derived from documentation, so re-verify if CoppeliaSim's
rendering convention ever changes.

This is intentionally the simplest possible perception approach (color blobs, not
a learned detector) to get an end-to-end pipeline working first. Swapping in a
real object detector later should only require replacing `detect_color_centroid`.

## Action (`action/pick_and_place.py`)

Picks up `fire_cube` and places it on top of `water_cube`, using the XYZ
positions `perception.stereo_localization` estimates (not ground truth read
directly from the sim).

**The actual IK solving does not happen in this file, or from Python at all.**
It happens inside CoppeliaSim, in a child script attached to the UR5 by
`simulation.build_scene.load_robot` (see that section above), which re-solves
`simIK.handleGroup` on every single simulation step. `Arm` just moves that
script's `ik_target` dummy smoothly and calls `sim.step()`. This replaced an
earlier version where `Arm` called `simIK.findConfigs` itself, once per
Cartesian sub-step, from the external Python client — that version is why
several of the gotchas below reference "the old approach"; understanding why
it was replaced matters for not reintroducing it.

- **Why the whole approach changed: `findConfigs` could track the *tip* along
  a smooth path, but not the *rest of the arm*.** `simIK.findConfigs` is a
  randomized search that returns *a* valid configuration for the current
  target, not the same configuration smoothly evolved from the previous one.
  Calling it once per Cartesian sub-step kept the gripper on a straight line,
  but let the shoulder/elbow jump between different valid configurations
  frame to frame — visible as the whole body twitching/flicking while the
  gripper itself stayed put (caught by the user watching a screen recording,
  not visible in a single before/after screenshot). A "closest configuration
  in joint space" heuristic reduced this but never eliminated it, because
  `findConfigs` has no concept of "the previous solve" to begin with — it
  fundamentally isn't a continuous tracker. This was found by loading
  CoppeliaSim's own bundled example scenes
  (`scenes/kinematics/smoothMovementsInFkAndIk.ttt`,
  `scenes/messaging/movementViaRemoteApi.ttt`) and inspecting their embedded
  scripts: neither uses `findConfigs` at all for continuous motion. Both use
  `simIK.handleGroup`, a genuinely differential/incremental solver — every
  call refines the *current* joint configuration one small step further
  toward the target, so consecutive calls stay continuous by construction.
  `simIK.handleGroup` was tried once before, directly from Python, and
  appeared to just get the arm stuck — that turned out to be because nothing
  was calling `sim.step()` (see below), not because the solver itself doesn't
  work.
- **`simIK.handleGroup` must run inside CoppeliaSim's own `sysCall_actuation`,
  not be called from the external Python client.** Both bundled reference
  scenes attach a small child script to the robot that does exactly one
  thing every simulation step: `simIK.handleGroup(ikEnv, ikGroup, {syncWorlds = true})`,
  tracking whatever pose an `ik_target` dummy currently has. `syncWorlds =
  true` is what makes it read the *current* real-scene pose of that dummy
  each call (confirmed empirically: moving the dummy via plain
  `sim.setObjectPosition` while the simulation runs makes the arm follow it
  in real time, no `simIK.setObjectMatrix`/mapped-handle indirection needed —
  unlike `findConfigs`, see below). `simulation.build_scene.load_robot`
  creates this script (`_IK_ACTUATION_SCRIPT_TEMPLATE`) once, at scene-build
  time, associated with the UR5 model.
  - **Embedded child scripts must `require` every API namespace at the very
    top of the script, outside any function** (`sim = require("sim")`,
    `simIK = require("simIK")`) — this specific CoppeliaSim version (4.10)
    doesn't provide `sim`/`simIK` as bare globals otherwise. Getting this
    wrong doesn't raise a visible error: it silently pauses the simulation
    the instant it starts (`sim.getSimulationState()` reads
    `sim.simulation_paused` and never advances) with no exception on the
    Python side at all — this cost a lot of confused debugging (assuming
    `simIK.handleGroup` itself was broken) before actually checking
    `sim.getSimulationState()`/`sim.getSimulationTime()` mid-run and finding
    the sim wasn't ticking at all.
- **`Arm` drives the simulation with `sim.setStepping(True)` + explicit
  `sim.step()` calls, one per waypoint** — this is what makes the embedded
  actuation script's `handleGroup` call actually re-run each time `move_to`
  moves the target a little further. Without stepped mode, the target's pose
  updates but the simulation clock never advances, so `sysCall_actuation`
  never fires again and the arm stays frozen (this is exactly what an
  earlier attempt at calling `sim.moveToPose`/`sim.moveToConfig` directly
  from Python hit, and why that approach was abandoned as "too slow" — it
  wasn't slow, it just never advanced the sim at all, so it looked stuck).
  `move_to`'s per-step `time.sleep(dt - elapsed)` pacing is separate from
  this and only exists so the CoppeliaSim GUI shows real robot-speed motion
  instead of a sped-up blur when driven this fast over the remote API.
- **`move_to()` still interpolates position and orientation together as one
  pose** (`sim.interpolatePoses`, position lerp + quaternion slerp) from the
  gripper's *actual current* pose, not the previous commanded target pose —
  self-correcting if a previous move didn't fully converge, rather than
  compounding the error into a sudden jump on the next move. This part of
  the design carried over from the `findConfigs`-based version.
- **`simIK.handleGroup` is still a *local* solver and can still get stuck in
  a local minimum**, same as `findConfigs` could, just for a different
  reason: it's differential, so if the *starting* configuration's tip
  orientation is very far (in the worst observed case, a full 180 degrees)
  from the target orientation, it can fail to find any descent direction and
  sits at a constant, non-zero tracking error indefinitely — confirmed by
  testing with drastically more steps/time budget and seeing zero
  improvement, not slow improvement. This is exactly why
  `simulation.build_scene.ARM_READY_POSE_DEG` is the specific pose it is (see
  its comment) — an earlier bent-elbow pose was reachable enough for
  `findConfigs`'s random search but pointed the tip 180 degrees away from
  the fixed top-down grasp orientation, which silently broke the new
  differential approach's very first move until this was diagnosed and
  fixed. `Arm._verify_converged` checks final tracking error (position +
  quaternion angle) against a tolerance and raises rather than silently
  accepting an unconverged pose — same "explicit guard, don't force it"
  philosophy as before, just checking real tracking error instead of
  "did `findConfigs` return anything."
- **Because the IK environment/group are now created once, in the embedded
  script's `sysCall_init` (at simulation start), not once per `Arm()`
  construction, the old `simIK.createEnvironment()`/`createCollection()`
  leak risk is gone by construction** — there's no longer a Python-side
  environment for repeated `Arm()` instantiation to leak. `Arm` still
  creates/destroys its two *collision* collections per instance (see below),
  which was never the primary leak source, but `Arm.close()` still tears
  those down in a `finally` block out of the same caution.
- **`findConfigs`'s random search could return a configuration where the
  wrist swings back into the arm's own base even though the tip lands on
  target; the differential solver is far less prone to this** (it evolves
  smoothly from a collision-free configuration rather than jumping between
  disconnected solution branches), but `Arm._verify_converged` still checks
  a `gripper` vs. `shoulder` (link1+link2) collision collection after each
  move as a safety net and raises if it's hit. Checking the *whole* arm
  against itself doesn't work: adjacent, physically-touching links (e.g. the
  gripper mounted flush on the wrist) always report as "colliding" — this
  narrower pair was verified collision-free in the baseline ready pose
  first. A similar arm-body-vs-floor check was tried in the old approach and
  **removed**: it flagged configurations that, on visual inspection
  (screenshot), weren't touching the floor at all — this model's collision
  proxies are looser than their visual meshes, making bounding-volume checks
  unreliable this close to the work surface.
- **Grasping is done manually** (`Arm.attach`/`Arm.detach` reparent the object to
  the gripper tip / back to world). RG2's own `attachPoint` + `attachProxSensor`
  look purpose-built for automatic proximity-triggered attach, and the
  proximity sensor does fire, and the fingers do physically stop against the
  object — but nothing ever reparents it. Rather than reverse-engineer RG2's
  internal script, grasping is just done explicitly from Python.
- **`sim.stopSimulation()` reverts any API-driven changes** (joint positions,
  attachments, etc.) made while that simulation was running, back to how
  things were when it started. Scripts here call `startSimulation()` once and
  never stop it mid-sequence; interactive testing that starts/stops between
  each change will see confusing "reverts."

## Conventions

- Language: Python 3.10+
- Simulation: CoppeliaSim, driven standalone via its Python remote API (ZMQ remote API) — no ROS2/Gazebo for now
- LLM interface: external API (e.g. Anthropic/OpenAI) — no local model
- Robot/hardware target: simulation only — no physical robot execution
- Lint/format: Ruff; tests: pytest (config lives in `pyproject.toml`, not separate ini files)

## Next steps for this file

Once the reasoning stage exists, update this file to add:

- How the LLM is prompted for semantic grounding and action planning
- How the LLM's output plan maps onto `action.pick_and_place`'s pick/place calls
  (right now the pick/place objects and positions are hardcoded to
  fire_cube-onto-water_cube, not derived from an instruction)
