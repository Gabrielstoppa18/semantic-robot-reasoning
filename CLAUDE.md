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

Perception and simulation scaffolding exist; reasoning and action stages are not
started yet (their packages exist as empty stubs).

/perception    # color-based stereo XYZ localization of the cubes (perception/stereo_localization.py)
/reasoning     # not started — LLM integration, prompting, semantic grounding, planning
/action        # not started — action-to-robot-command translation, motion control
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
python -m perception.stereo_localization # requires the scene above to already be built
ruff check .                             # lint
ruff format .                            # format
pytest                                   # run tests (single test: pytest tests/test_geometry.py::test_normalize_unit_length)
```

## Simulation scene (`simulation/build_scene.py`)

Builds, in the currently running CoppeliaSim instance:

- `water_cube` (blue) and `fire_cube` (red) — the thesis's canonical example objects
- a UR5 arm with an RG2 gripper (loaded from CoppeliaSim's bundled model library,
  paths hardcoded to the default Windows install at
  `C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu`), base set back from the cubes
- a fixed stereo pair of vision sensors (`left_camera`, `right_camera`) overlooking
  the workspace

CoppeliaSim vision sensors/cameras look along their local **+Z** axis (confirmed
empirically — this is the opposite of the common OpenGL -Z convention). See
`simulation/geometry.py::look_at_euler` for the orientation helper this relies on.

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

## Conventions

- Language: Python 3.10+
- Simulation: CoppeliaSim, driven standalone via its Python remote API (ZMQ remote API) — no ROS2/Gazebo for now
- LLM interface: external API (e.g. Anthropic/OpenAI) — no local model
- Robot/hardware target: simulation only — no physical robot execution
- Lint/format: Ruff; tests: pytest (config lives in `pyproject.toml`, not separate ini files)

## Next steps for this file

Once the reasoning and action stages exist, update this file to add:

- How the LLM is prompted for semantic grounding and action planning
- How planned actions are translated into UR5/RG2 motion commands
