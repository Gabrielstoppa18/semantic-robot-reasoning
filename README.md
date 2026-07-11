# semantic-robot-reasoning

Master's thesis project integrating a Large Language Model (LLM) with computer
vision to let a simulated robot interpret natural-language instructions, reason
about semantic/contextual meaning assigned to objects, and execute the
corresponding manipulation actions — without hardcoding the semantic rules.

**Example.** Given the instruction *"the blue cube is water and the red cube is
fire, use the water to put out the fire"*, the system perceives the scene,
identifies the cubes, infers via LLM reasoning that "extinguishing fire" means
placing the water object on top of the fire object, plans the action sequence,
and executes it on the simulated robot.

## Architecture

Four separable stages, each meant to be developed and swapped independently:

| Stage | Package | Status |
|---|---|---|
| Perception (detect/localize objects) | [`perception/`](perception/) | stereo color-blob XYZ localization |
| Semantic grounding (LLM infers object roles) | [`reasoning/`](reasoning/) | local VLM describes the scene and locates a named object |
| Reasoning & planning (LLM produces action sequence) | [`reasoning/`](reasoning/) | not started (single grounded pick/place only) |
| Action execution (robot motion commands) | [`action/`](action/) | IK-driven pick-and-place (hardcoded fire-onto-water) |

[`simulation/`](simulation/) is the glue/infrastructure layer: it builds the
CoppeliaSim scene and holds the remote API connection helper shared by the
other stages. [`dashboard/`](dashboard/) is a Gradio web UI tying them
together: a chat box for task instructions, live stereo camera views with
VLM bounding-box overlays, and perceived cube positions.

## Stack

- **Language:** Python 3.10+
- **Simulator:** [CoppeliaSim](https://www.coppeliarobotics.com/) (Edu), driven
  standalone via its Python ZeroMQ remote API — no ROS2/Gazebo for now
- **LLM:** local vision-language model via [Ollama](https://ollama.com)
  (`llava`) — free and offline, no API key
- **Robot/hardware target:** simulation only (UR5 arm + RG2 gripper) — no
  physical robot execution
- **Tooling:** [Ruff](https://docs.astral.sh/ruff/) (lint + format),
  [pytest](https://docs.pytest.org/) (tests)

## Setup

Requires [CoppeliaSim Edu](https://www.coppeliarobotics.com/) installed locally
(the ZeroMQ remote API server starts automatically with it — no extra
configuration needed), and [Ollama](https://ollama.com) installed and running
locally with a vision model pulled.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"

ollama pull llava
```

## Usage

With CoppeliaSim open on a new/empty scene:

```bash
python -m simulation.build_scene         # builds the cubes, UR5+RG2, and stereo cameras
python -m perception.stereo_localization # estimates each cube's XYZ from the two cameras
python -m action.pick_and_place          # picks up fire_cube, places it on top of water_cube

# IK/motion smoke test, robot only (no perception/grasping):
python -m simulation.build_scene --no-cubes --no-cameras
python -m action.test_ik_motion

# VLM scene recognition + grounded pick-and-place (requires Ollama running):
python -m reasoning.scene_recognition
python -m reasoning.vlm_pick_and_place

# Web dashboard: chat + live camera views + bounding boxes (requires Ollama running):
python -m dashboard.app   # serves at http://127.0.0.1:7860
```

## Development

```bash
ruff check .      # lint
ruff format .     # format
pytest            # run tests
```

## Project structure

```
simulation/    CoppeliaSim scene-building and remote API connection
perception/    object detection / 3D localization
reasoning/     local VLM (Ollama) scene description + object grounding; open-instruction planning not started
action/        IK-driven pick-and-place (simIK + RG2 gripper control)
dashboard/     Gradio web UI: chat-driven task input + live camera/bbox/position display
tests/         unit tests for the pure-logic parts of each stage
docs/          thesis notes, references, experiment logs
```

See [CLAUDE.md](CLAUDE.md) for implementation notes and non-obvious gotchas
(e.g. CoppeliaSim's camera orientation/rendering conventions) aimed at
whoever — human or AI — picks this codebase up next.
