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
| Semantic grounding (LLM infers object roles) | [`reasoning/`](reasoning/) | not started |
| Reasoning & planning (LLM produces action sequence) | [`reasoning/`](reasoning/) | not started |
| Action execution (robot motion commands) | [`action/`](action/) | not started |

[`simulation/`](simulation/) is the glue/infrastructure layer: it builds the
CoppeliaSim scene and holds the remote API connection helper shared by the
other stages.

## Stack

- **Language:** Python 3.10+
- **Simulator:** [CoppeliaSim](https://www.coppeliarobotics.com/) (Edu), driven
  standalone via its Python ZeroMQ remote API — no ROS2/Gazebo for now
- **LLM:** external API (e.g. Anthropic/OpenAI) — no local model
- **Robot/hardware target:** simulation only (UR5 arm + RG2 gripper) — no
  physical robot execution
- **Tooling:** [Ruff](https://docs.astral.sh/ruff/) (lint + format),
  [pytest](https://docs.pytest.org/) (tests)

## Setup

Requires [CoppeliaSim Edu](https://www.coppeliarobotics.com/) installed locally
(the ZeroMQ remote API server starts automatically with it — no extra
configuration needed).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"
```

## Usage

With CoppeliaSim open on a new/empty scene:

```bash
python -m simulation.build_scene         # builds the cubes, UR5+RG2, and stereo cameras
python -m perception.stereo_localization # estimates each cube's XYZ from the two cameras
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
reasoning/     LLM integration, semantic grounding, action planning (not started)
action/        action-to-robot-command translation, motion control (not started)
tests/         unit tests for the pure-logic parts of each stage
docs/          thesis notes, references, experiment logs
```

See [CLAUDE.md](CLAUDE.md) for implementation notes and non-obvious gotchas
(e.g. CoppeliaSim's camera orientation/rendering conventions) aimed at
whoever — human or AI — picks this codebase up next.
