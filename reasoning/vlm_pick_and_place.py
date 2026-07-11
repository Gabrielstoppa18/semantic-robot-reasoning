"""Task 2: use the VLM to identify "the blue cube" in the scene, then pick
it up and place it at a new position -- a minimal end-to-end test of the
perceive -> ground -> act loop, without the fire/water semantic-reasoning
example yet (see CLAUDE.md's project overview for that).

The VLM only identifies *which* object is "the blue cube" (and gives a
rough bounding box, printed for inspection); the actual pick/place XYZ still
comes from `perception.stereo_localization`'s validated stereo pipeline, not
from the VLM's bounding box -- see `scene_recognition.py`'s docstring for why.

`locate_and_move()` is the reusable piece of this -- `dashboard/app.py` calls
it directly (given a color extracted from a chat message) rather than
shelling out to this script.

Requires simulation.build_scene to have already built the (full) scene, and
Ollama running locally with a vision model pulled.

Run:
    python -m simulation.build_scene
    python -m reasoning.vlm_pick_and_place
"""

import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action.pick_and_place import pick_and_place
from perception.stereo_localization import locate_cubes
from reasoning.scene_recognition import describe_scene, locate_object
from simulation.build_scene import CUBE_SIZE
from simulation.connection import connect

# Move the picked cube 15cm along +Y from wherever it currently is --
# comfortably within the same reachable region action.pick_and_place already
# exercises, and clear of the other cube (which stays at y=0).
PLACE_OFFSET_Y = 0.15

# Maps the color word the VLM is expected to use back to this scene's
# ground-truth cube names (perception.stereo_localization.COLOR_RANGES).
_COLOR_TO_CUBE_NAME = {"blue": "water_cube", "red": "fire_cube"}


def _cube_name_for_label(label):
    label_lower = label.lower()
    for color, cube_name in _COLOR_TO_CUBE_NAME.items():
        if color in label_lower:
            return cube_name
    raise ValueError(f"couldn't map VLM label {label!r} to a known cube color")


def locate_and_move(sim, target_label, place_offset_y=PLACE_OFFSET_Y):
    """Ask the VLM to find `target_label` (e.g. "the blue cube"), map it to
    this scene's ground-truth cube, and move it `place_offset_y` along +Y via
    `action.pick_and_place`. Returns a dict describing what happened; raises
    `RuntimeError` if the VLM can't find it or perception can't localize it
    (`action.pick_and_place` itself raises `RuntimeError` if the arm can't
    converge -- left to propagate rather than caught here, same "explicit
    guard, don't paper over it" rule as the rest of the action stage)."""
    detection = locate_object(sim, target_label)
    if not detection["found"]:
        raise RuntimeError(f"VLM could not find {target_label!r} in the scene")

    cube_name = _cube_name_for_label(detection["label"])
    positions = locate_cubes(sim)
    pick_position = positions[cube_name]
    if pick_position is None:
        raise RuntimeError(f"perception.stereo_localization could not find {cube_name!r}")
    pick_position = pick_position.tolist()

    place_position = [pick_position[0], pick_position[1] + place_offset_y, CUBE_SIZE / 2]
    cube_handle = sim.getObject("/" + cube_name)
    pick_and_place(sim, cube_handle, pick_position, place_position)

    return {
        "detection": detection,
        "cube_name": cube_name,
        "pick_position": pick_position,
        "place_position": place_position,
    }


def main():
    sim = connect()
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        time.sleep(1.0)

    print("Scene description from the VLM:")
    print(describe_scene(sim))
    print()

    result = locate_and_move(sim, "the blue cube")
    print(f"VLM localization: {result['detection']}")
    print(f"picking up {result['cube_name']} at {result['pick_position']}, ", end="")
    print(f"placing at {result['place_position']}")
    print("done")


if __name__ == "__main__":
    main()
