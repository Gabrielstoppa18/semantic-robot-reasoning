"""Uses a local VLM (see `vlm_client.py`) to recognize the scene and give a
rough visual estimate of where a described object is in the camera image.

This is deliberately separate from `perception.stereo_localization`: that
module's color-blob + stereo triangulation is what `action.pick_and_place`
actually grasps with (~1mm accuracy, validated against ground truth). The
VLM's bounding box here is a coarse, language-driven identification /
cross-check signal, not a replacement for it -- vision-language models are
not reliable at precise spatial localization.

Run with CoppeliaSim running and the scene already built, and Ollama running
locally with a vision model pulled (see `vlm_client.DEFAULT_MODEL`):
    python -m reasoning.scene_recognition
"""

import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.stereo_localization import get_image
from reasoning.vlm_client import ask, ask_json

_BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "label": {"type": "string"},
        "bbox_normalized": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["found", "label", "bbox_normalized"],
}


def capture_image(sim, camera_name="left_camera"):
    camera = sim.getObject("/" + camera_name)
    return get_image(sim, camera)


def describe_scene(sim, camera_name="left_camera"):
    """Ask the VLM to describe, in its own words, what it sees -- the
    scene-recognition capability check before any grounding/planning."""
    image = capture_image(sim, camera_name)
    prompt = (
        "This is an image of a robot manipulation workspace. Briefly describe "
        "the objects you see: their color, shape, and approximate position in "
        "the image (e.g. left/right/center, near/far)."
    )
    return ask(image, prompt)


def locate_object(sim, label, camera_name="left_camera"):
    """Ask the VLM whether `label` (e.g. "the blue cube") is visible, and for
    its approximate bounding box. Returns a dict with pixel-space `bbox`
    (x_min, y_min, x_max, y_max) and `center`, or `{"found": False}` if the
    VLM couldn't find it."""
    image = capture_image(sim, camera_name)
    height, width = image.shape[:2]
    prompt = (
        f"This is an image of a robot manipulation workspace. Find {label}. "
        "Respond with whether it is visible, and its bounding box as "
        "[x_min, y_min, x_max, y_max], normalized to the image width and "
        "height (values from 0.0 to 1.0, origin at the top-left corner)."
    )
    result = ask_json(image, prompt, _BBOX_SCHEMA)

    if not result.get("found"):
        return {"found": False, "label": label}

    x_min, y_min, x_max, y_max = result["bbox_normalized"]
    bbox = (x_min * width, y_min * height, x_max * width, y_max * height)
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return {"found": True, "label": result.get("label", label), "bbox": bbox, "center": center}


def main():
    from simulation.connection import connect

    sim = connect()
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        time.sleep(1.0)  # let the vision sensors render at least one frame

    print("Scene description:")
    print(describe_scene(sim))

    print()
    print("Locating the blue cube:")
    print(locate_object(sim, "the blue cube"))


if __name__ == "__main__":
    main()
