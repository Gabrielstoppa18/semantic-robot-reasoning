"""Builds the thesis's example scene in the currently running CoppeliaSim
instance:

- a blue cube ("water") and a red cube ("fire") on the default floor
- a UR5 arm with an RG2 gripper, base set back from the cubes
- a fixed stereo pair of vision sensors overlooking the workspace, used by
  the perception stage to estimate the cubes' XYZ position

Run with CoppeliaSim already open on a new/empty scene:
    python -m simulation.build_scene
"""

import math
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.connection import connect
from simulation.geometry import look_at_euler

COPPELIASIM_DIR = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu"
UR5_MODEL = os.path.join(COPPELIASIM_DIR, "models", "robots", "non-mobile", "UR5.ttm")
RG2_MODEL = os.path.join(COPPELIASIM_DIR, "models", "components", "grippers", "RG2.ttm")

CUBE_SIZE = 0.05
WATER_CUBE_POS = [-0.1, 0.0, CUBE_SIZE / 2]
FIRE_CUBE_POS = [0.1, 0.0, CUBE_SIZE / 2]
BLUE = [0.0, 0.0, 1.0]
RED = [1.0, 0.0, 0.0]

ROBOT_BASE_POS = [0.0, 0.4, 0.0]

CAMERA_RESOLUTION = [640, 480]
CAMERA_FOV_DEG = 60
CAMERA_NEAR_CLIP = 0.01
CAMERA_FAR_CLIP = 5.0
WORKSPACE_TARGET = [0.0, 0.0, 0.03]
LEFT_CAMERA_POS = [-0.075, -0.6, 0.35]
RIGHT_CAMERA_POS = [0.075, -0.6, 0.35]


def create_cube(sim, name, position, color):
    handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [CUBE_SIZE] * 3)
    sim.setObjectAlias(handle, name)
    sim.setObjectPosition(handle, sim.handle_world, position)
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)
    return handle


def find_child_by_name(sim, root, name):
    for handle in sim.getObjectsInTree(root, sim.handle_all, 0):
        if sim.getObjectAlias(handle) == name:
            return handle
    raise LookupError(f"no child named {name!r} under object {root}")


def load_robot(sim, base_position):
    ur5 = sim.loadModel(UR5_MODEL)
    sim.setObjectPosition(ur5, sim.handle_world, base_position)
    connection = find_child_by_name(sim, ur5, "connection")

    gripper = sim.loadModel(RG2_MODEL)
    sim.setObjectPosition(gripper, connection, [0, 0, 0])
    sim.setObjectOrientation(gripper, connection, [0, 0, 0])
    sim.setObjectParent(gripper, connection, True)

    return ur5, gripper


def create_camera(sim, name, position, target):
    # options=2 -> perspective mode
    float_params = [
        CAMERA_NEAR_CLIP,
        CAMERA_FAR_CLIP,
        math.radians(CAMERA_FOV_DEG),
        0.02,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    handle = sim.createVisionSensor(2, [*CAMERA_RESOLUTION, 0, 0], float_params)
    sim.setObjectAlias(handle, name)
    sim.setObjectPosition(handle, sim.handle_world, position)
    sim.setObjectOrientation(handle, sim.handle_world, look_at_euler(sim, position, target))
    return handle


def create_stereo_cameras(sim):
    left = create_camera(sim, "left_camera", LEFT_CAMERA_POS, WORKSPACE_TARGET)
    right = create_camera(sim, "right_camera", RIGHT_CAMERA_POS, WORKSPACE_TARGET)
    return left, right


def main():
    sim = connect()

    water = create_cube(sim, "water_cube", WATER_CUBE_POS, BLUE)
    fire = create_cube(sim, "fire_cube", FIRE_CUBE_POS, RED)
    ur5, gripper = load_robot(sim, ROBOT_BASE_POS)
    left_cam, right_cam = create_stereo_cameras(sim)

    print(f"water_cube={water} fire_cube={fire}")
    print(f"UR5={ur5} RG2={gripper}")
    print(f"left_camera={left_cam} right_camera={right_cam}")


if __name__ == "__main__":
    main()
