"""Builds the thesis's example scene in the currently running CoppeliaSim
instance:

- a blue cube ("water") and a red cube ("fire") on the default floor
- a UR5 arm with an RG2 gripper, base set back from the cubes
- a fixed stereo pair of vision sensors overlooking the workspace, used by
  the perception stage to estimate the cubes' XYZ position

Run with CoppeliaSim already open on a new/empty scene:
    python -m simulation.build_scene

Pass --no-cubes and/or --no-cameras to build a reduced scene (e.g. just the
robot on its pedestal, for IK/motion testing in isolation):
    python -m simulation.build_scene --no-cubes --no-cameras
"""

import argparse
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

# Mounting the arm at floor level forces it to reach down almost to its own
# base height to grasp the cubes -- a near-singular, poorly-conditioned
# configuration that made full-pose IK (position + orientation) unsolvable.
# Raising the base onto a pedestal gives the arm a proper "reach down and
# out" geometry instead, which is comfortably within reach and well-posed.
PEDESTAL_HEIGHT = 0.2
PEDESTAL_SIZE = [0.15, 0.15, PEDESTAL_HEIGHT]
ROBOT_BASE_POS = [0.0, 0.85, PEDESTAL_HEIGHT]

# Relative sub-paths (from the UR5 model root) of the 6 arm joints, in
# base-to-wrist order. UR5.ttm nests each joint under the previous link.
ARM_JOINT_SUBPATHS = [
    "/joint",
    "/link/joint",
    "/link/joint/link/joint",
    "/link/joint/link/joint/link/joint",
    "/link/joint/link/joint/link/joint/link/joint",
    "/joint/link/joint/link/joint/link/joint/link/joint/link/joint",
]

# UR5's all-zero pose is a singular, fully-extended "candle" configuration --
# bad both for simIK.findConfigs's local search (see action/pick_and_place.py)
# and for the differential IK (simIK.handleGroup) the embedded actuation
# script below uses, which needs a well-conditioned Jacobian to start from.
# Seeded here, at scene-build time, so the arm is already in a good starting
# configuration the instant the embedded script's sysCall_init runs (i.e.
# before action.pick_and_place ever constructs an Arm).
#
# This specific pose was found by a small grid search (not hand-derived) for
# one whose *resulting tool-tip orientation* is already close to the fixed
# top-down grasp orientation action/pick_and_place.py always targets -- an
# earlier bent-elbow pose ([0,-90,90,-90,-90,0], reachable enough for
# findConfigs's random search) turned out to point the tip in an orientation
# 180 degrees away from top-down. That's invisible to a global/randomized
# solver, but simIK.handleGroup is a *local* differential solver: tracking a
# smooth path from a starting orientation 180 degrees off target got
# permanently stuck (constant ~150mm/~33deg error regardless of step count
# or time budget -- confirmed this was a stuck local minimum, not merely
# slow convergence). This pose's tip orientation matches top-down to
# <0.001 degrees, so the very first move a fresh Arm makes never needs a
# large orientation change, only a position one.
ARM_READY_POSE_DEG = [0, -120, 120, -60, 90, 0]

IK_TARGET_ALIAS = "ik_target"

# Damped least squares (not the undamped default): the arm passes through
# poorly-conditioned configurations reaching down to the workspace from the
# pedestal, and damping keeps the differential solver stable there instead of
# producing huge corrective joint jumps. Values match the bundled CoppeliaSim
# example this pattern is taken from (scenes/kinematics/smoothMovementsInFkAndIk.ttt).
IK_DAMPING = 0.05
IK_MAX_ITERATIONS = 10

# A child script attached to the UR5 that re-solves IK every simulation step
# (sysCall_actuation), continuously tracking whatever pose `ik_target` is
# currently at. This is CoppeliaSim's own documented pattern for smooth,
# whole-body IK-driven motion (see scenes/kinematics/smoothMovementsInFkAndIk.ttt
# and scenes/messaging/movementViaRemoteApi.ttt, both bundled with CoppeliaSim) --
# action/pick_and_place.py used to instead call simIK.findConfigs itself, once
# per Cartesian sub-step, from the external Python client. That approach never
# reliably produced smooth *whole-arm* motion: findConfigs is a randomized
# search that returns *a* valid configuration near the target, not the same
# configuration smoothly evolved from the previous step, so independent calls
# a few sub-steps apart could jump between different valid elbow/shoulder
# solutions -- visible as the rest of the arm twitching even while the
# gripper itself tracked the target fine. simIK.handleGroup is a genuinely
# differential/incremental solver: every call refines the *current* joint
# configuration a small step further toward the target, so consecutive calls
# stay continuous by construction. Running it in `sysCall_actuation` (rather
# than driving it from the external Python client) keeps it synced to the
# actual physics step, at zero remote-API round-trip cost per solve --
# `sim.moveToConfig`/`sim.moveToPose` driven directly from external Python
# were tried for this earlier and abandoned as "too slow" (see git history),
# but that was because nothing was calling `sim.step()` to actually advance
# the simulation; `action.pick_and_place.Arm` now does that itself, once per
# waypoint, via `sim.setStepping(True)` + `sim.step()`.
#
# Two important properties this buys, verified empirically against this exact
# scene: (1) simIK.addElementFromScene's constraint/element setup only needs
# to run once, at simulation start (sysCall_init), rather than once per
# `Arm()` construction -- so the ikEnv/ikGroup leak that used to require
# `Arm.close()` bookkeeping (simIK.createEnvironment()/createGroup() are not
# garbage-collected) can no longer happen from repeated Arm() construction,
# since there's no longer a Python-side environment to leak. (2) it is a
# *local* solver -- like findConfigs, it can still get stuck in a local
# minimum for some poses (observed: one out of five test orientations in
# action/test_ik_motion.py). `Arm.move_to` checks final tracking error against
# a tolerance and raises rather than silently accepting an unconverged pose.
_IK_ACTUATION_SCRIPT_TEMPLATE = """
sim = require("sim")
simIK = require("simIK")

function sysCall_init()
    simBase = sim.getObject("{base_path}")
    simTip = sim.getObject("{tip_path}")
    simTarget = sim.getObject("{target_path}")
    ikEnv = simIK.createEnvironment()
    ikGroup = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(
        ikEnv, ikGroup, simIK.method_damped_least_squares, {damping}, {max_iterations}
    )
    simIK.addElementFromScene(ikEnv, ikGroup, simBase, simTip, simTarget, simIK.constraint_pose)
end

function sysCall_actuation()
    simIK.handleGroup(ikEnv, ikGroup, {{syncWorlds = true}})
end

function sysCall_cleanup()
    simIK.eraseEnvironment(ikEnv)
end
"""

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


def create_pedestal(sim, robot_base_position):
    """Static stand the UR5 is mounted on (see the comment on ROBOT_BASE_POS)."""
    handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, PEDESTAL_SIZE)
    sim.setObjectAlias(handle, "robot_pedestal")
    center = [robot_base_position[0], robot_base_position[1], robot_base_position[2] / 2]
    sim.setObjectPosition(handle, sim.handle_world, center)
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, [0.5, 0.5, 0.5])
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

    # The bundled UR5 model has its own demo-behavior script that fights any
    # externally-driven joint control (it silently resets joint angles every
    # step). Remove it, and disable dynamics on the whole arm so the base
    # doesn't fall/drift once its joints are switched to kinematic mode below
    # -- action/pick_and_place.py drives the arm directly via IK instead.
    ur5_script = find_child_by_name(sim, ur5, "Script")
    sim.removeObjects([ur5_script])
    sim.setModelProperty(ur5, sim.modelproperty_not_dynamic)

    root_path = sim.getObjectAlias(ur5, 1)
    arm_joints = [sim.getObject(root_path + subpath) for subpath in ARM_JOINT_SUBPATHS]
    for joint in arm_joints:
        sim.setJointMode(joint, sim.jointmode_kinematic, 0)
    for joint, angle_deg in zip(arm_joints, ARM_READY_POSE_DEG, strict=True):
        sim.setJointPosition(joint, math.radians(angle_deg))

    tip_path = root_path + "/RG2/attachPoint"
    tip = sim.getObject(tip_path)
    target = sim.createDummy(0.02)
    sim.setObjectAlias(target, IK_TARGET_ALIAS)
    sim.setObjectPose(target, sim.handle_world, sim.getObjectPose(tip, sim.handle_world))

    script_code = _IK_ACTUATION_SCRIPT_TEMPLATE.format(
        base_path=root_path,
        tip_path=tip_path,
        target_path="/" + IK_TARGET_ALIAS,
        damping=IK_DAMPING,
        max_iterations=IK_MAX_ITERATIONS,
    )
    script_handle = sim.addScript(sim.scripttype_childscript)
    sim.setScriptText(script_handle, script_code)
    sim.associateScriptWithObject(script_handle, ur5)

    return ur5, gripper, arm_joints, target


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cubes", action="store_true", help="skip water_cube/fire_cube")
    parser.add_argument("--no-cameras", action="store_true", help="skip the stereo camera pair")
    args = parser.parse_args()

    sim = connect()

    if not args.no_cubes:
        water = create_cube(sim, "water_cube", WATER_CUBE_POS, BLUE)
        fire = create_cube(sim, "fire_cube", FIRE_CUBE_POS, RED)
        print(f"water_cube={water} fire_cube={fire}")

    create_pedestal(sim, ROBOT_BASE_POS)
    ur5, gripper, _arm_joints, _target = load_robot(sim, ROBOT_BASE_POS)
    print(f"UR5={ur5} RG2={gripper}")

    if not args.no_cameras:
        left_cam, right_cam = create_stereo_cameras(sim)
        print(f"left_camera={left_cam} right_camera={right_cam}")


if __name__ == "__main__":
    main()
