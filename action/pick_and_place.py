"""Picks up the red ("fire") cube and places it on top of the blue ("water")
cube, using the XYZ positions perception.stereo_localization estimates from
the two stereo cameras (not ground-truth positions read from the sim).

Requires simulation.build_scene to have already built the scene. The actual
IK solving happens inside CoppeliaSim itself, in a child script attached to
the UR5 by simulation.build_scene.load_robot -- this module only moves that
script's `ik_target` dummy smoothly and steps the simulation; see CLAUDE.md
for why (in short: driving simIK.findConfigs from here, once per Cartesian
sub-step, could not produce smooth whole-arm motion -- only the gripper
itself reliably tracked the target).

Run with CoppeliaSim running and the scene already built:
    python -m action.pick_and_place
"""

import math
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.stereo_localization import locate_cubes
from simulation.build_scene import ARM_JOINT_SUBPATHS, CUBE_SIZE, IK_TARGET_ALIAS
from simulation.connection import connect
from simulation.geometry import cross, normalize

SAFE_HEIGHT = 0.2
GRASP_APPROACH_OFFSET = 0.1  # above the cube before/after grasping
GRIPPER_SETTLE_TIME = 1.0

MOVE_STEPS = 40
POSITION_TOLERANCE = 0.02  # meters
ORIENTATION_TOLERANCE_DEG = 8.0

# The gripper's tip (RG2/attachPoint) approach axis is its local +Z; pointing
# it at world (0, 0, -1) gives a top-down grasp (see CLAUDE.md for how this
# was derived/verified).
_UP_HINT = [0.0, 1.0, 0.0]
_Z_AXIS = [0.0, 0.0, -1.0]
_X_AXIS = normalize(cross(_UP_HINT, _Z_AXIS))
_Y_AXIS = cross(_Z_AXIS, _X_AXIS)


def _pose_matrix(position):
    return [
        _X_AXIS[0],
        _Y_AXIS[0],
        _Z_AXIS[0],
        position[0],
        _X_AXIS[1],
        _Y_AXIS[1],
        _Z_AXIS[1],
        position[1],
        _X_AXIS[2],
        _Y_AXIS[2],
        _Z_AXIS[2],
        position[2],
    ]


class Arm:
    """Drives the UR5+RG2 by moving the scene's `ik_target` dummy; the
    embedded child script simulation.build_scene.load_robot attaches to the
    UR5 solves IK for it every simulation step via simIK.handleGroup, a
    differential solver that evolves the *current* joint configuration
    smoothly toward the target rather than jumping to an independently-found
    one. This class just has to move the target smoothly and step the
    simulation -- it no longer manages any simIK environment/group itself."""

    def __init__(self, sim):
        self.sim = sim
        self.ur5 = sim.getObject("/UR5")
        self.tip = sim.getObject("/UR5/RG2/attachPoint")
        self.target = sim.getObject("/" + IK_TARGET_ALIAS)
        root_path = sim.getObjectAlias(self.ur5, 1)
        self.joints = [sim.getObject(root_path + subpath) for subpath in ARM_JOINT_SUBPATHS]

        # Only the base/shoulder links, not the whole arm: adjacent,
        # physically touching links (e.g. the gripper mounted flush on the
        # wrist) always report as "colliding", so checking the whole tree
        # against itself is unusable -- this narrower pair still catches the
        # realistic failure (gripper swinging back into its own base).
        gripper_handle = sim.getObject(root_path + "/RG2")
        self.gripper_collection = sim.createCollection(0)
        sim.addItemToCollection(self.gripper_collection, sim.handle_tree, gripper_handle, 0)
        self.shoulder_collection = sim.createCollection(0)
        for name in ("link1_visible", "link2_visible"):
            link_handle = sim.getObject(root_path + "/" + name)
            sim.addItemToCollection(self.shoulder_collection, sim.handle_single, link_handle, 0)

        self._default_orientation_quaternion = sim.getQuaternionFromMatrix(
            _pose_matrix([0.0, 0.0, 0.0])
        )
        self._dt = sim.getSimulationTimeStep()
        # The embedded actuation script only re-solves IK on an actual
        # simulation step; without stepped mode, moving the target and
        # sleeping in Python wouldn't advance the simulation at all.
        sim.setStepping(True)

    def close(self):
        self.sim.destroyCollection(self.gripper_collection)
        self.sim.destroyCollection(self.shoulder_collection)
        self.sim.setStepping(False)

    def move_to(self, position, orientation=None, steps=MOVE_STEPS, pace_realtime=True):
        """Smoothly move the gripper to world-frame `position` (and, if
        given, world-frame Euler-angle `orientation`; defaults to the fixed
        top-down grasp pose).

        Interpolates position and orientation together as a single pose
        (`sim.interpolatePoses`, position lerp + quaternion slerp) starting
        from the tip's *actual* current pose, not wherever the target was
        last commanded to -- if the previous move didn't fully converge,
        this keeps the next one grounded in reality instead of compounding
        the error into a sudden jump. One simulation step is advanced per
        waypoint so the embedded actuation script's simIK.handleGroup call
        can track it; `pace_realtime` sleeps out the difference between that
        step and the simulation's own timestep so playback looks like real
        robot speed rather than a sped-up blur.
        """
        if orientation is None:
            target_quaternion = self._default_orientation_quaternion
        else:
            target_quaternion = self.sim.getQuaternionFromMatrix(
                self.sim.buildMatrix([0.0, 0.0, 0.0], orientation)
            )
        target_pose = [*position, *target_quaternion]
        start_pose = self.sim.getObjectPose(self.tip, self.sim.handle_world)

        for step in range(1, steps + 1):
            t = step / steps
            eased = t * t * (3 - 2 * t)  # smoothstep: eases in/out instead of a linear crawl
            waypoint_pose = self.sim.interpolatePoses(start_pose, target_pose, eased)
            self.sim.setObjectPose(self.target, self.sim.handle_world, waypoint_pose)
            self._step(pace_realtime)

        self._verify_converged(position, target_quaternion, steps)

    def _step(self, pace_realtime):
        step_start = time.time()
        self.sim.step()
        if pace_realtime:
            time.sleep(max(0.0, self._dt - (time.time() - step_start)))

    def _verify_converged(self, position, target_quaternion, steps):
        """`simIK.handleGroup` is a local/differential solver -- like
        `simIK.findConfigs` before it, it can get stuck in a local minimum
        for some poses even though they're nominally within reach (observed
        empirically: 1 of 5 test orientations in action/test_ik_motion.py
        failed to converge). Check the actual tracking error rather than
        assume the move succeeded just because it ran; per the project's
        established rule, don't force a bad pose through, flag it clearly."""
        if not self._is_collision_free():
            raise RuntimeError(
                f"gripper self-collision detected while moving toward {position} -- "
                "the wrist likely swung back into the arm's own base for this pose."
            )
        tip_position = self.sim.getObjectPosition(self.tip, self.sim.handle_world)
        position_error = math.sqrt(sum((tip_position[i] - position[i]) ** 2 for i in range(3)))
        tip_quaternion = self.sim.getObjectQuaternion(self.tip, self.sim.handle_world)
        dot = min(1.0, abs(sum(tip_quaternion[i] * target_quaternion[i] for i in range(4))))
        orientation_error_deg = math.degrees(2 * math.acos(dot))
        if position_error > POSITION_TOLERANCE or orientation_error_deg > ORIENTATION_TOLERANCE_DEG:
            raise RuntimeError(
                f"failed to converge to {position} (position error {position_error * 1000:.1f}mm, "
                f"orientation error {orientation_error_deg:.1f} deg) after {steps} steps -- "
                "likely a local minimum for this differential IK solver, "
                "or a genuinely hard/near-singular pose; try an intermediate waypoint instead of "
                "forcing more iterations/damping."
            )

    def _is_collision_free(self):
        self_hit, _ = self.sim.checkCollision(self.gripper_collection, self.shoulder_collection)
        return not self_hit

    def set_gripper(self, open_gripper, pace_realtime=True):
        self.sim.setInt32Signal("RG2_open", 1 if open_gripper else 0)
        steps = max(1, round(GRIPPER_SETTLE_TIME / self._dt))
        for _ in range(steps):
            self._step(pace_realtime)

    def attach(self, object_handle):
        """Parent `object_handle` to the gripper so it moves along with it.

        RG2's own attachPoint/attachProxSensor look like they should do this
        automatically on close, but empirically they don't (the fingers do
        physically stop against the object, they just never reparent it) --
        so grasping is done explicitly here instead of relying on RG2's script.
        """
        self.sim.setObjectParent(object_handle, self.tip, True)

    def detach(self, object_handle):
        self.sim.setObjectParent(object_handle, self.sim.handle_world, True)


def pick_and_place(sim, pick_object, pick_position, place_position):
    arm = Arm(sim)
    try:
        above_pick = [pick_position[0], pick_position[1], pick_position[2] + GRASP_APPROACH_OFFSET]
        above_place = [
            place_position[0],
            place_position[1],
            place_position[2] + GRASP_APPROACH_OFFSET,
        ]
        safe = [pick_position[0], pick_position[1], SAFE_HEIGHT]

        arm.set_gripper(open_gripper=True)
        arm.move_to(above_pick)
        arm.move_to(pick_position)
        arm.set_gripper(open_gripper=False)
        arm.attach(pick_object)
        arm.move_to(above_pick)
        arm.move_to(safe)
        arm.move_to(above_place)
        arm.move_to(place_position)
        arm.detach(pick_object)
        arm.set_gripper(open_gripper=True)
        arm.move_to(above_place)
    finally:
        arm.close()


def main():
    sim = connect()
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        time.sleep(0.5)

    positions = locate_cubes(sim)
    fire = np.array(positions["fire_cube"])
    water = np.array(positions["water_cube"])
    print(f"fire_cube (pick) at {fire}")
    print(f"water_cube (place base) at {water}")

    fire_cube = sim.getObject("/fire_cube")
    place_position = [water[0], water[1], water[2] + CUBE_SIZE]
    pick_and_place(sim, fire_cube, fire.tolist(), place_position)
    print("done")


if __name__ == "__main__":
    main()
