"""Standalone IK/motion smoke test for the UR5+RG2, independent of
perception or grasping. Connects to a scene containing just the robot and
drives the gripper through several waypoints with varying position AND
orientation, to confirm the embedded IK actuation script (simIK.handleGroup,
see simulation.build_scene.load_robot) tracks smoothly across a range of
poses, not just the fixed top-down grasp action.pick_and_place uses -- and
that the *whole arm* moves continuously, not just the gripper (see CLAUDE.md
for why action.pick_and_place.Arm no longer solves IK itself).

Run:
    python -m simulation.build_scene --no-cubes --no-cameras
    python -m action.test_ik_motion
"""

import math
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action.pick_and_place import Arm, _pose_matrix
from simulation.connection import connect

# Positions stay within the workspace region action.pick_and_place already
# exercises successfully (roughly where water_cube/fire_cube sit). Orientation
# is given as Euler-angle offsets (degrees) from the default top-down grasp
# pose -- not a rigorous roll/pitch/yaw relative to the gripper, just a way to
# generate a few distinct, non-top-down test orientations.
#
# Unlike the old simIK.findConfigs-based Arm, these don't need to be staged
# through tiny hops -- Arm.move_to interpolates the whole path itself and
# steps the simulation through it, so simIK.handleGroup only ever has to
# track a small, smooth, continuous motion, never a discontinuous jump.
# A *very* large simultaneous position+orientation change in one move_to
# call (tried: straight from x=+0.1/top-down to x=-0.1/roll+15/yaw-45) can
# still land the interpolated path on a real local minimum for this
# differential solver -- expected for any local/gradient-based IK method,
# and exactly what Arm._verify_converged's tolerance check is for. These
# waypoints stay moderate for a reliable smoke test.
WAYPOINTS_DEG = [
    ([0.1, 0.0, 0.15], [0, 0, 0]),
    ([0.05, 0.05, 0.18], [0, 0, 20]),
    ([-0.02, 0.03, 0.22], [10, 0, -15]),
    ([-0.1, 0.0, 0.2], [0, 0, 30]),
    ([0.0, -0.05, 0.16], [0, 15, 0]),
]


def main():
    sim = connect()
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        time.sleep(0.5)

    arm = Arm(sim)
    try:
        default_euler = sim.getEulerAnglesFromMatrix(_pose_matrix([0.0, 0.0, 0.0]))

        for position, delta_deg in WAYPOINTS_DEG:
            orientation = [default_euler[i] + math.radians(delta_deg[i]) for i in range(3)]
            print(f"moving to {position}, orientation delta(deg)={delta_deg}")
            arm.move_to(position, orientation=orientation)
        print("done")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
