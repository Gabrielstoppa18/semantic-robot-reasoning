"""Estimates the 3D (world) XYZ position of the blue ("water") and red
("fire") cubes by triangulating their color-blob centroid from the two
fixed stereo vision sensors created by simulation.build_scene.

This is a first, deliberately simple perception stage: color segmentation
instead of a learned detector. Swapping in a real object detector later
should only require replacing `detect_color_centroid`.

Run with CoppeliaSim running and the scene already built:
    python -m perception.stereo_localization
"""

import os
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.connection import connect

CAMERA_NAMES = ["left_camera", "right_camera"]

# HSV thresholds for the cube colors used in simulation.build_scene.
COLOR_RANGES = {
    "water_cube": ((100, 120, 60), (140, 255, 255)),  # blue
    "fire_cube": ((0, 120, 60), (10, 255, 255)),  # red (low hue wrap)
}


def get_image(sim, camera_handle):
    buffer, resolution = sim.getVisionSensorImg(camera_handle)
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    return np.flipud(image)  # CoppeliaSim images are bottom-up


def detect_color_centroid(image, hsv_low, hsv_high):
    """Return the (u, v) pixel centroid (top-down v) of the largest blob
    matching the given HSV range, or None if nothing matches."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def camera_intrinsics(sim, camera_handle):
    res_x = sim.getObjectInt32Param(camera_handle, sim.visionintparam_resolution_x)
    res_y = sim.getObjectInt32Param(camera_handle, sim.visionintparam_resolution_y)
    fov = sim.getObjectFloatParam(camera_handle, sim.visionfloatparam_perspective_angle)
    # CoppeliaSim applies `fov` to the larger resolution dimension.
    fx = (max(res_x, res_y) / 2) / np.tan(fov / 2)
    fy = fx
    return {"res_x": res_x, "res_y": res_y, "fx": fx, "fy": fy, "cx": res_x / 2, "cy": res_y / 2}


def pixel_ray(pixel, intrinsics):
    """Unit direction vector (in the camera's local frame) through `pixel`.

    Local frame convention: +X left, +Y up, +Z forward (verified empirically
    against known cube positions: image column u increases as local X
    decreases, i.e. the rendered image is mirrored left/right relative to
    the camera's local +X axis; a vision sensor at identity orientation
    looks toward world +Z).
    """
    u, v = pixel
    x = (intrinsics["cx"] - u) / intrinsics["fx"]
    y = (intrinsics["cy"] - v) / intrinsics["fy"]  # image v grows downward, local Y is up
    direction = np.array([x, y, 1.0])
    return direction / np.linalg.norm(direction)


def camera_pose(sim, camera_handle):
    """Return (position, rotation) of the camera in world coordinates.
    `rotation` columns are the camera's local X/Y/Z axes expressed in world."""
    matrix = np.array(sim.getObjectMatrix(camera_handle, sim.handle_world)).reshape(3, 4)
    return matrix[:, 3], matrix[:, :3]


def triangulate(p1, d1, p2, d2):
    """Closest point between two rays (p1 + s*d1) and (p2 + t*d2)."""
    a = d1.dot(d1)
    b = d1.dot(d2)
    c = d2.dot(d2)
    d = d1.dot(p1 - p2)
    e = d2.dot(p1 - p2)
    denom = a * c - b * b
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    return (closest1 + closest2) / 2


def locate_cubes(sim):
    cameras = {name: sim.getObject("/" + name) for name in CAMERA_NAMES}
    images = {name: get_image(sim, handle) for name, handle in cameras.items()}

    results = {}
    for cube_name, (hsv_low, hsv_high) in COLOR_RANGES.items():
        rays = {}
        for name, handle in cameras.items():
            pixel = detect_color_centroid(images[name], hsv_low, hsv_high)
            if pixel is None:
                rays[name] = None
                continue
            intrinsics = camera_intrinsics(sim, handle)
            position, rotation = camera_pose(sim, handle)
            direction_world = rotation @ pixel_ray(pixel, intrinsics)
            rays[name] = (position, direction_world)

        if any(r is None for r in rays.values()):
            results[cube_name] = None
            continue

        (p1, d1), (p2, d2) = rays.values()
        results[cube_name] = triangulate(p1, d1, p2, d2)

    return results


def main():
    sim = connect()
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        time.sleep(1.0)  # let the vision sensors render at least one frame

    positions = locate_cubes(sim)
    for name, position in positions.items():
        if position is None:
            print(f"{name}: not detected in both cameras")
        else:
            print(f"{name}: x={position[0]:.3f} y={position[1]:.3f} z={position[2]:.3f}")


if __name__ == "__main__":
    main()
