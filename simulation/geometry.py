"""Small geometry helpers shared by scene-building and perception code."""

import math


def normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def scale(a, s):
    return [c * s for c in a]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def look_at_euler(sim, position, target, up=(0, 0, 1)):
    """Euler angles (for sim.setObjectOrientation) that make an object's
    local +Z axis point from `position` toward `target`.

    CoppeliaSim vision sensors (and cameras) look along their local +Z axis,
    confirmed empirically: identity orientation looks toward world +Z.
    """
    z_axis = normalize(sub(target, position))
    x_axis = normalize(cross(up, z_axis))
    y_axis = cross(z_axis, x_axis)
    matrix = [
        x_axis[0],
        y_axis[0],
        z_axis[0],
        position[0],
        x_axis[1],
        y_axis[1],
        z_axis[1],
        position[1],
        x_axis[2],
        y_axis[2],
        z_axis[2],
        position[2],
    ]
    return sim.getEulerAnglesFromMatrix(matrix)
