import math

from simulation.geometry import add, cross, dot, normalize, scale, sub


def test_normalize_unit_length():
    result = normalize([3.0, 4.0, 0.0])
    assert math.isclose(math.sqrt(sum(c * c for c in result)), 1.0)
    assert result == [0.6, 0.8, 0.0]


def test_cross_of_orthonormal_axes():
    assert cross([1, 0, 0], [0, 1, 0]) == [0, 0, 1]
    assert cross([0, 1, 0], [0, 0, 1]) == [1, 0, 0]


def test_sub_and_add_are_inverses():
    a, b = [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]
    assert add(sub(a, b), b) == a


def test_scale():
    assert scale([1.0, -2.0, 3.0], 2) == [2.0, -4.0, 6.0]


def test_dot_orthogonal_vectors_is_zero():
    assert dot([1, 0, 0], [0, 1, 0]) == 0
    assert dot([2, 3, 4], [2, 3, 4]) == 29
