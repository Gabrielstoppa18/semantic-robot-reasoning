import numpy as np

from action.pick_and_place import _pose_matrix


def test_pose_matrix_places_position_in_last_column():
    matrix = _pose_matrix([1.0, 2.0, 3.0])
    assert matrix[3] == 1.0
    assert matrix[7] == 2.0
    assert matrix[11] == 3.0


def test_pose_matrix_rotation_part_is_orthonormal():
    matrix = _pose_matrix([0.0, 0.0, 0.0])
    rotation = np.array(matrix).reshape(3, 4)[:, :3]
    should_be_identity = rotation.T @ rotation
    assert np.allclose(should_be_identity, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)
