import numpy as np

from perception.stereo_localization import detect_color_centroid, pixel_ray, triangulate

RED_RANGE = ((0, 120, 60), (10, 255, 255))


def test_detect_color_centroid_finds_blob_center():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[20:40, 60:100] = (255, 0, 0)  # red block, columns 60-99, rows 20-39

    centroid = detect_color_centroid(image, *RED_RANGE)

    assert centroid is not None
    u, v = centroid
    assert 75 < u < 85
    assert 25 < v < 35


def test_detect_color_centroid_returns_none_when_absent():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert detect_color_centroid(image, *RED_RANGE) is None


def test_pixel_ray_at_principal_point_is_straight_forward():
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    direction = pixel_ray((320.0, 240.0), intrinsics)
    assert np.allclose(direction, [0.0, 0.0, 1.0])


def test_triangulate_recovers_known_intersection_point():
    target = np.array([0.0, 0.0, 5.0])

    p1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([0.0, 0.0, 1.0])

    p2 = np.array([1.0, 0.0, 0.0])
    d2 = target - p2
    d2 = d2 / np.linalg.norm(d2)

    estimated = triangulate(p1, d1, p2, d2)

    assert np.allclose(estimated, target, atol=1e-9)
