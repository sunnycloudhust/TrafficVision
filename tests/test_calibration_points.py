from itertools import combinations
import numpy as np
import pytest
from trafficvision.geometry.calibration_points import (
    metric_coordinates,
    locate_from_three_anchors,
    rectangle_coordinates,
    validate_correspondences,
)


def distances(points):
    return {
        (i, j): float(np.linalg.norm(points[i] - points[j])) for i, j in combinations(range(len(points)), 2)
    }


def test_pairwise_reconstruction_preserves_measured_distances():
    world = np.array([[0.0, 0.0], [14, 0], [14, 30], [0, 30], [3, 8]])
    pairs = distances(world)
    reconstructed = metric_coordinates(len(world), pairs, max_distance_error_m=1e-6)
    np.testing.assert_allclose(list(distances(reconstructed).values()), list(pairs.values()), atol=1e-8)
    np.testing.assert_allclose(reconstructed[0], [0, 0], atol=1e-8)
    assert reconstructed[1, 0] > 0 and abs(reconstructed[1, 1]) < 1e-8


def test_impossible_distances_are_rejected():
    pairs = distances(np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]]))
    pairs[0, 1] = 1000
    with pytest.raises(ValueError, match="mâu thuẫn"):
        metric_coordinates(4, pairs)
    pairs = distances(np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]]))
    pairs[1, 0] = pairs.pop((0, 1))  # reversed keys represent the same undirected pair
    metric_coordinates(4, pairs)
    pairs[0, 1] = pairs[1, 0]
    with pytest.raises(ValueError, match="hai lần"):
        metric_coordinates(4, pairs)


def test_trilateration_checks_original_distances():
    anchors = np.array([[0.0, 0.0], [10, 0], [10, 20]])
    expected = np.array([4.0, 6.0])
    point, error = locate_from_three_anchors(anchors, np.linalg.norm(anchors - expected, axis=1))
    np.testing.assert_allclose(point, expected)
    assert error < 1e-8
    with pytest.raises(ValueError, match="mâu thuẫn"):
        locate_from_three_anchors(anchors, [1, 1, 1])


@pytest.mark.parametrize(
    "image", [[], [[1, 2]] * 4, [[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 0], [1, float("nan")], [0, 1]]]
)
def test_invalid_point_arrays_raise_clear_errors(image):
    with pytest.raises(ValueError):
        validate_correspondences(image, [[0, 0], [10, 0], [10, 20], [0, 20]])


def test_rectangle_requires_ordered_corners_and_consistent_extra_points():
    image = [[0, 0], [100, 0], [100, 200], [0, 200]]
    world = rectangle_coordinates(image, 10, 20)
    validate_correspondences(image, world)
    with pytest.raises(ValueError, match="tứ giác lồi"):
        rectangle_coordinates([image[i] for i in (0, 2, 1, 3)], 10, 20)
    with pytest.raises(ValueError, match="mâu thuẫn"):
        rectangle_coordinates(image, 10, 20, {(0, 1): 100})
    extended = image + [[50, 100]]
    with pytest.raises(ValueError, match="Nhập đủ"):
        rectangle_coordinates(extended, 10, 20)
    pairs = {(i, 4): float(np.linalg.norm(world[i] - [5, 10])) for i in range(3)}
    solution = rectangle_coordinates(extended, 10, 20, pairs)
    np.testing.assert_allclose(solution[-1], [5, 10])
    validate_correspondences(extended, solution)


def test_correspondences_must_fit_one_homography():
    image = [[0, 0], [100, 0], [100, 200], [0, 200], [50, 100], [30, 50]]
    world = [[0, 0], [10, 0], [10, 20], [0, 20], [100, 100], [3, 5]]
    with pytest.raises(ValueError):
        validate_correspondences(image, world, max_error_m=0.01)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_measurement_tolerance_is_validated(tolerance):
    points = np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])
    with pytest.raises(ValueError):
        metric_coordinates(4, distances(points), tolerance)
