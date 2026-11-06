"""Metric point reconstruction and validation, independent of the GUI toolkit."""

import numpy as np
from .homography import Homography


def point_array(points, minimum=4):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"Cần ít nhất {minimum} điểm có dạng [N, 2].")
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Điểm phải hữu hạn và không trùng nhau.")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Các điểm thẳng hàng; cần phủ một vùng 2D.")
    return points


def distance_pairs(point_count, measured_distances):
    pairs = {}
    for (first, second), distance in measured_distances.items():
        if not (0 <= first < point_count and 0 <= second < point_count) or first == second:
            raise ValueError("Cặp điểm khoảng cách không hợp lệ.")
        pair = tuple(sorted((first, second)))
        if pair in pairs:
            raise ValueError("Một cặp khoảng cách được nhập hai lần.")
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Khoảng cách phải là số mét dương.")
        pairs[pair] = float(distance)
    return pairs


def check_distance_residuals(points, measured_distances, max_distance_error_m=0.25):
    if not np.isfinite(max_distance_error_m) or max_distance_error_m < 0:
        raise ValueError("Ngưỡng sai lệch mét phải hữu hạn và không âm.")
    errors = [
        abs(np.linalg.norm(points[i] - points[j]) - distance)
        for (i, j), distance in measured_distances.items()
    ]
    error = float(max(errors, default=0.0))
    if error > max_distance_error_m + 1e-8:
        raise ValueError(f"Khoảng cách mâu thuẫn: sai lệch {error:.3f} m vượt {max_distance_error_m:g} m.")
    return error


def metric_coordinates(point_count, measured_distances, max_distance_error_m=0.25):
    """Classical MDS, accepting only a 2D reconstruction within measurement tolerance."""
    if point_count < 4:
        raise ValueError("Homography cần ít nhất 4 điểm.")
    pairs = distance_pairs(point_count, measured_distances)
    if len(pairs) != point_count * (point_count - 1) // 2:
        raise ValueError("Cần nhập khoảng cách cho đủ mọi cặp điểm.")
    distances = np.zeros((point_count, point_count))
    for (i, j), distance in pairs.items():
        distances[i, j] = distances[j, i] = distance
    centering = np.eye(point_count) - np.ones((point_count, point_count)) / point_count
    with np.errstate(over="raise", invalid="raise"):
        try:
            gram = -0.5 * centering @ (distances**2) @ centering
        except FloatingPointError as exc:
            raise ValueError("Khoảng cách quá lớn để tính toán ổn định.") from exc
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    if eigenvalues[order[1]] <= 1e-9:
        raise ValueError("Các khoảng cách không xác định được mặt phẳng 2D.")
    coords = eigenvectors[:, order] * np.sqrt(eigenvalues[order])
    coords -= coords[0]
    angle = np.arctan2(coords[1, 1], coords[1, 0])
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    coords = coords @ rotation.T
    # Choose a deterministic mirror orientation. Reflection does not change distances.
    for y in coords[2:, 1]:
        if abs(y) > 1e-9:
            if y < 0:
                coords[:, 1] *= -1
            break
    check_distance_residuals(coords, pairs, max_distance_error_m)
    return point_array(coords)


def locate_from_three_anchors(anchors, distances, max_distance_error_m=0.25):
    anchors = point_array(anchors, minimum=3)
    distances = np.asarray(distances, dtype=float)
    if anchors.shape != (3, 2) or distances.shape != (3,):
        raise ValueError("Cần đúng 3 điểm mốc và 3 khoảng cách.")
    if not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError("Các khoảng cách phải là số mét dương.")
    matrix = 2 * (anchors[1:] - anchors[0])
    rhs = np.sum(anchors[1:] ** 2, axis=1) - np.sum(anchors[0] ** 2) + distances[0] ** 2 - distances[1:] ** 2
    point = np.linalg.solve(matrix, rhs)
    combined = np.vstack([anchors, point])
    pairs = {(i, 3): distance for i, distance in enumerate(distances)}
    error = check_distance_residuals(combined, pairs, max_distance_error_m)
    return point, error


def rectangle_coordinates(image_points, length, width, measured_distances=None, max_distance_error_m=0.25):
    image_points = point_array(image_points)
    if not all(np.isfinite(value) and value > 0 for value in (length, width)):
        raise ValueError("Chiều dài và chiều rộng phải là số mét dương.")
    edges = np.roll(image_points[:4], -1, axis=0) - image_points[:4]
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError("Bốn góc đầu phải tạo tứ giác lồi theo thứ tự quanh hình, không chéo nhau.")
    world = [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]]
    pairs = distance_pairs(len(image_points), measured_distances or {})
    for index in range(4, len(image_points)):
        required = [(i, index) for i in range(3)]
        if any(pair not in pairs for pair in required):
            raise ValueError(f"Nhập đủ khoảng cách từ P{index+1} tới P1, P2, P3.")
        point, _ = locate_from_three_anchors(
            world[:3], [pairs[pair] for pair in required], max_distance_error_m
        )
        world.append(point.tolist())
    world = point_array(world)
    # Check optional measurements too, including the rectangle's own edges/diagonals.
    check_distance_residuals(world, pairs, max_distance_error_m)
    return world


def validate_correspondences(image_points, world_points, max_error_m=0.25):
    image = point_array(image_points)
    world = point_array(world_points)
    if image.shape != world.shape:
        raise ValueError("Image/world points phải có cùng số lượng và thứ tự.")
    if not np.isfinite(max_error_m) or max_error_m < 0:
        raise ValueError("Ngưỡng sai lệch mét phải hữu hạn và không âm.")
    homography = Homography.from_points(image, world, max_reprojection_error=max_error_m + 1e-8)
    denominators = np.c_[image, np.ones(len(image))] @ homography.matrix[2]
    if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
        raise ValueError("Vùng calibration cắt qua đường chân trời của homography.")
    return homography
