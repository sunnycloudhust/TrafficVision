import cv2
import numpy as np


class Homography:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        if (
            self.matrix.shape != (3, 3)
            or not np.isfinite(self.matrix).all()
            or np.linalg.matrix_rank(self.matrix) != 3
        ):
            raise ValueError("Homography must be a finite, nonsingular 3x3 matrix")

    @classmethod
    def from_points(cls, image_points, world_points, max_reprojection_error=0.5):
        image = np.asarray(image_points, dtype=float)
        world = np.asarray(world_points, dtype=float)
        if image.ndim != 2 or image.shape != world.shape or image.shape[1] != 2 or len(image) < 4:
            raise ValueError("Need >=4 corresponding image/world points with shape [N,2]")
        if not np.isfinite(image).all() or not np.isfinite(world).all():
            raise ValueError("Calibration points must be finite")
        for points in (image, world):
            if (
                len(np.unique(points, axis=0)) != len(points)
                or np.linalg.matrix_rank(points - points.mean(0)) < 2
            ):
                raise ValueError("Duplicate/collinear calibration points")
        matrix, _ = cv2.findHomography(image, world, method=0)
        if matrix is None:
            raise ValueError("Unable to estimate homography")
        result = cls(matrix)
        error = np.linalg.norm(result.project(image) - world, axis=1)
        if error.max() > max_reprojection_error:
            raise ValueError(f"Calibration reprojection exceeds tolerance: {error.max():.4f} m")
        result.reprojection_error_m = float(error.mean())
        return result

    def project(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ValueError("Projection needs finite 2D points")
        homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1) @ self.matrix.T
        if np.any(abs(homogeneous[..., 2]) < 1e-8):
            raise ValueError("Point lies on the homography horizon")
        return homogeneous[..., :2] / homogeneous[..., 2:]
