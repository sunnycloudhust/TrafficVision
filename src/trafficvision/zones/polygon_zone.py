from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PolygonZone:
    name: str
    polygon: list
    confidence_threshold: float = 0.25
    inference_scale: float = 1.0
    measure_speed: bool = True

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, dtype=np.float32)
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2 or len(self.polygon) < 3:
            raise ValueError("Polygon needs at least three 2D points")
        if not np.isfinite(self.polygon).all() or abs(cv2.contourArea(self.polygon)) < 1e-6:
            raise ValueError("Polygon is degenerate")
        if not 0 <= self.confidence_threshold <= 1 or self.inference_scale <= 0:
            raise ValueError("Invalid zone threshold/scale")

    def contains(self, point):
        return cv2.pointPolygonTest(self.polygon, tuple(map(float, point)), False) >= 0

    def contains_box(self, box):
        return self.contains(((box[0] + box[2]) / 2, box[3]))
