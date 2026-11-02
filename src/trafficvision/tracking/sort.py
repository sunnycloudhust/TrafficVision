"""SORT variant: linear Kalman filter on xyxy+velocity and Hungarian IoU matching.

Tracks survive max_age missing frames, but only observed tracks are returned.
This avoids reporting extrapolated positions as measured speed observations.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .base import Track


def iou_matrix(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    return intersection / np.maximum(area_a[:, None] + area_b[None] - intersection, 1e-9)


class KalmanBox:
    def __init__(self, box, track_id, label, score):
        self.x = np.r_[box, np.zeros(4)].astype(float)
        self.P = np.diag([10.0] * 4 + [1000.0] * 4)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q = np.diag([0.1] * 4 + [0.01] * 4)
        self.R = np.eye(4)
        self.id, self.label, self.score = track_id, label, score
        self.age, self.hits = 0, 1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def correct(self, box, score):
        innovation = box - self.H @ self.x
        covariance = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(covariance, self.H @ self.P).T
        self.x += gain @ innovation
        residual = np.eye(8) - gain @ self.H
        self.P = residual @ self.P @ residual.T + gain @ self.R @ gain.T
        self.score, self.age, self.hits = float(score), 0, self.hits + 1


class SORT:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, confidence=0.25):
        if max_age < 0 or min_hits < 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("Invalid SORT settings")
        self.max_age, self.min_hits = max_age, min_hits
        self.iou_threshold, self.confidence = iou_threshold, confidence
        self.reset()

    def reset(self):
        self.tracks, self.next_id = [], 1

    def update(self, detections, frame=None):
        boxes = np.asarray(detections["boxes"], dtype=float).reshape(-1, 4)
        scores = np.asarray(detections["scores"], dtype=float)
        labels = np.asarray(detections["labels"], dtype=int)
        keep = scores >= self.confidence
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        for track in self.tracks:
            track.predict()
        similarity = iou_matrix([t.x[:4] for t in self.tracks], boxes)
        if len(self.tracks) and len(boxes):
            same_class = np.array([t.label for t in self.tracks])[:, None] == labels[None]
            similarity[~same_class] = -1
            cost = 1 - similarity
            cost[similarity < self.iou_threshold] = 1e6
            rows, cols = linear_sum_assignment(cost)
            matches = [(r, c) for r, c in zip(rows, cols) if similarity[r, c] >= self.iou_threshold]
        else:
            matches = []
        used = set()
        for row, col in matches:
            self.tracks[row].correct(boxes[col], scores[col])
            used.add(col)
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        for index in range(len(boxes)):
            if index not in used:
                self.tracks.append(
                    KalmanBox(boxes[index], self.next_id, int(labels[index]), float(scores[index]))
                )
                self.next_id += 1
        return [
            Track(t.id, t.x[:4].copy(), t.label, t.score)
            for t in self.tracks
            if t.age == 0 and t.hits >= self.min_hits
        ]
