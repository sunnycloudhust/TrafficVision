"""Lightweight tracking-by-detection implementations for vehicle sequences.

All trackers consume xyxy boxes, confidence scores and class IDs. SORT uses
motion/IoU association, ByteTrack adds high/low confidence association,
BoT-SORT adds camera motion compensation and color appearance, and OC-SORT
adds observation-centric direction association and reactivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-8)


def _assignment(cost: np.ndarray, maximum_cost: float):
    if not cost.size:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= maximum_cost]
    used_r, used_c = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in used_r], [i for i in range(cost.shape[1]) if i not in used_c]


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    score: float
    class_id: int
    hits: int = 1
    age: int = 1
    missed: int = 0
    previous_center: np.ndarray | None = None
    feature: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    @property
    def center(self):
        return (self.box[:2] + self.box[2:]) * 0.5


class _Tracker:
    def __init__(self, *, max_age=30, min_hits=2, iou_threshold=0.2,
                 high_threshold=0.5, low_threshold=0.1, appearance_weight=0.25):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.appearance_weight = float(appearance_weight)
        if self.max_age < 0 or self.min_hits < 1:
            raise ValueError("max_age must be non-negative and min_hits must be at least one")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be between zero and one")
        if not 0 <= self.low_threshold <= self.high_threshold <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= low <= high <= 1")
        if not 0 <= self.appearance_weight <= 1:
            raise ValueError("appearance_weight must be between zero and one")
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.frame_index = 0

    def _compensate_camera(self, image):
        return

    def _features(self, image, boxes):
        return [None] * len(boxes)

    def _association_cost(self, tracks, boxes, scores, classes, features):
        iou = _iou_matrix(np.asarray([t.box for t in tracks]), boxes)
        cost = 1.0 - iou
        for i, track in enumerate(tracks):
            cost[i, classes != track.class_id] = 1e6
        return cost

    def _match(self, track_indices, det_indices, boxes, scores, classes, features, maximum_cost=None):
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)
        tracks = [self.tracks[i] for i in track_indices]
        det_boxes = boxes[det_indices]
        cost = self._association_cost(tracks, det_boxes, scores[det_indices], classes[det_indices],
                                      [features[i] for i in det_indices])
        return _assignment(cost, 1.0 - self.iou_threshold if maximum_cost is None else maximum_cost)

    @staticmethod
    def _update_track(track, box, score, feature):
        old_center = track.center.copy()
        track.previous_center = old_center
        measured = np.asarray(box, dtype=np.float32)
        # Constant-velocity box prediction with a damped observation update.
        track.velocity = 0.8 * track.velocity + 0.2 * (measured - track.box)
        track.box = measured.copy()
        track.score = float(score)
        track.hits += 1
        track.missed = 0
        if feature is not None:
            track.feature = feature

    def update(self, boxes, scores, classes, image=None):
        self.frame_index += 1
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(classes)):
            raise ValueError("boxes, scores and classes must have equal lengths")

        self._compensate_camera(image)
        features = self._features(image, boxes)
        for track in self.tracks:
            track.age += 1
            track.missed += 1
            track.box = track.box + track.velocity
        active_tracks = [i for i, track in enumerate(self.tracks) if track.missed <= self.max_age]
        unmatched_dets = list(range(len(boxes)))
        matched_all = []
        if self.algorithm in {"bytetrack", "botsort"}:
            high = [i for i, score in enumerate(scores) if score >= self.high_threshold]
            low = [i for i, score in enumerate(scores)
                   if self.low_threshold <= score < self.high_threshold]
            matched, unmatched_tracks, unmatched_high = self._match(
                active_tracks, high, boxes, scores, classes, features)
            matched_all += [(active_tracks[t], high[d]) for t, d in matched]
            remaining = [active_tracks[i] for i in unmatched_tracks
                         if self.tracks[active_tracks[i]].missed == 1]
            matched_low, _, _ = self._match(remaining, low, boxes, scores, classes, features,
                                             maximum_cost=0.7)
            matched_all += [(remaining[t], low[d]) for t, d in matched_low]
            unmatched_dets = [high[i] for i in unmatched_high]
        else:
            matched, unmatched_tracks, unmatched_dets = self._match(
                active_tracks, list(range(len(boxes))), boxes, scores, classes, features)
            matched_all = [(active_tracks[t], d) for t, d in matched]

        for track_index, det_index in matched_all:
            self._update_track(self.tracks[track_index], boxes[det_index], scores[det_index], features[det_index])
        for det_index in unmatched_dets:
            if self.algorithm in {"bytetrack", "botsort"} and scores[det_index] < self.high_threshold:
                continue
            self.tracks.append(_Track(self.next_id, boxes[det_index].copy(), float(scores[det_index]),
                                      int(classes[det_index]), feature=features[det_index]))
            self.next_id += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_age]
        visible = [track for track in self.tracks if track.missed == 0 and track.hits >= self.min_hits]
        if self.frame_index <= self.min_hits:
            visible = [track for track in self.tracks if track.missed == 0]
        return [{"track_id": t.track_id, "box": t.box.copy(), "score": t.score,
                 "class_id": t.class_id} for t in visible]


class Sort(_Tracker):
    """SORT: IoU matching with online track lifecycle."""
    algorithm = "sort"


class ByteTrack(_Tracker):
    """ByteTrack-style two-pass association using high and low scores."""
    algorithm = "bytetrack"


class BotSort(ByteTrack):
    """BoT-SORT-style association with camera compensation and color cues.

    Color histograms are a lightweight appearance descriptor; this does not
    include a learned ReID network.
    """
    algorithm = "botsort"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._previous_gray = None
        self._camera_affine = None

    def _compensate_camera(self, image):
        if image is None:
            return
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("BoT-SORT camera compensation requires opencv-contrib-python") from exc
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self._previous_gray is not None:
            points = cv2.goodFeaturesToTrack(self._previous_gray, maxCorners=300, qualityLevel=0.01,
                                             minDistance=8)
            if points is not None and len(points) >= 6:
                current, status, _ = cv2.calcOpticalFlowPyrLK(self._previous_gray, gray, points, None)
                valid = status.reshape(-1).astype(bool)
                if valid.sum() >= 6:
                    affine, _ = cv2.estimateAffinePartial2D(points[valid], current[valid], method=cv2.RANSAC)
                    self._camera_affine = affine
                    if affine is not None:
                        for track in self.tracks:
                            corners = np.array([[track.box[0], track.box[1], 1],
                                                [track.box[2], track.box[3], 1]], dtype=np.float32) @ affine.T
                            track.box = np.array([corners[0, 0], corners[0, 1],
                                                  corners[1, 0], corners[1, 1]], dtype=np.float32)
        self._previous_gray = gray

    def _features(self, image, boxes):
        if image is None:
            return [None] * len(boxes)
        import cv2
        height, width = image.shape[:2]
        result = []
        for x1, y1, x2, y2 in boxes:
            crop = image[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))]
            if crop.size == 0:
                result.append(None)
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
            result.append(hist / max(float(np.linalg.norm(hist)), 1e-8))
        return result

    def _association_cost(self, tracks, boxes, scores, classes, features):
        cost = super()._association_cost(tracks, boxes, scores, classes, features)
        for i, track in enumerate(tracks):
            for j, feature in enumerate(features):
                if feature is not None and track.feature is not None:
                    similarity = float(np.dot(feature, track.feature))
                    cost[i, j] = (1 - self.appearance_weight) * cost[i, j] + self.appearance_weight * (1 - similarity)
        return cost


class OCSort(_Tracker):
    """OC-SORT-style observation-centric direction matching and reactivation."""
    algorithm = "ocsort"

    def _association_cost(self, tracks, boxes, scores, classes, features):
        cost = super()._association_cost(tracks, boxes, scores, classes, features)
        det_centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        for i, track in enumerate(tracks):
            if track.previous_center is None:
                continue
            velocity = track.center - track.previous_center
            velocity_norm = np.linalg.norm(velocity)
            if velocity_norm < 1e-4:
                continue
            for j, center in enumerate(det_centers):
                direction = center - track.center
                norm = np.linalg.norm(direction)
                if norm > 1e-4:
                    cosine = float(np.dot(velocity, direction) / (velocity_norm * norm))
                    # Penalize detections moving opposite to observed trajectory.
                    cost[i, j] += 0.15 * (1.0 - cosine)
        return cost


def build_tracker(name: str, **kwargs) -> _Tracker:
    trackers = {"sort": Sort, "bytetrack": ByteTrack, "botsort": BotSort, "ocsort": OCSort}
    try:
        return trackers[name.lower()](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown tracker {name!r}; choose from {', '.join(trackers)}") from exc
