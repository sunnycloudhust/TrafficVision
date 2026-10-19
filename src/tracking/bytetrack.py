"""Explicit adapter from Torchvision detections to Supervision's ByteTrack."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TrackedObject:
    box: list[float]
    class_id: int
    score: float
    track_id: int


class ByteTrackAdapter:
    """One tracker per video; call update once for every sequential frame."""
    def __init__(self, config: dict):
        self.config = dict(config)
        self.reset()

    def reset(self):
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError("Install the 'supervision' dependency to use ByteTrack") from exc
        self._sv = sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=float(self.config.get("track_activation_threshold", .25)),
            lost_track_buffer=int(self.config.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(self.config.get("minimum_matching_threshold", .8)),
            frame_rate=int(self.config.get("frame_rate", 30)),
            minimum_consecutive_frames=int(self.config.get("minimum_consecutive_frames", 1)),
        )

    def update(self, detections: dict) -> list[TrackedObject]:
        boxes = detections["boxes"].detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
        scores = detections["scores"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        classes = detections["labels"].detach().cpu().numpy().astype(np.int32).reshape(-1)
        sv_detections = self._sv.Detections(xyxy=boxes, confidence=scores, class_id=classes)
        tracked = self.tracker.update_with_detections(sv_detections)
        if len(tracked) == 0 or tracked.tracker_id is None:
            return []
        results = []
        for box, score, label, track_id in zip(tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id):
            if track_id is None or int(track_id) < 0:
                continue
            results.append(TrackedObject(box.astype(float).tolist(), int(label), float(score), int(track_id)))
        return results
