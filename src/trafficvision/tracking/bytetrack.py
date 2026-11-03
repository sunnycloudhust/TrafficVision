"""Ultralytics tracker adapter; one tracker per class prevents cross-class IDs."""

import inspect
from types import SimpleNamespace
import numpy as np
from .base import Track


class UltralyticsTracker:
    def __init__(self, name="bytetrack", config=None, fps=25):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT

        if name not in ("bytetrack", "botsort"):
            raise ValueError(name)
        settings = dict(
            track_high_thresh=0.25,
            track_low_thresh=0.05,
            new_track_thresh=0.3,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method="none",
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
        )
        settings.update(config or {})
        if settings["with_reid"]:
            raise ValueError("Pinned BoT-SORT adapter has no ReID encoder; use with_reid: false")
        if not 0 <= settings["track_low_thresh"] <= settings["track_high_thresh"] <= 1:
            raise ValueError("Invalid association confidence thresholds")
        self.constructor = BOTSORT if name == "botsort" else BYTETracker
        self.settings, self.fps = settings, fps
        self.reset()

    def _new_tracker(self):
        """Support Ultralytics releases with and without the frame_rate argument."""
        parameters = inspect.signature(self.constructor).parameters
        args = SimpleNamespace(**self.settings)
        if "frame_rate" in parameters:
            return self.constructor(args, frame_rate=self.fps)
        return self.constructor(args)

    def reset(self):
        self.trackers, self.ids, self.next_id = {}, {}, 1

    def update(self, detections, frame=None):
        from ultralytics.engine.results import Boxes

        boxes = np.asarray(detections["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(detections["scores"], dtype=np.float32)
        labels = np.asarray(detections["labels"], dtype=np.int64)
        result = []
        for label in sorted(set(labels.tolist()) | set(self.trackers)):
            if label not in self.trackers:
                self.trackers[label] = self._new_tracker()
            chosen = labels == label
            packed = np.column_stack([boxes[chosen], scores[chosen], labels[chosen]]).astype(np.float32)
            shape = frame.shape[:2] if frame is not None else (1, 1)
            observations = self.trackers[label].update(Boxes(packed, shape), img=frame)
            for row in observations:
                key = (label, int(row[4]))
                if key not in self.ids:
                    self.ids[key] = self.next_id
                    self.next_id += 1
                result.append(Track(self.ids[key], row[:4].copy(), label, float(row[5])))
        return result
