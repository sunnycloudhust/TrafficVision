"""Timestamp-based regression over observed world positions; no pixel/kmh fallback."""

from collections import deque
import math
import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        smoothing_window=5,
        min_track_observations=3,
        max_reasonable_kmh=160,
        max_gap_seconds=0.5,
        min_duration_seconds=0.08,
    ):
        if smoothing_window < min_track_observations or min_track_observations < 2:
            raise ValueError("Window must cover at least two required observations")
        if max_gap_seconds <= 0 or min_duration_seconds < 0 or max_reasonable_kmh <= 0:
            raise ValueError("Invalid speed settings")
        self.window, self.minimum, self.maximum = smoothing_window, min_track_observations, max_reasonable_kmh
        self.max_gap, self.min_duration = max_gap_seconds, min_duration_seconds
        self.history = {}

    def reset(self, track_id=None):
        if track_id is None:
            self.history.clear()
        else:
            self.history.pop(track_id, None)

    def update(self, track_id, point, timestamp):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or not math.isfinite(timestamp):
            self.reset(track_id)
            return None
        history = self.history.setdefault(track_id, deque(maxlen=self.window))
        if history:
            delta = timestamp - history[-1][0]
            if delta <= 0 or delta > self.max_gap:
                history.clear()
            elif np.linalg.norm(point - history[-1][1]) / delta * 3.6 > self.maximum:
                # Reset at discontinuities; never smooth an ID jump into plausible speed.
                history.clear()
                history.append((timestamp, point.copy()))
                return None
        history.append((timestamp, point.copy()))
        if len(history) < self.minimum or history[-1][0] - history[0][0] < self.min_duration:
            return None
        times = np.array([t for t, _ in history])
        positions = np.stack([p for _, p in history])
        centered = times - times.mean()
        velocity = (centered[:, None] * (positions - positions.mean(0))).sum(0) / (centered**2).sum()
        speed = float(np.linalg.norm(velocity) * 3.6)
        return speed if speed <= self.maximum else None

    def prune(self, timestamp):
        for track_id in list(self.history):
            if timestamp - self.history[track_id][-1][0] > self.max_gap:
                self.reset(track_id)
