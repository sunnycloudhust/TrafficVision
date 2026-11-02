from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    label: int
    score: float


class Tracker(Protocol):
    def update(self, detections, frame=None) -> list[Track]: ...

    def reset(self): ...
