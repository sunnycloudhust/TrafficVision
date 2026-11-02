import numpy as np
import pytest
from trafficvision.registry import build_tracker


def detection(x=0, score=0.9, label=1):
    return dict(
        boxes=np.array([[x, 10, x + 30, 30]], dtype=np.float32),
        scores=np.array([score]),
        labels=np.array([label]),
    )


EMPTY = dict(boxes=np.empty((0, 4)), scores=np.empty(0), labels=np.empty(0))


def test_sort_gap_recovery_and_expiry():
    tracker = build_tracker("sort", dict(min_hits=1, max_age=2))
    first = tracker.update(detection())[0].track_id
    assert tracker.update(EMPTY) == []
    assert tracker.update(detection(2))[0].track_id == first
    for _ in range(3):
        tracker.update(EMPTY)
    assert tracker.update(detection(3))[0].track_id != first


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_two_stage_low_score_association(name):
    tracker = build_tracker(name)
    frame = np.zeros((64, 64, 3), np.uint8)
    first = tracker.update(detection(), frame)[0].track_id
    second = tracker.update(detection(1, score=0.15), frame)
    assert second[0].track_id == first
    assert tracker.update(EMPTY, frame) == []
    recovered = tracker.update(detection(2), frame)
    assert recovered[0].track_id == first
