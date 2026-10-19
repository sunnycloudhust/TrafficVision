import sys
import types
import numpy as np
import torch

from src.tracking.bytetrack import ByteTrackAdapter


def test_torchvision_detections_are_adapted(monkeypatch):
    class Detections:
        def __init__(self, xyxy, confidence, class_id):
            self.xyxy, self.confidence, self.class_id = xyxy, confidence, class_id
            self.tracker_id = np.arange(1, len(xyxy) + 1)
        def __len__(self): return len(self.xyxy)

    class FakeByteTrack:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def update_with_detections(self, detections): return detections

    monkeypatch.setitem(sys.modules, "supervision", types.SimpleNamespace(Detections=Detections, ByteTrack=FakeByteTrack))
    adapter = ByteTrackAdapter({"frame_rate": 25})
    objects = adapter.update({"boxes": torch.tensor([[1., 2., 10., 12.]]),
                              "scores": torch.tensor([.9]), "labels": torch.tensor([2])})
    assert len(objects) == 1
    assert objects[0].box == [1., 2., 10., 12.]
    assert objects[0].class_id == 2 and objects[0].track_id == 1
