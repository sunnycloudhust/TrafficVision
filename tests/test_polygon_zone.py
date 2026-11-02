import pytest
import torch
from trafficvision.zones.polygon_zone import PolygonZone
from trafficvision.zones.zone_manager import ZoneManager


def test_bottom_center_and_boundary():
    zone = PolygonZone("road", [[0, 20], [100, 20], [100, 100], [0, 100]])
    assert zone.contains_box([10, 0, 30, 20])
    assert not zone.contains_box([10, 0, 30, 19])
    with pytest.raises(ValueError):
        PolygonZone("bad", [[0, 0], [1, 1], [2, 2]])


def test_far_crop_maps_to_original_coordinates():
    class Detector:
        calls = 0

        def predict(self, image):
            self.calls += 1
            if self.calls == 1:
                return dict(
                    boxes=torch.empty((0, 4)), scores=torch.empty(0), labels=torch.empty(0, dtype=torch.long)
                )
            return dict(
                boxes=torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
                scores=torch.tensor([0.2]),
                labels=torch.tensor([1]),
            )

    zones = ZoneManager(
        dict(
            frame_size=[100, 100],
            zones=dict(
                far=dict(
                    polygon=[[20, 20], [40, 20], [40, 40], [20, 40]],
                    inference_scale=2.0,
                    confidence_threshold=0.1,
                )
            ),
        )
    )
    result = zones.detect(Detector(), torch.zeros(3, 100, 100))
    torch.testing.assert_close(result["boxes"], torch.tensor([[20.0, 20.0, 30.0, 30.0]]))
    with pytest.raises(ValueError, match="frame size"):
        zones.validate_frame(200, 100)
