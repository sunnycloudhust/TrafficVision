import importlib.util
from pathlib import Path
import numpy as np
import pytest
import yaml
from trafficvision.zones.zone_manager import ZoneManager

spec = importlib.util.spec_from_file_location(
    "zone_labeler", Path(__file__).parents[1] / "data-labelling/zone_labeler.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_labeler_rejects_invalid_polygons_and_incomplete_save(tmp_path):
    labeler = module.ZoneLabeler(np.zeros((540, 960, 3), np.uint8), "test", tmp_path / "zones.yaml")
    with pytest.raises(ValueError, match="at least one"):
        labeler.save()
    labeler.current = [[1, 1], [2, 2], [3, 3]]
    with pytest.raises(ValueError, match="nonzero area"):
        labeler.finish_zone("road")
    with pytest.raises(ValueError, match="Finish"):
        labeler.save()
    labeler.current = [[0, 0], [100, 0], [100, 100]]
    labeler.finish_zone("road")
    labeler.save()
    config = yaml.safe_load(labeler.output.read_text())
    manager = ZoneManager(config)
    manager.validate_frame(960, 540)
    assert manager.zones[0].contains([50, 25])
    labeler.current = [[0, 0], [10, 0], [10, 10]]
    with pytest.raises(ValueError, match="unique"):
        labeler.finish_zone("road")


def test_display_coordinates_scale_and_clip(tmp_path):
    labeler = module.ZoneLabeler(np.zeros((2160, 3840, 3), np.uint8), "test", tmp_path / "zones.yaml")
    labeler.display_frame()
    assert labeler.to_frame_point(700, 350) == [1920, 960]
    assert labeler.to_frame_point(-10, 5000) == [0, 2159]
