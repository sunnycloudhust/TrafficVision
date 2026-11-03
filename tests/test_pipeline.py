import csv
import json
import cv2
import numpy as np
import pytest
from helpers import SyntheticDetector
from trafficvision.registry import build_tracker
from trafficvision.geometry.homography import Homography
from trafficvision.pipeline.detection_tracking import DetectionTrackingPipeline
from trafficvision.pipeline.video_pipeline import run_video
from trafficvision.visualization.video_writer import VideoWriter


@pytest.mark.parametrize("calibrated", [True, False])
def test_video_exports_and_speed(tmp_path, calibrated):
    source = tmp_path / "input.mp4"
    writer = VideoWriter(source, 25, (320, 180))
    for _ in range(12):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.close()
    homography = Homography(np.diag([0.1, 0.1, 1])) if calibrated else None
    pipeline = DetectionTrackingPipeline(
        SyntheticDetector(), build_tracker("sort", dict(min_hits=1)), homography=homography
    )
    output = tmp_path / "output.mp4"
    summary = run_video(source, output, pipeline)
    assert summary["frames"] == 12
    with output.with_suffix(".csv").open() as handle:
        rows = list(csv.DictReader(handle))
    data = json.loads(output.with_suffix(".json").read_text())
    assert len(rows) == len(data) == 12 and len({r["track_id"] for r in rows}) == 1
    if calibrated:
        assert float(rows[-1]["speed_kmh"]) == pytest.approx(18, abs=0.5)
    else:
        assert all(row["speed_kmh"] is None and row["world_x"] is None for row in data)
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 12
    cap.release()
