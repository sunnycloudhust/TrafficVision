"""Self-contained synthetic demo. Oracle detections test plumbing, not accuracy."""

from pathlib import Path
import json
import cv2
import numpy as np
import torch
from .registry import build_model, build_tracker
from .geometry.homography import Homography
from .pipeline.detection_tracking import DetectionTrackingPipeline
from .pipeline.video_pipeline import run_video
from .visualization.video_writer import VideoWriter
from .zones.zone_manager import ZoneManager


class SyntheticDetector:
    """Known trajectory of a generated rectangle; deliberately not a learned model."""

    def __init__(self):
        self.index = 0

    def predict(self, image):
        x = 20 + self.index * 2
        self.index += 1
        return dict(
            boxes=torch.tensor([[x, 50.0, x + 30, 80.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([1]),
        )


def run_smoke(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    checks = {}
    for name in ("faster_rcnn", "fast_rcnn", "rcnn", "rtdetr"):
        params = (
            dict(image_size=64, queries=10, decoder_layers=2)
            if name == "rtdetr"
            else dict(min_size=64, max_size=64, proposals=20)
        )
        if name in ("fast_rcnn", "rcnn"):
            params["proposal_strategy"] = "grid"
        model = build_model(name, **params).train()
        image = torch.rand(3, 48, 64)
        loss = sum(
            model(
                [image], [dict(boxes=torch.tensor([[8.0, 5.0, 40.0, 30.0]]), labels=torch.tensor([1]))]
            ).values()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss for {name}")
        loss.backward()
        checks[name] = dict(loss=float(loss.detach()), parameters=sum(p.numel() for p in model.parameters()))
    source = output / "synthetic_input.mp4"
    writer = VideoWriter(source, 25, (320, 180))
    try:
        for index in range(50):
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            cv2.rectangle(frame, (20 + index * 2, 50), (50 + index * 2, 80), (180, 180, 180), -1)
            writer.write(frame)
    finally:
        writer.close()
    # Synthetic scene is defined to have 0.1 m/pixel: 2px/frame at 25FPS = 18km/h.
    homography = Homography([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]])
    pipeline = DetectionTrackingPipeline(
        SyntheticDetector(),
        build_tracker("sort", dict(min_hits=1)),
        ZoneManager(),
        homography,
        dict(smoothing_window=5, min_track_observations=3),
    )
    summary = run_video(source, output / "synthetic_demo.mp4", pipeline)
    result = dict(synthetic_only=True, models=checks, video=summary, expected_speed_kmh=18.0)
    (output / "checks.json").write_text(json.dumps(result, indent=2))
    return result
