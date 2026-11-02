"""Tracking on ordered local or remote validation/test frames."""

import json
from pathlib import Path
import cv2
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from ..data.datasets.ua_detrac import build_dataset
from ..visualization.result_exporter import ResultExporter


def track_dataset(config, split, pipeline, output, fps=25):
    if fps <= 0:
        raise ValueError("Dataset FPS must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = build_dataset(config["data"], split).source
    exporter = ResultExporter(output)
    gt_path = output.with_suffix(".gt.jsonl")
    previous_video, previous_frame, seen = None, 0, set()
    frames = 0
    try:
        with gt_path.open("w") as ground_truth:
            for image, record in source:
                video, frame_id = record["video_id"], record["frame_id"]
                if video != previous_video:
                    if video in seen:
                        raise ValueError("Streaming shards must group each sequence contiguously")
                    seen.add(video)
                    pipeline.tracker.reset()
                    pipeline.speed.reset()
                    previous_video, previous_frame = video, 0
                if frame_id != previous_frame + 1:
                    raise ValueError(
                        "Tracking evaluation requires every frame in ascending order starting at 1"
                    )
                previous_frame = frame_id
                timestamp = (frame_id - 1) / fps
                tensor = pil_to_tensor(image).float() / 255
                bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                results = pipeline.process(tensor, timestamp, bgr)
                exporter.write(video, frame_id, timestamp, results)
                ground_truth.write(json.dumps(record) + "\n")
                frames += 1
    finally:
        exporter.close()
    return dict(
        frames=frames,
        sequences=len(seen),
        predictions=str(output.with_suffix(".csv")),
        ground_truth=str(gt_path),
    )
