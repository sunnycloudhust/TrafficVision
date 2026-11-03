"""Video decoding, timestamp policy and resource cleanup around the core pipeline."""

import json
import math
from pathlib import Path
import time
import cv2
import torch
from tqdm.auto import tqdm
from ..visualization.annotator import annotate
from ..visualization.video_writer import VideoWriter
from ..visualization.result_exporter import ResultExporter


def run_video(
    source, output, pipeline, camera_config=None, max_frames=None, timestamp_mode="fps", video_id=None
):
    camera = camera_config or {}
    if timestamp_mode not in ("fps", "pts"):
        raise ValueError("timestamp_mode must be fps or pts")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    writer = exporter = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = float(camera.get("fps", 0))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Valid source FPS or camera fps required")
        declared_fps = camera.get("fps")
        if declared_fps and abs(fps - declared_fps) > max(0.05, fps * 0.01):
            raise ValueError("Camera FPS differs from source; check the video/calibration pairing")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if camera.get("frame_size") and camera["frame_size"] != [width, height]:
            raise ValueError("Calibration frame size differs from source")
        pipeline.zones.validate_frame(width, height)
        if (
            camera.get("camera_id")
            and pipeline.zones.camera_id
            and camera["camera_id"] != pipeline.zones.camera_id
        ):
            raise ValueError("Camera and zone configs refer to different cameras")
        pipeline.tracker.reset()
        pipeline.speed.reset()
        writer = VideoWriter(output, fps, (width, height))
        exporter = ResultExporter(output)
        frame_id, previous_timestamp = 0, None
        start = time.perf_counter()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = frame_count if frame_count > 0 else None
        if max_frames is not None:
            total_frames = min(total_frames, max_frames) if total_frames is not None else max_frames
        with tqdm(total=total_frames, desc=f"Processing {Path(str(source)).name}", unit="frame") as progress:
            while max_frames is None or frame_id < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if timestamp_mode == "pts" else frame_id / fps
                if not math.isfinite(timestamp) or (
                    previous_timestamp is not None and timestamp <= previous_timestamp
                ):
                    raise ValueError("Non-monotonic/missing PTS; use --timestamp-mode fps for constant-FPS video")
                previous_timestamp = timestamp
                frame_id += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
                results = pipeline.process(tensor, timestamp, frame)
                processing_fps = frame_id / max(time.perf_counter() - start, 1e-9)
                writer.write(annotate(frame, results, pipeline.zones, processing_fps))
                exporter.write(video_id or Path(str(source)).stem, frame_id, timestamp, results)
                progress.update(1)
                progress.set_postfix(fps=f"{processing_fps:.1f}")
        if frame_id == 0:
            raise ValueError("Source has no readable video frames")
        summary = dict(
            frames=frame_id,
            fps=fps,
            processing_fps=frame_id / (time.perf_counter() - start),
            frame_size=[width, height],
            calibrated=pipeline.homography is not None,
            timestamp_mode=timestamp_mode,
            output=str(output),
        )
        Path(output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        cap.release()
        if writer:
            writer.close()
        if exporter:
            exporter.close()
