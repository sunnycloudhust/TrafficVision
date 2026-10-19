"""Run Faster R-CNN + ByteTrack on a video and save annotated video and CSV."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import sys
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection.predictor import Predictor
from src.models.loading import load_model_checkpoint
from src.tracking.bytetrack import ByteTrackAdapter
from src.visualization.visualizer import draw_tracks
from src.utils.config import choose_device, load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tracking-config", default="configs/tracking.yaml")
    parser.add_argument("--score-threshold", type=float)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = choose_device("auto")
    model, state = load_model_checkpoint(checkpoint_path, device)
    classes = state["model_config"]["class_names"]
    track_cfg = load_yaml(args.tracking_config)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened(): raise RuntimeError(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or float(track_cfg.get("frame_rate", 30))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path("outputs/videos"); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem; video_out = out_dir / f"{stem}_{checkpoint_path.parent.name}.mp4"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {video_out}")
    tracker = ByteTrackAdapter({**track_cfg, "frame_rate": round(fps)})
    score_threshold = args.score_threshold if args.score_threshold is not None else float(track_cfg.get("score_threshold", .05))
    predictor = Predictor(model, device, score_threshold)
    csv_path = out_dir / f"{stem}_{checkpoint_path.parent.name}_tracks.csv"
    count = 0; start = time.perf_counter()
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            rows = csv.writer(stream); rows.writerow(["frame", "track_id", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"])
            while True:
                ok, frame = capture.read()
                if not ok: break
                detections = predictor.predict_bgr(frame)
                tracks = tracker.update(detections)
                draw_tracks(frame, tracks, classes); writer.write(frame)
                for obj in tracks:
                    rows.writerow([count + 1, obj.track_id, obj.class_id, classes[obj.class_id - 1], obj.score, *obj.box])
                count += 1
    finally:
        capture.release(); writer.release()
    elapsed = time.perf_counter() - start
    result = {"source": args.video, "output_video": str(video_out), "tracks_csv": str(csv_path),
              "frames": count, "source_fps": fps, "processing_seconds": elapsed,
              "processing_fps": count / max(elapsed, 1e-9), "device": str(device)}
    video_out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
