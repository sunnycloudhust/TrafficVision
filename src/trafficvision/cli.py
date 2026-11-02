import argparse
import csv
import json
from pathlib import Path
from .config import load_config


def write_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="TrafficVision: detection, tracking and calibrated speed")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--model", choices=["faster_rcnn", "yolov8", "fast_rcnn", "rcnn", "rtdetr"])
    train.add_argument("--init", choices=["random", "imagenet", "pretrained"])
    train.add_argument("--resume")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", default="outputs/evaluation/detection.json")
    demo = sub.add_parser("demo")
    demo.add_argument("--source", required=True)
    demo.add_argument("--model", required=True)
    demo.add_argument("--checkpoint")
    demo.add_argument("--config", help="YOLO weights/label_space or detector config")
    demo.add_argument("--tracker", choices=["sort", "bytetrack", "botsort"], default="bytetrack")
    demo.add_argument("--tracker-config")
    demo.add_argument("--camera-config")
    demo.add_argument("--zone-config")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--output", required=True)
    demo.add_argument("--max-frames", type=int)
    demo.add_argument("--timestamp-mode", choices=["fps", "pts"], default="fps")
    demo.add_argument("--video-id", help="Must match XML sequence name when evaluating tracking")
    tracking = sub.add_parser("evaluate-tracking")
    tracking.add_argument("--ground-truth", required=True, help="Canonical frames.jsonl")
    tracking.add_argument("--predictions", required=True, help="Concatenated demo CSV, one header")
    tracking.add_argument("--split", default="test", choices=["val", "test"])
    tracking.add_argument("--output", default="outputs/evaluation/tracking.json")
    track_data = sub.add_parser("track-dataset", help="Run and evaluate all sequences in a canonical split")
    track_data.add_argument("--config", required=True)
    track_data.add_argument("--model", required=True)
    track_data.add_argument("--checkpoint", required=True)
    track_data.add_argument("--tracker", choices=["sort", "bytetrack", "botsort"], default="bytetrack")
    track_data.add_argument("--tracker-config")
    track_data.add_argument("--split", choices=["val", "test"], default="test")
    track_data.add_argument("--device", default="cpu")
    track_data.add_argument("--fps", type=float, default=25)
    track_data.add_argument("--output", required=True, help="CSV path; JSON and metrics written alongside")
    speed = sub.add_parser("evaluate-speed")
    speed.add_argument("--pairs", required=True, help="CSV with matched gt_kmh,predicted_kmh columns")
    speed.add_argument("--output", default="outputs/evaluation/speed.json")
    smoke = sub.add_parser("smoke", help="Synthetic functional check; not real model quality evaluation")
    smoke.add_argument("--output", default="outputs/smoke")
    args = parser.parse_args(argv)
    if args.command == "train":
        from .training.trainer import train

        train(load_config(args.config), args.model, args.init, args.resume)
    elif args.command == "evaluate":
        from .registry import load_detector
        from .training.trainer import data_loader
        from .evaluation.detection_metrics import evaluate_detector

        config = load_config(args.config)
        detector = load_detector(args.model, args.checkpoint, config, args.device, confidence=0.0)
        result = evaluate_detector(detector, data_loader(config, args.split))
        result.update(model=args.model, checkpoint=args.checkpoint, split=args.split)
        write_result(result, args.output)
    elif args.command == "demo":
        from .registry import load_detector, build_tracker
        from .geometry.calibration import load_calibration
        from .zones.zone_manager import ZoneManager
        from .pipeline.detection_tracking import DetectionTrackingPipeline
        from .pipeline.video_pipeline import run_video

        camera = load_config(args.camera_config) if args.camera_config else {}
        zones = ZoneManager(load_config(args.zone_config) if args.zone_config else {})
        tracker_path = args.tracker_config or f"configs/trackers/{args.tracker}.yaml"
        settings = load_config(tracker_path)
        detector = load_detector(
            args.model,
            args.checkpoint,
            load_config(args.config) if args.config else {},
            args.device,
            confidence=0.01,
        )
        tracker = build_tracker(args.tracker, settings, camera.get("fps", 25))
        pipeline = DetectionTrackingPipeline(
            detector, tracker, zones, load_calibration(camera), camera.get("speed")
        )
        print(
            json.dumps(
                run_video(
                    args.source,
                    args.output,
                    pipeline,
                    camera,
                    args.max_frames,
                    args.timestamp_mode,
                    args.video_id,
                ),
                indent=2,
            )
        )
    elif args.command == "evaluate-tracking":
        from .evaluation.tracking_metrics import tracking_metrics

        write_result(tracking_metrics(args.ground_truth, args.predictions, args.split), args.output)
    elif args.command == "track-dataset":
        from .registry import load_detector, build_tracker
        from .pipeline.detection_tracking import DetectionTrackingPipeline
        from .pipeline.dataset_pipeline import track_dataset
        from .evaluation.tracking_metrics import tracking_metrics

        config = load_config(args.config)
        detector = load_detector(args.model, args.checkpoint, config, args.device, confidence=0.01)
        settings = load_config(args.tracker_config or f"configs/trackers/{args.tracker}.yaml")
        pipeline = DetectionTrackingPipeline(detector, build_tracker(args.tracker, settings, args.fps))
        report = track_dataset(config, args.split, pipeline, args.output, args.fps)
        report.update(tracking_metrics(report["ground_truth"], report["predictions"], args.split))
        report.update(model=args.model, tracker=args.tracker)
        write_result(report, Path(args.output).with_suffix(".metrics.json"))
    elif args.command == "evaluate-speed":
        from .evaluation.speed_metrics import speed_metrics

        with open(args.pairs) as handle:
            rows = list(csv.DictReader(handle))
        write_result(
            speed_metrics([float(r["gt_kmh"]) for r in rows], [float(r["predicted_kmh"]) for r in rows]),
            args.output,
        )
    elif args.command == "smoke":
        from .smoke import run_smoke

        print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == "__main__":
    main()
