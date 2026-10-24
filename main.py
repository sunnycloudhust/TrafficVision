"""TrafficVision command line for training, evaluating and tracking DETRAC."""
from __future__ import annotations

import argparse
from src.utils import PROJECT_ROOT

TRAIN_CONFIG = PROJECT_ROOT / "configs/train.yaml"
TEST_CONFIG = PROJECT_ROOT / "configs/test.yaml"
TRACKING_CONFIG = PROJECT_ROOT / "configs/tracking.yaml"
MODEL_ARCHITECTURES = {
    "faster_rcnn": "fasterrcnn_resnet50_fpn",
    "fast_rcnn": "fast_rcnn",
    "rcnn": "rcnn",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate and track DETRAC vehicle detections."
    )
    parser.add_argument(
        "--command",
        choices=("models", "trackers", "train", "evaluate", "track", "track-all"),
        help="Command to run."
        )
    parser.add_argument(
        "--model", 
        choices=tuple(MODEL_ARCHITECTURES), help="Detector architecture to use."
        )
    parser.add_argument(
        "--initialization", choices=("scratch", "pretrained")
        )
    parser.add_argument(
        "--tracker", choices=("sort", "bytetrack", "botsort", "ocsort"),
        help="Tracking algorithm to use with --track.",
        )
    
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "models":
        print("Supported detector models:")
        print("  faster_rcnn  - Faster R-CNN ResNet-50-FPN (RPN proposals)")
        print("  fast_rcnn    - Fast R-CNN with Selective Search proposals")
        print("  rcnn         - R-CNN with independent region crops and Selective Search")
        return

    if args.command == "trackers":
        print("Supported trackers:")
        print("  sort      - constant-velocity motion and IoU association")
        print("  bytetrack - high/low confidence two-pass association")
        print("  botsort   - ByteTrack association with camera compensation and color cues")
        print("  ocsort    - observation-centric direction association")
        return

    if args.command == "track":
        from src.tracking.run_tracking import run as run_tracking
        run_tracking(TRACKING_CONFIG, tracker_override=args.tracker)
        return

    if args.command == "track-all":
        from src.tracking.run_tracking import run_all
        run_all(TRACKING_CONFIG)
        return

    if args.command == "train":
        from src.training.train_detector import train as train_detector

        checkpoint = train_detector(
            TRAIN_CONFIG,
            architecture=MODEL_ARCHITECTURES.get(args.model) if args.model else None,
            initialization=args.initialization,
        )
        print(f"Best checkpoint: {checkpoint}")
        return

    if args.command == "evaluate":
        from src.evaluation.evaluate_detector import run as evaluate_detector

        model_name = args.model or "faster_rcnn"
        initialization = args.initialization or "pretrained"
        expected_architecture = MODEL_ARCHITECTURES.get(args.model) if args.model else None
        checkpoint = PROJECT_ROOT / f"outputs/detrac_{model_name}_{initialization}/best.pth"
        evaluate_detector(
            TEST_CONFIG,
            checkpoint_override=checkpoint,
            expected_architecture=expected_architecture,
        )
        return

if __name__ == "__main__":
    main()
