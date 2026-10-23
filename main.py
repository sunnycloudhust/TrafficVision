"""TrafficVision command line for training, evaluating and tracking DETRAC."""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ARCHITECTURES = {
    "faster_rcnn": "fasterrcnn_resnet50_fpn",
    "fast_rcnn": "fast_rcnn",
    "rcnn": "rcnn",
}


def _project_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _model_output_name(model: str) -> str:
    return {"faster_rcnn": "faster_rcnn", "fast_rcnn": "fast_rcnn", "rcnn": "rcnn"}[model]


def _add_model_option(parser, required=False):
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_ARCHITECTURES),
        required=required,
        help="Detector architecture to use.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate and track DETRAC vehicle detections."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="List supported detector architectures.")

    train_parser = commands.add_parser("train", help="Train one detector.")
    _add_model_option(train_parser)
    train_parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    train_parser.add_argument("--initialization", choices=("scratch", "pretrained"))
    train_parser.add_argument("--output-dir", type=Path)

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a checkpoint on DETRAC Test.")
    _add_model_option(evaluate_parser)
    evaluate_parser.add_argument("--config", type=Path, default=Path("configs/test.yaml"))
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--output-dir", type=Path)

    pipeline_parser = commands.add_parser(
        "pipeline", help="Train a detector, then evaluate its best checkpoint on DETRAC Test."
    )
    _add_model_option(pipeline_parser)
    pipeline_parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    pipeline_parser.add_argument("--test-config", type=Path, default=Path("configs/test.yaml"))
    pipeline_parser.add_argument("--initialization", choices=("scratch", "pretrained"))
    pipeline_parser.add_argument("--train-output-dir", type=Path)
    pipeline_parser.add_argument("--evaluation-output-dir", type=Path)

    track_parser = commands.add_parser("track", help="Track detector detections on DETRAC Test.")
    track_parser.add_argument("--tracker", choices=("sort", "bytetrack", "botsort", "ocsort"))
    track_parser.add_argument("--config", type=Path, default=Path("configs/tracking.yaml"))
    track_parser.add_argument("--detections-dir", type=Path)
    track_parser.add_argument("--output-dir", type=Path)

    track_all_parser = commands.add_parser(
        "track-all", help="Run and compare SORT, ByteTrack, BoT-SORT and OC-SORT."
    )
    track_all_parser.add_argument("--config", type=Path, default=Path("configs/tracking.yaml"))
    track_all_parser.add_argument("--detections-dir", type=Path)
    track_all_parser.add_argument("--output-dir", type=Path)

    commands.add_parser("trackers", help="List supported tracking algorithms.")

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

        run_tracking(
            _project_path(args.config), tracker_override=args.tracker,
            detections_dir_override=_project_path(args.detections_dir) if args.detections_dir else None,
            output_dir_override=_project_path(args.output_dir) if args.output_dir else None,
        )
        return

    if args.command == "track-all":
        from src.tracking.run_tracking import run_all

        run_all(
            _project_path(args.config),
            detections_dir_override=_project_path(args.detections_dir) if args.detections_dir else None,
            output_dir_override=_project_path(args.output_dir) if args.output_dir else None,
        )
        return

    if args.command == "train":
        from src.training.train_detector import train as train_detector

        checkpoint = train_detector(
            _project_path(args.config),
            architecture=MODEL_ARCHITECTURES.get(args.model) if args.model else None,
            initialization=args.initialization,
            output_dir=args.output_dir,
        )
        print(f"Best checkpoint: {checkpoint}")
        return

    if args.command == "evaluate":
        from src.evaluation.evaluate_detector import run as evaluate_detector

        expected_architecture = MODEL_ARCHITECTURES.get(args.model) if args.model else None
        checkpoint = _project_path(args.checkpoint) if args.checkpoint else None
        if checkpoint is None and args.model:
            checkpoint = PROJECT_ROOT / f"outputs/detrac_{_model_output_name(args.model)}" / "best.pth"
        output_dir = _project_path(args.output_dir) if args.output_dir else None
        evaluate_detector(
            _project_path(args.config),
            checkpoint_override=checkpoint,
            output_dir_override=output_dir,
            expected_architecture=expected_architecture,
        )
        return

    if args.command == "pipeline":
        from src.evaluation.evaluate_detector import run as evaluate_detector
        from src.training.train_detector import train as train_detector

        architecture = MODEL_ARCHITECTURES.get(args.model) if args.model else None
        checkpoint = train_detector(
            _project_path(args.train_config),
            architecture=architecture,
            initialization=args.initialization,
            output_dir=args.train_output_dir,
        )
        evaluate_detector(
            _project_path(args.test_config),
            checkpoint_override=checkpoint,
            output_dir_override=args.evaluation_output_dir,
            expected_architecture=architecture,
        )


if __name__ == "__main__":
    main()
