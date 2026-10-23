"""Run a trained detector on DETRAC Test and save detections plus metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import DETRACDataset, detection_collate_fn
from src.evaluation import DetectionMetricAccumulator
from src.models import build_model


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(config_path: Path, *, checkpoint_override=None, output_dir_override=None,
        expected_architecture=None) -> dict:
    config = _load_yaml(config_path)
    project_root = Path(__file__).resolve().parents[2]

    def resolve_path(key: str, default: str) -> Path:
        path = Path(config.get(key, default))
        return path if path.is_absolute() else project_root / path

    dataset_root = resolve_path("dataset_root", "dataset/DETRAC")
    configured_checkpoint = checkpoint_override or config.get("checkpoint") or \
        "outputs/detrac_faster_rcnn/best.pth"
    checkpoint_path = Path(configured_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the detector first.")

    device = _resolve_device(str(config.get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint["model_config"])
    class_names = checkpoint["class_names"]
    architecture = model_config.get("architecture", "fasterrcnn_resnet50_fpn")
    if expected_architecture and architecture != expected_architecture:
        raise ValueError(
            f"Requested {expected_architecture}, but checkpoint contains {architecture}"
        )
    model_name = {
        "fasterrcnn_resnet50_fpn": "faster_rcnn",
        "fast_rcnn": "fast_rcnn",
        "rcnn": "rcnn",
    }.get(architecture)
    if model_name is None:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    configured_output = output_dir_override or config.get("output_dir")
    if configured_output:
        output_dir = Path(configured_output)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        output_dir = project_root / f"outputs/evaluation_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Build the architecture without downloading initialization weights; the
    # trained checkpoint immediately supplies every parameter.
    model_config["initialization"] = "scratch"
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = DETRACDataset(root=dataset_root, split="test", class_names=class_names)
    workers = int(config.get("num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        persistent_workers=workers > 0,
    )
    class_id_to_name = {index + 1: name for index, name in enumerate(class_names)}
    save_threshold = float(config.get("save_score_threshold", 0.05))
    metrics = DetectionMetricAccumulator(
        iou_threshold=float(config.get("iou_threshold", 0.5)),
        score_threshold=float(config.get("metric_score_threshold", 0.5)),
    )
    detection_dir = output_dir / "detections"
    detection_dir.mkdir(parents=True, exist_ok=True)
    open_files = {}
    writers = {}
    try:
        for sequence in sorted({record["sequence"] for record in dataset.records}):
            file = (detection_dir / f"{sequence}.csv").open("w", newline="", encoding="utf-8")
            open_files[sequence] = file
            writer = csv.writer(file)
            writer.writerow(["frame", "x", "y", "width", "height", "score", "class_id", "class_name"])
            writers[sequence] = writer

        frame_number = 0
        with torch.inference_mode():
            for images, targets in loader:
                device_images = [image.to(device) for image in images]
                architecture = model_config.get("architecture", "fasterrcnn_resnet50_fpn")
                if architecture in {"fast_rcnn", "rcnn"}:
                    image_paths = [dataset.records[int(target["image_id"])]["image_path"]
                                   for target in targets]
                    predictions = model(device_images, image_paths=image_paths)
                else:
                    predictions = model(device_images)
                for prediction, target in zip(predictions, targets):
                    info = dataset.get_frame_info(int(target["image_id"]))
                    sequence = str(info["sequence"])
                    frame = int(info["frame_number"])
                    metrics.update(prediction, target)
                    boxes = prediction["boxes"].detach().cpu()
                    labels = prediction["labels"].detach().cpu()
                    scores = prediction["scores"].detach().cpu()
                    for box, label, score in zip(boxes, labels, scores):
                        if float(score) < save_threshold:
                            continue
                        x1, y1, x2, y2 = box.tolist()
                        label_id = int(label)
                        writers[sequence].writerow([
                            frame, x1, y1, x2 - x1, y2 - y1,
                            float(score), label_id, class_id_to_name[label_id],
                        ])
                    frame_number += 1
                    if frame_number % 1000 == 0:
                        print(f"Processed {frame_number}/{len(dataset)} test frames")
    finally:
        for file in open_files.values():
            file.close()

    result = metrics.compute()
    result.update({
        "checkpoint": str(checkpoint_path),
        "test_sequences": len({record["sequence"] for record in dataset.records}),
        "test_frames": frame_number,
        "class_names": class_names,
        "iou_threshold": float(config.get("iou_threshold", 0.5)),
        "metric_score_threshold": float(config.get("metric_score_threshold", 0.5)),
    })
    result["ap50_per_class"] = {
        class_id_to_name[int(label)]: ap for label, ap in result["ap50_per_class"].items()
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Per-video detections: {detection_dir}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/test.yaml"))
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
