"""Run one online tracker over saved DETRAC detections and evaluate tracks."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import DETRACDataset
from src.evaluation.tracking_metrics import evaluate_tracking
from src.utils import ensure_dir, ensure_parent_dir, load_yaml, project_path
from .trackers import build_tracker


def run(config_path: Path, *, tracker_override=None, detections_dir_override=None,
        output_dir_override=None) -> dict:
    config = load_yaml(config_path)

    tracker_name = (tracker_override or config.get("tracker", "sort")).lower()
    detections_dir = project_path(detections_dir_override or config.get(
        "detections_dir", "outputs/evaluation_faster_rcnn_pretrained/detections"))
    dataset_root = project_path(config.get("dataset_root", "dataset/DETRAC"))
    output_root = project_path(output_dir_override or config.get("output_dir", "outputs/tracking"))
    output_dir = ensure_dir(output_root / tracker_name)
    dataset = DETRACDataset(root=dataset_root, split="test", include_empty=True)

    records_by_sequence = defaultdict(list)
    for record in dataset.records:
        records_by_sequence[record["sequence"]].append(record)
    class_names = list(dataset.class_names)
    tracker_options = dict(config.get("tracker_options", {}))
    minimum_score = float(config.get("minimum_score", 0.05))
    per_sequence = {}

    for sequence in sorted(records_by_sequence):
        csv_path = detections_dir / f"{sequence}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing detections for {sequence}: {csv_path}. Run `python main.py evaluate` first."
            )
        detections = defaultdict(list)
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                score = float(row["score"])
                if score < minimum_score:
                    continue
                x, y, width, height = (float(row[key]) for key in ("x", "y", "width", "height"))
                class_id = int(row["class_id"])
                if not 1 <= class_id <= len(class_names):
                    raise ValueError(f"Invalid class_id {class_id} in {csv_path}")
                if width <= 0 or height <= 0:
                    continue
                detections[int(row["frame"])].append({
                    "box": [x, y, x + width, y + height], "score": score,
                    "class_id": class_id,
                })

        tracker = build_tracker(tracker_name, **tracker_options)
        gt_frames, pred_frames = [], []
        output_path = output_dir / f"{sequence}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "track_id", "x", "y", "width", "height",
                             "score", "class_id", "class_name"])
            for record in records_by_sequence[sequence]:
                frame = int(record["frame_number"])
                current = detections.get(frame, [])
                boxes = np.asarray([item["box"] for item in current], dtype=np.float32).reshape(-1, 4)
                scores = np.asarray([item["score"] for item in current], dtype=np.float32)
                labels = np.asarray([item["class_id"] for item in current], dtype=np.int64)
                image = None
                if tracker_name == "botsort":
                    with Image.open(record["image_path"]) as pil_image:
                        image = np.asarray(pil_image.convert("RGB"))
                tracks = tracker.update(boxes, scores, labels, image=image)
                for item in tracks:
                    x1, y1, x2, y2 = item["box"].tolist()
                    class_id = int(item["class_id"])
                    writer.writerow([frame, item["track_id"], x1, y1, x2 - x1, y2 - y1,
                                     item["score"], class_id, class_names[class_id - 1]])
                gt_frames.append({"boxes": record["boxes"], "labels": record["labels"],
                                  "track_ids": record["track_ids"]})
                pred_frames.append({
                    "boxes": [item["box"] for item in tracks],
                    "labels": [item["class_id"] for item in tracks],
                    "track_ids": [item["track_id"] for item in tracks],
                })
        per_sequence[sequence] = evaluate_tracking(
            gt_frames, pred_frames, iou_threshold=float(config.get("iou_threshold", 0.5)))
        print(f"{sequence}: {per_sequence[sequence]}")

    metric_names = ("hota", "idf1", "mota")
    summary = {
        "tracker": tracker_name,
        "sequences": len(per_sequence),
        "macro_average": {key: float(np.mean([metrics[key] for metrics in per_sequence.values()]))
                          if per_sequence else 0.0 for key in metric_names},
        "id_switches": int(sum(metrics["id_switches"] for metrics in per_sequence.values())),
        "iou_threshold": float(config.get("iou_threshold", 0.5)),
        "per_sequence": per_sequence,
        "detections_dir": str(detections_dir),
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["macro_average"], indent=2))
    print(f"Tracks: {output_dir}/*.csv")
    print(f"Metrics: {result_path}")
    return summary


def run_all(config_path: Path, *, detections_dir_override=None, output_dir_override=None) -> dict:
    """Run every supported tracker into a separate output directory."""
    results = {}
    for tracker_name in ("sort", "bytetrack", "botsort", "ocsort"):
        print(f"\n=== {tracker_name} ===")
        results[tracker_name] = run(
            config_path,
            tracker_override=tracker_name,
            detections_dir_override=detections_dir_override,
            output_dir_override=output_dir_override,
        )
    config = load_yaml(config_path)
    base_dir = project_path(output_dir_override or config.get("output_dir", "outputs/tracking"))
    comparison = {
        "metric_note": "Unweighted sequence macro averages; compare within this dataset and setup.",
        "trackers": {
            name: {"macro_average": result["macro_average"],
                  "id_switches": result["id_switches"], "sequences": result["sequences"]}
            for name, result in results.items()
        },
    }
    comparison_path = base_dir / "comparison.json"
    ensure_parent_dir(comparison_path)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Comparison: {comparison_path}")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tracking.yaml"))
    parser.add_argument("--tracker", choices=("sort", "bytetrack", "botsort", "ocsort"))
    parser.add_argument("--detections-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(project_path(args.config), tracker_override=args.tracker,
        detections_dir_override=args.detections_dir, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
