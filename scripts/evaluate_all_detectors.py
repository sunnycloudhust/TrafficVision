"""Evaluate all available detection checkpoints on the same local GPU and generate comparison tables."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

RUNS = [
    ("faster_rcnn_scratch", "faster_rcnn", "configs/experiments/faster_rcnn_scratch.yaml", "outputs/faster_rcnn/best.pt", None),
    ("faster_rcnn_imagenet", "faster_rcnn", "configs/experiments/faster_rcnn_imagenet.yaml", "outputs/faster_rcnn_imagenet/best.pt", None),
    ("rtdetr_scratch", "rtdetr", "configs/experiments/rtdetr_scratch.yaml", "outputs/rtdetr/best.pt", None),
    ("yolov8_coco", "yolov8", "configs/experiments/yolov8_coco.yaml", "yolov8n.pt", None),
    ("yolov8_finetuned", "yolov8", "configs/experiments/yolov8_baseline.yaml", "outputs/yolov8/train/weights/best.pt", None),
    ("fast_rcnn_scratch", "fast_rcnn", "configs/experiments/fast_rcnn_scratch.yaml", "outputs/fast_rcnn/best.pt", 200),
    ("rcnn_scratch", "rcnn", "configs/experiments/rcnn_scratch.yaml", "outputs/rcnn/best.pt", 400),
]


def sync_ultralytics_runs(root: Path):
    """If Ultralytics saved runs to ~/runs/detect/outputs/yolov8/train, copy into outputs/yolov8/train."""
    target = root / "outputs" / "yolov8" / "train"
    fallback = Path.home() / "runs" / "detect" / "outputs" / "yolov8" / "train"
    if not (target / "weights" / "best.pt").exists() and (fallback / "weights" / "best.pt").exists():
        print(f"Syncing YOLOv8 weights from {fallback} -> {target}")
        shutil.copytree(fallback, target, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if output JSON already exists")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sync_ultralytics_runs(root)

    reports = []
    for name, model_name, config, ckpt, max_batches in RUNS:
        ckpt_path = root / ckpt
        if ckpt != "yolov8n.pt" and not ckpt_path.exists():
            print(f"[SKIP] {name}: checkpoint not found at {ckpt}")
            continue
        out_json = root / "outputs" / "evaluation" / f"{name}.json"
        if out_json.exists() and not args.force:
            print(f"[DONE] {name}: reusing existing evaluation at {out_json.relative_to(root)}")
            reports.append(str(out_json.relative_to(root)))
            continue
        print(f"\n=== Evaluating {name} ({ckpt}) on split={args.split}, device={args.device} ===")
        cmd = [
            sys.executable,
            str(root / "scripts" / "run_trafficvision.py"),
            "evaluate",
            "--model",
            model_name,
            "--config",
            config,
            "--checkpoint",
            ckpt,
            "--split",
            args.split,
            "--device",
            args.device,
            "--output",
            str(out_json.relative_to(root)),
        ]
        if max_batches is not None:
            cmd.extend(["--max-batches", str(max_batches)])
        subprocess.run(cmd, cwd=root, check=True)
        reports.append(str(out_json.relative_to(root)))

    if reports:
        out_csv = "outputs/evaluation/detection_comparison.csv"
        print(f"\n=== Generating comparison table -> {out_csv} ===")
        subprocess.run(
            [sys.executable, str(root / "scripts" / "compare_runs.py"), *reports, "--output", out_csv],
            cwd=root,
            check=True,
        )
    else:
        print("\nNo checkpoints found to evaluate yet.")


if __name__ == "__main__":
    main()
