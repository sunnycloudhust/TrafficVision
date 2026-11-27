"""Aggregate training loss and validation mAP history across all trained models into CSV and PNG plots."""

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

CUSTOM_RUNS = [
    ("Faster R-CNN (Scratch)", "outputs/faster_rcnn/metrics.jsonl"),
    ("Faster R-CNN (ResNet18 ImageNet)", "outputs/faster_rcnn_imagenet/metrics.jsonl"),
    ("RT-DETR (Scratch)", "outputs/rtdetr/metrics.jsonl"),
    ("Fast R-CNN (Scratch)", "outputs/fast_rcnn/metrics.jsonl"),
    ("R-CNN (Scratch)", "outputs/rcnn/metrics.jsonl"),
]

YOLO_CSV = "outputs/yolov8/train/results.csv"


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    series = {}

    for label, rel_path in CUSTOM_RUNS:
        p = root / rel_path
        if not p.exists():
            continue
        epochs, losses, map50s, maps = [], [], [], []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ep = int(rec["epoch"]) + 1
            loss = float(rec["loss"])
            m50 = float(rec["metrics"]["map_50"])
            m50_95 = float(rec["metrics"]["map"])
            epochs.append(ep)
            losses.append(loss)
            map50s.append(m50)
            maps.append(m50_95)
            rows.append(
                {
                    "model": label,
                    "epoch": ep,
                    "train_loss": round(loss, 5),
                    "val_map_50": round(m50, 5),
                    "val_map_50_95": round(m50_95, 5),
                    "lr": rec.get("lr", ""),
                    "source_file": rel_path,
                }
            )
        series[label] = (epochs, losses, map50s, maps)

    yolo_p = root / YOLO_CSV
    if yolo_p.exists():
        epochs, losses, map50s, maps = [], [], [], []
        with yolo_p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r = {k.strip(): v.strip() for k, v in r.items()}
                ep = int(float(r["epoch"]))
                box_l = float(r["train/box_loss"])
                cls_l = float(r["train/cls_loss"])
                dfl_l = float(r["train/dfl_loss"])
                total_l = box_l + cls_l + dfl_l
                m50 = float(r["metrics/mAP50(B)"])
                m50_95 = float(r["metrics/mAP50-95(B)"])
                epochs.append(ep)
                losses.append(total_l)
                map50s.append(m50)
                maps.append(m50_95)
                rows.append(
                    {
                        "model": "YOLOv8n (Fine-tuned)",
                        "epoch": ep,
                        "train_loss": round(total_l, 5),
                        "val_map_50": round(m50, 5),
                        "val_map_50_95": round(m50_95, 5),
                        "lr": r.get("lr/pg0", ""),
                        "source_file": YOLO_CSV,
                    }
                )
        series["YOLOv8n (Fine-tuned)"] = (epochs, losses, map50s, maps)

    csv_path = out_dir / "training_loss_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "epoch", "train_loss", "val_map_50", "val_map_50_95", "lr", "source_file"]
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_loss, ax_rtdetr, ax_map = axes

    for label, (ep, loss, m50, _) in series.items():
        if "RT-DETR" in label:
            ax_rtdetr.plot(ep, loss, marker="o", linewidth=2, color="tab:red", label=label)
        else:
            ax_loss.plot(ep, loss, marker="o", linewidth=2, label=label)
        ax_map.plot(ep, [v * 100 for v in m50], marker="o", linewidth=2, label=label)

    ax_loss.set_title("Training Loss per Epoch (CNN & YOLO Models)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_rtdetr.set_title("Training Loss per Epoch (RT-DETR Hungarian Loss)")
    ax_rtdetr.set_xlabel("Epoch")
    ax_rtdetr.set_ylabel("Hungarian Matching Loss")
    ax_rtdetr.grid(True, alpha=0.3)
    ax_rtdetr.legend(fontsize=9)

    ax_map.set_title("Validation mAP@0.5 (%) per Epoch")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP@0.5 (%)")
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(fontsize=8)

    fig.tight_layout()
    png_path = out_dir / "training_loss_and_map_curves.png"
    fig.savefig(png_path, dpi=160)
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
