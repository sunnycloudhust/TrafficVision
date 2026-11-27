"""Cross-check the published aggregate training history against stored source logs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
history = list(csv.DictReader((ROOT / "outputs/evaluation/training_loss_history.csv").open()))
mismatches = []
for row in history:
    source = ROOT / row["source_file"]
    epoch = int(row["epoch"])
    if source.suffix == ".jsonl":
        logs = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        record = next(item for item in logs if item["epoch"] + 1 == epoch)
        values = [record["loss"], record["metrics"]["map_50"], record["metrics"]["map"]]
    else:
        with source.open(newline="") as handle:
            record = next(item for item in csv.DictReader(handle) if int(float(item["epoch"])) == epoch)
        values = [sum(float(record[key]) for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss")),
                  float(record["metrics/mAP50(B)"]), float(record["metrics/mAP50-95(B)"])]
    for key, actual in zip(("train_loss", "val_map_50", "val_map_50_95"), values):
        if abs(float(row[key]) - actual) > 0.0000051:
            mismatches.append((row["model"], epoch, key, row[key], actual))

print(f"Checked {len(history)} rows; {len(mismatches)} mismatched fields")
for mismatch in mismatches:
    print(mismatch)
if mismatches:
    raise SystemExit(1)
