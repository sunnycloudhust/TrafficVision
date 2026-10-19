"""Evaluate a checkpoint on the untouched official test sequences."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import build_dataset, detection_collate
from src.evaluation.evaluator import evaluate_detector
from src.models.loading import load_model_checkpoint
from src.utils.config import choose_device, load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-config")
    args = parser.parse_args()
    cfg = load_yaml(args.config); base = Path(args.config).resolve().parent.parent
    ds_cfg = load_yaml(base / (args.dataset_config or cfg["dataset_config"]))
    checkpoint_path = Path(args.checkpoint or cfg["checkpoint"])
    device = choose_device(cfg.get("device", "auto"))
    model, state = load_model_checkpoint(checkpoint_path, device)
    classes = state["model_config"]["class_names"]
    if classes != ds_cfg["class_names"]:
        raise ValueError("Checkpoint and test dataset class mappings differ")
    dataset = build_dataset(ds_cfg, "test", training=False)
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 1)), shuffle=False,
                        num_workers=int(cfg.get("num_workers", 0)), collate_fn=detection_collate)
    predictions, targets, metrics = evaluate_detector(model, loader, classes, device,
                                                       float(cfg.get("confidence_threshold", .5)),
                                                       float(cfg.get("iou_threshold", .5)))
    output_dir = base / cfg.get("output_dir", "outputs/metrics"); output_dir.mkdir(parents=True, exist_ok=True)
    name = checkpoint_path.parent.name or checkpoint_path.stem
    (output_dir / f"{name}_detection.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    raw = []
    for target, pred in zip(targets, predictions):
        raw.append({"sequence": target["sequence"], "frame": target["frame"],
                    "boxes": pred["boxes"].tolist(), "labels": pred["labels"].tolist(), "scores": pred["scores"].tolist()})
    (output_dir / f"{name}_predictions.json").write_text(json.dumps(raw), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
