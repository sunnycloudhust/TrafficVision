"""Evaluate COCO Faster R-CNN only after an explicit COCO-name to dataset-name mapping."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import build_dataset, detection_collate
from src.evaluation.detection_metrics import compute_detection_metrics
from src.utils.config import choose_device, load_yaml


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, help="JSON mapping COCO label names to configured class names")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config); root = Path(args.config).resolve().parent.parent
    ds = load_yaml(root / cfg["dataset_config"]); classes = ds["class_names"]
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    categories = weights.meta["categories"]
    category_to_dataset = {}
    for label_id, source_name in enumerate(categories):
        target = mapping.get(source_name)
        if target is not None:
            if target not in classes: raise ValueError(f"Mapping target {target!r} is not in configured classes {classes}")
            category_to_dataset[label_id] = classes.index(target) + 1
    if not category_to_dataset:
        raise ValueError("Mapping has no category names present in the Torchvision COCO weights")
    device = choose_device(cfg.get("device", "auto")); model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    model.roi_heads.score_thresh = 0.0
    dataset = build_dataset(ds, "test", training=False)
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 1)), shuffle=False,
                        num_workers=int(cfg.get("num_workers", 0)), collate_fn=detection_collate)
    predictions, targets = [], []
    for images, batch_targets in loader:
        results = model([x.to(device) for x in images])
        for out in results:
            labels = out["labels"].detach().cpu()
            keep = torch.tensor([int(x) in category_to_dataset for x in labels], dtype=torch.bool)
            out = {key: value.detach().cpu()[keep] for key, value in out.items()}
            out["labels"] = torch.tensor([category_to_dataset[int(x)] for x in labels[keep]], dtype=torch.int64)
            predictions.append(out)
        targets.extend({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in t.items()} for t in batch_targets)
    metrics = compute_detection_metrics(predictions, targets, classes,
                                        float(cfg.get("confidence_threshold", .5)), float(cfg.get("iou_threshold", .5)))
    result = {"experiment": "COCO pretrained, no fine-tuning", "source_to_target_class_mapping": mapping,
              "metrics": metrics, "checkpoint": weights.url}
    output = root / cfg.get("output_dir", "outputs/metrics"); output.mkdir(parents=True, exist_ok=True)
    (output / "coco_pretrained_only_detection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
