"""COCO-style AP and score-thresholded class-aware precision/recall at IoU .5."""

import time
import torch
from torchvision.ops import box_iou
from ..data.ignore import centers_in_regions


def filter_ignored(prediction, target):
    ignored = centers_in_regions(prediction["boxes"], target.get("ignored_regions", torch.empty((0, 4))))
    # A prediction matching a labeled positive takes precedence over ignore regions.
    if len(target["boxes"]) and len(prediction["boxes"]):
        same_class = prediction["labels"][:, None] == target["labels"][None]
        positive = ((box_iou(prediction["boxes"], target["boxes"]) >= 0.5) & same_class).any(1)
        ignored &= ~positive
    return {k: v[~ignored] for k, v in prediction.items()}


def precision_counts(pred, target, threshold=0.25):
    used, tp, fp = set(), 0, 0
    for index in pred["scores"].argsort(descending=True).tolist():
        if pred["scores"][index] < threshold:
            continue
        candidates = [
            j
            for j in range(len(target["boxes"]))
            if j not in used and target["labels"][j] == pred["labels"][index]
        ]
        if candidates:
            ious = box_iou(pred["boxes"][index : index + 1], target["boxes"][candidates])[0]
            best = int(ious.argmax())
            if ious[best] >= 0.5:
                used.add(candidates[best])
                tp += 1
                continue
        fp += 1
    return tp, fp, len(target["boxes"]) - tp


@torch.inference_mode()
def evaluate_detector(detector, loader, max_batches=None, confidence=0.25):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tp = fp = fn = frames = 0
    duration = 0.0
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        predictions, truth = [], []
        for image, target in zip(images, targets):
            start = time.perf_counter()
            prediction = detector.predict(image)
            duration += time.perf_counter() - start
            prediction = filter_ignored(prediction, target)
            predictions.append(prediction)
            truth.append({k: target[k] for k in ("boxes", "labels", "area", "iscrowd")})
            a, b, c = precision_counts(prediction, target, confidence)
            tp, fp, fn = tp + a, fp + b, fn + c
            frames += 1
        metric.update(predictions, truth)
    if not frames:
        raise ValueError("Cannot evaluate an empty dataset")
    result = {k: v.tolist() for k, v in metric.compute().items()}
    result.update(
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        frames=frames,
        prediction_fps=frames / max(duration, 1e-9),
        latency_ms=duration / frames * 1000,
        confidence_threshold=confidence,
        ignore_policy="bbox center; labeled positives take precedence",
    )
    module = getattr(detector, "model", None)
    if not isinstance(module, torch.nn.Module):
        module = getattr(module, "model", None)
    if isinstance(module, torch.nn.Module):
        result["parameter_count"] = sum(p.numel() for p in module.parameters())
        result["parameter_bytes"] = sum(p.numel() * p.element_size() for p in module.parameters())
    result["gpu_peak_memory_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    if hasattr(detector, "label_space"):
        result["label_space"] = detector.label_space
        if detector.label_space == "coco":
            result["mapping_note"] = "COCO car->car, bus->bus, truck->others; no pretrained van category"
    return result
