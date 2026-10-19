"""Shared evaluation interface for Faster R-CNN checkpoints."""
from src.evaluation.detection_metrics import compute_detection_metrics, predict_dataset


def evaluate_detector(model, loader, class_names, device, confidence_threshold=.5, iou_threshold=.5):
    """Return raw predictions, targets and aggregate/per-class detection metrics."""
    predictions, targets = predict_dataset(model, loader, device)
    metrics = compute_detection_metrics(predictions, targets, class_names, confidence_threshold, iou_threshold)
    return predictions, targets, metrics
